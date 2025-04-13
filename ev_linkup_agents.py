"""
EV LinkUp Agents System - Multi-agent architecture for handling EV-related queries and operations.

This module implements a modular agent system for the EV LinkUp platform, providing:
1. Intent routing - Determines whether queries are Firestore CRUD or general knowledge
2. Firestore schema exploration - Dynamically discovers the Firestore schema
3. Specialized agent components for different operations
4. Response formatting - Converts technical outputs to user-friendly text
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import asyncio
import json
import re
import os
import logging
from contextlib import AsyncExitStack
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import traceback

# Import Pydantic AI components
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

# Import our custom MCP client
import mcp_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ev_linkup_agents")

# Load environment variables
load_dotenv()

# Define models for our agent system
class Intent(BaseModel):
    """Represents the detected intent of a user query"""
    intent_type: str = Field(..., description="Type of intent: 'firestore_crud', 'general_ev_info', 'help', 'unknown'")
    confidence: float = Field(..., description="Confidence score for the intent classification, 0.0-1.0")
    collection: Optional[str] = Field(None, description="Target Firestore collection, if applicable")
    subcollection: Optional[str] = Field(None, description="Target Firestore subcollection, if applicable")
    operation: Optional[str] = Field(None, description="CRUD operation: 'create', 'read', 'update', or 'delete'")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter criteria for Firestore queries")
    entities: Optional[Dict[str, Any]] = Field(None, description="Extracted entities from user query")

class FirestoreDocument(BaseModel):
    """Represents a Firestore document structure"""
    path: str = Field(..., description="Path to the document in Firestore")
    fields: Dict[str, Any] = Field(..., description="Fields and values in the document")
    subcollections: Optional[List[str]] = Field(None, description="Subcollections under this document")

class EVLinkupAgentSystem:
    """Main agent system orchestrating multiple specialized agents"""
    
    def __init__(self):
        """Initialize the multi-agent system"""
        self.mcp_client = None
        self.model = self._get_model()
        self.agents = {}
        self.firestore_schema = {}
        self.console = Console()
        self.last_schema_refresh = None
        
        # Agent prompts - keeping them as class attributes for easy updating
        self.INTENT_ROUTER_PROMPT = """
        You are an Intent Classification agent for the EV LinkUp platform. Your job is to analyze user messages and determine:
        
        1. What type of query is this? (Firestore operation, general EV information, help request, etc.)
        2. If it's a Firestore operation, which collection and subcollection does it target?
        3. What operation is being requested (create, read, update, delete)
        
        EV LinkUp has these main Firestore collections:
        - Owner: Information about charging station owners
          - Subcollections: EV_Station, Payments, Rating, Notify
        - User: Information about platform users
          - Subcollections: History
        - Notification: System notifications
        - userDetails: User financial and personal details
          - Subcollections: Food Expense, Salary, etc.
        
        Provide your analysis as a structured output with: intent_type, confidence, collection (if applicable), 
        subcollection (if applicable), operation (if applicable), and any extracted entities.
        """
        
        self.FIRESTORE_EXPLORER_PROMPT = """
        You are a Firestore Schema Explorer agent for the EV LinkUp platform. Your job is to:
        
        1. Use MCP tools to discover the structure of Firestore collections
        2. Map user requests to the correct collections, documents, and fields
        3. Generate the appropriate Firestore query structure
        
        When called, you should first explore the schema if needed, then determine the exact query structure.
        
        Current known collections:
        - Owner (with subcollections: EV_Station, Payments, Rating, Notify)
        - User (with subcollection: History)
        - Notification
        - userDetails (with subcollections: Food Expense, Salary)
        
        Return your analysis as structured data with the appropriate collection paths, query parameters, and operations.
        """
        
        self.FIRESTORE_AGENT_PROMPT = """
        You are a Firestore Operation agent for the EV LinkUp platform. Your job is to:
        
        1. Execute Firestore CRUD operations via MCP tools
        2. Handle queries for EV stations, user information, payments, ratings, etc.
        3. Return formatted results or confirmation messages
        
        You have access to these collections:
        - Owner (with subcollections: EV_Station, Payments, Rating, Notify)
        - User (with subcollection: History)
        - Notification
        - userDetails (with subcollections: Food Expense, Salary)
        
        For operations, always first check if the collection/subcollection exists,
        then execute the operation, and finally format the output for the user.
        """
        
        self.GENERAL_EV_AGENT_PROMPT = """
        You are an EV Knowledge agent for the EV LinkUp platform. Your job is to:
        
        1. Answer general questions about electric vehicles and charging
        2. Explain how the EV LinkUp platform works
        3. Provide step-by-step guidance on using EV charging stations
        
        Focus on being helpful, informative, and accurate. Use conversational language.
        Format responses with clear sections using Markdown (headers, bullet points, etc.).
        
        Do NOT try to access Firestore data for general information questions.
        """
        
        self.FORMATTER_AGENT_PROMPT = """
        You are a Response Formatter agent for the EV LinkUp platform. Your job is to:
        
        1. Take raw outputs (especially data from Firestore) and transform them into user-friendly messages
        2. Ensure consistent style and formatting with Markdown
        3. Convert technical language into conversational responses
        
        ALWAYS format with Markdown:
        - Use ### for main section headers
        - Use **bold text** for field names or important information
        - Use bullet points for lists of items
        - Use numbered lists for step-by-step instructions
        
        Respond as if you are having a conversation - be helpful and friendly.
        """
    
    def _get_model(self) -> OpenAIModel:
        """Get the LLM model based on environment variables"""
        llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
        base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')
        
        return OpenAIModel(
            llm,
            base_url=base_url,
            api_key=api_key
        )
    
    async def initialize(self) -> None:
        """Initialize the agent system and MCP client"""
        try:
            logger.info("Initializing EV LinkUp Agent System...")
            
            # Initialize MCP client
            self.mcp_client = mcp_client.MCPClient()
            
            # Load configuration from the standard location
            import pathlib
            SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
            CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"
            self.mcp_client.load_servers(str(CONFIG_FILE))
            
            # Start MCP servers and get tools
            tools = await self.mcp_client.start()
            
            if not tools:
                logger.warning("No tools were loaded from MCP servers")
                return False
            
            logger.info(f"Successfully loaded {len(tools)} tools from MCP servers")
            
            # Create specialized agents with shared tools
            # This ensures all agents have access to the same MCP tools
            self.agents = {
                "intent_router": Agent(
                    model=self.model, 
                    tools=tools,
                    system_prompt=self.INTENT_ROUTER_PROMPT
                ),
                "firestore_explorer": Agent(
                    model=self.model, 
                    tools=tools,
                    system_prompt=self.FIRESTORE_EXPLORER_PROMPT
                ),
                "firestore_agent": Agent(
                    model=self.model, 
                    tools=tools,
                    system_prompt=self.FIRESTORE_AGENT_PROMPT
                ),
                "general_ev_agent": Agent(
                    model=self.model, 
                    tools=tools,
                    system_prompt=self.GENERAL_EV_AGENT_PROMPT
                ),
                "formatter_agent": Agent(
                    model=self.model, 
                    tools=tools,
                    system_prompt=self.FORMATTER_AGENT_PROMPT
                ),
            }
            
            # Initial schema refresh
            await self.refresh_firestore_schema()
            
            logger.info("EV LinkUp Agent System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def refresh_firestore_schema(self) -> None:
        """
        Refresh the Firestore schema by exploring collections and subcollections
        This should be called periodically to keep the schema up to date
        """
        try:
            logger.info("Refreshing Firestore schema...")
            
            # Use the firestore_explorer agent to discover the schema
            # We'll use the agent's ability to call tools to do this
            schema_prompt = """
            Please explore the Firestore database structure and return a complete schema.
            
            1. First, list all top-level collections using firestore_list_collections
            2. For each collection, list a sample of documents (limit 5)
            3. For each document, check if it has subcollections and list them
            4. For each subcollection, list a sample document to understand its structure
            
            Return the full schema as a structured JSON object. Format the schema as:
            {
                "collection_name": {
                    "documents": {
                        "sample_doc_id": {
                            "fields": {"field1": "value1", "field2": "value2"},
                            "subcollections": {
                                "subcollection_name": {
                                    "documents": {...}
                                }
                            }
                        }
                    }
                }
            }
            """
            
            result = await self.agents["firestore_explorer"].run(schema_prompt)
            
            # Try to extract JSON from the response
            try:
                # Look for a JSON structure in the response
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result.data)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no markdown code block, try to find JSON directly
                    json_str = re.search(r'(\{[\s\S]*\})', result.data).group(1)
                
                schema = json.loads(json_str)
                self.firestore_schema = schema
                logger.info(f"Firestore schema refreshed with {len(schema)} collections")
                
                # Update the system prompts with the new schema information
                self._update_agent_prompts()
                
                self.last_schema_refresh = datetime.now()
            except Exception as e:
                logger.error(f"Failed to parse Firestore schema: {e}")
                logger.debug(f"Schema response was: {result.data}")
        
        except Exception as e:
            logger.error(f"Error refreshing Firestore schema: {e}")
            logger.error(traceback.format_exc())
    
    def _update_agent_prompts(self) -> None:
        """Update the agent prompts with the latest schema information"""
        
        # Generate schema description
        schema_description = "Current Firestore collections:\n"
        
        for collection, details in self.firestore_schema.items():
            schema_description += f"- {collection}"
            
            # Add sample fields if available
            sample_doc = next(iter(details.get("documents", {}).values()), None)
            if sample_doc and "fields" in sample_doc:
                fields = list(sample_doc["fields"].keys())[:5]  # Limit to 5 fields
                if fields:
                    schema_description += f" (fields: {', '.join(fields)})"
            
            # Add subcollections
            subcollections = []
            if sample_doc and "subcollections" in sample_doc:
                subcollections = list(sample_doc["subcollections"].keys())
            
            if subcollections:
                schema_description += f" (subcollections: {', '.join(subcollections)})"
            
            schema_description += "\n"
        
        # Update the agent prompts with the new schema information
        self.FIRESTORE_EXPLORER_PROMPT = self.FIRESTORE_EXPLORER_PROMPT.replace(
            "Current known collections:", 
            f"Current known collections (as of {datetime.now().strftime('%Y-%m-%d %H:%M')}):\n{schema_description}"
        )
        
        self.FIRESTORE_AGENT_PROMPT = self.FIRESTORE_AGENT_PROMPT.replace(
            "You have access to these collections:", 
            f"You have access to these collections (as of {datetime.now().strftime('%Y-%m-%d %H:%M')}):\n{schema_description}"
        )
        
        # Refresh agents with updated prompts
        if self.agents:
            self.agents["firestore_explorer"].system_prompt = self.FIRESTORE_EXPLORER_PROMPT
            self.agents["firestore_agent"].system_prompt = self.FIRESTORE_AGENT_PROMPT
    
    async def detect_intent(self, query: str, message_history: List = None) -> Intent:
        """
        Analyze the user query to determine intent
        
        Args:
            query: The user's query text
            message_history: Previous conversation messages
        
        Returns:
            Intent object with classification details
        """
        try:
            # Check if we need to refresh the schema (once per day)
            if (self.last_schema_refresh is None or 
                (datetime.now() - self.last_schema_refresh).total_seconds() > 86400):
                await self.refresh_firestore_schema()
            
            # Prepare a specific prompt for intent detection
            intent_prompt = f"""
            Analyze this user query and classify its intent: "{query}"
            
            Return your analysis as a JSON object with these fields:
            - intent_type: 'firestore_crud', 'general_ev_info', 'help', or 'unknown'
            - confidence: A score from 0.0 to 1.0 indicating confidence in the classification
            - collection: The target Firestore collection (if applicable)
            - subcollection: The target Firestore subcollection (if applicable)
            - operation: The CRUD operation ('create', 'read', 'update', or 'delete')
            - entities: Any extracted entities like locations, emails, amounts, etc.
            
            Example response:
            ```json
            {
                "intent_type": "firestore_crud",
                "confidence": 0.92,
                "collection": "Owner",
                "subcollection": "EV_Station",
                "operation": "read",
                "entities": {
                    "location": "Mumbai",
                    "email": null
                }
            }
            ```
            
            Return ONLY the JSON object without any additional text.
            """
            
            # Use the intent router agent to analyze
            result = await self.agents["intent_router"].run(intent_prompt, message_history=message_history)
            
            # Extract JSON from the response
            try:
                # Try to find JSON in a markdown code block first
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result.data)
                if json_match:
                    intent_data = json.loads(json_match.group(1))
                else:
                    # Otherwise try to parse the whole response as JSON
                    intent_data = json.loads(result.data)
                
                # Create an Intent model
                intent = Intent(**intent_data)
                logger.info(f"Intent detected: {intent.intent_type} (confidence: {intent.confidence:.2f})")
                return intent
                
            except Exception as e:
                logger.error(f"Failed to parse intent response: {e}")
                logger.debug(f"Intent response was: {result.data}")
                
                # Return default intent as fallback
                return Intent(
                    intent_type="unknown",
                    confidence=0.0,
                    operation=None,
                    collection=None,
                    subcollection=None,
                    entities={"error": str(e)}
                )
                
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            logger.error(traceback.format_exc())
            
            # Return default intent as fallback
            return Intent(
                intent_type="unknown",
                confidence=0.0,
                operation=None,
                collection=None,
                subcollection=None,
                entities={"error": str(e)}
            )
    
    async def process_firestore_query(self, intent: Intent, query: str, message_history: List = None) -> str:
        """
        Process a Firestore CRUD query
        
        Args:
            intent: The detected intent
            query: The original user query
            message_history: Previous conversation messages
        
        Returns:
            The formatted response
        """
        try:
            # Step 1: Use the explorer agent to build the precise Firestore query
            collection_path = intent.collection
            if intent.subcollection:
                # We need to first find a valid document ID in the parent collection
                # This requires explorer agent to determine the right approach
                collection_path += "/" + intent.subcollection
            
            # We need to tell the explorer agent about our intent
            explorer_prompt = f"""
            I need to perform a {intent.operation} operation on {collection_path}.
            
            User query: "{query}"
            
            Please determine:
            1. The exact Firestore collection/subcollection path to use
            2. The specific MCP tool to call
            3. The parameters to provide to the tool
            
            If this is a subcollection query, you may need to:
            - First query the parent collection to get valid document IDs
            - Then access the subcollection with the correct path (e.g., "Owner/doc_id/EV_Station")
            
            Return a structured JSON with:
            - tool_name: The MCP tool to use
            - parameters: The parameters to pass to the tool
            
            Important: Be very specific with document paths and ensure they are complete.
            If the operation fails with FAILED_PRECONDITION, we need to verify the exact document path exists.
            """
            
            explorer_result = await self.agents["firestore_explorer"].run(explorer_prompt, message_history=message_history)
            
            # Step 2: Use the Firestore agent to execute the query with error handling
            # We'll pass the explorer's response to guide the Firestore agent
            firestore_prompt = f"""
            Execute the following Firestore operation:
            
            User query: "{query}"
            Intent: {intent.operation} on {collection_path}
            
            Explorer agent's analysis:
            {explorer_result.data}
            
            Please:
            1. Execute the appropriate Firestore operation using MCP tools
            2. Return the results or confirmation
            3. If the operation fails with FAILED_PRECONDITION, try to:
               - Verify the document path exists first by querying the collection
               - Check if the collection exists using firestore_list_collections
               - If updating a field that doesn't exist, try creating it first
            4. Provide clear error feedback if operations cannot be completed
            
            Format your response appropriately with any retrieved data.
            """
            
            firestore_result = await self.agents["firestore_agent"].run(firestore_prompt, message_history=message_history)
            
            # Check if the result contains error messages and handle accordingly
            if "FAILED_PRECONDITION" in firestore_result.data or "Error" in firestore_result.data:
                # Try alternate approach with more direct collection access
                recovery_prompt = f"""
                It seems there was an error accessing the Firebase data. Let's try a different approach:
                
                1. First, list all collections to make sure we have the right names
                2. For the collection "{intent.collection}", try to retrieve documents without filters first
                3. If this works, then apply the specific filters or operations from the original query: "{query}"
                4. If we still get errors, provide a clear explanation of what might be wrong with the database setup
                
                Remember to handle FAILED_PRECONDITION errors by checking if the collections and documents actually exist.
                """
                
                recovery_result = await self.agents["firestore_agent"].run(recovery_prompt, message_history=message_history)
                firestore_result = recovery_result
            
            # Step 3: Use the formatter agent to make the response user-friendly
            formatter_prompt = f"""
            The user asked: "{query}"
            
            The Firestore operation result is:
            {firestore_result.data}
            
            Please format this response in a user-friendly way:
            1. Use clear, conversational language
            2. Structure with Markdown for readability
            3. Highlight important information
            4. Remove any technical jargon or internal references
            5. If there's an error, explain it simply and suggest alternatives
            6. If the operation wasn't possible due to database issues, suggest manual steps the user could take
            
            Your response should be complete and helpful to the user.
            """
            
            formatter_result = await self.agents["formatter_agent"].run(formatter_prompt)
            return formatter_result.data
            
        except Exception as e:
            logger.error(f"Error processing Firestore query: {e}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error when trying to access the database. The Firebase MCP server might be experiencing issues. You may want to check your database configuration or try again later. Error details: {str(e)}"
    
    async def process_general_ev_query(self, query: str, message_history: List = None) -> str:
        """
        Process a general EV knowledge query
        
        Args:
            query: The user's query
            message_history: Previous conversation messages
        
        Returns:
            The formatted response
        """
        try:
            # Use the general EV agent to answer
            general_prompt = f"""
            Please answer this question about electric vehicles or the EV LinkUp platform:
            
            "{query}"
            
            Provide helpful, accurate information. Format your response using Markdown for readability.
            Do NOT attempt to access any database information for this answer - use your general knowledge.
            """
            
            result = await self.agents["general_ev_agent"].run(general_prompt, message_history=message_history)
            return result.data
            
        except Exception as e:
            logger.error(f"Error processing general EV query: {e}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error when trying to answer your question: {str(e)}"
    
    async def process_help_request(self, query: str, message_history: List = None) -> str:
        """
        Process a help/guidance request
        
        Args:
            query: The user's query
            message_history: Previous conversation messages
        
        Returns:
            The formatted response
        """
        try:
            # Use the general EV agent with specific help focus
            help_prompt = f"""
            The user is asking for help or guidance with:
            
            "{query}"
            
            Please provide clear, step-by-step assistance about the EV LinkUp platform features.
            Format your response using Markdown with headings, bullet points, and numbered steps as appropriate.
            Focus on being friendly, helpful, and straightforward.
            """
            
            result = await self.agents["general_ev_agent"].run(help_prompt, message_history=message_history)
            return result.data
            
        except Exception as e:
            logger.error(f"Error processing help request: {e}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error when trying to provide help: {str(e)}"
    
    async def handle_query(self, query: str, message_history: List = None) -> str:
        """
        Main entry point for handling user queries
        
        Args:
            query: The user's query text
            message_history: Previous conversation messages (optional)
        
        Returns:
            The agent's response
        """
        try:
            # Step 1: Detect the intent of the query
            intent = await self.detect_intent(query, message_history)
            
            # Step 2: Route to the appropriate processing function based on intent
            if intent.intent_type == "firestore_crud" and intent.confidence > 0.6:
                return await self.process_firestore_query(intent, query, message_history)
                
            elif intent.intent_type == "general_ev_info" and intent.confidence > 0.6:
                return await self.process_general_ev_query(query, message_history)
                
            elif intent.intent_type == "help" and intent.confidence > 0.6:
                return await self.process_help_request(query, message_history)
                
            else:
                # If confidence is low or intent is unknown, use the general EV agent as fallback
                fallback_prompt = f"""
                I'm not entirely sure what you're asking about, but I'll try to help with:
                
                "{query}"
                
                Please provide helpful information or suggest what the user might be looking for.
                If this appears to be a database query but you're not sure, clearly explain that to the user.
                """
                
                result = await self.agents["general_ev_agent"].run(fallback_prompt, message_history=message_history)
                return result.data
                
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
    
    async def cleanup(self):
        """Clean up resources when shutting down"""
        if self.mcp_client:
            try:
                await self.mcp_client.cleanup()
            except Exception as e:
                logger.error(f"Error during MCP client cleanup: {e}")

# Utility functions for use with the agent system

def format_conversation_history(messages: List[Dict[str, Any]]) -> List:
    """
    Format conversation history from Supabase into the format expected by Pydantic AI
    
    Args:
        messages: List of message dictionaries from Supabase
        
    Returns:
        List of Pydantic AI message objects
    """
    formatted_messages = []
    
    for msg in messages:
        msg_data = msg["message"]
        msg_type = msg_data["type"] 
        msg_content = msg_data["content"]
        
        if msg_type == "human":
            formatted_messages.append(ModelRequest(parts=[UserPromptPart(content=msg_content)]))
        else:
            formatted_messages.append(ModelResponse(parts=[TextPart(content=msg_content)]))
    
    return formatted_messages

async def get_agent_system():
    """Initialize and return the agent system instance"""
    agent_system = EVLinkupAgentSystem()
    success = await agent_system.initialize()
    
    if not success:
        raise RuntimeError("Failed to initialize the EV LinkUp agent system")
    
    return agent_system