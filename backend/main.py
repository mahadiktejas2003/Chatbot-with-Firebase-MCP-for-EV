import sys
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import json
import os
from pydantic_ai.messages import ModelRequest, UserPromptPart
from typing import Dict, Any
from pydantic_ai.messages import ModelResponse, TextPart
import re
import spacy

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_mcp_agent import get_pydantic_ai_agent

# Ensure ProactorEventLoop is used on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for MCP client and agent
mcp_client = None
mcp_agent = None
initialization_lock = asyncio.Lock()
initialization_complete = False

async def initialize_agent():
    """Initialize the MCP client and agent with proper error handling"""
    global mcp_client, mcp_agent, initialization_complete
    
    if initialization_complete:
        return True
        
    async with initialization_lock:
        if initialization_complete:
            return True
            
        try:
            print("Initializing MCP client and agent...")
            mcp_client, mcp_agent = await get_pydantic_ai_agent()
            
            # A small delay to ensure everything is properly loaded
            await asyncio.sleep(1)
            initialization_complete = True
            print("MCP client and agent successfully initialized")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize MCP client and agent: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return False

@app.on_event("startup")
async def startup():
    """Initialize the MCP client and agent during startup"""
    await initialize_agent()

@app.on_event("shutdown")
async def shutdown():
    """Properly clean up resources during shutdown"""
    global mcp_client
    if mcp_client:
        try:
            # Use a timeout to prevent hanging during cleanup
            try:
                await asyncio.wait_for(mcp_client.cleanup(), timeout=5.0)
                print("MCP client successfully cleaned up")
            except asyncio.TimeoutError:
                print("MCP client cleanup timed out after 5 seconds")
            except Exception as e:
                print(f"Error during MCP client cleanup: {e}")
        finally:
            mcp_client = None

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server status."""
    global initialization_complete
    
    if initialization_complete:
        return {"status": "healthy", "mcp_initialized": True}
    else:
        return {"status": "initializing", "mcp_initialized": False}

def extract_response_text(result):
    """
    Extract the response text from agent results in a safe way,
    handling different response formats and types.
    """
    try:
        # Check different attributes where the response might be
        if hasattr(result, 'data') and result.data is not None:
            return str(result.data)
        elif hasattr(result, 'content') and result.content is not None:
            return str(result.content)
        elif hasattr(result, 'text') and result.text is not None:
            return str(result.text)
        elif hasattr(result, 'response') and result.response is not None:
            return str(result.response)
        elif hasattr(result, 'result') and result.result is not None:
            return str(result.result)
        else:
            # Default to string representation as last resort
            return str(result)
    except Exception as e:
        print(f"Error extracting response text: {str(e)}")
        return "I processed your request but had trouble formatting the response."

# Firestore Intent Mapping with expanded vocabulary
FIRESTORE_INTENTS = {
    "create": re.compile(r"create|add|register|insert|new", re.IGNORECASE),
    "read": re.compile(r"show|find|get|search|list|query|check|locate|display|view|fetch", re.IGNORECASE),
    "update": re.compile(r"update|modify|change|edit|revise|book|reserve|set", re.IGNORECASE),
    "delete": re.compile(r"delete|remove|cancel|clear", re.IGNORECASE)
}

# Expanded collection mapping to include keywords and subcollections
FIRESTORE_COLLECTIONS = {
    # Main collections
    "notification": "Notification",
    "owner": "Owner",
    "user": "User",
    "userdetails": "userDetails",
    
    # EV Station related terms (mapped to Owner collection)
    "station": "Owner",
    "charging": "Owner",
    "ev": "Owner", 
    "evs": "Owner",
    "charger": "Owner",
    "ev_station": "Owner",
    "charging_station": "Owner",
    
    # User related terms
    "profile": "User",
    "account": "User",
    "history": "User",
    
    # Payment related terms (mapped to Owner -> Payments subcollection)
    "payment": "Owner",
    "transaction": "Owner",
    "bill": "Owner",
    
    # Rating related terms (mapped to Owner -> Rating subcollection)
    "rating": "Owner",
    "review": "Owner",
    "feedback": "Owner"
}

# EV-specific keywords for intent recognition
EV_KEYWORDS = [
    "station", "charging", "ev", "electric vehicle", "charger", "charge", 
    "slot", "book", "reserve", "energy", "battery", "nearest", "near", 
    "nearby", "available", "free", "occupied", "price", "cost", "rate"
]

# Add location keywords to help with geographic queries
LOCATION_KEYWORDS = ["in", "at", "near", "around", "closest", "nearest"]

async def extract_location(query: str) -> str:
    """Extract location information from a query string."""
    query_lower = query.lower()
    
    # Try to find location after location keywords
    for keyword in LOCATION_KEYWORDS:
        pattern = rf"{keyword}\s+([a-zA-Z\s]+)(?:,|\.|$| for| to)"
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1).strip()
    
    return None

async def extract_email(query: str) -> str:
    """Extract email address from a query string."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, query)
    if match:
        return match.group(0)
    return None

async def enhanced_intent_recognition(query: str) -> Dict[str, Any]:
    """Enhanced intent recognition for EV assistance chatbot queries."""
    # Initialize the result dictionary
    result = {
        "intent": None,
        "collection": None,
        "subcollection": None,
        "location": None,
        "email": None,
        "is_ev_related": False
    }
    
    # Extract location and email
    result["location"] = await extract_location(query)
    result["email"] = await extract_email(query)
    
    # Check if query is EV-related
    query_lower = query.lower()
    result["is_ev_related"] = any(keyword in query_lower for keyword in EV_KEYWORDS)
    
    # Identify intent (CRUD operation)
    for intent_name, pattern in FIRESTORE_INTENTS.items():
        if pattern.search(query_lower):
            result["intent"] = intent_name
            break
    
    # If no intent was found but query is about finding something, default to "read"
    if not result["intent"] and any(word in query_lower for word in ["where", "find", "show", "locate", "search", "get"]):
        result["intent"] = "read"
    
    # Identify collection from the query
    for keyword, collection in FIRESTORE_COLLECTIONS.items():
        if keyword in query_lower:
            result["collection"] = collection
            
            # Identify subcollections based on specific keywords
            if keyword in ["station", "charging", "ev", "evs", "charger", "ev_station", "charging_station"]:
                result["subcollection"] = "EV_Station"
            elif keyword in ["payment", "transaction", "bill"]:
                result["subcollection"] = "Payments"
            elif keyword in ["rating", "review", "feedback"]:
                result["subcollection"] = "Rating"
            elif keyword in ["history"]:
                result["subcollection"] = "History"
            
            break
    
    # If the query is EV-related but no collection was identified, default to Owner/EV_Station
    if result["is_ev_related"] and not result["collection"]:
        result["collection"] = "Owner"
        result["subcollection"] = "EV_Station"
    
    # Default intent to "read" for most EV-related queries if not specified
    if result["is_ev_related"] and not result["intent"]:
        result["intent"] = "read"
    
    print(f"Intent recognition results: {result}")
    return result

async def build_firebase_query(recognition_result: Dict[str, Any], query: str) -> str:
    """Build a structured query for the Firebase MCP server based on intent recognition."""
    intent = recognition_result.get("intent")
    collection = recognition_result.get("collection")
    subcollection = recognition_result.get("subcollection")
    location = recognition_result.get("location")
    email = recognition_result.get("email")
    is_ev_related = recognition_result.get("is_ev_related", False)
    
    # Handle EV Station queries with priority
    if is_ev_related and subcollection == "EV_Station":
        if intent == "read":
            if location:
                # Query for EV stations in a specific location
                return f"Query the Owner collection for EV_Station subcollection where ev_station_location contains '{location}'"
            elif email:
                # Query for EV stations owned by a specific user
                return f"Query the Owner collection where owner_email equals '{email}' and get EV_Station subcollection"
            else:
                # General query for all EV stations
                return "Query the Owner collection and list all documents in the EV_Station subcollection"
        elif intent == "update" and "book" in query.lower():
            # Handle booking a slot
            if email:
                return f"Update the EV_Station subcollection to add '{email}' to the slot array where evs_available is true"
            else:
                return "Update the EV_Station subcollection to reserve a slot"
    
    # Handle other Firestore operations based on intent
    if intent and collection:
        base_query = f"{intent.capitalize()} data from {collection}"
        if subcollection:
            base_query += f" -> {subcollection}"
        if email:
            base_query += f" for user {email}"
        if location:
            base_query += f" near {location}"
        return base_query
    
    # For general queries, just pass the original query to the MCP agent
    return query

async def handle_firestore_query(query: str, mcp_agent, message_history: list) -> Dict[str, Any]:
    """Enhanced handler for Firestore queries that better supports EV-related operations
    and ensures all user queries get a response."""
    recognition_result = await enhanced_intent_recognition(query)
    
    # For EV-related queries or specific Firestore operation queries, try structured query first
    if recognition_result["is_ev_related"] or (recognition_result["intent"] and recognition_result["collection"]):
        # Build specialized Firebase query
        firebase_query = await build_firebase_query(recognition_result, query)
        print(f"Executing Firebase query: {firebase_query}")
        
        try:
            # Execute through MCP agent
            result = await mcp_agent.run(firebase_query, message_history=message_history)
            response_text = extract_response_text(result)
            
            # If response doesn't seem to contain useful data, try with the original query
            if "couldn't find" in response_text.lower() or "invalid" in response_text.lower():
                print("Firebase query returned no results, trying with original query...")
                result = await mcp_agent.run(query, message_history=message_history)
                response_text = extract_response_text(result)
                
            return {"type": "response", "message": response_text}
            
        except Exception as e:
            error_msg = f"Error executing Firestore operation: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            # Don't return error to user, fall through to general query handling
    
    # For general queries or if Firestore query failed, just pass the original query to the MCP agent
    try:
        print("Processing as general query:", query)
        result = await mcp_agent.run(query, message_history=message_history)
        return {"type": "response", "message": extract_response_text(result)}
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return {"type": "error", "message": "I'm having trouble processing your request. Please try rephrasing your question."}

# Import libraries needed for topic classification
import re
from typing import Dict, Any, List, Tuple
from pydantic_ai.messages import ModelRequest, UserPromptPart

# Define topics that the chatbot should understand
EV_TOPICS = {
    "database": ["database", "db", "firestore", "collection", "document", "query", "record", "data", "evs_id", "owner_id", "user_id", "store", "save"],
    "station_info": ["station", "charging", "ev", "evs", "charger", "slot", "energy", "points", "available", "nearest", "location"],
    "user_management": ["user", "owner", "profile", "account", "email", "mobile", "register", "login", "authentication"],
    "payments": ["payment", "transaction", "billing", "cost", "price", "rate", "fee", "charge", "money", "subscribe"],
    "general_ev_info": ["environment", "impact", "benefit", "electric vehicle", "technology", "range", "battery", "sustainable", "green", "emissions", "carbon"],
    "driving": ["drive", "driving", "operate", "charge", "start", "stop", "use", "parking", "acceleration", "brake", "steer", "maintenance"]
}

async def classify_query(query: str) -> Dict[str, float]:
    """Classify the query into topics with confidence scores."""
    query_lower = query.lower()
    topics = {}
    
    # Analyze query for each topic
    for topic, keywords in EV_TOPICS.items():
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        if matches > 0:
            confidence = min(1.0, matches / 3)  # Cap at 1.0, require 3 matches for full confidence
            topics[topic] = confidence
    
    # If no topics were matched, default to general conversation
    if not topics:
        topics["general_conversation"] = 1.0
        
    return topics

async def is_database_query(query: str, topics: Dict[str, float]) -> bool:
    """Determine if this is a database-related query."""
    # Strong database indicators
    db_indicators = ["show me", "find", "list", "get", "display", "retrieve", "query", "search", "lookup"]
    explicit_db = ["from database", "from db", "from firestore", "in our database", "in our db", "in the database", "stored", "records"]
    
    # Check if query contains both station keywords and database indicators
    if "database" in topics and topics["database"] > 0.2:
        return True
        
    # Check for explicit database mentions
    if any(indicator in query.lower() for indicator in explicit_db):
        return True
        
    # Check if query is asking for specific information in a way that suggests database lookup
    if any(indicator in query.lower() for indicator in db_indicators):
        # Only return true if we also have station_info or user_management topics
        if ("station_info" in topics and topics["station_info"] > 0.3) or ("user_management" in topics and topics["user_management"] > 0.3):
            return True
    
    return False

async def create_mcp_prompt(query: str, topics: Dict[str, float]) -> str:
    """Create a prompt for the MCP agent based on topic classification."""
    top_topic = max(topics.items(), key=lambda x: x[1])[0] if topics else "general_conversation"
    
    if await is_database_query(query, topics):
        # For database queries, create a structured prompt
        if "station_info" in topics and topics["station_info"] > 0.3:
            return f"Query the Firestore database for EV station information related to: {query}"
        elif "user_management" in topics and topics["user_management"] > 0.3:
            return f"Query the Firestore database for user information related to: {query}"
        else:
            return f"Query the Firestore database for information related to: {query}"
    
    # For general EV questions, provide context and instruct the LLM
    if top_topic == "general_ev_info":
        return (
            f"This is a general EV information question about: {query}\n"
            f"Provide factual information about EVs and their benefits. Do NOT query the database for this request."
        )
    
    if top_topic == "driving":
        return (
            f"This is a question about how to drive or operate an EV: {query}\n"
            f"Provide helpful information about EV operation. Do NOT query the database for this request."
        )
    
    # Return the original query for general conversation
    return query

async def handle_user_query(query: str, mcp_agent, message_history: list) -> Dict[str, Any]:
    """Enhanced query handler with better topic detection, prompt engineering and contextual awareness."""
    # Step 1: Classify the query to understand the topic
    topics = await classify_query(query)
    print(f"Query topics: {topics}")
    
    # Step 2: Extract important entities from the query
    entities = {
        "location": await extract_location(query),
        "email": await extract_email(query),
        "ev_related": any(keyword in query.lower() for keyword in EV_KEYWORDS)
    }
    print(f"Extracted entities: {entities}")
    
    # Step 3: Determine if this is a database-specific query
    is_db_query = await is_database_query(query, topics)
    
    # Step 4: Create appropriate prompts based on the query type
    if is_db_query:
        # Handle specific database query cases
        if "station_info" in topics and topics["station_info"] > 0.3:
            if entities["location"]:
                firebase_prompt = f"""
                This is a query for EV stations in {entities["location"]}. 
                
                1. Check the 'Owner' collection for documents.
                2. For each owner document, check the 'EV_Station' subcollection.
                3. Filter stations where location contains '{entities["location"]}'.
                4. Return station details including name, availability, energy, and type.

                Format the response with proper Markdown (bold headers, bullet points for stations).
                """
            elif entities["email"]:
                firebase_prompt = f"""
                This is a query for EV stations associated with user {entities["email"]}.
                
                1. Check the 'Owner' collection for documents where owner_email equals '{entities["email"]}'.
                2. For each matching owner, retrieve their 'EV_Station' subcollection.
                3. Return full details of these stations.

                Format the response with proper Markdown (bold headers, bullet points for stations).
                """
            else:
                firebase_prompt = """
                This is a query for all EV stations in the database.
                
                1. Get all documents from the 'Owner' collection.
                2. For each owner document, retrieve the 'EV_Station' subcollection.
                3. Return a well-formatted summary of all stations.

                Format the response with proper Markdown (bold headers, bullet points for stations).
                """
        elif "user_management" in topics and topics["user_management"] > 0.3:
            firebase_prompt = f"""
            This is a query for user information in the database.
            
            1. Check the 'User' collection for user documents.
            2. If a specific email {entities["email"] if entities["email"] else ""} was provided, filter for that user.
            3. Return user details including name, email, and history if available.

            Format the response with proper Markdown (bold headers, bullet points for users).
            """
        elif "owner" in query.lower() or "owners" in query.lower():
            firebase_prompt = """
            This is a query for owner information in the database.
            
            1. Retrieve all documents from the 'Owner' collection.
            2. Return owner details including name, email, and their station information.

            Format the response with proper Markdown (bold headers, bullet points for owners).
            """
        else:
            # General database query
            firebase_prompt = f"""
            This is a general database query about: {query}
            
            1. Determine which collections need to be queried based on the user's request.
            2. Execute the appropriate Firestore queries.
            3. Return the results in a well-formatted, organized manner.

            Format the response with proper Markdown (bold headers, bullet points for results).
            """
        
        print(f"Executing Firebase query: {firebase_prompt}")
        
        try:
            # Send the specialized Firebase prompt to the MCP agent
            result = await mcp_agent.run(firebase_prompt, message_history=message_history)
            response_text = extract_response_text(result)
            
            # If the response indicates an error or no data, try with general knowledge
            if ("couldn't find" in response_text.lower() or "no documents" in response_text.lower() or 
                "error" in response_text.lower() or "failed" in response_text.lower()):
                # Fall back to general knowledge about the topic
                fallback_prompt = f"""
                The database query didn't return useful results. The user asked: "{query}"
                
                Please provide a helpful general response about this topic.
                If this was about EV stations or users, explain that the requested information 
                might not be available in the database.
                
                Format your response nicely using Markdown for headings and emphasis.
                """
                print(f"Using fallback prompt: {fallback_prompt}")
                result = await mcp_agent.run(fallback_prompt, message_history=message_history)
                response_text = extract_response_text(result)
            
            return {"type": "response", "message": response_text}
            
        except Exception as e:
            error_msg = f"Error executing Firestore operation: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            # Fall through to general query handling on error
    
    # Handle non-database queries (general knowledge questions)
    try:
        # Determine the main topic for proper prompt engineering
        main_topic = max(topics.items(), key=lambda x: x[1])[0] if topics else "general_conversation"
        
        if "general_ev_info" in topics and topics["general_ev_info"] > 0.3:
            general_prompt = f"""
            This is a general question about electric vehicles: "{query}"
            
            Provide a comprehensive, informative response about this EV-related topic.
            Do NOT query the database for this request as it's a general knowledge question.
            
            Format your response professionally using Markdown:
            - Use ### for main section headers
            - Use **bold text** for emphasis on important points
            - Use bullet points for lists
            - Organize information into clear sections
            """
        elif "driving" in topics and topics["driving"] > 0.3:
            general_prompt = f"""
            This is a question about how to drive or operate an electric vehicle: "{query}"
            
            Provide step-by-step instructions and practical advice.
            Do NOT query the database for this request as it's a general knowledge question.
            
            Format your response professionally using Markdown:
            - Use ### for main section headers
            - Use **bold text** for emphasis on key instructions
            - Use numbered lists for sequential steps
            - Add safety warnings where appropriate
            """
        else:
            # General conversation prompt
            general_prompt = f"""
            The user asked: "{query}"
            
            Provide a helpful, informative response to this query.
            
            Format your response professionally using Markdown where appropriate:
            - Use ### for main section headers
            - Use **bold text** for emphasis on important points
            - Use bullet points or numbered lists when appropriate
            """
        
        print(f"Processing as general query with prompt: {general_prompt}")
        result = await mcp_agent.run(general_prompt, message_history=message_history)
        return {"type": "response", "message": extract_response_text(result)}
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"type": "error", "message": "I'm having trouble processing your request. Please try rephrasing your question."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for handling chat messages."""
    connection_open = False
    message_history = []
    
    try:
        # Accept the connection
        await websocket.accept()
        connection_open = True
        print("WebSocket connection established")
        
        # Make sure agent is initialized
        if not initialization_complete:
            success = await initialize_agent()
            if not success:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "The chat service failed to initialize. Please try again later."
                }))
                await websocket.close()
                connection_open = False
                return
        
        while True:
            # Receive user message
            data = await websocket.receive_text()
            print(f"Received message: {data}")
            
            if not data or not data.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "I didn't receive any message. Please try again."
                }))
                continue
            
            try:
                # Notify client that processing has started
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Processing your message..."
                }))
                
                # Process message with enhanced query handler
                response = await handle_user_query(data, mcp_agent, message_history)
                
                # Update message history with the new exchange
                user_message = ModelRequest(parts=[UserPromptPart(content=data)])
                message_history.append(user_message)
                
                # Send the response back to the client
                await websocket.send_text(json.dumps(response))
                    
            except Exception as e:
                error_msg = f"Error processing message '{data}': {str(e)}"
                print(f"{error_msg}\n{traceback.format_exc()}")
                
                if connection_open:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "I encountered an error processing your request. Please try again."
                        }))
                    except Exception:
                        # Connection might be closed
                        connection_open = False
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        connection_open = False
    except Exception as e:
        print(f"Error in WebSocket connection: {e}\n{traceback.format_exc()}")
        if connection_open:
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "A connection error occurred. Please refresh and try again."
                }))
            except Exception:
                pass
    finally:
        # Only try to close if the connection is still open
        if connection_open:
            try:
                await websocket.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Find an available port
    port = 8000
    max_port = 8020
    
    while port <= max_port:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(0.2)
            test_socket.bind(('0.0.0.0', port))
            test_socket.close()
            
            print(f"Starting backend server on port {port}")
            uvicorn.run(app, host="0.0.0.0", port=port)
            break
        except OSError:
            print(f"Port {port} is not available")
            port += 1
        except Exception as e:
            print(f"Unexpected error when starting server: {e}")
            traceback.print_exc()
            break
    
    if port > max_port:
        print(f"Could not find an available port in range {8000}-{max_port}.")