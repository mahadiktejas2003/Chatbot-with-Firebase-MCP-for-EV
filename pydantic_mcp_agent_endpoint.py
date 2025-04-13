from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import httpx
import sys
import os
import logging

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

# Import our custom agent system
from ev_linkup_agents import EVLinkupAgentSystem, format_conversation_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ev_linkup_api")

# Load environment variables
load_dotenv()

# Global variable for our agent system
agent_system = None

# Define a lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    global agent_system
    
    try:
        logger.info("Initializing EV LinkUp Agent System...")
        agent_system = EVLinkupAgentSystem()
        await agent_system.initialize()
        logger.info("EV LinkUp Agent System successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agent system: {e}")
        # Continue even with error as some endpoints might not require the agent
    
    yield
    
    # Shutdown: clean up resources
    if agent_system:
        try:
            await agent_system.cleanup()
            logger.info("Agent system successfully cleaned up")
        except Exception as e:
            logger.error(f"Error during agent system cleanup: {e}")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentResponse(BaseModel):
    success: bool

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True    

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.post("/api/ev-linkup-agent", response_model=AgentResponse)
async def ev_linkup_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Check if agent system is initialized
        global agent_system
        if not agent_system:
            raise HTTPException(
                status_code=503, 
                detail="Agent system is not initialized"
            )
        
        # Fetch conversation history
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = format_conversation_history(conversation_history)

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        )        

        # Process the query using our agent system
        result = await agent_system.handle_query(request.query, messages)

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=result,
            data={"request_id": request.request_id}
        )

        return AgentResponse(success=True)

    except Exception as e:
        logger.error(f"Error processing agent request: {str(e)}")
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    global agent_system
    
    if agent_system:
        return {"status": "healthy", "system_initialized": True}
    else:
        return {"status": "degraded", "system_initialized": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)