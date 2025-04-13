import sys
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import json
import os
from pydantic_ai.messages import ModelRequest, UserPromptPart
from typing import Dict, Any, List
from pydantic_ai.messages import ModelResponse, TextPart
import re
import logging

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom agent system
from ev_linkup_agents import EVLinkupAgentSystem, format_conversation_history

# Ensure ProactorEventLoop is used on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ev_linkup_backend")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for agent system
agent_system = None
initialization_lock = asyncio.Lock()
initialization_complete = False

async def initialize_agent_system():
    """Initialize the EV LinkUp Agent System with proper error handling"""
    global agent_system, initialization_complete
    
    if initialization_complete:
        return True
        
    async with initialization_lock:
        if initialization_complete:
            return True
            
        try:
            logger.info("Initializing EV LinkUp Agent System...")
            agent_system = EVLinkupAgentSystem()
            success = await agent_system.initialize()
            
            if not success:
                logger.error("Agent system initialization failed")
                return False
                
            # A small delay to ensure everything is properly loaded
            await asyncio.sleep(1)
            initialization_complete = True
            logger.info("EV LinkUp Agent System successfully initialized")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize agent system: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False

@app.on_event("startup")
async def startup():
    """Initialize the agent system during startup"""
    await initialize_agent_system()

@app.on_event("shutdown")
async def shutdown():
    """Properly clean up resources during shutdown"""
    global agent_system
    if agent_system:
        try:
            # Use a timeout to prevent hanging during cleanup
            try:
                await asyncio.wait_for(agent_system.cleanup(), timeout=5.0)
                logger.info("Agent system successfully cleaned up")
            except asyncio.TimeoutError:
                logger.warning("Agent system cleanup timed out after 5 seconds")
            except Exception as e:
                logger.error(f"Error during agent system cleanup: {e}")
        finally:
            agent_system = None

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server status."""
    global initialization_complete
    
    if initialization_complete:
        return {"status": "healthy", "system_initialized": True}
    else:
        return {"status": "initializing", "system_initialized": False}

async def store_message_in_memory(message_history: List, is_user: bool, content: str) -> List:
    """Store a message in the message history list and return the updated list."""
    if is_user:
        message = ModelRequest(parts=[UserPromptPart(content=content)])
    else:
        message = ModelResponse(parts=[TextPart(content=content)])
    
    message_history.append(message)
    return message_history

async def handle_user_query(query: str, message_history: List) -> Dict[str, Any]:
    """Process the user query using the EV LinkUp Agent System."""
    global agent_system
    
    try:
        # Add the user query to message history for context
        message_history = await store_message_in_memory(message_history, True, query)
        
        # Process the query using our agent system
        response = await agent_system.handle_query(query, message_history)
        
        # Add the agent's response to the message history
        message_history = await store_message_in_memory(message_history, False, response)
        
        return {"type": "response", "message": response}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a more user-friendly error message
        error_message = "I'm having trouble processing your request. Please try again or rephrase your question."
        return {"type": "error", "message": error_message}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for handling chat messages."""
    connection_open = False
    message_history = []
    
    try:
        # Accept the connection
        await websocket.accept()
        connection_open = True
        logger.info("WebSocket connection established")
        
        # Make sure agent system is initialized
        if not initialization_complete:
            success = await initialize_agent_system()
            if not success:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "The EV LinkUp service failed to initialize. Please try again later."
                }))
                await websocket.close()
                connection_open = False
                return
        
        while True:
            # Receive user message
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")
            
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
                
                # Process message with agent system
                response = await handle_user_query(data, message_history)
                
                # Send the response back to the client
                await websocket.send_text(json.dumps(response))
                    
            except Exception as e:
                error_msg = f"Error processing message '{data}': {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
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
        logger.info("WebSocket disconnected")
        connection_open = False
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}\n{traceback.format_exc()}")
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
                logger.error(f"Error closing WebSocket: {e}")

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
            
            logger.info(f"Starting EV LinkUp backend server on port {port}")
            uvicorn.run(app, host="0.0.0.0", port=port)
            break
        except OSError:
            logger.info(f"Port {port} is not available")
            port += 1
        except Exception as e:
            logger.error(f"Unexpected error when starting server: {e}")
            traceback.print_exc()
            break
    
    if port > max_port:
        logger.error(f"Could not find an available port in range {8000}-{max_port}.")