import sys
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import json
import os
from pydantic_ai.messages import ModelRequest, UserPromptPart

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint that handles chat messages"""
    # Track connection state to prevent double closing
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
            
            # Process message with MCP agent
            try:
                # Notify client that processing has started
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Processing your message..."
                }))
                
                # Process the full message with the agent
                result = await mcp_agent.run(data, message_history=message_history)
                
                # Update message history with the new exchange
                # Add user message to history
                user_message = ModelRequest(parts=[UserPromptPart(content=data)])
                message_history.append(user_message)
                
                # Safely update message history from result
                try:
                    if hasattr(result, 'all_messages') and callable(result.all_messages):
                        # If all_messages() method exists, use it
                        result_messages = result.all_messages()
                        if result_messages:
                            message_history.extend(result_messages)
                except Exception as hist_error:
                    print(f"Error updating message history: {str(hist_error)}")
                
                # Extract response text safely and send back to client
                response_text = extract_response_text(result)
                
                # Send the response back to the client
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": response_text
                }))
                    
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