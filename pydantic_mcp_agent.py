from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from dotenv import load_dotenv
import asyncio
import pathlib
import sys
import os

from pydantic_ai import Agent
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel

import mcp_client

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

load_dotenv()

def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(
        llm,
        base_url=base_url,
        api_key=api_key
    )

async def get_pydantic_ai_agent():
    """
    Create and initialize the MCP agent with proper error handling.
    Returns a tuple of (client, agent) for use in applications.
    """
    try:
        client = mcp_client.MCPClient()
        client.load_servers(str(CONFIG_FILE))
        tools = await client.start()
        
        # Validate tools list
        if not tools:
            print("Warning: No tools were loaded from MCP servers")
        else:
            print(f"Successfully loaded {len(tools)} tools from MCP servers")
            
        # Create the agent with the loaded tools
        agent = Agent(model=get_model(), tools=tools)
        return client, agent
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ Main Function with CLI Chat ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    print("=== Pydantic AI MCP CLI Chat ===")
    print("Type 'exit' to quit the chat")
    
    # Initialize the agent and message history
    try:
        mcp_client, mcp_agent = await get_pydantic_ai_agent()
        console = Console()
        messages = []
        
        try:
            while True:
                # Get user input
                user_input = input("\n[You] ")
                
                # Check if user wants to exit
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("Goodbye!")
                    break
                
                if not user_input.strip():
                    print("Please enter a message.")
                    continue
                
                try:
                    # Process the user input and output the response
                    print("\n[Assistant]")
                    with Live('', console=console, vertical_overflow='visible') as live:
                        async with mcp_agent.run_stream(
                            user_input, message_history=messages
                        ) as result:
                            curr_message = ""
                            async for message in result.stream_text(delta=True):
                                curr_message += message
                                live.update(Markdown(curr_message))
                        
                        # Add the new messages to the chat history
                        messages.extend(result.all_messages())
                    
                except Exception as e:
                    print(f"\n[Error] An error occurred: {str(e)}")
        finally:
            # Ensure proper cleanup of MCP client resources when exiting
            await mcp_client.cleanup()
    except Exception as e:
        print(f"Failed to initialize the agent: {str(e)}")
        print("Please check your configuration settings and try again.")

if __name__ == "__main__":
    asyncio.run(main())