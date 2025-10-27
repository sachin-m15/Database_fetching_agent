from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import agent  # This imports agent.py and runs get_agent_executor() on startup

# Initialize FastAPI app
app = FastAPI(
    title="AI Database Agent API",
    description="An API for interacting with a Supabase database using natural language.",
    version="1.0.0"
)

# Configure CORS to allow requests from any origin
# This is fine for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., file://, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for the chat query
class ChatQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"message": "AI Database Agent is running. Post to /chat to interact."}

@app.post("/chat")
async def handle_chat(query: ChatQuery = Body(...)):
    """
    Main chat endpoint to interact with the LangChain SQL agent.
    """
    print(f"Received query: {query.query}")
    
    # Check if agent is initialized
    if agent.agent_executor is None:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: The AI agent is not initialized. Check server logs for database connection errors."
        )

    try:
        # Run the query through the agent
        # We use invoke for a simple request/response
        response = agent.agent_executor.invoke({"input": query.query})
        
        # The agent's response is in the 'output' key
        ai_response = response.get("output", "Sorry, I couldn't process that request.")
        
        print(f"Agent response: {ai_response}")
        return {"response": ai_response}

    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
