



# main.py
#
# This script creates a FastAPI web server to expose the academic article
# analysis agent as an API endpoint.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import the compiled LangGraph app from your agent.py file
from agent import app as article_agent_app
from agent import AgentState # Import the state definition for type hinting

# --- API Data Models ---

class QueryRequest(BaseModel):
    """Defines the structure of the incoming request body."""
    query: str
    
    class Config:
        schema_extra = {
            "example": {
                "query": 'check this article and give me information on "amino acid Mutation", "Name of the pathogen" and "Name of the drug" the url is https://journals.asm.org/doi/10.1128/aac.00637-15'
            }
        }

class QueryResponse(BaseModel):
    """Defines the structure of the successful response body."""
    answer: str


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Academic Article Analysis Agent",
    description="An API that uses a LangGraph agent to retrieve and analyze scientific articles from a URL, handling paywalls by searching open-access sources.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


# --- API Endpoint Definition ---

@app.post("/analyze-article", response_model=QueryResponse)
async def analyze_article_endpoint(request: QueryRequest):
    """
    Receives a query with a URL, processes it with the LangGraph agent,
    and returns the analysis.
    """
    # The initial state for the agent is the user's query.
    initial_state: AgentState = {"original_question": request.query}
    
    print(f"Received request to analyze query: {request.query}")
    
    try:
        # Invoke the LangGraph agent synchronously. For long-running tasks,
        # you might use background tasks, but this is fine for this use case.
        final_state = article_agent_app.invoke(initial_state)

        # Check the final state for errors reported by the agent
        if error := final_state.get("execution_error"):
            print(f"Agent execution failed with error: {error}")
            raise HTTPException(status_code=400, detail=f"Agent Error: {error}")

        # On success, return the result
        result = final_state.get("execution_result", "No result was produced.")
        return {"answer": result}

    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

# --- Main Execution Block (for direct running) ---

if __name__ == "__main__":
    # This allows you to run the server directly using `python main.py`
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

