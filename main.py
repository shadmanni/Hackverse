import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Streamlit frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def mock_token_stream():
    """
    Simulates IBM Granite streaming a response.
    It streams a factual sentence, then simulates an entropy spike on a hallucination.
    """
    safe_context = "According to the Celonis event logs, the Q3 compliance cycle time for vendor onboarding is exactly "
    tokens = safe_context.split(" ")
    
    # Stream the safe tokens normally
    for token in tokens:
        yield f"data: {token} \n\n"
        await asyncio.sleep(0.3) # Simulate LLM generation latency
        
    # Trigger the simulated firewall interception
    await asyncio.sleep(0.5)
    yield "data: [INTERCEPTION: SEMANTIC ENTROPY > \u03c4. ABORTING HALLLUCINATED TOKEN GENERATION.]\n\n"

@app.get("/stream")
async def stream_tokens():
    return StreamingResponse(mock_token_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)