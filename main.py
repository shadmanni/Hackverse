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

from entropy_engine import EntropyEngine

engine = EntropyEngine(threshold_tau=0.65, window_size=5)

async def mock_token_stream():
    """
    Simulates IBM Granite streaming a response.
    It streams a factual sentence, then evaluates each token using Riddhi's EntropyEngine.
    When entropy/variance crosses tau, the circuit breaker trips.
    """
    engine.reset()
    safe_context = "According to the Celonis event logs, the Q3 compliance cycle time for vendor onboarding is exactly "
    tokens = safe_context.split(" ")
    
    # Stream the safe tokens normally (simulating high log-probability around -0.05 to -0.15)
    for token in tokens:
        # Factual tokens have high probability
        is_hallucinating, uncertainty, var = engine.evaluate_token(token, logprob=-0.1)
        yield f"data: {token} \n\n"
        await asyncio.sleep(0.3) # Simulate LLM generation latency
        
    # Now simulate a hallucinated token with low log probability (high uncertainty spike)
    hallucinated_token = "42.8_days_and_unverified_vendor_cost_$5M"
    is_hallucinating, uncertainty, var = engine.evaluate_token(hallucinated_token, logprob=-2.85)
    
    if is_hallucinating:
        await asyncio.sleep(0.3)
        yield f"data: [INTERCEPTION: SEMANTIC ENTROPY ({uncertainty:.2f}) > \u03c4 (0.65). ABORTING HALLUCINATED TOKEN GENERATION.]\n\n"
    else:
        yield f"data: {hallucinated_token} \n\n"

@app.get("/stream")
async def stream_tokens():
    return StreamingResponse(mock_token_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)