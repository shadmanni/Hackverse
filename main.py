import asyncio
from fastapi import FastAPI, Query, HTTPException, Request, Response, status, Depends, Header, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from entropy_engine import EntropyEngine

app = FastAPI(title="Sentinel-RAG Interception Proxy")

# Allow Streamlit frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EntropyEngine(threshold_tau=0.65, window_size=5)

async def mock_token_stream(query: str = None):
    """
    Simulates IBM Granite streaming a response with intra-generation entropy monitoring.
    It streams grounded tokens, and evaluates each token using Riddhi's EntropyEngine.
    If an ungrounded token or poison prompt causes an entropy spike, the circuit breaker trips.
    """
    engine.reset()
    q = (query or "").lower()
    is_poison = any(k in q for k in ["poison", "unverified", "forecast", "override", "hallucinate", "hack", "q4"])

    context_history = []
    if is_poison:
        safe_context = "Analyzing Celonis event logs... Accessing Q4 draft projections: Vendor contract override values indicate "
        tokens = safe_context.split(" ")
        for token in tokens:
            is_hallucinating, uncertainty, var = engine.evaluate_token(token, logprob=-0.1, context_history=context_history)
            context_history.append(token)
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.18)
        
        # Simulate poison / hallucinated token logprob drop
        poison_token = "42.8_days_unverified_$5M"
        is_hallucinating, uncertainty, var = engine.evaluate_token(poison_token, logprob=-2.85, context_history=context_history)
        await asyncio.sleep(0.3)
        yield f"data: [INTERCEPTION: SEMANTIC ENTROPY ({uncertainty:.2f}) > \u03c4 (0.65). ABORTING HALLUCINATED TOKEN GENERATION.]\n\n"
    else:
        q_text = query if query else "the exact Q3 compliance cycle time for vendor onboarding"
        safe_context = f"According to verified Celonis event logs, query analysis for '{q_text}' confirms a mean cycle time of 4.2 business days with 99.4% SLA compliance."
        tokens = safe_context.split(" ")
        for token in tokens:
            is_hallucinating, uncertainty, var = engine.evaluate_token(token, logprob=-0.05, context_history=context_history)
            context_history.append(token)
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.12)
            
        yield "data: [COMPLETED: GROUND TRUTH VERIFIED]\n\n"

@app.get("/stream")
async def stream_tokens(query: str = Query(None)):
    return StreamingResponse(mock_token_stream(query), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
