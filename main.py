import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentinel-RAG Interception Proxy")

# Allow Streamlit frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def mock_token_stream(query: str = None):
    """
    Simulates IBM Granite streaming a response with intra-generation entropy monitoring.
    It streams grounded tokens, and if a poison prompt / ungrounded request is detected,
    simulates a semantic entropy spike mid-sentence.
    """
    q = (query or "").lower()
    is_poison = any(k in q for k in ["poison", "unverified", "forecast", "override", "hallucinate", "hack", "q4"])

    if is_poison:
        safe_context = "Analyzing Celonis event logs... Accessing Q4 draft projections: Vendor contract override values indicate "
        tokens = safe_context.split(" ")
        for token in tokens:
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.18)
        
        await asyncio.sleep(0.3)
        yield "data: [INTERCEPTION: SEMANTIC ENTROPY > \u03c4. ABORTING HALLLUCINATED TOKEN GENERATION.]\n\n"
    else:
        q_text = query if query else "the exact Q3 compliance cycle time for vendor onboarding"
        safe_context = f"According to verified Celonis event logs, query analysis for '{q_text}' confirms a mean cycle time of 4.2 business days with 99.4% SLA compliance."
        tokens = safe_context.split(" ")
        for token in tokens:
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.12)
            
        yield "data: [COMPLETED: GROUND TRUTH VERIFIED]\n\n"

@app.get("/stream")
async def stream_tokens(query: str = Query(None)):
    return StreamingResponse(mock_token_stream(query), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)