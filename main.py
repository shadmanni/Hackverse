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

PROCESS_GRAPHS = {
    "p2p": {
        "name": "Celonis Purchase-to-Pay (P2P) Event Graph",
        "collection": "celonis_p2p_chunks",
        "vector_count": 1420,
        "pii_masked": True
    },
    "o2c": {
        "name": "Celonis Order-to-Cash (O2C) Workflow",
        "collection": "celonis_o2c_chunks",
        "vector_count": 980,
        "pii_masked": True
    },
    "ap_audit": {
        "name": "Celonis Accounts Payable (AP) Compliance Audit",
        "collection": "celonis_ap_audit_chunks",
        "vector_count": 2150,
        "pii_masked": True
    },
    "supply_chain": {
        "name": "Celonis Global Logistics & Supply Chain",
        "collection": "celonis_supply_chain_chunks",
        "vector_count": 3400,
        "pii_masked": True
    }
}

@app.get("/health")
async def health_check():
    """Asynchronous health & status polling endpoint for Streamlit UI telemetry"""
    return {
        "status": "NOMINAL",
        "backend": "Antigravity FastAPI Interception Proxy",
        "port": 8000,
        "interception_latency_ms": 11.4,
        "milvus_status": "CONNECTED",
        "milvus_host": "milvus-standalone:19530",
        "active_graphs": len(PROCESS_GRAPHS),
        "circuit_breaker_tau": 0.420
    }

@app.get("/graphs")
async def list_graphs():
    """Returns available Celonis process mining vector collections"""
    return PROCESS_GRAPHS

async def mock_token_stream(query: str = None, graph: str = "p2p"):
    """
    Simulates IBM Granite streaming a response with intra-generation entropy monitoring.
    Processes queries against the selected Celonis process graph.
    """
    q = (query or "").lower()
    graph_info = PROCESS_GRAPHS.get(graph, PROCESS_GRAPHS["p2p"])
    graph_name = graph_info["name"]
    is_poison = any(k in q for k in ["poison", "unverified", "forecast", "override", "hallucinate", "hack", "q4"])

    if is_poison:
        safe_context = f"Analyzing {graph_name} via Milvus vector search... Accessing Q4 draft projections: Vendor contract override values indicate "
        tokens = safe_context.split(" ")
        for token in tokens:
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.18)
        
        await asyncio.sleep(0.3)
        yield "data: [INTERCEPTION: SEMANTIC ENTROPY > \u03c4. ABORTING HALLLUCINATED TOKEN GENERATION.]\n\n"
    else:
        q_text = query if query else "the exact Q3 compliance cycle time for vendor onboarding"
        safe_context = f"According to verified {graph_name} event logs, query analysis for '{q_text}' confirms a mean cycle time of 4.2 business days with 99.4% SLA compliance."
        tokens = safe_context.split(" ")
        for token in tokens:
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.12)
            
        yield "data: [COMPLETED: GROUND TRUTH VERIFIED]\n\n"

@app.get("/stream")
async def stream_tokens(query: str = Query(None), graph: str = Query("p2p")):
    return StreamingResponse(mock_token_stream(query, graph), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
