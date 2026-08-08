import os
import json
import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.credentials import Credentials

load_dotenv()

app = FastAPI(title="Sentinel-RAG Interception Proxy")

# Allow Streamlit frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")

async def live_watsonx_stream(query: str = None):
    """
    Connects to IBM Granite via Watsonx AI, streams tokens and extracts log-probabilities.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        yield "data: [ERROR: Watsonx credentials not found in environment]\n\n"
        return

    q_text = query if query else "the exact Q3 compliance cycle time for vendor onboarding"
    prompt = f"Answer the following query using Celonis event logs: {q_text}\n"

    generate_params = {
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.LOGPROBS: True,
        GenParams.RETURN_OPTIONS: {
            "input_text": False,
            "generated_tokens": True,
            "token_logprobs": True,
            "token_ranks": False,
            "top_n_tokens": False
        }
    }

    credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY
    )

    try:
        model = Model(
            model_id=WATSONX_MODEL_ID,
            params=generate_params,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )

        for chunk in model.generate_stream(prompt=prompt):
            results = chunk.get("results", [])
            if not results:
                continue
            
            result = results[0]
            generated_text = result.get("generated_text", "")
            
            gen_tokens = result.get("generated_tokens", [])
            logprob_val = None
            if gen_tokens:
                logprob_val = gen_tokens[0].get("logprob")
            
            if logprob_val is not None:
                # Phase 3: We will inject mathematical threshold evaluation here
                print(f"[Intercept] Token: '{generated_text}', LogProb: {logprob_val}")
            
            if generated_text:
                yield f"data: {generated_text} \n\n"
            
            await asyncio.sleep(0.01)
            
        yield "data: [COMPLETED: STREAM FINISHED]\n\n"

    except Exception as e:
        print(f"Error calling Watsonx API: {e}")
        yield f"data: [ERROR: {str(e)}]\n\n"

@app.get("/stream")
async def stream_tokens(query: str = Query(None)):
    return StreamingResponse(live_watsonx_stream(query), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)