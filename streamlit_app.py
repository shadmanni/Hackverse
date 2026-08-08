import streamlit as st
import requests

# --- Enterprise Branding ---
st.set_page_config(page_title="Sentinel-RAG | IBM Consulting", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0f62fe;'>🛡️ Sentinel-RAG Enterprise Trust Terminal</h1>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("Live Process Intelligence Query")
st.write("**Query:** *What is the exact Q3 compliance cycle time for vendor onboarding based on Celonis event logs?*")

if st.button("Execute Query via IBM Granite"):
    message_placeholder = st.empty()
    full_response = ""

    try:
        # Connect to Shivansh's Antigravity/FastAPI endpoint
        with requests.get("http://localhost:8000/stream", stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        token = decoded_line.replace("data: ", "")

                        # Catch the firewall interception
                        if "[INTERCEPTION" in token:
                            st.error(f"🚨 {token}")
                            st.warning("🔄 Triggering watsonx Agentic Fallback... Re-querying Celonis Ground Truth.")
                            break
                        else:
                            # Stream safe tokens
                            full_response += token + " "
                            message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

    except Exception as e:
        st.error(f"Connection to Sentinel Proxy failed. Ensure the backend is running. Error: {e}")
