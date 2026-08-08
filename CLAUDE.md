# 🛡️ CLAUDE.md: Sentinel-RAG Architecture & Context Master

**Project Name:** Sentinel-RAG (Enterprise Trust Middleware for Process Intelligence)  
**Track:** AI for Business Transformation  
**Sponsor Integrations:** IBM Granite, IBM Data Prep Kit, IBM watsonx.data, Celonis EMS  
**Development Stack:** IBM Bob (Orchestration & Code Acceleration) + Antigravity Framework (Low-Latency Execution)

---

## 1. PROJECT OVERVIEW & THE CRITICAL ENTERPRISE PROBLEM

### The "Confident Liar" Liability
Major corporations rely on platforms like **Celonis** (Process Mining) to establish absolute, deterministic ground truth—mapping end-to-end digital twins of their ERP workflows, supply chains, and financial transactions. As enterprises build Generative AI Copilots on top of this data, a fundamental architectural mismatch occurs.

**Stochastic Guesswork in Deterministic Domains:** Large Language Models (LLMs) operate on probabilistic next-token prediction mechanisms. When an LLM encounters missing metadata in rigid process graphs, its decoding engine probabilistically fills contextual gaps rather than failing gracefully. This produces "hallucinations"—syntactically polished, highly confident statements that are factually false. 

**The Intra-Generation Deficit:** Current RAG guardrails rely almost exclusively on post-generation evaluations (e.g., "LLM-as-a-judge"). Evaluating responses only *after* generation completes introduces severe latency, wastes compute cycles, and allows ungrounded outputs to reach decision engines before intervention occurs.

**Business Impact:**
* **Financial Risk:** Studies show AI hallucinations cost enterprises over **$67 billion globally**. A fabricated metric in an enterprise supply chain can lead to catastrophic operational decisions.
* **Operational Bottleneck:** Employees currently spend an average of **4.3 hours per week** verifying AI outputs, destroying the efficiency gains promised by GenAI.

---

## 2. THE SENTINEL-RAG PARADIGM SHIFT (OUR SOLUTION)

Sentinel-RAG introduces a complete paradigm shift: moving hallucination detection from post-hoc Natural Language Processing (text checking) to **real-time mathematical uncertainty quantification**. We act as an adversarial firewall between the LLM and the enterprise data.

### Core Innovations & Differentiators
1. **The "Circuit Breaker" Architecture:** Current RAG pipelines fail silently. When semantic uncertainty spikes during generation, Sentinel-RAG physically trips a circuit breaker, **halting the generation stream mid-sentence**. It then autonomously routes the failure to an IBM watsonx self-healing agent.
2. **Token-Level Log-Probability Variance:** Instead of waiting for full sentences, we analyze the time-series variance of token log-probabilities. If grounded, log-probabilities are concentrated. If guessing, the distribution flattens. We mathematically catch the model the millisecond it guesses.
3. **Semantic Entropy vs. Token Entropy:** Grounded in 2024 Nature research (*Detecting hallucinations in large language models using semantic entropy*), we measure uncertainty across the *semantic meaning space* to reliably isolate confabulations.
4. **Near-Zero Latency Overhead:** Our C++/CUDA proxy sidecar computes these metrics in parallel during intra-generation decoding, eliminating the latency penalty of standard semantic entropy sampling.

---

## 3. TECHNICAL EXCELLENCE & FEASIBILITY (THE ARCHITECTURE)

Sentinel-RAG is a highly optimized, 5-layer architecture utilizing the best of the IBM ecosystem, orchestrated via **IBM Bob**.

* **Layer 1: IBM Data Prep Kit (Ingestion & Privacy)** - Intercepts raw Celonis EMS event logs. Executes parsing, PII redaction, and semantic chunking to ensure strict data privacy (Zero-Knowledge Compliance).
* **Layer 2: IBM watsonx.data / Milvus (Storage)** - The dense vector database storing ground-truth process metadata.
* **Layer 3: IBM Granite (Generation)** - The primary generative engine (e.g., Granite-13b-chat), chosen for enterprise-grade licensing.
* **Layer 4: Antigravity FastAPI Middleware (Interception)** - The framework-free Python proxy orchestrating Server-Sent Events (SSE) token streams and executing the real-time circuit breaker in sub-15ms.
* **Layer 5: IBM watsonx.ai Agents & Streamlit UI (Recovery)** - The Security Terminal dashboard and autonomous fallback orchestrators triggered by the circuit breaker.

---

## 4. TEAM ROSTER, DOMAIN OWNERSHIP & END GOALS

The team is fully certified under the IBM SkillsBuild "Future Forward: AI for Innovation" program (PLAN-FA448E08E2A9).

* **Shivansh Srivastava** (Lead Middleware & Interception Orchestrator)
  * *Domain:* Antigravity Backend, Socket Management, and FastAPI Proxy.
  * *Tasks:* Construct the high-speed proxy; manage the SSE token stream from IBM Granite; implement the socket-level circuit breaker that drops the connection when a hallucination is detected.
* **Shadman Nishat** (Data Pipeline & Ingestion Lead)
  * *Domain:* IBM Data Prep Kit, Celonis ETL, and Vector Storage.
  * *Tasks:* Synthesize Celonis logs; build PII masking pipeline; integrate embeddings into Milvus. Design "Poison Prompts" to intentionally trigger hallucinations during demos.
* **Riddhi Jain** (Entropy Analytics & Optimization)
  * *Domain:* Mathematical Modeling, Draft Models, and Threshold Calibration.
  * *Tasks:* Translate Semantic Entropy math into an optimized Python engine; calculate predictive variance ($V(y_t)$); calibrate the $\tau$ threshold to eliminate false positives.
* **Shaurya Arvind** (Visualizations & IBM Bob Workflows)
  * *Domain:* Enterprise UI, Fallback Agents, and Streamlit Dashboard.
  * *Tasks:* Build the interactive "Security Terminal" dashboard that visually renders the incoming stream and explicitly highlights the interception events.

---

## 5. ELABORATE PHASE-WISE EXECUTION ROADMAP

### PHASE 1: The "Smoke & Mirrors" MVP (Now – Sat 8:30 PM)
* **Goal:** Deliver a 3-minute visual proof-of-concept making the "invisible" firewall visible.
* **Execution:** Shivansh builds a local FastAPI endpoint simulating an SSE token stream that drops the socket with a `406 Not Acceptable: Entropy Spike` error. Shaurya builds the Streamlit UI to visualize the stream halting mid-sentence. Shadman creates `mock_celonis_data.json`. Riddhi prepares the predictive variance math breakdown.

### PHASE 2: Core Pipeline & Vectorization (Sat 8:30 PM – Sun 2:00 AM)
* **Goal:** Establish genuine data flow from raw synthetic event logs to IBM Granite.
* **Execution:** Connect Antigravity proxy to live IBM Granite API. Feed synthetic JSON logs into IBM Data Prep Kit for PII masking and embed into Milvus. Set up parallel draft model for log-probabilities.

### PHASE 3: The Interception Engine (Sun 2:00 AM – Sun 11:00 AM)
* **Goal:** Live circuit breaker functions accurately on real token variance.
* **Execution:** Merge log-probability calculator into Antigravity streaming loop. If $V(y_t) > \tau$, socket is aggressively closed. Milvus retriever connected to prompt pipeline. Self-healing trigger built into UI for fallback querying.

### PHASE 4: Hardening & Enterprise Polish (Sun 11:00 AM – Sun 6:00 PM)
* **Goal:** Secure "Technical Excellence" and "UX & Design" rubric marks.
* **Execution:** Benchmark proxy loop for < 15ms latency. Finalize demo scripting (factual vs poison prompts). Automate threshold calibration for 0% false positives. Brand the Streamlit dashboard as a deployable SaaS Security Terminal.

---

## 6. HACKATHON EVALUATION STRATEGY

### Evaluation 1 (Saturday 8:30 PM) - The Foundation (40/40 points)
* **Problem Understanding:** Hook judges with the "Confident Liar" liability in deterministic Celonis domains.
* **Innovation & Creativity:** Emphasize the speed shift (mid-stream mathematical entropy vs post-hoc NLP).
* **Feasibility & Planning:** Showcase the IBM Bob + Antigravity stack.
* **Team Coordination:** Flawless handoffs during the pitch; mention IBM SkillsBuild certifications.

### Evaluation 2 (Monday Morning) - The Final Pitch (80/80 points)
* **Working Prototype (25pts):** The Split-Screen Demo. Left: Unprotected Granite hallucinating. Right: Sentinel-RAG catching the hallucination mid-word, cutting the stream, and self-correcting.
* **Problem-Solution Fit (20pts):** The Buyer Narrative. Built for IBM Consulting to derisk Watsonx integrations in highly regulated sectors.
* **Technical Excellence (15pts):** Highlight the IBM Data Prep Kit (PII compliance), Milvus, and sub-15ms Antigravity latency.
* **UX & Design (10pts):** The Security Terminal Feel. Live Semantic Entropy Variance graph flashing on breaches.
* **Impact & Scalability (10pts):** Model-Agnostic. Sentinel-RAG can deploy in front of *any* Granite instance globally without retraining.