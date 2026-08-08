# 🛡️ CLAUDE.md: Sentinel-RAG Architecture & Context Master

**Project Name:** Sentinel-RAG (Enterprise Trust Middleware for Process Intelligence)  
**Track:** AI for Business Transformation  
**Sponsor Integrations:** IBM Granite, IBM Data Prep Kit, IBM watsonx.data, Celonis EMS  
**Development Stack:** IBM Bob (Orchestration & Code Acceleration) + Antigravity Framework (Low-Latency Execution)

---

## 1. PROJECT OVERVIEW & THE CRITICAL ENTERPRISE PROBLEM

### The "Confident Liar" Liability
Major corporations rely on platforms like **Celonis** (Process Mining) to establish absolute, deterministic ground truth—mapping end-to-end digital twins of their ERP workflows, supply chains, and financial transactions[cite: 8]. As enterprises build Generative AI Copilots on top of this data[cite: 8], a fundamental architectural mismatch occurs.

**Stochastic Guesswork in Deterministic Domains:** Large Language Models (LLMs) operate on probabilistic (stochastic) next-token prediction mechanisms[cite: 8]. When an LLM is tasked with reasoning over rigid, highly structured process graphs and encounters missing metadata or complex process dependencies, its decoding engine probabilistically fills contextual gaps rather than failing gracefully[cite: 8]. This produces "hallucinations"—syntactically polished, highly confident statements that are factually false[cite: 8].

**The Intra-Generation Deficit:** Current RAG guardrails rely almost exclusively on post-generation evaluations (such as "LLM-as-a-judge" or asynchronous external verification pipelines)[cite: 8]. Because these checks evaluate responses only *after* generation completes, they introduce severe streaming latency, consume unnecessary compute cycles, and allow ungrounded, hallucinated outputs to reach downstream decision engines or end users before intervention occurs[cite: 8].

**Business Impact:**
* **Financial Risk:** Studies show AI hallucinations cost enterprises over **$67 billion globally**[cite: 8]. In consumer apps, a hallucination is a funny typo; in enterprise supply chains or finance, a fabricated number can lead to tremendous losses[cite: 8].
* **Operational Bottleneck:** Employees currently spend an average of **4.3 hours per week** verifying AI outputs due to a lack of trust, creating an expensive operational bottleneck[cite: 8].

---

## 2. THE SENTINEL-RAG PARADIGM SHIFT (OUR SOLUTION)

Sentinel-RAG introduces a complete paradigm shift: moving hallucination detection from post-hoc Natural Language Processing (text checking) to **real-time mathematical uncertainty quantification**[cite: 8]. We act as an adversarial firewall between the LLM and the enterprise data.

### Core Innovations & Differentiators
1. **The "Circuit Breaker" Architecture (Active Interception):** Current RAG pipelines fail silently, passing hallucinations directly to downstream systems[cite: 8]. Our solution flips the "LLM-as-a-judge" model by acting as a true middleware proxy[cite: 8]. When semantic entropy crosses a specific safety threshold, it physically trips a circuit breaker, **halting the generation stream during the decoding phase**[cite: 8]. It does not just flag the error; it stops execution instantly and routes the failure to an autonomous fallback agent[cite: 8].
2. **Token-Level Mathematical Detection (Log-Probability Variance):** Standard models wait for complete sentences to evaluate them, but our approach evaluates the underlying math by looking for large logprobs[cite: 8]. In standard LLM generation, an LLM assigns a probability to every single token it produces[cite: 8]. We analyze the time-series variance of the token log-probabilities[cite: 8]. Sentinel-RAG intercepts data token-by-token[cite: 8]. If the model is grounded in Celonis data, log-probabilities are highly concentrated[cite: 8]. If it starts guessing (extrinsic hallucination), the distribution flattens[cite: 8]. Sentinel mathematically catches the model the millisecond it begins to guess[cite: 8].
3. **Semantic Entropy vs. Token Entropy:** Grounded in recent research (like *Detecting hallucinations in large language models using semantic entropy* [doi:10.1038/s41586-024-07421-0]), semantic entropy measures how much an LLM's responses vary in meaning, rather than just wording, by considering semantic diversity across multiple sampled responses[cite: 8]. By measuring the Shannon entropy of the token distribution on-the-fly, we mathematically differentiate between when the IBM Granite model confidently reasons over process data versus when it is blindly guessing[cite: 8].
4. **Near-Zero Latency Overhead (Production-Grade Speeds):** Calculating semantic entropy via standard sampling imposes a 5x to 10x computational overhead and latency penalty because it requires sampling multiple full responses before evaluating uncertainty[cite: 8]. We used this finding to design Sentinel-RAG's C++/CUDA proxy sidecar[cite: 8]. Instead of waiting for full responses to generate sequentially, we compute log-probabilities and entropy in parallel during intra-generation decoding, giving us the accuracy of semantic entropy with near-zero added latency[cite: 8].

---

## 3. TECHNICAL EXCELLENCE & FEASIBILITY (THE ARCHITECTURE)

Sentinel-RAG is a highly optimized architecture utilizing the best of the IBM ecosystem, orchestrated via **IBM Bob**.

* **Layer 1: IBM Data Prep Kit (Ingestion & Privacy)** - Intercepts raw Celonis EMS event logs. Executes parsing, PII data masking to discover, remove, or hide sensitive information from datasets, and semantic chunking to ensure strict data privacy (Zero-Knowledge Compliance) while still allowing the data to be securely utilized for data analysis and machine learning[cite: 8].
* **Layer 2: IBM watsonx.data / Milvus (Storage)** - The dense vector database storing ground-truth process metadata.
* **Layer 3: IBM Granite (Generation)** - The primary generative engine (e.g., Granite-13b-chat), chosen for enterprise-grade licensing.
* **Layer 4: Antigravity FastAPI Middleware (Interception)** - The framework-free Python proxy orchestrating Server-Sent Events (SSE) token streams and executing the real-time circuit breaker in sub-15ms. Moving hallucination detection to the middleware layer natively supports real-time visibility, enterprise controls, and continuous accountability[cite: 8].
* **Layer 5: IBM watsonx.ai Agents & Streamlit UI (Recovery)** - The Security Terminal dashboard and autonomous fallback orchestrators triggered by the circuit breaker. The proxy architecture directly aligns with enterprise strategies, like IBM watsonx.governance, transforming AI governance into a continuous, feedback-driven process[cite: 8].

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
* **Problem Understanding:** Hook judges with the "Confident Liar" liability in deterministic Celonis domains. Highlight structural vulnerabilities of using secondary LLMs as post-hoc evaluators, proving that post-generation verification models frequently exhibit bias, confirmation errors, and struggle on complex, multi-hop reasoning tasks[cite: 8].
* **Innovation & Creativity:** Emphasize the speed shift (mid-stream mathematical entropy vs post-hoc NLP).
* **Feasibility & Planning:** Showcase the IBM Bob + Antigravity stack. Use findings from research to highlight how Sentinel-RAG's parallel computation achieves semantic entropy accuracy with near-zero added latency, overcoming traditional sampling bottlenecks[cite: 8].
* **Team Coordination:** Flawless handoffs during the pitch; mention IBM SkillsBuild certifications.

### Evaluation 2 (Monday Morning) - The Final Pitch (80/80 points)
* **Working Prototype (25pts):** The Split-Screen Demo. Left: Unprotected Granite hallucinating. Right: Sentinel-RAG catching the hallucination mid-word, cutting the stream, and self-correcting.
* **Problem-Solution Fit (20pts):** The Buyer Narrative. Built for IBM Consulting to derisk Watsonx integrations in highly regulated sectors.
* **Technical Excellence (15pts):** Highlight the IBM Data Prep Kit (PII compliance), Milvus, and sub-15ms Antigravity latency.
* **UX & Design (10pts):** The Security Terminal Feel. Live Semantic Entropy Variance graph flashing on breaches.
* **Impact & Scalability (10pts):** Model-Agnostic. Sentinel-RAG can deploy in front of *any* Granite instance globally without retraining.