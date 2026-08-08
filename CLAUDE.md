# 🛡️ CLAUDE.md: Sentinel-RAG Architecture & Context Master

**Project Name:** Sentinel-RAG (Enterprise Trust Middleware for Process Intelligence)  
**Track:** AI for Business Transformation  
**Sponsor Integrations:** IBM Granite, IBM Data Prep Kit, IBM watsonx.data, Celonis EMS  
**Development Stack:** IBM Bob (Orchestration & Code Acceleration) + Antigravity Framework (Low-Latency Execution)

---

## 1. PROJECT OVERVIEW & THE CRITICAL ENTERPRISE PROBLEM

### The "Confident Liar" Liability
Major corporations rely on platforms like **Celonis** to map their end-to-end digital twin (ERP workflows, supply chain logs, financial transactions) and identify hidden bottlenecks in real time[span_1](start_span)[span_1](end_span). As enterprises build Generative AI Copilots on top of this data, a fundamental architectural mismatch occurs[span_2](start_span)[span_2](end_span).

**Stochastic Guesswork in Deterministic Domains:** Large Language Models (LLMs) operate on probabilistic (stochastic) next-token prediction mechanisms[span_3](start_span)[span_3](end_span). When an LLM is tasked with reasoning over rigid, highly structured process graphs and encounters missing metadata or complex process dependencies, its decoding engine probabilistically fills contextual gaps rather than failing gracefully[span_4](start_span)[span_4](end_span). This produces "hallucinations"—syntactically polished, highly confident statements that are factually false[span_5](start_span)[span_5](end_span).

**The Intra-Generation Deficit:** Current RAG guardrails rely almost exclusively on post-generation evaluations (such as "LLM-as-a-judge" or asynchronous external verification pipelines)[span_6](start_span)[span_6](end_span). Because these checks evaluate responses only *after* generation completes, they introduce severe streaming latency, consume unnecessary compute cycles, and allow ungrounded, hallucinated outputs to reach downstream decision engines or end users before intervention occurs[span_7](start_span)[span_7](end_span).

**Business Impact:**
* **Financial Risk:** Studies show AI hallucinations cost enterprises over **$67 billion globally**[span_8](start_span)[span_8](end_span). In consumer apps, a hallucination is a funny typo; in enterprise supply chains or finance, a fabricated number can lead to tremendous losses[span_9](start_span)[span_9](end_span).
* **Operational Bottleneck:** Employees currently spend an average of **4.3 hours per week** verifying AI outputs due to a lack of trust, creating an expensive operational bottleneck[span_10](start_span)[span_10](end_span).

---

## 2. THE SENTINEL-RAG PARADIGM SHIFT (OUR SOLUTION)

Sentinel-RAG introduces a complete paradigm shift: moving hallucination detection from post-hoc Natural Language Processing (text checking) to **real-time mathematical uncertainty quantification**[span_11](start_span)[span_11](end_span). We act as an adversarial firewall between the LLM and the enterprise data.

### Core Innovations & Differentiators
1. **The "Circuit Breaker" Architecture (Active Interception):** Current RAG pipelines fail silently, passing hallucinations directly to downstream systems[span_12](start_span)[span_12](end_span). Our solution flips the "LLM-as-a-judge" model by acting as a true middleware proxy[span_13](start_span)[span_13](end_span). When semantic entropy crosses a specific safety threshold, it physically trips a circuit breaker, **halting the generation stream during the decoding phase**[span_14](start_span)[span_14](end_span). It does not just flag the error; it stops execution instantly and routes the failure to an autonomous fallback agent[span_15](start_span)[span_15](end_span).
2. **Token-Level Mathematical Detection (Log-Probability Variance):** Standard models wait for complete sentences to evaluate them, but our approach evaluates the underlying math by looking for large logprobs[span_16](start_span)[span_16](end_span). In standard LLM generation, an LLM assigns a probability to every single token it produces[span_17](start_span)[span_17](end_span). We analyze the time-series variance of the token log-probabilities[span_18](start_span)[span_18](end_span). Sentinel-RAG intercepts data token-by-token[span_19](start_span)[span_19](end_span). If the model is grounded in Celonis data, log-probabilities are highly concentrated[span_20](start_span)[span_20](end_span). If it starts guessing (extrinsic hallucination), the distribution flattens[span_21](start_span)[span_21](end_span). Sentinel mathematically catches the model the millisecond it begins to guess[span_22](start_span)[span_22](end_span).
3. **Semantic Entropy vs. Token Entropy:** Grounded in recent research (*Detecting hallucinations in large language models using semantic entropy*), semantic entropy measures how much an LLM's responses vary in meaning, rather than just wording, by considering semantic diversity across multiple sampled responses[span_23](start_span)[span_23](end_span). By measuring the Shannon entropy of the token distribution on-the-fly, we mathematically differentiate between when the IBM Granite model confidently reasons over process data versus when it is blindly guessing[span_24](start_span)[span_24](end_span).
4. **Near-Zero Latency Overhead (Production-Grade Speeds):** Calculating semantic entropy via standard sampling imposes a 5x to 10x computational overhead and latency penalty because it requires sampling multiple full responses before evaluating uncertainty[span_25](start_span)[span_25](end_span). We used this finding to design Sentinel-RAG's C++/CUDA proxy sidecar[span_26](start_span)[span_26](end_span). Instead of waiting for full responses to generate sequentially, we compute log-probabilities and entropy in parallel during intra-generation decoding, giving us the accuracy of semantic entropy with near-zero added latency[span_27](start_span)[span_27](end_span).

---

## 3. TECHNICAL EXCELLENCE & FEASIBILITY (THE ARCHITECTURE)

Sentinel-RAG is a highly optimized architecture utilizing the best of the IBM ecosystem, orchestrated via **IBM Bob**.

* **Layer 1: IBM Data Prep Kit (Ingestion & Privacy)** - Intercepts raw Celonis EMS event logs. Executes parsing, PII data masking to discover, remove, or hide sensitive information from datasets, and semantic chunking to ensure strict data privacy (Zero-Knowledge Compliance) while still allowing the data to be securely utilized for data analysis and machine learning[span_28](start_span)[span_28](end_span).
* **Layer 2: IBM watsonx.data / Milvus (Storage)** - The dense vector database storing ground-truth process metadata.
* **Layer 3: IBM Granite (Generation)** - The primary generative engine (e.g., Granite-13b-chat), chosen for enterprise-grade licensing.
* **Layer 4: Antigravity FastAPI Middleware (Interception)** - The framework-free Python proxy orchestrating Server-Sent Events (SSE) token streams and executing the real-time circuit breaker in sub-15ms. Moving hallucination detection to the middleware layer natively supports real-time visibility, enterprise controls, and continuous accountability[span_29](start_span)[span_29](end_span).
* **Layer 5: IBM watsonx.ai Agents & Streamlit UI (Recovery)** - The Security Terminal dashboard and autonomous fallback orchestrators triggered by the circuit breaker. The proxy architecture directly aligns with enterprise strategies, like IBM watsonx.governance, transforming AI governance into a continuous, feedback-driven process[span_30](start_span)[span_30](end_span).

---

## 4. TEAM ROSTER, DOMAIN OWNERSHIP & END GOALS

The team is fully certified under the IBM SkillsBuild "Future Forward: AI for Innovation" program[span_31](start_span)[span_31](end_span)[span_32](start_span)[span_32](end_span).

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

## 6. HACKATHON EVALUATION STRATEGY & RUBRIC MAPPING

### Evaluation 1 (Saturday 8:30 PM) - The Foundation (40/40 points)

* **Problem Understanding (10/10):** The core conflict is that LLMs use probabilistic guesswork in deterministic domains, producing confident hallucinations[span_33](start_span)[span_33](end_span). Current RAG relies on post-generation verification, which introduces latency and fails to stop errors[span_34](start_span)[span_34](end_span). We highlight structural vulnerabilities using the *JudgeBench* paper to prove post-hoc models exhibit bias and struggle on complex reasoning tasks[span_35](start_span)[span_35](end_span).
* **Innovation & Creativity (10/10):** We are shifting hallucination detection from post-hoc NLP to real-time mathematical uncertainty quantification[span_36](start_span)[span_36](end_span). Instead of failing silently, our middleware physically trips a circuit breaker and halts generation during the decoding phase, then autonomously routes to a self-healing agent[span_37](start_span)[span_37](end_span).
* **Feasibility & Planning (10/10):** The *Semantic Entropy Probes* paper showed standard sampling imposes a 5x-10x latency penalty[span_38](start_span)[span_38](end_span). Our custom C++/CUDA proxy sidecar overcomes this by computing log-probabilities in parallel, achieving accuracy with near-zero latency[span_39](start_span)[span_39](end_span). We execute this via the IBM Bob + Antigravity stack.
* **Team Coordination (10/10):** Strict domain isolation (FastAPI proxy, Data Prep Kit, Math/Thresholds, and UI). Flawless handoffs during the pitch; explicitly state that every member holds the IBM SkillsBuild "Future Forward: AI for Innovation" certification[span_40](start_span)[span_40](end_span)[span_41](start_span)[span_41](end_span).

### Evaluation 2 (Monday Morning) - The Final Pitch (80/80 points)

* **Working Prototype (25pts):** The Split-Screen Demo. Left: Unprotected Granite confidently hallucinating a metric. Right: Sentinel-RAG catching the hallucination mid-word, cutting the stream, and self-correcting. 
* **Problem-Solution Fit (20pts):** The Buyer Narrative. Built for IBM Consulting to derisk Watsonx integrations in highly regulated sectors. AI hallucinations cost $67 billion globally; this eliminates that liability[span_42](start_span)[span_42](end_span).
* **Technical Excellence (15pts):** Highlight the IBM Data Prep Kit (PII compliance)[span_43](start_span)[span_43](end_span), Milvus, and sub-15ms Antigravity latency.
* **UX & Design (10pts):** The Security Terminal Feel. Live Semantic Entropy Variance graph flashing on breaches.
* **Impact & Scalability (10pts):** Model-Agnostic. Sentinel-RAG can deploy in front of *any* Granite instance globally without retraining, natively supporting real-time enterprise controls[span_44](start_span)[span_44](end_span).
