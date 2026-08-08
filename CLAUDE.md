# 🛡️ CLAUDE.md: Sentinel-RAG Architecture & Context Master

**Project Name:** Sentinel-RAG (Enterprise Trust Middleware for Process Intelligence)  
**Track:** AI for Business Transformation  
**Sponsor Integrations:** IBM Granite, IBM Data Prep Kit, IBM watsonx.data, Celonis EMS  
**Development Stack:** IBM Bob (Orchestration & Code Acceleration) + Antigravity Framework (Low-Latency Execution)

---

## 1. PROJECT OVERVIEW & THE CRITICAL ENTERPRISE PROBLEM

### The "Confident Liar" Liability
Major corporations rely on platforms like **Celonis** to map their end-to-end digital twin (ERP workflows, supply chain logs, financial transactions) and identify hidden bottlenecks in real time. As enterprises build Generative AI Copilots on top of this data, a fundamental architectural mismatch occurs.

**Stochastic Guesswork in Deterministic Domains:** Large Language Models (LLMs) operate on probabilistic (stochastic) next-token prediction mechanisms. When an LLM is tasked with reasoning over rigid, highly structured process graphs and encounters missing metadata or complex process dependencies, its decoding engine probabilistically fills contextual gaps rather than failing gracefully. This produces "hallucinations"—syntactically polished, highly confident statements that are factually false.

**The Intra-Generation Deficit:** Current RAG guardrails rely almost exclusively on post-generation evaluations (such as "LLM-as-a-judge" or asynchronous external verification pipelines). Because these checks evaluate responses only *after* generation completes, they introduce severe streaming latency, consume unnecessary compute cycles, and allow ungrounded, hallucinated outputs to reach downstream decision engines or end users before intervention occurs.

**Business Impact:**
* **Financial Risk:** Studies show AI hallucinations cost enterprises over **$67 billion globally**. In consumer apps, a hallucination is a funny typo; in enterprise supply chains or finance, a fabricated number can lead to tremendous losses.
* **Operational Bottleneck:** Employees currently spend an average of **4.3 hours per week** verifying AI outputs due to a lack of trust, creating an expensive operational bottleneck.

---

## 2. THE SENTINEL-RAG PARADIGM SHIFT (OUR SOLUTION)

Sentinel-RAG introduces a complete paradigm shift: moving hallucination detection from post-hoc Natural Language Processing (text checking) to **real-time mathematical uncertainty quantification**. We act as an adversarial firewall between the LLM and the enterprise data.

### Core Innovations & Differentiators
1. **The "Circuit Breaker" Architecture (Active Interception):** Current RAG pipelines fail silently, passing hallucinations directly to downstream systems. Our solution flips the "LLM-as-a-judge" model by acting as a true middleware proxy. When semantic entropy crosses a specific safety threshold, it physically trips a circuit breaker, **halting the generation stream during the decoding phase**. It does not just flag the error; it stops execution instantly and routes the failure to an autonomous fallback agent.
2. **Token-Level Mathematical Detection (Log-Probability Variance):** Standard models wait for complete sentences to evaluate them, but our approach evaluates the underlying math by looking for large logprobs. In standard LLM generation, an LLM assigns a probability to every single token it produces. We analyze the time-series variance of the token log-probabilities. Sentinel-RAG intercepts data token-by-token. If the model is grounded in Celonis data, log-probabilities are highly concentrated. If it starts guessing (extrinsic hallucination), the distribution flattens. Sentinel mathematically catches the model the millisecond it begins to guess.
3. **Semantic Entropy vs. Token Entropy:** Grounded in recent research (*Detecting hallucinations in large language models using semantic entropy*), semantic entropy measures how much an LLM's responses vary in meaning, rather than just wording, by considering semantic diversity across multiple sampled responses. By measuring the Shannon entropy of the token distribution on-the-fly, we mathematically differentiate between when the IBM Granite model confidently reasons over process data versus when it is blindly guessing.
4. **Near-Zero Latency Overhead (Production-Grade Speeds):** Calculating semantic entropy via standard sampling imposes a 5x to 10x computational overhead and latency penalty because it requires sampling multiple full responses before evaluating uncertainty. We used this finding to design Sentinel-RAG's C++/CUDA proxy sidecar. Instead of waiting for full responses to generate sequentially, we compute log-probabilities and entropy in parallel during intra-generation decoding, giving us the accuracy of semantic entropy with near-zero added latency.

---

## 3. TECHNICAL EXCELLENCE & FEASIBILITY (THE ARCHITECTURE)

Sentinel-RAG is a highly optimized architecture utilizing the best of the IBM ecosystem, orchestrated via **IBM Bob**.

* **Layer 1: IBM Data Prep Kit (Ingestion & Privacy)** - Intercepts raw Celonis EMS event logs. Executes parsing, PII data masking to discover, remove, or hide sensitive information from datasets, and semantic chunking to ensure strict data privacy (Zero-Knowledge Compliance) while still allowing the data to be securely utilized for data analysis and machine learning.
* **Layer 2: IBM watsonx.data / Milvus (Storage)** - The dense vector database storing ground-truth process metadata.
* **Layer 3: IBM Granite (Generation)** - The primary generative engine (e.g., Granite-13b-chat), chosen for enterprise-grade licensing.
* **Layer 4: Antigravity FastAPI Middleware (Interception)** - The framework-free Python proxy orchestrating Server-Sent Events (SSE) token streams and executing the real-time circuit breaker in sub-15ms. Moving hallucination detection to the middleware layer natively supports real-time visibility, enterprise controls, and continuous accountability.
* **Layer 5: IBM watsonx.ai Agents & Streamlit UI (Recovery)** - The Security Terminal dashboard and autonomous fallback orchestrators triggered by the circuit breaker. The proxy architecture directly aligns with enterprise strategies, like IBM watsonx.governance, transforming AI governance into a continuous, feedback-driven process.

---

## 4. TEAM ROSTER, DOMAIN OWNERSHIP & END GOALS

The team is fully certified under the IBM SkillsBuild "Future Forward: AI for Innovation" program.

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

## 5. DETAILED PHASE-WISE EXECUTION ROADMAP

### PHASE 1: The "Smoke & Mirrors" MVP (Foundation)
**Goal:** Deliver a visual proof-of-concept making the "invisible" firewall visible to secure the 8:30 PM evaluation score.
* **Shivansh's Role:** Build a local FastAPI endpoint simulating an SSE token stream. Program it to intentionally drop the socket halfway through the stream with an `Entropy Spike` error to mock the interception.
* **Shadman's Role:** Scaffold a `mock_celonis_data.json` file to prove the team's understanding of enterprise deterministic data structures.
* **Riddhi's Role:** Map out the predictive variance math breakdown and threshold limits to verbally defend the logic to the judges.
* **Shaurya's Role:** Build the initial Streamlit UI. Connect it to Shivansh's mock API to visually demonstrate the text stream halting mid-sentence when the error is thrown.

### PHASE 2: Core Pipeline & Dockerization
**Goal:** Establish genuine data flow and package the architecture into a portable, universal runtime.
* **Shivansh's Role:** Connect the FastAPI proxy to the live IBM Granite API. Write the backend `Dockerfile`.
* **Shadman's Role:** Feed the synthetic JSON logs into the IBM Data Prep Kit to execute PII masking. Embed the sanitized data into the local Milvus vector database.
* **Riddhi's Role:** Set up the parallel draft model logic capable of extracting real-time log-probabilities from Granite's context window.
* **Shaurya's Role:** Use IBM Bob to auto-generate the `docker-compose.yml` file, linking the Milvus database, the FastAPI backend, and the frontend into a single deployable microservices network.

### PHASE 3: The Interception Engine (The Core Integration)
**Goal:** Prove the live circuit breaker functions accurately on real token variance during active generation.
* **Shivansh's Role:** Merge Riddhi's log-probability calculator directly into the FastAPI streaming loop. Ensure that if the variance threshold is crossed, the socket connection to the client is aggressively and cleanly closed.
* **Shadman's Role:** Connect the Milvus retriever to the prompt pipeline, ensuring Granite is strictly reasoning over the ingested Celonis data. 
* **Riddhi's Role:** Fine-tune the math engine to feed real-time variance data smoothly to Shivansh's proxy without causing CPU bottlenecks.
* **Shaurya's Role:** Build the self-healing trigger into the UI. When the proxy severs the socket, the frontend must instantly display a fallback loading state and query the recovery agent.

### PHASE 4: Hardening & Enterprise Polish
**Goal:** Secure the "Technical Excellence" and "UX & Design" rubric marks for the final Monday presentation.
* **Shivansh's Role:** Benchmark the proxy loop to guarantee the mathematical interception adds less than 15ms of latency overhead per token.
* **Shadman's Role:** Finalize the demo script by writing explicit "Poison Prompts" designed to intentionally force the model to hallucinate during the live demo.
* **Riddhi's Role:** Automate the threshold calibration to guarantee a 0% false-positive rate on factual queries.
* **Shaurya's Role:** Brand the Streamlit dashboard as a sleek, deployable SaaS Security Terminal, ensuring the split-screen comparison is easily viewable by judges via local Wi-Fi.

---

## 6. HACKATHON EVALUATION STRATEGY & RUBRIC MAPPING

### Evaluation 1 (Saturday 8:30 PM) - The Foundation (40/40 points)

* **Problem Understanding (10/10):** The core conflict is that LLMs use probabilistic guesswork in deterministic domains, producing confident hallucinations. Current RAG relies on post-generation verification, which introduces latency and fails to stop errors. We highlight structural vulnerabilities using the *JudgeBench* paper to prove post-hoc models exhibit bias and struggle on complex reasoning tasks.
* **Innovation & Creativity (10/10):** We are shifting hallucination detection from post-hoc NLP to real-time mathematical uncertainty quantification. Instead of failing silently, our middleware physically trips a circuit breaker and halts generation during the decoding phase, then autonomously routes to a self-healing agent.
* **Feasibility & Planning (10/10):** The *Semantic Entropy Probes* paper showed standard sampling imposes a 5x-10x latency penalty. Our custom C++/CUDA proxy sidecar overcomes this by computing log-probabilities in parallel, achieving accuracy with near-zero latency. We execute this via the IBM Bob + Antigravity stack.
* **Team Coordination (10/10):** Strict domain isolation (FastAPI proxy, Data Prep Kit, Math/Thresholds, and UI). Flawless handoffs during the pitch; explicitly state that every member holds the IBM SkillsBuild "Future Forward: AI for Innovation" certification.

### Evaluation 2 (Monday Morning) - The Final Pitch (80/80 points)

* **Working Prototype (25pts):** The Split-Screen Demo. Left: Unprotected Granite confidently hallucinating a metric. Right: Sentinel-RAG catching the hallucination mid-word, cutting the stream, and self-correcting. 
* **Problem-Solution Fit (20pts):** The Buyer Narrative. Built for IBM Consulting to derisk Watsonx integrations in highly regulated sectors. AI hallucinations cost $67 billion globally; this eliminates that liability.
* **Technical Excellence (15pts):** Highlight the IBM Data Prep Kit (PII compliance), Milvus, and sub-15ms Antigravity latency.
* **UX & Design (10pts):** The Security Terminal Feel. Live Semantic Entropy Variance graph flashing on breaches.
* **Impact & Scalability (10pts):** Model-Agnostic. Sentinel-RAG can deploy in front of *any* Granite instance globally without retraining, natively supporting real-time enterprise controls.
