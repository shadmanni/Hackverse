# 🛡️ CLAUDE.md: Sentinel-RAG Architecture & Context Master

**Project Name:** Sentinel-RAG (Enterprise Trust Middleware for Process Intelligence)  
**Track:** AI for Business Transformation[cite: 1, 2]  
**Sponsor Integrations:** IBM Granite, Milvus (Lite), Celonis EMS[cite: 1, 2]  
**Development Stack:** IBM Bob (Orchestration & Code Acceleration) + Custom Python Evaluation Pipeline  

---

## 0. IMPLEMENTATION STATUS (what is actually built — read this first)

The roadmap in section 5 is the original plan. Where it disagrees with this
section, this section wins.

* **Generation:** `ibm-granite/granite-3.3-2b-instruct` running **locally via
  transformers on Apple MPS** (`granite_runner.py`). Not the watsonx API, not
  Granite-13b. Chosen so the demo has no network dependency on stage. Real
  per-token log-probabilities and top-5 distributions come out of the decode
  loop — the entropy engine is fed measured values, not constants.
* **Frontend:** Next.js "Security Terminal" (`frontend/`). **Streamlit is gone**
  — `streamlit_app.py` was deleted; two processes cannot hold the Milvus Lite
  lock at once.
* **Vector store:** Milvus **Lite**, embedded single-file at
  `data/sentinel_milvus.db`, collection `celonis_ground_truth`, 461 vectors from
  `data/mock_celonis_data_large.json`. Single-process: stop the backend before
  running anything else that opens it.
* **PII:** actors are pseudonymised to stable salted-hash aliases in
  `phase2_ingestion_pipeline.py`. **IBM Data Prep Kit is NOT wired in** —
  `dpk_ingest.py` stages it as an offline step in a separate venv, because DPK
  downgrades transformers to 4.57.6 and would break the model. Do not claim DPK
  is in the pipeline until that script has actually been run.
* **Detection is two layers**, not one: semantic entropy over real logprobs
  (requires a run of consecutive breaches, since a single flat token is usually
  just sentence-opening or hedging), plus a deterministic numeric-grounding
  check that refutes any figure the event log cannot produce.
* **Ground truth:** every aggregate the system may state comes from
  `celonis_metrics.py`. The declared mean compliance cycle time is **10.4 days**.
  Any hardcoded "4.2 business days" / "99.4% SLA" is a fabrication — the event
  log does not contain those figures.

**Operational rules:** stop the backend before running `pytest` (two Granite
instances on one MPS device segfault). Backend warm-up is ~35s at startup, paid
once, not per query.

---

## 1. PROJECT OVERVIEW & THE CRITICAL ENTERPRISE PROBLEM

### The "Confident Liar" Liability
Major corporations use platforms like Celonis to map their end-to-end digital twin (ERP workflows, supply chain logs, financial transactions)[cite: 3]. Celonis generates billions of dollars in enterprise value by identifying hidden bottlenecks in real time[cite: 3]. Companies want to build Generative AI Copilots and Autonomous Agents on top of this data[cite: 3]. However, a fundamental architectural mismatch occurs:

**Stochastic Guesswork in Deterministic Domains:** Large Language Models (LLMs) operate on probabilistic (stochastic) next-token prediction mechanisms[cite: 3]. When an LLM encounters missing metadata or complex process dependencies, its decoding engine fills contextual gaps probabilistically rather than failing gracefully[cite: 3]. This produces "hallucinations"—syntactically polished, highly confident statements that are factually false[cite: 3]. 

**The Intra-Generation Deficit:** Current RAG guardrails rely almost exclusively on post-generation evaluations (such as "LLM-as-a-judge" or asynchronous external verification pipelines)[cite: 3]. Because these checks evaluate responses only after generation completes, they introduce severe streaming latency, consume unnecessary compute cycles, and allow ungrounded, hallucinated outputs to reach downstream decision engines or end users before intervention occurs[cite: 3]. 

**Business Impact:**
* **Financial Risk:** Studies show AI hallucinations cost enterprises over $67 billion globally[cite: 3]. In enterprise supply chains or finance, a fabricated number can lead to tremendous losses[cite: 3].
* **Operational Bottleneck:** Employees currently spend an average of 4.3 hours per week verifying AI outputs due to a lack of trust, creating an expensive operational bottleneck[cite: 3]. 

---

## 2. THE SENTINEL-RAG PARADIGM SHIFT (OUR SOLUTION)

Sentinel-RAG introduces a complete paradigm shift: moving hallucination detection from post-hoc Natural Language Processing (text checking) to real-time mathematical uncertainty quantification[cite: 3]. 

### Core Innovations & Differentiators
1. **The "Circuit Breaker" Architecture (Active Interception):** Current RAG pipelines fail silently, passing hallucinations directly to downstream systems[cite: 3]. Our solution flips the "LLM-as-a-judge" model by acting as a true middleware proxy[cite: 3]. When semantic entropy crosses a specific safety threshold, it physically trips a circuit breaker, halting the generation stream during the decoding phase[cite: 3]. It stops execution instantly and routes the failure to an autonomous fallback agent[cite: 3].
2. **Token-Level Mathematical Detection (Log-Probability Variance):** In standard LLM generation, an LLM assigns a probability to every single token it produces[cite: 3]. We evaluate the underlying math by looking for large logprobs, analyzing the time-series variance of the token log-probabilities[cite: 3]. If the model is grounded in Celonis data, log-probabilities are highly concentrated; if it starts guessing, the distribution flattens[cite: 3]. Sentinel mathematically catches the model the millisecond it begins to guess[cite: 3].
3. **Semantic Entropy vs. Token Entropy:** Grounded in recent research (*Detecting hallucinations in large language models using semantic entropy*, doi:10.1038/s41586-024-07421-0), semantic entropy measures how much an LLM's responses vary in meaning, rather than just wording, by considering semantic diversity across multiple sampled responses[cite: 3]. 
4. **Autonomous Recovery:** Hallucinations can be detected by assessing the uncertainty of model outputs[cite: 3]. Once this mathematical uncertainty is flagged, instead of simply logging an error, our system autonomously reroutes the workflow to a self-healing agent specifically designed to repair the context gap[cite: 3].

---

## 3. TECHNICAL EXCELLENCE & FEASIBILITY (THE ARCHITECTURE)

Sentinel-RAG is a highly optimized architecture utilizing the best of the IBM ecosystem, orchestrated via **IBM Bob**.

* **Layer 1: Ingestion & Privacy** - Intercepts raw Celonis EMS event logs and pseudonymises direct identifiers before embedding, so no actor name reaches the vector store. Pseudonyms are stable per actor, which preserves handoff and segregation-of-duty analysis that blanket redaction would destroy. IBM Data Prep Kit is staged for this layer (`dpk_ingest.py`) but not yet run — see section 0.
* **Layer 2: Milvus Lite Vector Database (Storage)** - The dense vector database storing ground-truth process metadata.
* **Layer 3: IBM Granite (Generation)** - `granite-3.3-2b-instruct`, run locally on MPS, processing prompts containing retrieved Celonis process metrics[cite: 1, 2].
* **Layer 4: Semantic Entropy Evaluator (The Core Logic)** - A specialized Python evaluation layer that actively calculates predictive variance and log-probabilities on the generated output in real-time.
* **Layer 5: Autonomous Recovery (Self-Healing Agents)** - When the evaluator logs an entropy error, it triggers an autonomous agentic fallback to dynamically formulate a new vector search strategy and repair the context gap[cite: 1, 2].

---

## 4. TEAM ROSTER, DOMAIN OWNERSHIP & END GOALS

The team is fully certified under the IBM SkillsBuild "Future Forward: AI for Innovation" program (PLAN-FA448E08E2A9)[cite: 1, 2].

* **Shivansh Srivastava** (USN: 245819276 | Lead Evaluator & System Orchestrator)[cite: 1, 2]
  * *Tasks:* Construct the core Python evaluation pipeline; integrate the live token stream from IBM Granite; implement the threshold logic that actively logs the error and halts the flawed generation when semantic entropy spikes.
* **Shadman Nishat** (USN: 245816440 | Data Pipeline & Ingestion Lead)[cite: 1, 2]
  * *Tasks:* Synthesize Celonis logs; build PII masking pipeline; integrate embeddings into Milvus. Design "Poison Prompts" to intentionally trigger hallucinations during demos.
* **Riddhi Jain** (USN: 245890454 | Entropy Analytics & Optimization)[cite: 1, 2]
  * *Tasks:* Translate Semantic Entropy math into an optimized Python engine; calculate predictive variance; calibrate the mathematical threshold to eliminate false positives.
* **Shaurya Arvind** (USN: 245891406 | Visualizations & IBM Bob Workflows)[cite: 1, 2]
  * *Tasks:* Build the interactive "Security Terminal" dashboard that visually renders the incoming stream and explicitly highlights the interception and self-healing agent recovery events.

---

## 5. ELABORATE PHASE-WISE EXECUTION ROADMAP

### PHASE 1: The "Smoke & Mirrors" MVP (Foundation & UI Setup)
* **Goal:** Deliver a visual proof-of-concept that makes the "invisible" firewall visible to the judges for the Round 1 evaluation.
* **Shivansh (Lead Evaluator):** Build the initial mock Python evaluation script. Create a local execution loop that streams a hardcoded sentence, simulates a spike in log-probability variance, actively logs an error flag, and physically halts the stream at a predefined token.
* **Shadman (Data Pipeline):** Generate `mock_celonis_data.json` containing simulated supply chain metrics (e.g., order-to-cash cycle times). Draft a mock ingestion script to demonstrate the ETL pipeline plan to the judges.
* **Riddhi (Entropy Analytics):** Whiteboard the Semantic Entropy mathematical framework. Outline the exact predictive variance formula and document how token log-probabilities will be extracted from IBM Granite's output.
* **Shaurya (Visualizations):** Initialize the frontend (originally Streamlit; now Next.js). Build the chat interface that connects to Shivansh's mock stream, programming it to catch the error flag, highlight the exact halted token in red, and display a simulated "Self-Healing Triggered" loader.

### PHASE 2: Core Pipeline, Vectorization & Integration
* **Goal:** Establish the genuine data flow from raw synthetic event logs to the live IBM Granite model.
* **Shivansh (Lead Evaluator):** Replace the mock script with a real Granite decode loop (shipped as a LOCAL transformers/MPS runner, not the API) and extract actual token log-probabilities in real time.
* **Shadman (Data Pipeline):** Implement the ingestion pipeline (DPK staged, not yet run — see section 0). Parse the synthetic Celonis logs, execute PII data masking to discover, remove, or hide sensitive information from the datasets, and embed the cleaned semantic chunks into the Milvus vector database[cite: 3].
* **Riddhi (Entropy Analytics):** Write the core Python mathematical engine. Accept the streaming log-probabilities from Shivansh's pipeline and calculate the time-series variance ($V(y_t)$) continuously across the sequence.
* **Shaurya (Visualizations):** Use IBM Bob to rapidly scaffold asynchronous frontend API polling logic. Integrate the UI with Shadman's vector database, allowing the user to select which specific process graphs they want to query.

### PHASE 3: The Interception Engine & Autonomous Recovery
* **Goal:** The core technical milestone—live circuit breaking and self-healing.
* **Shivansh (Lead Evaluator):** Integrate Riddhi's math engine directly into the live token loop. Implement the active circuit breaker: if the variance threshold is breached, physically terminate the generation stream, log the error to the console, and emit a fallback signal to the frontend.
* **Shadman (Data Pipeline):** Connect the Milvus retriever to the Granite prompt context. Refine a set of "Poison Prompts" designed to ask for data *missing* from the vector store, forcing the LLM into stochastic guesswork and reliably triggering the firewall.
* **Riddhi (Entropy Analytics):** Test the live engine against both factual and hallucinated outputs. Begin calibrating the mathematical threshold ($\tau$) to perfectly differentiate between confident reasoning and ungrounded fabrication.
* **Shaurya (Visualizations):** Wire the fallback signal to the autonomous recovery subagent. When the UI receives the error log, trigger the self-healing agent to dynamically formulate a new vector search strategy, repair the context gap, and stream the corrected response[cite: 3].

### PHASE 4: Hardening, Enterprise Polish & Final Pitch Prep
* **Goal:** Secure the "Technical Excellence" and "UX & Design" rubric marks for the final Monday evaluation.
* **Shivansh (Lead Evaluator):** Optimize the Python evaluation loop. Ensure the real-time entropy calculations and threshold checks add near-zero latency overhead to the active token stream.
* **Shadman (Data Pipeline):** Finalize the demo run-of-show. Ensure a 100% success rate on the Poison Prompts triggering the firewall, while standard prompts execute smoothly against the Milvus data.
* **Riddhi (Entropy Analytics):** Lock in the threshold calibration for a 0% false-positive rate. Export log-probability variance graphs to show the judges exactly how the math behaved under the hood during the demo.
* **Shaurya (Visualizations):** Brand the Next.js dashboard as a deployable, enterprise-grade SaaS "Security Terminal." Add a live data visualization module that charts the Semantic Entropy Variance spiking in real-time right before the stream halts.

---

## 6. HACKATHON EVALUATION STRATEGY & RUBRIC MAPPING

### Round 1: Foundation Pitch
* **Problem Understanding:** Hook judges with the "Confident Liar" liability in deterministic Celonis domains[cite: 3]. Highlight structural vulnerabilities of using secondary LLMs as post-hoc evaluators, leveraging the *JudgeBench* paper to prove they exhibit bias and struggle on complex reasoning tasks[cite: 3].
* **Innovation & Creativity:** Emphasize the speed shift (mid-stream mathematical entropy vs post-hoc NLP) and the autonomous recovery mechanism[cite: 3].
* **Feasibility & Planning:** Showcase the integration of IBM Bob for development and the streamlined Python evaluator execution. 
* **Team Coordination:** Flawless handoffs during the pitch; explicitly mention IBM SkillsBuild certifications for domain authority[cite: 1, 2].

### Round 2: The Final Pitch
* **Working Prototype (25pts):** The Split-Screen Demo. Left: Unprotected Granite hallucinating. Right: Sentinel-RAG predicting the hallucination, logging the error, halting the stream, and self-correcting via a subagent.
* **Problem-Solution Fit (20pts):** The Buyer Narrative. Built for enterprise consulting to derisk GenAI integrations in highly regulated sectors, specifically addressing the 4.3 hours/week wasted on verification[cite: 3].
* **Technical Excellence (15pts):** Highlight the IBM Data Prep Kit (PII compliance)[cite: 3], Milvus, and the direct Python semantic entropy evaluation logic.
* **UX & Design (10pts):** The Security Terminal Feel. Live Semantic Entropy Variance graph flashing on breaches.
* **Impact & Scalability (10pts):** Model-Agnostic. Sentinel-RAG natively supports real-time visibility, enterprise controls, and continuous accountability[cite: 3].
