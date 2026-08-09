"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  ShieldCheck,
  Database,
  Terminal,
  Send,
  Lock,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import SplitScreenResponse from "@/components/SplitScreenResponse";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface ChatMessage {
  id: string;
  query: string;
  graphName: string;
  /** true while this message is actively streaming */
  streaming: boolean;
}

/** Shape returned by /graphs — all of it computed from the event log. */
interface GraphInfo {
  name: string;
  collection: string;
  source_system: string;
  vector_count: number;
  cases: number;
  pii_masked: boolean;
  mean_cycle_days: number;
  declared_compliance_cycle_days: number;
  activities: Record<string, number>;
}

interface Health {
  status: string;
  model: string;
  model_loaded: boolean;
  retriever_loaded: boolean;
  warmup_ms: number | null;
  circuit_breaker_tau: number;
  event_log_cases: number;
  event_log_events: number;
}

const PRESETS = [
  {
    label: "🚨 Fabricated aggregate",
    query: "What is the mean compliance cycle time?",
    hint: "unprotected invents ~15 days; the log says 10.4",
  },
  {
    label: "👻 Phantom entity",
    query: "State the exact off-contract discount percentage for warehouse node W-99 in Q4.",
    hint: "no node W-99 exists in the log",
  },
  {
    label: "✅ Grounded query",
    query: "Give the exact mean cycle time for Compliance Review to two decimal places.",
    hint: "answerable: 10.43 days",
  },
];

export default function Home() {
  const [queryInput, setQueryInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [graph, setGraph] = useState<GraphInfo | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${BACKEND}/graphs`)
      .then((r) => r.json())
      .then((d) => setGraph(d.o2c ?? null))
      .catch(() => setGraph(null));
    fetch(`${BACKEND}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    const q = queryInput.trim();
    if (!q) return;

    // Clear input immediately
    setQueryInput("");

    const newId = `${Date.now()}`;

    // Mark all previous messages as no longer streaming (they are collapsed while new one runs)
    setMessages((prev) => [
      // New message goes to front (index 0 = top of list)
      { id: newId, query: q, graphName: graph?.name ?? "Celonis event log", streaming: true },
      // Previous messages — mark as not streaming so they collapse
      ...prev.map((m) => ({ ...m, streaming: false })),
    ]);
  }

  /** Called by SplitScreenResponse when both SSE streams finish */
  function handleDone(id: string) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, streaming: false } : m))
    );
  }

  const online = health?.model_loaded ?? false;

  return (
    <div className="flex h-screen bg-[#080a0c] text-gray-100 overflow-hidden font-sans">
      {/* ── SIDEBAR ── */}
      <aside
        className={`relative flex-col justify-between bg-[#0c0f14] border-r border-[#1a212b] transition-all duration-300 overflow-hidden hidden md:flex ${
          sidebarOpen ? "w-80" : "w-0"
        } shrink-0`}
      >
        <div className={`w-80 flex flex-col h-full transition-opacity duration-200 ${sidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
          <div className="p-5 space-y-6 overflow-y-auto flex-1">
            <div className="flex items-center gap-3 pb-4 border-b border-[#1a212b]">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-600 to-amber-600 flex items-center justify-center shadow-lg shrink-0">
                <ShieldCheck className="w-6 h-6 text-white" />
              </div>
              <div className="min-w-0">
                <h1 className="font-mono text-base font-bold text-white">SENTINEL-RAG</h1>
                <span className="text-[11px] font-mono text-slate-400">
                  Intra-generation hallucination firewall
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-mono font-semibold uppercase text-slate-400 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-amber-500" />
                Celonis Vector Store
              </label>
              <div className="bg-[#11161d] border border-[#1d2530] p-3 rounded-md space-y-1.5 text-xs font-mono text-slate-400">
                {graph ? (
                  <>
                    <Row k="Process" v={graph.name.replace(" Compliance Cycle", "")} />
                    <Row k="Collection" v={graph.collection} />
                    <Row k="Events ingested" v={graph.vector_count.toLocaleString()} />
                    <Row k="Cases" v={graph.cases.toLocaleString()} />
                    <Row k="Mean cycle" v={`${graph.mean_cycle_days} days`} />
                    <Row k="Declared compliance" v={`${graph.declared_compliance_cycle_days} days`} />
                    <Row k="PII masking" v="Data Prep Kit" ok />
                  </>
                ) : (
                  <span className="text-slate-600 italic">backend offline</span>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-xs font-mono font-semibold uppercase text-slate-400 flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-emerald-400" />
                Guardrail Parameters
              </span>
              <div className="bg-[#11161d] border border-[#1d2530] p-3 rounded-md space-y-1.5 text-xs font-mono text-slate-300">
                {health ? (
                  <>
                    <Row k="Model" v={health.model.split("/").pop() ?? health.model} />
                    <Row k="Entropy τ" v={health.circuit_breaker_tau.toFixed(3)} />
                    <Row
                      k="Warm-up"
                      v={health.warmup_ms ? `${(health.warmup_ms / 1000).toFixed(1)}s (once)` : "—"}
                    />
                    <Row k="Retriever" v={health.retriever_loaded ? "Milvus Lite" : "in-memory"} />
                    <Row k="Ground truth" v={`${health.event_log_events} events`} />
                  </>
                ) : (
                  <span className="text-slate-600 italic">backend offline</span>
                )}
              </div>
            </div>

            {graph && (
              <div className="space-y-2">
                <span className="text-xs font-mono font-semibold uppercase text-slate-400">
                  Activities in log
                </span>
                <div className="bg-[#11161d] border border-[#1d2530] p-3 rounded-md space-y-1 text-[11px] font-mono text-slate-400">
                  {Object.entries(graph.activities).map(([a, n]) => (
                    <Row key={a} k={a} v={String(n)} />
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="p-5 border-t border-[#1a212b]">
            <div
              className={`flex items-center justify-between bg-[#11161d] border p-2.5 rounded-md ${
                online ? "border-emerald-900/60" : "border-rose-900/60"
              }`}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${online ? "bg-emerald-400 animate-pulse" : "bg-rose-500"}`}
                />
                <span
                  className={`text-xs font-mono font-bold ${online ? "text-emerald-400" : "text-rose-400"}`}
                >
                  {online ? "MODEL LOADED" : "OFFLINE"}
                </span>
              </div>
              <span className="text-[10px] font-mono text-slate-400">:8000</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Sidebar toggle button */}
      <button
        onClick={() => setSidebarOpen((v) => !v)}
        className="hidden md:flex items-center justify-center w-5 bg-[#0c0f14] border-r border-[#1a212b] hover:bg-[#141920] transition-colors shrink-0 cursor-pointer group z-10"
        title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {sidebarOpen ? (
          <ChevronLeft className="w-3.5 h-3.5 text-slate-500 group-hover:text-slate-300 transition-colors" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-slate-300 transition-colors" />
        )}
      </button>

      {/* ── MAIN ── */}
      <main className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        <header className="h-16 border-b border-[#1a212b] bg-[#0c0f14]/80 backdrop-blur-md px-6 flex items-center gap-3 shrink-0">
          <Terminal className="w-5 h-5 text-rose-500 shrink-0" />
          <div className="min-w-0">
            <h2 className="text-sm font-bold font-mono text-white truncate">
              SENTINEL-RAG INTERCEPTION TERMINAL
            </h2>
            <p className="text-[11px] text-slate-400 truncate">
              Real-time token-level hallucination interception over Celonis process data
            </p>
          </div>
        </header>

        {/* Messages — newest at top */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 flex flex-col">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-5">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-rose-950 to-slate-900 border border-rose-900/60 flex items-center justify-center">
                <ShieldCheck className="w-8 h-8 text-rose-500" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-100 font-mono">
                  Run a split-screen evaluation
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  The same local Granite model answers twice: once with no retrieved context and no
                  interception, once through the Sentinel proxy. Nothing is pre-scripted.
                </p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((m, idx) => {
                const isActive = m.streaming;
                const isCollapsed = !isActive && messages.some((x) => x.streaming);
                return (
                  <div
                    key={m.id}
                    className={`transition-all duration-500 ${
                      isCollapsed
                        ? "opacity-40 scale-[0.985] pointer-events-none"
                        : "opacity-100 scale-100"
                    }`}
                    style={{ order: idx }}
                  >
                    {/* Query bubble */}
                    <div className="flex justify-end mb-3">
                      <div className="max-w-[70%] bg-gradient-to-br from-rose-900/60 to-slate-800/80 border border-rose-800/40 rounded-2xl rounded-tr-sm px-4 py-2.5 shadow-lg">
                        <p className="text-sm text-slate-100 break-words leading-relaxed">{m.query}</p>
                        <span className="text-[10px] font-mono text-slate-500 mt-1 block">
                          {m.graphName}
                        </span>
                      </div>
                    </div>

                    {/* Collapsed placeholder while another query is active */}
                    {isCollapsed ? (
                      <div className="flex items-center gap-2 px-3 py-2 bg-[#0d1217] border border-[#1e2630] rounded-lg text-xs font-mono text-slate-500 italic">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-600 shrink-0" />
                        Previous response hidden while new query runs…
                      </div>
                    ) : (
                      <SplitScreenResponse
                        key={m.id}
                        query={m.query}
                        graphName={m.graphName}
                        onDone={() => handleDone(m.id)}
                      />
                    )}
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input bar */}
        <div className="p-4 border-t border-[#1a212b] bg-[#0c0f14] shrink-0">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-3">
            <div className="relative flex items-center bg-[#131820] border border-[#232c38] focus-within:border-rose-500/80 rounded-xl p-2 transition-colors">
              <input
                id="query-input"
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="Enter an enterprise query…"
                className="w-full bg-transparent px-3 py-2 text-sm text-slate-100 focus:outline-none placeholder:text-slate-500"
              />
              <button
                type="submit"
                disabled={!queryInput.trim()}
                className="bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-5 py-2 rounded-lg text-xs font-mono font-bold flex items-center gap-2 shrink-0 cursor-pointer transition-opacity"
              >
                Execute <Send className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="flex items-center gap-2 overflow-x-auto text-[11px] font-mono pt-1">
              <span className="text-slate-500 font-bold shrink-0">Presets:</span>
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  title={p.hint}
                  onClick={() => setQueryInput(p.query)}
                  className="px-2.5 py-1 bg-[#131820] border border-[#232c38] hover:border-slate-500 text-slate-300 rounded-md shrink-0 cursor-pointer transition-colors"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

function Row({ k, v, ok }: { k: string; v: string; ok?: boolean }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="truncate">{k}:</span>
      <span className={`font-bold shrink-0 ${ok ? "text-emerald-400" : "text-slate-200"}`}>{v}</span>
    </div>
  );
}
