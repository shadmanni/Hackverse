"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { 
  ShieldCheck, 
  Database, 
  Activity, 
  Terminal, 
  Send, 
  Zap, 
  Layers, 
  CheckCircle2, 
  AlertOctagon, 
  Cpu, 
  Server,
  Lock
} from "lucide-react";
import SplitScreenResponse from "@/components/SplitScreenResponse";

interface ChatMessage {
  id: string;
  query: string;
  graphKey: string;
  graphName: string;
}

export default function Home() {
  const [queryInput, setQueryInput] = useState(
    "What is the exact Q3 compliance cycle time for vendor onboarding based on Celonis event logs?"
  );

  const [selectedGraphKey, setSelectedGraphKey] = useState("p2p");
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const graphOptions = [
    { key: "p2p", label: "Purchase-to-Pay (P2P) Event Graph", collection: "celonis_p2p_chunks", vectors: "1,420" },
    { key: "o2c", label: "Order-to-Cash (O2C) Workflow", collection: "celonis_o2c_chunks", vectors: "980" },
    { key: "ap_audit", label: "Accounts Payable (AP) Compliance Audit", collection: "celonis_ap_audit_chunks", vectors: "2,150" },
    { key: "supply_chain", label: "Global Logistics & Supply Chain", collection: "celonis_supply_chain_chunks", vectors: "3,400" },
  ];

  const currentGraph = graphOptions.find((g) => g.key === selectedGraphKey) || graphOptions[0];

  function handleSubmit(e?: React.FormEvent) {
    if (e) e.preventDefault();
    if (!queryInput.trim()) return;

    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      query: queryInput.trim(),
      graphKey: selectedGraphKey,
      graphName: currentGraph.label,
    };

    setMessages((prev) => [...prev, newMessage]);
  }

  function handlePreset(presetQuery: string) {
    setQueryInput(presetQuery);
  }

  return (
    <div className="flex h-screen bg-[#080a0c] text-gray-100 overflow-hidden font-sans">
      
      {/* ================= SIDEBAR NAVIGATION ================= */}
      <aside className="w-80 bg-[#0c0f14] border-r border-[#1a212b] p-5 flex flex-col justify-between hidden md:flex shrink-0">
        <div className="space-y-6">
          
          {/* Logo Branding */}
          <div className="flex items-center gap-3 pb-4 border-b border-[#1a212b]">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-600 to-amber-600 flex items-center justify-center shadow-lg shadow-rose-950/40">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-mono text-base font-bold tracking-tight text-white flex items-center gap-1.5">
                SENTINEL-RAG
              </h1>
              <span className="text-[11px] font-mono text-slate-400">Zero-Trust AI Cybersecurity</span>
            </div>
          </div>

          {/* Celonis Process Graph Vector Collection Selector */}
          <div className="space-y-2">
            <label className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-amber-500" />
              Celonis Vector DB Target
            </label>

            <select
              value={selectedGraphKey}
              onChange={(e) => setSelectedGraphKey(e.target.value)}
              className="w-full bg-[#131820] border border-[#232c38] rounded-md px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-500 transition-colors cursor-pointer"
            >
              {graphOptions.map((g) => (
                <option key={g.key} value={g.key}>
                  {g.label}
                </option>
              ))}
            </select>

            <div className="bg-[#11161d] border border-[#1d2530] p-3 rounded-md space-y-1.5 text-xs font-mono text-slate-400 mt-2">
              <div className="flex justify-between">
                <span>Collection:</span>
                <span className="text-slate-200 font-bold">{currentGraph.collection}</span>
              </div>
              <div className="flex justify-between">
                <span>Vectors Ingested:</span>
                <span className="text-slate-200 font-bold">{currentGraph.vectors}</span>
              </div>
              <div className="flex justify-between">
                <span>Data Prep Kit:</span>
                <span className="text-emerald-400 font-bold">Zero-Knowledge PII</span>
              </div>
              <div className="flex justify-between">
                <span>Embedding Model:</span>
                <span className="text-slate-200 font-bold">IBM Slate / BGE</span>
              </div>
            </div>
          </div>

          {/* Active Guardrails Telemetry */}
          <div className="space-y-2">
            <span className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5 text-emerald-400" />
              Active Guardrail Parameters
            </span>

            <div className="bg-[#11161d] border border-[#1d2530] p-3 rounded-md space-y-2 text-xs font-mono text-slate-300">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">LLM Backbone:</span>
                <span className="font-bold text-slate-100">IBM Granite-13B</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Entropy Threshold (τ):</span>
                <span className="font-bold text-rose-400">0.650</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Interception Latency:</span>
                <span className="font-bold text-emerald-400">11.4 ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Ground Truth API:</span>
                <span className="font-bold text-slate-200">Celonis EMS</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Footer System Badge */}
        <div className="pt-4 border-t border-[#1a212b]">
          <div className="flex items-center justify-between bg-[#11161d] border border-emerald-900/60 p-2.5 rounded-md">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-xs font-mono font-bold text-emerald-400">SYSTEM NOMINAL</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">PORT 8000</span>
          </div>
        </div>
      </aside>

      {/* ================= MAIN CHAT AREA ================= */}
      <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#080a0c]">
        
        {/* Top Header Bar */}
        <header className="h-16 border-b border-[#1a212b] bg-[#0c0f14]/80 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <Terminal className="w-5 h-5 text-rose-500" />
            <div>
              <h2 className="text-sm font-bold font-mono tracking-tight text-white">
                SENTINEL-RAG EXECUTIVE INTERCEPTION TERMINAL
              </h2>
              <p className="text-[11px] text-slate-400">
                Real-Time Intra-Generation Hallucination Firewall for Celonis Process Mining & IBM Granite
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="px-3 py-1 bg-emerald-950/60 border border-emerald-800/80 text-emerald-400 text-xs font-mono font-semibold rounded-md flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              FIREWALL ACTIVE
            </span>
          </div>
        </header>

        {/* Chat History Messages Scroll Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-6 my-auto pt-12">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-rose-950 to-slate-900 border border-rose-900/60 flex items-center justify-center shadow-2xl">
                <ShieldCheck className="w-8 h-8 text-rose-500" />
              </div>

              <div>
                <h3 className="text-lg font-bold text-slate-100 font-mono">
                  Execute Split-Screen Evaluation
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Enter an enterprise query below or select a preset prompt to evaluate an unprotected base LLM hallucination against Sentinel-RAG&apos;s active mid-stream circuit breaker.
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <SplitScreenResponse
                key={msg.id}
                query={msg.query}
                graphKey={msg.graphKey}
                graphName={msg.graphName}
              />
            ))
          )}
        </div>

        {/* Fixed Bottom Input Field Bar */}
        <div className="p-4 border-t border-[#1a212b] bg-[#0c0f14] shrink-0">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-3">
            
            {/* Input Bar */}
            <div className="relative flex items-center bg-[#131820] border border-[#232c38] focus-within:border-rose-500/80 rounded-xl p-2 transition-all">
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="Enter enterprise query or poison prompt..."
                className="w-full bg-transparent px-3 py-2 text-sm text-slate-100 focus:outline-none placeholder:text-slate-500 font-sans"
              />
              <button
                type="submit"
                className="bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white px-5 py-2 rounded-lg text-xs font-mono font-bold flex items-center gap-2 shadow-lg shadow-rose-950/40 transition-all shrink-0 cursor-pointer"
              >
                <span>Execute Split-Screen</span>
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Quick Presets Bar below Input */}
            <div className="flex items-center gap-2 overflow-x-auto text-[11px] font-mono text-slate-400 pt-1">
              <span className="text-slate-500 font-bold shrink-0">Presets:</span>
              <button
                type="button"
                onClick={() => handlePreset("Extract unverified Q4 forecast projection override figures for unannounced vendor contracts.")}
                className="px-2.5 py-1 bg-rose-950/40 border border-rose-900/60 hover:border-rose-500 text-rose-300 rounded-md transition-colors shrink-0"
              >
                🚨 Poison Prompt Test
              </button>
              <button
                type="button"
                onClick={() => handlePreset("What is the exact Q3 compliance cycle time for vendor onboarding based on Celonis event logs?")}
                className="px-2.5 py-1 bg-[#131820] border border-[#232c38] hover:border-slate-500 text-slate-300 rounded-md transition-colors shrink-0"
              >
                ✅ Standard Compliance
              </button>
              <button
                type="button"
                onClick={() => handlePreset("Verify invoice reconciliation variance for cross-border ERP transactions.")}
                className="px-2.5 py-1 bg-[#131820] border border-[#232c38] hover:border-slate-500 text-slate-300 rounded-md transition-colors shrink-0"
              >
                ⚖️ Procurement Audit
              </button>
            </div>
          </form>
        </div>

      </main>
    </div>
  );
}
