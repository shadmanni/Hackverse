"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  ShieldCheck,
  Zap,
  Cpu,
  Clock,
  Bot,
  CheckCircle2,
  Activity,
} from "lucide-react";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/** One SSE event from the interception proxy. */
interface StreamEvent {
  kind: "token" | "intercept" | "recovery_start" | "recovery_token" | "recovery" | "done" | "error";
  text?: string;
  logprob?: number;
  probability?: number;
  uncertainty?: number;
  rolling_variance?: number;
  shannon_entropy?: number;
  index?: number;
  // intercept
  reason?: string;
  token?: string;
  token_index?: number;
  tau?: number;
  breach_run?: number;
  ungrounded_value?: number | null;
  tokens_before_halt?: number;
  semantic_entropy?: number;
  figure_at_stake?: boolean;
  /** the decoder's top-k where the figure was chosen, clustered by value */
  candidates?: { token: string; p: number; cluster: string }[] | null;
  candidates_token_index?: number | null;
  divergence_explained?: string | null;
  // done
  tokens?: number;
  elapsed_ms?: number;
  entropy_overhead_ms?: number;
  overhead_pct?: number;
  message?: string;
  // recovery
  strategy?: string;
  retrieval_reason?: string;
  regenerated?: boolean;
  all_figures_grounded?: boolean;
  /** the regeneration was rejected and replaced by the deterministic lookup */
  fell_back_to_lookup?: boolean;
}

interface PanelState {
  text: string;
  uncertainty: number[];
  done: boolean;
  intercept: StreamEvent | null;
  summary: StreamEvent | null;
  error: string | null;
}

const EMPTY: PanelState = {
  text: "",
  uncertainty: [],
  done: false,
  intercept: null,
  summary: null,
  error: null,
};

/**
 * Reads an SSE body, buffering across chunk boundaries.
 *
 * The previous implementation split each raw chunk on "\n\n" and dropped
 * whatever straddled two chunks, silently losing tokens. It then re-split every
 * token on spaces and slept 75 ms per word, so the UI animated at a fixed rate
 * that had nothing to do with the model and compounded the backend's own delay.
 * Tokens now render exactly when Granite emits them.
 */
async function readSSE(
  url: string,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch(url, { signal });
  if (!resp.ok || !resp.body) throw new Error(`backend returned ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      if (!frame.startsWith("data: ")) continue;
      try {
        onEvent(JSON.parse(frame.slice(6)) as StreamEvent);
      } catch {
        /* ignore a malformed frame rather than killing the stream */
      }
    }
  }
}

/** Uncertainty per token against tau. Breaching points are marked. */
function EntropyChart({ series, tau }: { series: number[]; tau: number }) {
  const W = 260;
  const H = 54;
  if (series.length < 2) {
    return (
      <div className="h-[54px] flex items-center justify-center text-[10px] text-slate-600 font-mono">
        awaiting tokens…
      </div>
    );
  }
  const peak = Math.max(tau * 1.6, ...series);
  const x = (i: number) => (i / (series.length - 1)) * W;
  const y = (v: number) => H - (v / peak) * H;
  const path = series.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[54px]" preserveAspectRatio="none"
         role="img" aria-label={`Uncertainty per token against threshold ${tau}`}>
      {/* threshold */}
      <line x1="0" y1={y(tau)} x2={W} y2={y(tau)} stroke="#f43f5e" strokeWidth="1"
            strokeDasharray="3 3" opacity="0.75" />
      <path d={path} fill="none" stroke="#34d399" strokeWidth="1.4" />
      {series.map((v, i) =>
        v > tau ? <circle key={i} cx={x(i)} cy={y(v)} r="1.9" fill="#f43f5e" /> : null,
      )}
    </svg>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="bg-[#141920] border border-slate-800 p-2.5 rounded-lg min-w-0">
      <span className="text-[10px] text-slate-400 font-mono block truncate">{label}</span>
      <span className={`text-xs font-bold font-mono mt-0.5 block truncate ${tone}`}>{value}</span>
    </div>
  );
}

interface Props {
  query: string;
  graphName: string;
  onDone?: () => void;
}

export default function SplitScreenResponse({ query, graphName, onDone }: Props) {
  const [left, setLeft] = useState<PanelState>(EMPTY);   // unprotected
  const [right, setRight] = useState<PanelState>(EMPTY); // sentinel
  const [healing, setHealing] = useState<{
    strategy: string;
    retrievalReason?: string;
    text: string;
    streaming: boolean;
    verified?: boolean;
    fellBack?: boolean;
  } | null>(null);
  const tau = right.intercept?.tau ?? 0.65;
  const startedAt = useRef<number>(0);

  useEffect(() => {
    const ctrl = new AbortController();
    startedAt.current = performance.now();
    setLeft(EMPTY);
    setRight(EMPTY);
    setHealing(null);

    const apply =
      (set: React.Dispatch<React.SetStateAction<PanelState>>) => (e: StreamEvent) =>
        set((p) => {
          switch (e.kind) {
            case "token":
              return {
                ...p,
                text: p.text + (e.text ?? ""),
                uncertainty: [...p.uncertainty, e.uncertainty ?? 0],
              };
            case "intercept":
              return { ...p, intercept: e, done: true };
            case "recovery_start":
            case "recovery_token":
            case "recovery":
              return p;
            case "done":
              return { ...p, done: true, summary: e };
            case "error":
              return { ...p, done: true, error: e.message ?? "stream error" };
            default:
              return p;
          }
        });

    const q = encodeURIComponent(query);

    // Sequential, not concurrent: one model on one device serves both panels, so
    // firing them together would just queue on the decode lock and make the
    // latency comparison meaningless.
    (async () => {
      try {
        await readSSE(`${BACKEND}/unprotected_stream?query=${q}`, apply(setLeft), ctrl.signal);
      } catch (err) {
        if (!ctrl.signal.aborted) {
          setLeft((p) => ({ ...p, done: true, error: String(err) }));
        }
      }
      if (ctrl.signal.aborted) return;
      try {
        await readSSE(
          `${BACKEND}/stream?query=${q}`,
          (e) => {
            apply(setRight)(e);
            if (e.kind === "recovery_start") {
              setHealing({
                strategy: e.strategy ?? "context_repair",
                retrievalReason: e.retrieval_reason,
                text: "",
                streaming: true,
              });
            } else if (e.kind === "recovery_token") {
              setHealing((h) => (h ? { ...h, text: h.text + (e.text ?? "") } : h));
            } else if (e.kind === "recovery") {
              setHealing((h) => ({
                strategy: e.strategy ?? h?.strategy ?? "context_repair",
                retrievalReason: e.retrieval_reason ?? h?.retrievalReason,
                text: e.text ?? h?.text ?? "",
                streaming: false,
                verified: e.all_figures_grounded,
                fellBack: e.fell_back_to_lookup,
              }));
            }
          },
          ctrl.signal,
        );
      } catch (err) {
        if (!ctrl.signal.aborted) {
          setRight((p) => ({ ...p, done: true, error: String(err) }));
        }
      }
      // Notify parent that both streams have finished
      if (!ctrl.signal.aborted) onDone?.();
    })();

    return () => ctrl.abort();
  }, [query]);

  const leftTokens = left.summary?.tokens ?? left.uncertainty.length;
  const rightTokens = right.intercept?.tokens_before_halt ?? right.summary?.tokens ?? right.uncertainty.length;
  const halted = right.intercept !== null;
  const saved = halted ? Math.max(0, leftTokens - rightTokens) : 0;
  const savedPct = halted && leftTokens > 0 ? Math.round((saved / leftTokens) * 100) : 0;

  const fmtMs = (v?: number) => (v === undefined ? "—" : `${v.toFixed(0)} ms`);
  const overhead =
    right.summary?.overhead_pct !== undefined
      ? `${right.summary.overhead_pct.toFixed(3)}%`
      : right.intercept?.entropy_overhead_ms !== undefined
        ? `${right.intercept.entropy_overhead_ms.toFixed(2)} ms`
        : "measuring…";

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ---------------- UNPROTECTED ---------------- */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-[#0e1217] border border-rose-900/40 rounded-xl p-5 shadow-2xl flex flex-col"
        >
          <div className="flex items-center justify-between border-b border-rose-900/40 pb-3 mb-4">
            <div className="flex items-center gap-2 min-w-0">
              <AlertTriangle className="w-5 h-5 text-rose-500 shrink-0" />
              <h3 className="text-sm font-bold uppercase text-rose-400 font-mono truncate">
                Unprotected Granite
              </h3>
            </div>
            <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-rose-950/60 border border-rose-800/60 text-rose-400 rounded shrink-0">
              NO CONTEXT · NO PROXY
            </span>
          </div>

          <div className="bg-[#06080a] border border-rose-950/80 rounded-lg p-4 font-mono text-xs leading-relaxed text-slate-300 min-h-[150px]">
            <div className="flex items-center justify-between text-[11px] text-slate-500 border-b border-slate-900 pb-2 mb-3">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                UNGUARDED DECODER
              </span>
              <span className="text-rose-400/80 font-bold">UNVERIFIED</span>
            </div>
            <div className="whitespace-pre-wrap break-words">
              {left.error ? (
                <span className="text-rose-400">backend unreachable — {left.error}</span>
              ) : left.text ? (
                <>
                  {left.text}
                  {!left.done && <span className="inline-block w-2 h-4 bg-rose-500 ml-1 animate-pulse" />}
                </>
              ) : (
                <span className="text-slate-600 italic">decoding…</span>
              )}
            </div>
          </div>

          <div className="mt-4 border-t border-slate-800/80 pt-3">
            <EntropyChart series={left.uncertainty} tau={tau} />
            <div className="grid grid-cols-3 gap-2 mt-3">
              <Stat label="TOKENS EMITTED" value={left.done ? String(leftTokens) : "…"} tone="text-rose-400" />
              <Stat label="DECODE TIME" value={fmtMs(left.summary?.elapsed_ms)} tone="text-amber-400" />
              <Stat label="INTERVENTION" value="none" tone="text-rose-400" />
            </div>
          </div>
        </motion.div>

        {/* ---------------- SENTINEL ---------------- */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className={`bg-[#0e1217] border rounded-xl p-5 shadow-2xl flex flex-col transition-colors duration-300 ${
            halted ? "border-rose-600/80" : "border-emerald-900/40"
          }`}
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
            <div className="flex items-center gap-2 min-w-0">
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
              <h3 className="text-sm font-bold uppercase text-emerald-400 font-mono truncate">
                Sentinel-RAG Firewall
              </h3>
            </div>
            <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 rounded flex items-center gap-1 shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              ACTIVE
            </span>
          </div>

          <div
            className={`bg-[#06080a] border rounded-lg p-4 font-mono text-xs leading-relaxed text-slate-300 min-h-[150px] transition-colors ${
              halted ? "border-rose-600/60" : "border-emerald-950/80"
            }`}
          >
            <div className="flex items-center justify-between text-[11px] text-slate-500 border-b border-slate-900 pb-2 mb-3">
              <span className="flex items-center gap-1.5 text-slate-400">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                ENTROPY MONITOR · τ = {tau}
              </span>
              {halted ? (
                <span className="text-rose-400 font-bold flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3" /> TRIPPED
                </span>
              ) : (
                <span className="text-emerald-400 font-bold">NOMINAL</span>
              )}
            </div>

            <div className="whitespace-pre-wrap break-words">
              {right.error ? (
                <span className="text-rose-400">backend unreachable — {right.error}</span>
              ) : (
                <>
                  {right.text}
                  {halted && (
                    <motion.span
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="inline-flex items-center gap-1 ml-1 bg-rose-600 text-white font-bold px-2 py-0.5 rounded text-[11px]"
                    >
                      ⏹ HALTED AT TOKEN #{right.intercept?.token_index}
                    </motion.span>
                  )}
                  {!right.done && <span className="inline-block w-2 h-4 bg-emerald-400 ml-1 animate-pulse" />}
                </>
              )}
            </div>

            <AnimatePresence>
              {halted && right.intercept && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 bg-rose-950/70 border border-rose-600/80 rounded-md p-3"
                >
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Zap className="w-4 h-4 text-rose-400" />
                    <span className="text-xs font-bold text-rose-400 font-mono">
                      CIRCUIT BREAKER TRIPPED MID-STREAM
                    </span>
                  </div>
                  <dl className="text-[11px] text-slate-300 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
                    <dt className="text-slate-500">reason</dt>
                    <dd>{right.intercept.reason}</dd>
                    {right.intercept.ungrounded_value != null && (
                      <>
                        <dt className="text-slate-500">figure not in log</dt>
                        <dd className="text-rose-300 font-bold">{right.intercept.ungrounded_value}</dd>
                      </>
                    )}
                    {right.intercept.reason === "semantic_entropy" && (
                      <>
                        <dt className="text-slate-500">uncertainty</dt>
                        <dd>
                          {right.intercept.uncertainty?.toFixed(3)} &gt; τ {right.intercept.tau}
                          {right.intercept.breach_run ? ` ×${right.intercept.breach_run}` : ""}
                        </dd>
                      </>
                    )}
                    <dt className="text-slate-500">token p</dt>
                    <dd>{right.intercept.probability?.toFixed(4)}</dd>
                    <dt className="text-slate-500">halted after</dt>
                    <dd>{right.intercept.tokens_before_halt} tokens</dd>
                  </dl>

                  {/* What the decoder was choosing between at the halt. The
                      scalar entropies say uncertainty was high; this says what
                      it was about, which is the only form of it a judge can
                      check by eye. */}
                  {!!right.intercept.candidates?.length && (
                    <div className="mt-3 pt-2.5 border-t border-rose-800/50">
                      {/* Deliberately not "at the halt": the breaker trips when
                          the figure terminates, one or more tokens after the
                          value was chosen. This is the choosing token. */}
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                        Candidates where the figure was chosen
                        {right.intercept.candidates_token_index != null
                          ? ` · token #${right.intercept.candidates_token_index}`
                          : ""}
                      </span>
                      <div className="mt-1.5 space-y-1">
                        {right.intercept.candidates.map((c, i) => {
                          const isFigure = c.cluster.startsWith("#");
                          return (
                            <div key={i} className="flex items-center gap-2 font-mono text-[11px]">
                              <span
                                className={`w-14 shrink-0 truncate text-right ${
                                  isFigure ? "text-rose-300 font-bold" : "text-slate-500"
                                }`}
                              >
                                {c.token.trim() || "␣"}
                              </span>
                              <div className="flex-1 h-2 bg-black/40 rounded-sm overflow-hidden">
                                <div
                                  className={`h-full ${isFigure ? "bg-rose-500" : "bg-slate-700"}`}
                                  style={{ width: `${Math.max(c.p * 100, 1)}%` }}
                                />
                              </div>
                              <span
                                className={`w-11 shrink-0 tabular-nums ${
                                  isFigure ? "text-rose-300" : "text-slate-500"
                                }`}
                              >
                                {(c.p * 100).toFixed(1)}%
                              </span>
                            </div>
                          );
                        })}
                      </div>
                      {right.intercept.divergence_explained && (
                        <p className="mt-2 text-[10px] leading-relaxed text-slate-400 italic">
                          {right.intercept.divergence_explained}
                        </p>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {healing && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ delay: 0.15 }}
                  className="mt-3 bg-[#0a1117] border-l-2 border-emerald-500 p-3 rounded-r-md overflow-hidden"
                >
                  <div className="flex items-center gap-1.5 text-amber-400 text-[11px] font-mono font-bold mb-1">
                    <Bot className={`w-4 h-4 ${healing.streaming ? "animate-pulse" : ""}`} />
                    Self-Healing Agent · {healing.strategy}
                  </div>
                  {healing.retrievalReason && (
                    <p className="text-[10px] text-slate-500 font-mono mb-1.5">
                      context gap repaired — {healing.retrievalReason}
                    </p>
                  )}
                  <p className="text-[11px] text-slate-200 leading-relaxed break-words">
                    {healing.text || (
                      <span className="text-slate-500 italic">re-querying Milvus and regenerating…</span>
                    )}
                    {healing.streaming && (
                      <span className="inline-block w-1.5 h-3 bg-amber-400 ml-1 animate-pulse" />
                    )}
                  </p>
                  {!healing.streaming && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] font-mono border-t border-slate-800/80 pt-1.5">
                      {/* Three states, not two. The backend emits
                          fell_back_to_lookup, and the UI ignored it: when the
                          regeneration was REJECTED and swapped for the log
                          lookup, all_figures_grounded described the replacement
                          and this still printed the green "regenerated under
                          grounding" tick. The panel claimed a regeneration that
                          had been thrown away, in a product about unverifiable
                          claims. Both paths end grounded, so nothing was unsafe
                          - but which one ran has to be visible. */}
                      {!healing.verified ? (
                        <span className="text-amber-400">
                          answer still contains an unverifiable figure
                        </span>
                      ) : healing.fellBack ? (
                        <span className="flex items-center gap-1 text-sky-400">
                          <CheckCircle2 className="w-3 h-3" />
                          regeneration rejected — answered from a deterministic log lookup
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-emerald-400">
                          <CheckCircle2 className="w-3 h-3" />
                          regenerated under grounding — every figure verified against the event log
                        </span>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="mt-4 border-t border-slate-800/80 pt-3">
            <EntropyChart series={right.uncertainty} tau={tau} />
            <div className="grid grid-cols-3 gap-2 mt-3">
              <Stat
                label="TOKENS SAVED"
                value={halted ? `${saved} (${savedPct}%)` : right.done ? "grounded" : "…"}
                tone="text-emerald-400"
              />
              <Stat
                label="TIME TO HALT"
                value={fmtMs(right.intercept?.elapsed_ms ?? right.summary?.elapsed_ms)}
                tone="text-emerald-300"
              />
              <Stat label="AUDIT OVERHEAD" value={overhead} tone="text-emerald-400" />
            </div>
          </div>
        </motion.div>
      </div>

      <p className="text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
        <Activity className="w-3 h-3" />
        Both panels are the same local Granite model running the identical prompt, and
        neither is given retrieved context up front — so exactly one variable differs, the
        audit itself. The right panel is scored per token and retrieves only once it
        intercepts, to repair the gap it found. Nothing is pre-scripted.
      </p>
    </div>
  );
}
