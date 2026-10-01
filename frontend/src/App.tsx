import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError, deleteSession, getArchitectures, getMethods, getResults, getRunStatus,
} from "./api";
import Downloads from "./components/Downloads";
import ModelUpload from "./components/ModelUpload";
import Optimize from "./components/Optimize";
import Rail, { type RailStep } from "./components/Rail";
import Results from "./components/Results";
import type { ArchitectureInfo, MethodInfo, RunResults, RunStatus, SessionResponse } from "./types";

export default function App() {
  const [archs, setArchs] = useState<ArchitectureInfo[]>([]);
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<RunResults | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const resultsRef = useRef<HTMLElement>(null);

  useEffect(() => {
    Promise.all([getArchitectures(), getMethods()])
      .then(([a, m]) => { setArchs(a); setMethods(m); })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Could not load configuration."));
  }, []);

  const runId = status?.run_id;
  const running = status?.status === "queued" || status?.status === "running";

  useEffect(() => {
    if (!runId || !running) return;
    const timer = window.setInterval(async () => {
      try {
        const s = await getRunStatus(runId);
        setStatus(s);
        if (s.status === "completed" || s.status === "failed") {
          setResults(await getResults(runId));
        }
      } catch (e) {
        setPollError(e instanceof ApiError ? e.message : "Lost contact with the server.");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [runId, running]);

  useEffect(() => {
    if (results?.status === "completed") resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [results]);

  const reset = useCallback(async () => {
    if (session) await deleteSession(session.session_id).catch(() => undefined);
    setSession(null); setStatus(null); setResults(null); setPollError(null);
  }, [session]);

  const runAgain = () => { setStatus(null); setResults(null); setPollError(null); };

  const finished = results?.status === "completed";
  const steps: RailStep[] = [
    { id: "model", label: "Model and data", state: session ? "done" : "current" },
    { id: "optimize", label: "Optimization", state: !session ? "locked" : finished ? "done" : "current" },
    { id: "results", label: "Benchmark results", state: finished ? "current" : "locked" },
    { id: "downloads", label: "Downloads", state: finished ? "current" : "locked" },
  ];

  return (
    <div className="shell">
      <Rail steps={steps} />
      <main>
        <header className="masthead">
          <h1>Find out what compression actually does to your model</h1>
          <p>
            Upload weights for a supported image classifier and a validation set. NeuralFit applies FP16 or INT8 conversion and measures accuracy, file size, latency and
            parameters against the original, including where things get worse.
          </p>
        </header>

        {loadError && <p role="alert" className="error">{loadError}</p>}

        <section id="model" className="block" aria-labelledby="h-model">
          <h2 id="h-model">Model and data</h2>
          <ModelUpload archs={archs} session={session} onCreated={setSession} onReset={reset} />
        </section>

        <section id="optimize" className={`block ${session ? "" : "locked"}`} aria-labelledby="h-opt">
          <h2 id="h-opt">Optimization</h2>
          {session ? (
            results ? (
              <p className="lede">Run finished. Use “Run another comparison” in the results section to change settings.</p>
            ) : (
              <Optimize methods={methods} session={session} status={status} onStarted={setStatus} />
            )
          ) : (
            <p className="hint">Validate a model and dataset first.</p>
          )}
          {pollError && <p role="alert" className="error">{pollError}</p>}
        </section>

        <section id="results" ref={resultsRef} className={`block ${results ? "" : "locked"}`} aria-labelledby="h-res">
          <h2 id="h-res">Benchmark results</h2>
          {results ? <Results results={results} onRunAgain={runAgain} /> : <p className="hint">Results appear here after a run.</p>}
        </section>

        <section id="downloads" className={`block ${finished ? "" : "locked"}`} aria-labelledby="h-dl">
          <h2 id="h-dl">Downloads</h2>
          {results && finished ? <Downloads results={results} /> : <p className="hint">Optimized models and the JSON report become available after a completed run.</p>}
        </section>
      </main>
    </div>
  );
}
