import { useMemo, useState } from "react";
import { ApiError, startRun } from "../api";
import type { BenchmarkConfig, MethodInfo, RunStatus, SessionResponse } from "../types";

interface Props {
  methods: MethodInfo[];
  session: SessionResponse;
  status: RunStatus | null;
  onStarted: (s: RunStatus) => void;
}

export default function Optimize({ methods, session, status, onStarted }: Props) {
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [values, setValues] = useState<Record<string, Record<string, string>>>({});
  const [bench, setBench] = useState<BenchmarkConfig>({ warmup_runs: 20, timed_runs: 100, threads: 1 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chosen = useMemo(() => methods.filter((m) => selected[m.key] && m.available), [methods, selected]);
  const active = status && (status.status === "queued" || status.status === "running");
  const setOpt = (method: string, name: string, v: string) =>
    setValues((prev) => ({ ...prev, [method]: { ...prev[method], [name]: v } }));

  const start = async () => {
    setBusy(true);
    setError(null);
    const method_configs: Record<string, Record<string, string | number>> = {};
    for (const m of chosen) {
      const cfg: Record<string, string | number> = {};
      for (const o of m.options) {
        const raw = values[m.key]?.[o.name];
        if (raw !== undefined && raw !== "") cfg[o.name] = o.type === "integer" ? Number(raw) : raw;
      }
      method_configs[m.key] = cfg;
    }
    try {
      onStarted(await startRun({ session_id: session.session_id, methods: chosen.map((m) => m.key), method_configs, benchmark: bench }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the run.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <p className="lede">
        The original FP32 model is always measured first. Each method below is then applied and measured with the same
        images, preprocessing, batch size and timing procedure.
      </p>

      {methods.map((m) => (
        <fieldset key={m.key} className="method" disabled={!m.available || !!active}>
          <legend className="sr-only">{m.label}</legend>
          <label className="method-head">
            <input type="checkbox" checked={!!selected[m.key]} onChange={(e) => setSelected({ ...selected, [m.key]: e.target.checked })} />
            <span className="method-name">{m.label}</span>
          </label>
          <p className="method-desc">{m.description}</p>
          {!m.available && <p className="notice">Unavailable: {m.unavailable_reason}</p>}
          {selected[m.key] && m.available && (
            <div className="method-body">
              <div className="grid2">
                {m.options.map((o) => (
                  <div className="field" key={o.name}>
                    <label htmlFor={`${m.key}-${o.name}`}>{o.label}</label>
                    {o.type === "choice" ? (
                      <select id={`${m.key}-${o.name}`} value={values[m.key]?.[o.name] ?? String(o.default)} onChange={(e) => setOpt(m.key, o.name, e.target.value)}>
                        {o.choices?.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    ) : (
                      <input id={`${m.key}-${o.name}`} type="number" min={o.minimum ?? undefined} max={o.maximum ?? undefined} value={values[m.key]?.[o.name] ?? String(o.default)} onChange={(e) => setOpt(m.key, o.name, e.target.value)} />
                    )}
                    <p className="hint">{o.description}</p>
                  </div>
                ))}
              </div>
              <details className="plain">
                <summary>Limitations</summary>
                <ul className="bullets">{m.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
              </details>
            </div>
          )}
        </fieldset>
      ))}

      <details className="plain">
        <summary>Benchmark settings</summary>
        <div className="grid3">
          <div className="field">
            <label htmlFor="warm">Warm-up runs</label>
            <input id="warm" type="number" min={1} max={500} value={bench.warmup_runs} onChange={(e) => setBench({ ...bench, warmup_runs: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="timed">Timed runs</label>
            <input id="timed" type="number" min={5} max={2000} value={bench.timed_runs} onChange={(e) => setBench({ ...bench, timed_runs: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="threads">CPU threads</label>
            <input id="threads" type="number" min={1} max={64} value={bench.threads} onChange={(e) => setBench({ ...bench, threads: Number(e.target.value) })} />
          </div>
        </div>
        <p className="hint">Latency is one image per forward pass on CPU. Warm-up runs are discarded.</p>
      </details>

      {error && <p role="alert" className="error">{error}</p>}
      {status?.status === "failed" && <p role="alert" className="error">{status.error?.message}</p>}

      {active ? (
        <div className="progress" role="status" aria-live="polite">
          <div className="bar"><span style={{ width: `${(status!.completed_steps / Math.max(status!.total_steps, 1)) * 100}%` }} /></div>
          <p>{status!.stage} <span className="hint inline">step {Math.min(status!.completed_steps + 1, status!.total_steps)} of {status!.total_steps}</span></p>
        </div>
      ) : (
        <div>
          <button type="button" className="btn" disabled={busy || chosen.length === 0} onClick={start}>Run benchmark</button>
          {chosen.length === 0 && <span className="hint inline">Select at least one method.</span>}
        </div>
      )}
    </div>
  );
}
