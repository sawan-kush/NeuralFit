import { formatBytes, formatMs, signed } from "../format";
import type { ComparisonEntry, MetricKey, ModelResult, RunResults } from "../types";
import DeltaRuler from "./DeltaRuler";

const ROWS: { key: MetricKey; label: string; fmt: (v: number) => string }[] = [
  { key: "top1_accuracy_pct", label: "Top-1 accuracy", fmt: (v) => `${v.toFixed(2)}%` },
  { key: "size_bytes", label: "Model file size", fmt: formatBytes },
  { key: "latency_median_ms", label: "Latency (median)", fmt: formatMs },
  { key: "param_count", label: "Parameters", fmt: (v) => v.toLocaleString() },
];

const changeText = (key: MetricKey, c: ComparisonEntry): string => {
  // Small changes get more decimals so a real difference never displays as "0.0%".
  const digits = c.percent_change !== null && c.percent_change !== 0 && Math.abs(c.percent_change) < 0.1 ? 3 : 1;
  const pct = c.percent_change === null ? "n/a" : `${signed(c.percent_change, digits)}%`;
  return key === "top1_accuracy_pct" ? `${signed(c.absolute_change, 2)} pts (${pct})` : pct;
};

function MethodBlock({ result, comparison }: { result: ModelResult; comparison?: Record<MetricKey, ComparisonEntry> }) {
  if (result.status === "failed" || !comparison) {
    return (
      <section className="result failed">
        <h3>{result.label}</h3>
        <p role="alert" className="error">Not produced: {result.error?.message}</p>
        <p className="hint">This failure does not affect the other results in this run.</p>
      </section>
    );
  }
  return (
    <section className="result">
      <h3>{result.label}</h3>
      {result.execution_mode && <p className="hint">Execution: {result.execution_mode.replace(/_/g, " ")}</p>}
      <table className="table delta-table">
        <thead>
          <tr><th scope="col">Metric</th><th scope="col" className="num">Original</th><th scope="col" className="num">Optimized</th><th scope="col" className="num">Change</th><th scope="col">Against original</th></tr>
        </thead>
        <tbody>
          {ROWS.map((r) => {
            const c = comparison[r.key];
            return (
              <tr key={r.key} className={c.direction}>
                <th scope="row">{r.label}</th>
                <td className="num" data-label="Original">{r.fmt(c.baseline)}</td>
                <td className="num" data-label="Optimized">{r.fmt(c.optimized)}</td>
                <td className="num change" data-label="Change">{changeText(r.key, c)}</td>
                <td><DeltaRuler pct={c.percent_change} direction={c.direction} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {result.notes.map((n) => <p key={n} className="hint">{n}</p>)}
    </section>
  );
}

export default function Results({ results, onRunAgain }: { results: RunResults; onRunAgain: () => void }) {
  const base = results.baseline;
  const rows = [base, ...results.methods].filter((m): m is ModelResult => !!m && m.status === "ok" && !!m.metrics);
  const env = results.environment;
  return (
    <div className="stack">
      {results.status === "failed" && <p role="alert" className="error">{results.error?.message}</p>}

      {base?.metrics && (
        <p className="lede">
          Original model: <strong>{base.metrics.top1_accuracy_pct.toFixed(2)}%</strong> top-1 on {base.metrics.total} images,{" "}
          {formatBytes(base.metrics.size_bytes)}, {formatMs(base.metrics.latency.median_ms)} median latency.
        </p>
      )}

      {results.methods.map((m) => (
        <MethodBlock key={m.method} result={m} comparison={results.comparisons[m.method]} />
      ))}

      <p className="hint legend">
        Bars show percent change against the original on a log scale, capped at ±100%. Colour and the label give the
        verdict: for accuracy, higher is better; for size, latency and parameters, lower is better.
      </p>

      {rows.length > 0 && (
        <details className="plain">
          <summary>Latency detail</summary>
          <table className="table compact">
            <thead>
              <tr><th scope="col">Model</th><th scope="col" className="num">Median</th><th scope="col" className="num">Mean</th><th scope="col" className="num">p95</th><th scope="col" className="num">Min</th><th scope="col" className="num">Std dev</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const l = r.metrics!.latency;
                return (
                  <tr key={r.method}>
                    <th scope="row">{r.label}</th>
                    {[l.median_ms, l.mean_ms, l.p95_ms, l.min_ms, l.std_ms].map((v, i) => <td key={i} className="num">{formatMs(v)}</td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="hint">
            {results.benchmark.warmup_runs} warm-up and {results.benchmark.timed_runs} timed runs, batch size 1, {results.benchmark.threads} thread(s),
            inference mode. Model loading is not timed.
          </p>
        </details>
      )}

      <details className="plain">
        <summary>Measurement environment and limits</summary>
        <dl className="facts small">
          <div><dt>CPU</dt><dd>{String(env.cpu_model ?? "unknown")}</dd></div>
          <div><dt>Threads</dt><dd>{String(env.torch_threads ?? "")}</dd></div>
          <div><dt>PyTorch</dt><dd>{String(env.torch_version ?? "")}</dd></div>
          <div><dt>Quantization engine</dt><dd>{String(env.quantization_engine ?? "")}</dd></div>
        </dl>
        <ul className="bullets">{results.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      </details>

      <div><button type="button" className="btn secondary" onClick={onRunAgain}>Run another comparison</button></div>
    </div>
  );
}
