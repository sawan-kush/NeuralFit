import { artifactUrl } from "../api";
import { formatBytes } from "../format";
import type { RunResults } from "../types";

export default function Downloads({ results }: { results: RunResults }) {
  const label = (name: string): string => {
    if (name === "report.json") return "Results report (JSON)";
    if (name === "baseline_fp32.pt") return "Original model, re-serialized (FP32)";
    const m = results.methods.find((x) => x.artifact_name === name);
    return m ? `${m.label} model` : name;
  };
  const ordered = [...results.artifacts].sort((a, b) => Number(b.name !== "baseline_fp32.pt") - Number(a.name !== "baseline_fp32.pt"));
  return (
    <div className="stack">
      <ul className="downloads">
        {ordered.map((a) => (
          <li key={a.name}>
            <a className="btn secondary" href={artifactUrl(results.run_id, a.name)} download={a.name}>{label(a.name)}</a>
            <span className="hint inline">{a.name}, {formatBytes(a.size_bytes)}</span>
          </li>
        ))}
      </ul>
      <p className="hint">
        Model files are NeuralFit checkpoints: a state dict plus the metadata needed to rebuild the model. The README shows how to
        reload them. INT8 models run only on the CPU backend they were converted for.
      </p>
    </div>
  );
}
