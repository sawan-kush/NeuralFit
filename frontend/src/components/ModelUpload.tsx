import { useState } from "react";
import { ApiError, createSession } from "../api";
import { formatBytes } from "../format";
import type { ArchitectureInfo, SessionResponse } from "../types";

interface Props {
  archs: ArchitectureInfo[];
  session: SessionResponse | null;
  onCreated: (s: SessionResponse) => void;
  onReset: () => void;
}

const TREE = `validation_dataset/
  cats/
    img001.jpg
    img002.jpg
  dogs/
    img001.jpg`;

export default function ModelUpload({ archs, session, onCreated, onReset }: Props) {
  const [arch, setArch] = useState("");
  const [weights, setWeights] = useState<File | null>(null);
  const [dataset, setDataset] = useState<File | null>(null);
  const [labelsText, setLabelsText] = useState("");
  const [inputSize, setInputSize] = useState("");
  const [resizeSize, setResizeSize] = useState("");
  const [mean, setMean] = useState<string[] | null>(null);
  const [std, setStd] = useState<string[] | null>(null);
  const [batch, setBatch] = useState("32");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const spec = archs.find((a) => a.key === (arch || archs[0]?.key));
  const labels = labelsText.split("\n").map((l) => l.trim()).filter(Boolean);
  const meanVals = mean ?? spec?.default_mean.map(String) ?? ["", "", ""];
  const stdVals = std ?? spec?.default_std.map(String) ?? ["", "", ""];
  const ready = !!spec && !!weights && !!dataset && labels.length >= 2 && !busy;

  const loadLabels = async (file: File | undefined) => {
    if (file) setLabelsText((await file.text()).trim());
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!spec || !weights || !dataset) return;
    setBusy(true);
    setError(null);
    const config = {
      class_labels: labels,
      batch_size: Number(batch),
      preprocessing: {
        input_size: Number(inputSize || spec.default_input_size),
        ...(resizeSize ? { resize_size: Number(resizeSize) } : {}),
        mean: meanVals.map(Number),
        std: stdVals.map(Number),
      },
    };
    const form = new FormData();
    form.append("architecture", spec.key);
    form.append("config", JSON.stringify(config));
    form.append("weights", weights);
    form.append("dataset_zip", dataset);
    try {
      onCreated(await createSession(form));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  if (session) {
    const arch = archs.find((a) => a.key === session.architecture);
    return (
      <div>
        <dl className="facts">
          <div><dt>Architecture</dt><dd>{arch?.display_name ?? session.architecture}</dd></div>
          <div><dt>Weights file</dt><dd>{formatBytes(session.uploaded_weights_size_bytes)}</dd></div>
          <div><dt>Classes</dt><dd>{session.num_classes}</dd></div>
          <div><dt>Validation images</dt><dd>{session.total_images.toLocaleString()}</dd></div>
          <div><dt>Input</dt><dd>{session.preprocessing.input_size}×{session.preprocessing.input_size}</dd></div>
        </dl>
        <details className="plain">
          <summary>Images per class</summary>
          <table className="table compact">
            <tbody>
              {Object.entries(session.class_counts).map(([k, v]) => (
                <tr key={k}><th scope="row">{k}</th><td className="num">{v}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
        {session.warnings.map((w) => (
          <p key={w} className="notice">{w}</p>
        ))}
        <p className="hint">Inputs are kept on the server for {session.retention_hours} hours unless you remove them.</p>
        <button type="button" className="btn secondary" onClick={onReset}>Replace model and dataset</button>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="stack">
      <div className="field">
        <label htmlFor="arch">Architecture</label>
        <select id="arch" value={spec?.key ?? ""} onChange={(e) => { setArch(e.target.value); setInputSize(""); setResizeSize(""); setMean(null); setStd(null); }}>
          {archs.map((a) => <option key={a.key} value={a.key}>{a.display_name}</option>)}
        </select>
        <p className="hint">{spec?.description} Only these architectures are supported.</p>
      </div>

      <div className="field">
        <label htmlFor="weights">Model weights</label>
        <input id="weights" type="file" accept=".pt,.pth" onChange={(e) => setWeights(e.target.files?.[0] ?? null)} />
        <p className="hint">
          A state dict saved with <code>torch.save(model.state_dict(), "weights.pt")</code>. Full pickled models are rejected
          because loading them could run code.
        </p>
      </div>

      <div className="field">
        <label htmlFor="dataset">Validation dataset (.zip)</label>
        <input id="dataset" type="file" accept=".zip" onChange={(e) => setDataset(e.target.files?.[0] ?? null)} />
        <p className="hint">One folder per class, named exactly like the class labels below. JPG, PNG or BMP images.</p>
        <pre className="tree">{TREE}</pre>
      </div>

      <div className="field">
        <label htmlFor="labels">Class labels, in the model's output order</label>
        <textarea id="labels" rows={5} value={labelsText} onChange={(e) => setLabelsText(e.target.value)} placeholder={"cats\ndogs"} />
        <p className="hint">
          One per line. Line 1 is output index 0. {labels.length > 0 && <strong>{labels.length} labels.</strong>}{" "}
          <label className="link-file">
            Load from a .txt file
            <input type="file" accept=".txt" onChange={(e) => loadLabels(e.target.files?.[0])} />
          </label>
        </p>
      </div>

      <details className="plain">
        <summary>Preprocessing (defaults: ImageNet)</summary>
        <div className="grid2">
          <div className="field">
            <label htmlFor="size">Input size (px)</label>
            <input id="size" type="number" min={32} max={512} value={inputSize} placeholder={String(spec?.default_input_size ?? 224)} onChange={(e) => setInputSize(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="resize">Resize shorter side to (px)</label>
            <input id="resize" type="number" min={32} max={1024} value={resizeSize} placeholder="auto" onChange={(e) => setResizeSize(e.target.value)} />
          </div>
        </div>
        <fieldset className="triple">
          <legend>Normalization mean (R, G, B)</legend>
          {meanVals.map((v, i) => (
            <input key={i} aria-label={`mean ${i + 1}`} type="number" step="any" value={v} onChange={(e) => setMean(meanVals.map((x, j) => (j === i ? e.target.value : x)))} />
          ))}
        </fieldset>
        <fieldset className="triple">
          <legend>Normalization std (R, G, B)</legend>
          {stdVals.map((v, i) => (
            <input key={i} aria-label={`std ${i + 1}`} type="number" step="any" min="0" value={v} onChange={(e) => setStd(stdVals.map((x, j) => (j === i ? e.target.value : x)))} />
          ))}
        </fieldset>
        <div className="field narrow">
          <label htmlFor="batch">Evaluation batch size</label>
          <input id="batch" type="number" min={1} max={256} value={batch} onChange={(e) => setBatch(e.target.value)} />
        </div>
      </details>

      {error && <p role="alert" className="error">{error}</p>}
      <div>
        <button type="submit" className="btn" disabled={!ready}>{busy ? "Validating\u2026" : "Validate inputs"}</button>
        {!ready && !busy && <span className="hint inline">Add weights, a dataset ZIP and at least two labels.</span>}
      </div>
    </form>
  );
}
