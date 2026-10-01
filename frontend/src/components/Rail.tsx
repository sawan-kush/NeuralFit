export type StepState = "done" | "current" | "locked";

export interface RailStep {
  id: string;
  label: string;
  state: StepState;
}

export default function Rail({ steps }: { steps: RailStep[] }) {
  return (
    <nav className="rail" aria-label="Workflow">
      <div className="brand">
        <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
          <rect width="32" height="32" rx="7" fill="var(--cobalt)" />
          <path d="M8 21h16M8 16h10M8 11h5" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
        </svg>
        <span>NeuralFit</span>
      </div>
      <ol>
        {steps.map((s, i) => (
          <li key={s.id} className={`rail-step ${s.state}`} aria-current={s.state === "current" ? "step" : undefined}>
            <span className="rail-mark" aria-hidden="true">
              {s.state === "done" ? "\u2713" : i + 1}
            </span>
            {s.state === "locked" ? (
              <span className="rail-label">{s.label}</span>
            ) : (
              <a className="rail-label" href={`#${s.id}`}>
                {s.label}
              </a>
            )}
            <span className="sr-only">{s.state === "done" ? " (complete)" : s.state === "locked" ? " (locked)" : ""}</span>
          </li>
        ))}
      </ol>
      <p className="rail-note">
        Measurements come from one CPU on the machine running the server. Nothing is estimated.
      </p>
    </nav>
  );
}
