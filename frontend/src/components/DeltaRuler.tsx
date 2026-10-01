import type { Direction } from "../types";

// Bar length is log-scaled and capped at +/-100% so both 0.5% and 75% changes stay legible.
const scale = (pct: number): number => Math.sign(pct) * Math.min(Math.log1p(Math.abs(pct)) / Math.log1p(100), 1);

const LABEL: Record<Direction, string> = { improved: "Better", worse: "Worse", unchanged: "Same" };

export default function DeltaRuler({ pct, direction }: { pct: number | null; direction: Direction }) {
  const s = pct === null ? 0 : scale(pct);
  const width = Math.abs(s) * 50;
  const minWidth = pct !== null && pct !== 0 ? Math.max(width, 1.5) : 0;
  return (
    <div className={`ruler ${direction}`}>
      <div className="ruler-track" aria-hidden="true">
        <span className="ruler-zero" />
        <span className="ruler-bar" style={{ width: `${minWidth}%`, [s < 0 ? "right" : "left"]: "50%" }} />
      </div>
      <span className="ruler-verdict">{LABEL[direction]}</span>
    </div>
  );
}
