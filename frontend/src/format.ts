export const formatBytes = (n: number): string =>
  n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} KB` : `${n} B`;

export const formatMs = (v: number): string => `${v.toFixed(2)} ms`;

export const signed = (v: number, digits = 1): string =>
  `${v > 0 ? "+" : v < 0 ? "\u2212" : ""}${Math.abs(v).toFixed(digits)}`;
