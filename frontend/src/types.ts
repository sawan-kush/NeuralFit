export interface ArchitectureInfo {
  key: string;
  display_name: string;
  description: string;
  default_input_size: number;
  default_mean: [number, number, number];
  default_std: [number, number, number];
  default_resize_size: number;
}

export interface OptionInfo {
  name: string;
  label: string;
  type: "integer" | "choice";
  default: number | string;
  minimum?: number | null;
  maximum?: number | null;
  choices?: string[] | null;
  description: string;
}

export interface MethodInfo {
  key: string;
  label: string;
  description: string;
  limitations: string[];
  options: OptionInfo[];
  available: boolean;
  unavailable_reason?: string | null;
}

export interface Preprocessing {
  input_size: number;
  resize_size?: number | null;
  mean: [number, number, number];
  std: [number, number, number];
}

export interface SessionResponse {
  session_id: string;
  architecture: string;
  num_classes: number;
  total_images: number;
  class_counts: Record<string, number>;
  preprocessing: Preprocessing;
  batch_size: number;
  uploaded_weights_size_bytes: number;
  warnings: string[];
  retention_hours: number;
}

export interface BenchmarkConfig {
  warmup_runs: number;
  timed_runs: number;
  threads: number;
}

export interface RunRequest {
  session_id: string;
  methods: string[];
  method_configs: Record<string, Record<string, string | number>>;
  benchmark: BenchmarkConfig;
}

export interface ErrorInfo {
  code: string;
  message: string;
}

export interface RunStatus {
  run_id: string;
  session_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  completed_steps: number;
  total_steps: number;
  error?: ErrorInfo | null;
}

export interface LatencyStats {
  median_ms: number;
  mean_ms: number;
  p95_ms: number;
  min_ms: number;
  std_ms: number;
  warmup_runs: number;
  timed_runs: number;
  batch_size: number;
}

export interface Metrics {
  top1_accuracy_pct: number;
  correct: number;
  total: number;
  size_bytes: number;
  param_count: number;
  latency: LatencyStats;
}

export interface ModelResult {
  method: string;
  label: string;
  config: Record<string, string | number>;
  status: "ok" | "failed";
  metrics?: Metrics | null;
  execution_mode?: string | null;
  notes: string[];
  error?: ErrorInfo | null;
  artifact_name?: string | null;
}

export type Direction = "improved" | "worse" | "unchanged";

export interface ComparisonEntry {
  baseline: number;
  optimized: number;
  absolute_change: number;
  percent_change: number | null;
  direction: Direction;
}

export type MetricKey = "top1_accuracy_pct" | "size_bytes" | "latency_median_ms" | "param_count";

export interface ArtifactInfo {
  name: string;
  kind: "model" | "report";
  size_bytes: number;
}

export interface RunResults {
  run_id: string;
  status: "completed" | "failed";
  benchmark: BenchmarkConfig;
  environment: Record<string, string | number>;
  baseline?: ModelResult | null;
  methods: ModelResult[];
  comparisons: Record<string, Record<MetricKey, ComparisonEntry>>;
  artifacts: ArtifactInfo[];
  limitations: string[];
  error?: ErrorInfo | null;
}
