import type {
  ArchitectureInfo, MethodInfo, RunRequest, RunResults, RunStatus, SessionResponse,
} from "./types";

const BASE = "/api/v1";

export class ApiError extends Error {
  code: string;
  constructor(message: string, code: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the NeuralFit server. Is the backend running on port 8000?", "network_error");
  }
  if (!res.ok) {
    let code = "http_error";
    let message = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      code = body.error.code;
      message = body.error.message;
    } catch {
      /* keep defaults */
    }
    throw new ApiError(message, code);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const getArchitectures = () => request<ArchitectureInfo[]>("/architectures");
export const getMethods = () => request<MethodInfo[]>("/methods");
export const createSession = (form: FormData) => request<SessionResponse>("/sessions", { method: "POST", body: form });
export const deleteSession = (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" });
export const startRun = (body: RunRequest) =>
  request<RunStatus>("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
export const getRunStatus = (id: string) => request<RunStatus>(`/runs/${id}`);
export const getResults = (id: string) => request<RunResults>(`/runs/${id}/results`);
export const artifactUrl = (runId: string, name: string) => `${BASE}/runs/${runId}/artifacts/${name}`;
