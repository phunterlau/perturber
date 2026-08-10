import type {
  EvidenceSummary,
  ExperimentSpec,
  JobEvent,
  RankSpec,
  ResearchCase,
  ResearchCasePlan,
  ResearchIntent,
  ResearchWorkflow,
  RunManifest,
} from "./types";

const token = () => window.__PROBE_TOKEN__ ?? "";

const headers = (content = false): HeadersInit => ({
  Authorization: `Bearer ${token()}`,
  ...(content ? { "Content-Type": "application/json" } : {}),
});

async function checked(response: Response): Promise<Response> {
  if (response.ok) return response;
  const body = await response.text();
  throw new Error(`HTTP ${response.status}: ${body}`);
}

export async function submitJob(spec: RankSpec | ExperimentSpec): Promise<{ job_id: string }> {
  return (await checked(await fetch("/api/v1/jobs", { method: "POST", headers: headers(true), body: JSON.stringify(spec) }))).json();
}

export async function streamJob(jobId: string, onEvent: (event: JobEvent) => void): Promise<void> {
  const response = await checked(await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/events`, { headers: { ...headers(), Accept: "application/x-ndjson" } }));
  if (!response.body) throw new Error("The job event stream has no body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  while (true) {
    const { done, value } = await reader.read();
    buffered += decoder.decode(value, { stream: !done });
    const lines = buffered.split("\n");
    buffered = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffered.trim()) onEvent(JSON.parse(buffered));
}

export async function listRuns(): Promise<RunManifest[]> {
  return (await checked(await fetch("/api/v1/runs", { headers: headers() }))).json();
}

export async function loadManifest(runId: string): Promise<RunManifest> {
  return (await checked(await fetch(`/api/v1/runs/${encodeURIComponent(runId)}`, { headers: headers() }))).json();
}

export async function loadSummary(runId: string): Promise<EvidenceSummary> {
  return (await checked(await fetch(`/api/v1/runs/${encodeURIComponent(runId)}/summary`, { headers: headers() }))).json();
}

export async function listCases(): Promise<ResearchCase[]> {
  return (await checked(await fetch("/api/v1/cases", { headers: headers() }))).json();
}

export async function createCase(intent: ResearchIntent, workflow: ResearchWorkflow, rankRunId?: string): Promise<ResearchCase> {
  return (await checked(await fetch("/api/v1/cases", {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify({ schema_version: "probe.research-case-create/v1", intent, workflow, rank_run_id: rankRunId }),
  }))).json();
}

export async function updateCase(value: ResearchCase, intent: ResearchIntent, workflow: ResearchWorkflow): Promise<ResearchCase> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(value.case_id)}`, {
    method: "PUT",
    headers: headers(true),
    body: JSON.stringify({ schema_version: "probe.research-case-update/v1", revision: value.revision, intent, workflow }),
  }))).json();
}

export async function loadCase(caseId: string): Promise<ResearchCase> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}`, { headers: headers() }))).json();
}

export async function loadCasePlan(caseId: string): Promise<ResearchCasePlan> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/plan`, { headers: headers() }))).json();
}

export async function preflightStage(caseId: string, stageKey: string): Promise<Record<string, unknown>> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/stages/${encodeURIComponent(stageKey)}/preflight`, { method: "POST", headers: headers() }))).json();
}

export async function startCaseStage(caseId: string, stageKey: string): Promise<{ job_id: string }> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/stages/${encodeURIComponent(stageKey)}/start`, { method: "POST", headers: headers() }))).json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await checked(await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", headers: headers() }));
}

export async function loadHandoff(caseId: string): Promise<{ prompt: string; ready_stages: string[] }> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/handoff`, { headers: headers() }))).json();
}

export async function downloadPacket(caseId: string): Promise<void> {
  const response = await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/packet`, { headers: headers() }));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${caseId}-research-packet.zip`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function loadTokenPreview(caseId: string, pairId: string): Promise<Record<string, unknown>> {
  return (await checked(await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/tokenize/${encodeURIComponent(pairId)}`, { headers: headers() }))).json();
}
