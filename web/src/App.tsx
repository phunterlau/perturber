import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import type { ComponentType } from "react";
import {
  cancelJob,
  createCase,
  downloadPacket,
  listCases,
  listRuns,
  loadCase,
  loadCasePlan,
  loadHandoff,
  loadRunSpec,
  loadSummary,
  loadTokenPreview,
  preflightStage,
  startCaseStage,
  streamJob,
  submitJob,
  updateCase,
} from "./api";
import {
  isAttentionInterventionSummary,
  isAttentionRankSummary,
  isAttentionTraceSummary,
  isDirectionSummary,
  isFFNCouplingSummary,
  isInterventionSummary,
  isRankSummary,
  isTrajectorySummary,
} from "./types";
import type {
  AttentionHead,
  AttentionInterventionSummary,
  AttentionRankSummary,
  AttentionTraceSummary,
  DirectionSummary,
  Dose,
  EvidenceSummary,
  FFNCouplingSummary,
  JobEvent,
  RankSpec,
  RankSummary,
  ResearchCase,
  ResearchCasePlan,
  ResearchIntent,
  ResearchWorkflow,
  RunManifest,
  Split,
  TrajectorySummary,
} from "./types";
import {
  applyConfirmedTrajectoryBand,
  couplingDisagreements,
  controlledDose,
  DEFAULT_INTENT,
  defaultAttentionWorkflow,
  isCausalPath,
  interventionTrajectoryRows,
  matchedControlTrajectoryRows,
  parseWorkflowYaml,
  rankedCouplingNeurons,
  rankedNeurons,
  rankingComparison,
  rankingObjectiveLabel,
  resolveNeuronEvidence,
  serializeWorkflowYaml,
  stageTitle,
  strongestSelectedPath,
  suggestedTrajectoryBand,
  trajectoryMetricView,
  workflowFromQuick,
} from "./research";
import type { TrajectoryCheckpointFilter, TrajectoryMetric } from "./research";
import type { RankingObjective } from "./types";

const Plot = lazy<ComponentType<any>>(async () => {
  const [{ default: Plotly }, { default: createPlotlyComponent }] = await Promise.all([
    import("plotly.js-basic-dist-min"),
    import("react-plotly.js/factory"),
  ]);
  return { default: createPlotlyComponent(Plotly) };
});
const ORIGINAL = "The capital of France is Paris, right? Answer only Yes or No, with no explanation.";
const PERTURBED = "The capital of France is London, right? Answer only Yes or No, with no explanation.";
const fmt = (value: number | null | undefined, digits = 4) => value == null ? "undefined" : `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const label = (value: string) => value.replaceAll("_", " ");

const plotLayout = {
  paper_bgcolor: "transparent", plot_bgcolor: "transparent",
  font: { color: "#465963", family: "IBM Plex Mono, monospace", size: 11 },
  margin: { l: 52, r: 18, t: 18, b: 46 },
  xaxis: { gridcolor: "#dce5e8", zerolinecolor: "#91a7ae" },
  yaxis: { gridcolor: "#dce5e8", zerolinecolor: "#91a7ae" }, autosize: true,
};

function App() {
  const [mode, setMode] = useState<"quick" | "research">("research");
  const [status, setStatus] = useState("Ready for a controlled research case.");
  return <main>
    <header className="topbar">
      <div><a className="paper-link" href="https://arxiv.org/abs/2604.27401" target="_blank" rel="noreferrer">arXiv:2604.27401</a><h1>Perturbation Probing</h1></div>
      <div className="mode-switch" aria-label="Workbench mode">
        <button className={mode === "quick" ? "active" : ""} onClick={() => setMode("quick")}>Quick Probe</button>
        <button className={mode === "research" ? "active" : ""} onClick={() => setMode("research")}>Research Cases</button>
      </div>
      <div className="system-state"><span className="pulse" />{status}</div>
    </header>
    <Suspense fallback={<div className="chart-loading">Loading research charts…</div>}>{mode === "quick" ? <QuickProbe setStatus={setStatus} onPromoted={() => setMode("research")} /> : <ResearchWorkbench setStatus={setStatus} />}</Suspense>
  </main>;
}

function QuickProbe({ setStatus, onPromoted }: { setStatus: (value: string) => void; onPromoted: () => void }) {
  const [original, setOriginal] = useState(ORIGINAL);
  const [perturbed, setPerturbed] = useState(PERTURBED);
  const [target, setTarget] = useState("No");
  const [control, setControl] = useState("Yes");
  const [model, setModel] = useState("Qwen/Qwen3-0.6B");
  const [device, setDevice] = useState<"auto" | "cpu" | "mps" | "cuda">("auto");
  const [topK, setTopK] = useState(500);
  const [summary, setSummary] = useState<RankSummary | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [runSpec, setRunSpec] = useState<RankSpec | null>(null);
  const [runs, setRuns] = useState<RunManifest[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"overview" | "layers" | "neurons" | "runs">("overview");
  useEffect(() => { void listRuns().then(setRuns); }, []);
  const spec = useMemo<RankSpec>(() => ({
    schema_version: "probe.rank/v1", kind: "rank", name: "web-prompt-pair",
    model: { id: model, revision: null, adapter: "auto", device, dtype: "auto", chat_template: true, enable_thinking: false },
    pairs: [{ id: "pair-1", original, perturbed, split: "discovery" }],
    observable: { name: "custom:web", target_tokens: target.split(",").map((x) => x.trim()).filter(Boolean), control_tokens: control.split(",").map((x) => x.trim()).filter(Boolean), reduction: "mean_logit_gap", decision_position: 0 },
    capture: { activation: "post_swiglu", position: -1, layers: "all" },
    ranking: { top_k: topK, select_by: "absolute_importance", pair_aggregation: "single_pair" },
    execution: { max_forward_passes: 2, max_artifact_bytes: 50_000_000, allow_download: false, trust_remote_code: false, seed: 0 }, tags: { surface: "web" },
  }), [original, perturbed, target, control, model, device, topK]);

  const analyze = async () => {
    setBusy(true); setError(null); setSummary(null); setStatus("Submitting bounded two-pass job…");
    try {
      const job = await submitJob(spec); let completed: string | null = null;
      await streamJob(job.job_id, (event) => { if (event.event === "model.loading") setStatus("Loading cached model…"); if (event.event === "pair.started") setStatus("Running paired forward passes…"); if (event.event === "job.completed") completed = String(event.payload.run_id); });
      if (!completed) throw new Error("Job stream ended without a committed run.");
      const loaded = await loadSummary(completed); if (!isRankSummary(loaded)) throw new Error("Unexpected run result.");
      setSummary(loaded); setRunId(completed); setRunSpec(spec); setStatus(`Completed immutable run ${completed}`); setTab("overview"); setRuns(await listRuns());
    } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); setStatus("Quick Probe failed."); } finally { setBusy(false); }
  };
  const promote = async () => {
    if (!runId) return;
    try { await createCase({ ...DEFAULT_INTENT, hypothesis: "Investigate the promoted Quick Probe perturbation." }, workflowFromQuick(runSpec ?? spec), runId); setStatus("Promoted immutable rank run into a research case."); onPromoted(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const openRun = async (id: string) => { const loaded = await loadSummary(id); if (isRankSummary(loaded)) { const savedSpec = await loadRunSpec(id); setSummary(loaded); setRunId(id); setRunSpec(savedSpec); setTab("overview"); } };

  return <div className="quick-shell">
    <section className="control-strip">
      <label>MODEL<input value={model} onChange={(e) => setModel(e.target.value)} /></label>
      <label>DEVICE<select value={device} onChange={(e) => setDevice(e.target.value as typeof device)}><option>auto</option><option>mps</option><option>cpu</option><option>cuda</option></select></label>
      <label>TOP K<input type="number" min="1" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label>
      <button className="run-button" disabled={busy} onClick={analyze}>{busy ? "RUNNING" : "ANALYZE PAIR"}</button>
    </section>
    <section className="prompt-grid"><PromptCard index="01" title="Original / control" value={original} onChange={setOriginal} /><PromptCard index="02" title="Perturbed / treatment" value={perturbed} onChange={setPerturbed} perturbed /></section>
    <section className="observable-strip"><label>TARGET TOKENS<input value={target} onChange={(e) => setTarget(e.target.value)} /></label><div className="formula">F = mean(logits[target]) − mean(logits[control])</div><label>CONTROL TOKENS<input value={control} onChange={(e) => setControl(e.target.value)} /></label></section>
    {error && <pre className="error-panel">{error}</pre>}
    <nav className="tabs">{(["overview", "layers", "neurons", "runs"] as const).map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}{runId && <button className="promote" onClick={promote}>Promote to research case →</button>}</nav>
    <section className="workspace">{tab === "overview" && <RankOverview summary={summary} />}{tab === "layers" && <FFNLayers summary={summary} />}{tab === "neurons" && <FFNNeurons summary={summary} />}{tab === "runs" && <RunList runs={runs} openRun={openRun} />}</section>
  </div>;
}

function ResearchWorkbench({ setStatus }: { setStatus: (value: string) => void }) {
  const [cases, setCases] = useState<ResearchCase[]>([]);
  const [active, setActive] = useState<ResearchCase | null>(null);
  const [intent, setIntent] = useState<ResearchIntent>(DEFAULT_INTENT);
  const [workflow, setWorkflow] = useState<ResearchWorkflow>(defaultAttentionWorkflow());
  const [yamlText, setYamlText] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [view, setView] = useState<"define" | "evidence" | "trajectory" | "ffn" | "attention" | "provenance">("define");
  const [plan, setPlan] = useState<ResearchCasePlan | null>(null);
  const [summaries, setSummaries] = useState<Record<string, EvidenceSummary>>({});
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshList = async (select?: string) => {
    const values = await listCases(); setCases(values);
    const target = values.find((item) => item.case_id === (select ?? active?.case_id)) ?? values[0] ?? null;
    if (target) await selectCase(target); else setActive(null);
  };
  const selectCase = async (value: ResearchCase) => {
    const current = await loadCase(value.case_id); setActive(current); setIntent(current.intent); setWorkflow(current.workflow); setYamlText(serializeWorkflowYaml(current.workflow));
    const [casePlan, loaded] = await Promise.all([loadCasePlan(current.case_id), loadCaseSummaries(current)]); setPlan(casePlan); setSummaries(loaded); setError(null);
  };
  useEffect(() => { void refreshList(); }, []);

  const create = async () => {
    setBusy(true); setError(null);
    try { const created = await createCase(DEFAULT_INTENT, defaultAttentionWorkflow()); await refreshList(created.case_id); setView("define"); setStatus(`Created research case ${created.case_id}`); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!active) return; setBusy(true); setError(null);
    try { const updated = await updateCase(active, intent, workflow); await refreshList(updated.case_id); setStatus("Saved canonical research workflow draft."); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusy(false); }
  };
  const applyYaml = () => {
    try { setWorkflow(parseWorkflowYaml(yamlText)); setError(null); }
    catch (caught) { setError(`YAML was not applied: ${caught instanceof Error ? caught.message : String(caught)}`); }
  };
  const reviewStage = async (key: string) => {
    if (!active) return; setSelectedStage(key); setPreflight(null); setError(null);
    try { setPreflight(await preflightStage(active.case_id, key)); } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const runStage = async () => {
    if (!active || !selectedStage) return;
    setBusy(true); setError(null); setStatus(`Starting ${stageTitle(selectedStage)}…`);
    try {
      const job = await startCaseStage(active.case_id, selectedStage);
      await streamJob(job.job_id, (event: JobEvent) => { if (event.event.endsWith("started")) setStatus(`${stageTitle(selectedStage)} · ${label(event.event)}`); });
      const refreshed = await loadCase(active.case_id); await refreshList(refreshed.case_id); setStatus(`${stageTitle(selectedStage)} completed and was verified.`); setPreflight(null);
    } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); setStatus(`${stageTitle(selectedStage)} failed.`); await refreshList(active.case_id); } finally { setBusy(false); }
  };
  const stop = async () => { const stage = active?.stages.find((item) => item.key === selectedStage); if (stage?.job_id) await cancelJob(stage.job_id); };
  const configurePath = async (config: { sender: { layer: number; head: number }; receiver: { layer: number; head: number }; pairId: string; operation: "patch" | "restore"; alignment: "identity" | "explicit"; positions: Array<{ original: number; perturbed: number }>; controls: number; seed: number }) => {
    if (!active) return;
    const traces = [...(workflow.attention_traces ?? [])];
    const index = traces.findIndex((item) => item.trace_kind === "head_paths");
    if (index < 0) throw new Error("The case has no configured head-path stage.");
    const current = traces[index];
    traces[index] = { ...current, pair_ids: [config.pairId], senders: [config.sender], receivers: [config.receiver], operation: config.operation, alignments: [{ pair_id: config.pairId, mode: config.alignment, positions: config.alignment === "explicit" ? config.positions : [] }], controls: { samples: config.controls }, execution: { ...(current.execution as Record<string, unknown>), seed: config.seed } };
    const next = { ...workflow, attention_traces: traces };
    const updated = await updateCase(active, intent, next);
    setWorkflow(next); await refreshList(updated.case_id); setStatus("Saved sender, receiver, alignment, controls, and seed to the canonical path stage.");
  };
  const confirmTrajectoryBand = async (layers: number[], pairId: string) => {
    if (!active) return;
    setBusy(true); setError(null);
    try {
      const next = applyConfirmedTrajectoryBand(workflow, layers, pairId);
      const updated = await updateCase(active, intent, next);
      setWorkflow(next); await refreshList(updated.case_id); setView("ffn");
      setStatus(`Confirmed L${layers[0]}–L${layers.at(-1)} for the FFN coupling checkpoint.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally { setBusy(false); }
  };

  return <div className="research-shell">
    <aside className="case-sidebar"><div className="sidebar-heading"><span>RESEARCH CASES</span><button onClick={create} disabled={busy}>＋</button></div>{cases.map((item) => <button className={item.case_id === active?.case_id ? "case-item active" : "case-item"} key={item.case_id} onClick={() => void selectCase(item)}><strong>{item.workflow.name}</strong><small>{label(item.evidence_label)} · {item.stages.filter((stage) => stage.status === "verified").length}/{item.stages.length} verified</small></button>)}{!cases.length && <p className="sidebar-empty">Create a case to move beyond a single observational ranking.</p>}</aside>
    <section className="case-main">
      {!active ? <Empty>Create a guided research case from the Qwen attention-path template.</Empty> : <>
        <div className="case-hero"><div><span className={`evidence-badge ${active.evidence_label}`}>{label(active.evidence_label)}</span><h2>{active.workflow.name}</h2><p>{active.intent.hypothesis}</p></div><div className="case-id">CASE<br/><strong>{active.case_id}</strong></div></div>
        <EvidenceRail stages={active.stages} selected={selectedStage} onSelect={reviewStage} />
        <nav className="case-tabs">{(["define", "evidence", "trajectory", "ffn", "attention", "provenance"] as const).map((item) => <button className={view === item ? "active" : ""} onClick={() => setView(item)} key={item}>{item === "ffn" ? "FFN Circuit" : item === "attention" ? "Attention Path" : item === "trajectory" ? "Trajectory" : item}</button>)}</nav>
        {error && <pre className="error-panel">{error}</pre>}
        <div className="case-workspace">
          {view === "define" && <DefineCase active={active} intent={intent} setIntent={setIntent} workflow={workflow} setWorkflow={setWorkflow} advanced={advanced} setAdvanced={setAdvanced} yamlText={yamlText} setYamlText={setYamlText} applyYaml={applyYaml} save={save} busy={busy} />}
          {view === "evidence" && <EvidenceView caseValue={active} plan={plan} selectedStage={selectedStage} preflight={preflight} reviewStage={reviewStage} runStage={runStage} stop={stop} busy={busy} />}
          {view === "trajectory" && <TrajectoryWorkspace caseValue={active} summaries={summaries} confirmBand={confirmTrajectoryBand} busy={busy} />}
          {view === "ffn" && <FFNCircuit caseValue={active} summaries={summaries} />}
          {view === "attention" && <AttentionWorkspace caseValue={active} summaries={summaries} configurePath={configurePath} />}
          {view === "provenance" && <Provenance caseValue={active} setStatus={setStatus} />}
        </div>
      </>}
    </section>
  </div>;
}

async function loadCaseSummaries(value: ResearchCase): Promise<Record<string, EvidenceSummary>> {
  const entries = await Promise.all(value.stages.filter((item) => item.run_id).map(async (item) => [item.key, await loadSummary(item.run_id!)] as const));
  return Object.fromEntries(entries);
}

function DefineCase(props: { active: ResearchCase; intent: ResearchIntent; setIntent: (value: ResearchIntent) => void; workflow: ResearchWorkflow; setWorkflow: (value: ResearchWorkflow) => void; advanced: boolean; setAdvanced: (value: boolean) => void; yamlText: string; setYamlText: (value: string) => void; applyYaml: () => void; save: () => void; busy: boolean }) {
  const { intent, setIntent, workflow, setWorkflow } = props;
  const updateRank = (rank: RankSpec) => setWorkflow({ ...workflow, rank });
  const updatePair = (index: number, patch: Record<string, unknown>) => updateRank({ ...workflow.rank, pairs: workflow.rank.pairs.map((item, i) => i === index ? { ...item, ...patch } : item) });
  return <div className="define-layout">
    <section className="panel form-panel"><div className="section-heading"><div><span>01</span><h3>Research contract</h3></div><button onClick={() => props.setAdvanced(!props.advanced)}>{props.advanced ? "Guided form" : "Advanced YAML"}</button></div>
      {!props.advanced ? <div className="field-grid">
        <Field label="HYPOTHESIS" value={intent.hypothesis} onChange={(value) => setIntent({ ...intent, hypothesis: value })} wide />
        <Field label="INTENDED PERTURBATION" value={intent.intended_perturbation} onChange={(value) => setIntent({ ...intent, intended_perturbation: value })} wide />
        <Field label="INVARIANTS · ONE PER LINE" value={intent.invariants.join("\n")} onChange={(value) => setIntent({ ...intent, invariants: value.split("\n").map((x) => x.trim()).filter(Boolean) })} wide multiline />
        <Field label="FALSIFYING OUTCOME" value={intent.falsifying_outcome} onChange={(value) => setIntent({ ...intent, falsifying_outcome: value })} wide />
        <Field label="WORKFLOW NAME" value={workflow.name} onChange={(value) => setWorkflow({ ...workflow, name: value })} />
        <Field label="MODEL" value={workflow.rank.model.id} onChange={(value) => updateRank({ ...workflow.rank, model: { ...workflow.rank.model, id: value } })} />
        <Field label="MODEL REVISION" value={workflow.rank.model.revision ?? ""} onChange={(value) => updateRank({ ...workflow.rank, model: { ...workflow.rank.model, revision: value || null } })} />
        <Field label="SEED" value={String(workflow.rank.execution.seed)} onChange={(value) => updateRank({ ...workflow.rank, execution: { ...workflow.rank.execution, seed: Number(value) } })} />
        <Field label="TARGET TOKENS · COMMA SEPARATED" value={workflow.rank.observable.target_tokens.join(", ")} onChange={(value) => updateRank({ ...workflow.rank, observable: { ...workflow.rank.observable, target_tokens: value.split(",").map((x) => x.trim()).filter(Boolean) } })} />
        <Field label="CONTROL TOKENS · COMMA SEPARATED" value={workflow.rank.observable.control_tokens.join(", ")} onChange={(value) => updateRank({ ...workflow.rank, observable: { ...workflow.rank.observable, control_tokens: value.split(",").map((x) => x.trim()).filter(Boolean) } })} />
      </div> : <div className="yaml-editor"><textarea value={props.yamlText} onChange={(e) => props.setYamlText(e.target.value)} spellCheck={false} /><div><button onClick={props.applyYaml}>Apply validated YAML</button><button onClick={() => props.setYamlText(serializeWorkflowYaml(workflow))}>Export current draft</button></div></div>}
    </section>
    {!props.advanced && <section className="panel pair-editor"><div className="section-heading"><div><span>02</span><h3>Controlled prompt pairs</h3></div><button onClick={() => updateRank({ ...workflow.rank, pairs: [...workflow.rank.pairs, { id: `pair-${workflow.rank.pairs.length + 1}`, split: "discovery", original: "", perturbed: "" }] })}>＋ Pair</button></div>{workflow.rank.pairs.map((pair, index) => <article className="pair-row" key={`${pair.id}-${index}`}><div className="pair-meta"><input aria-label={`Pair ${index + 1} ID`} value={pair.id} onChange={(e) => updatePair(index, { id: e.target.value })}/><select aria-label={`Pair ${index + 1} split`} value={pair.split} onChange={(e) => updatePair(index, { split: e.target.value as Split })}><option value="discovery">discovery</option><option value="validation">validation</option><option value="heldout">heldout</option></select><button aria-label={`Remove pair ${index + 1}`} disabled={workflow.rank.pairs.length === 1} onClick={() => updateRank({ ...workflow.rank, pairs: workflow.rank.pairs.filter((_, i) => i !== index) })}>×</button></div><div className="pair-prompts"><textarea aria-label={`${pair.id} original`} value={pair.original} onChange={(e) => updatePair(index, { original: e.target.value })}/><textarea aria-label={`${pair.id} perturbed`} value={pair.perturbed} onChange={(e) => updatePair(index, { perturbed: e.target.value })}/></div></article>)}</section>}
    <div className="save-bar"><span>Saving validates the canonical workflow; executed stages remain immutable.</span><button className="primary" disabled={props.busy} onClick={props.save}>Save research case</button></div>
  </div>;
}

function Field({ label: name, value, onChange, wide, multiline }: { label: string; value: string; onChange: (value: string) => void; wide?: boolean; multiline?: boolean }) {
  return <label className={wide ? "wide" : ""}>{name}{multiline ? <textarea value={value} onChange={(e) => onChange(e.target.value)} /> : <input value={value} onChange={(e) => onChange(e.target.value)} />}</label>;
}

function EvidenceRail({ stages, selected, onSelect }: { stages: ResearchCase["stages"]; selected: string | null; onSelect: (key: string) => void }) {
  return <div className="evidence-rail" aria-label="Evidence pipeline">{stages.map((stage, index) => <button key={stage.key} className={`${stage.status} ${selected === stage.key ? "selected" : ""}`} onClick={() => onSelect(stage.key)}><span className="stage-index">{String(index + 1).padStart(2, "0")}</span><strong>{stageTitle(stage.key)}</strong><small>{label(stage.status)}</small></button>)}</div>;
}

function EvidenceView({ caseValue, plan, selectedStage, preflight, reviewStage, runStage, stop, busy }: { caseValue: ResearchCase; plan: ResearchCasePlan | null; selectedStage: string | null; preflight: Record<string, unknown> | null; reviewStage: (key: string) => void; runStage: () => void; stop: () => void; busy: boolean }) {
  const stage = caseValue.stages.find((item) => item.key === selectedStage) ?? caseValue.stages[0];
  const stagePlan = plan?.stages.find((item) => item.key === stage?.key);
  return <div className="evidence-layout"><section className="panel evidence-map"><h3>Evidence progression</h3><div className="branch-map"><div>Rank</div><i>→</i><div>Behavior gate</div><i>→</i><div className="branch"><span>FFN circuit</span><span>Attention route</span></div><i>→</i><div className="branch"><span>Matched intervention</span><span>Sender → receiver path</span></div></div><p className="muted">The browser displays claim strength emitted by verified artifacts. Ranking alone remains a hypothesis.</p><div className="claim-list">{caseValue.stages.flatMap((item) => item.claims.map((claim) => <article key={`${item.key}-${claim.claim_id}`} className={`claim ${claim.status}`}><span>{claim.status}</span><strong>{claim.claim_type}</strong><p>{claim.statement}</p></article>))}</div></section>
    <aside className="panel checkpoint"><h3>Stage checkpoint</h3>{stage ? <><strong className="checkpoint-title">{stageTitle(stage.key)}</strong><dl><dt>Status</dt><dd>{label(stage.status)}</dd><dt>Parents</dt><dd>{stage.parent_run_ids.join(", ") || "resolved at execution"}</dd><dt>Model calls</dt><dd>{stagePlan?.plan?.forward_passes ?? "—"}</dd><dt>Artifact budget</dt><dd>{formatBytes(Number((preflight?.plan as Record<string, unknown> | undefined)?.max_artifact_bytes ?? 0)) || "declared in spec"}</dd><dt>Cached model</dt><dd>{stagePlan?.plan ? String(stagePlan.plan.model_cached) : "—"}</dd><dt>Device</dt><dd>{stagePlan?.plan?.resolved_device ?? "—"}</dd></dl>{stage.warnings.map((item) => <div className="warning" key={item}>{item}</div>)}{stage.status === "ready" && <button className="primary full" onClick={() => preflight ? runStage() : reviewStage(stage.key)} disabled={busy}>{preflight ? "Confirm and run stage" : "Review preflight"}</button>}{stage.status === "running" && <button className="danger full" onClick={stop}>Cancel running job</button>}{stage.run_id && <code className="run-code">{stage.run_id}</code>}</> : <Empty>Select a configured stage.</Empty>}</aside></div>;
}

function FFNCircuit({ caseValue, summaries }: { caseValue: ResearchCase; summaries: Record<string, EvidenceSummary> }) {
  const rank = Object.values(summaries).find(isRankSummary) ?? null;
  const coupling = Object.values(summaries).find(isFFNCouplingSummary);
  const intervention = Object.values(summaries).find(isInterventionSummary);
  const direction = Object.values(summaries).find(isDirectionSummary);
  const [split, setSplit] = useState<"all" | Split>("all");
  const [sign, setSign] = useState<"all" | "target" | "control">("all");
  const [selectedObjective, setSelectedObjective] = useState<RankingObjective>("shared_direction");
  const [selected, setSelected] = useState<string | null>(null);
  if (!rank) return <Empty>Run the rank stage to inspect the FFN circuit.</Empty>;
  const objective = rank.shared_direction_neurons?.length
    ? selectedObjective
    : (rank.ranking_objective ?? "effect_magnitude");
  const neurons = rankedNeurons(rank, objective).filter((item) => sign === "all" || (sign === "target" ? item.importance_mean > 0 : item.importance_mean < 0));
  const couplingNeurons = coupling ? rankedCouplingNeurons(coupling, objective) : [];
  const comparison = rankingComparison(rank);
  const dualAvailable = Boolean(rank.shared_direction_neurons?.length && rank.effect_magnitude_neurons?.length);
  const key = (layer: number, neuron: number) => `${layer}:${neuron}`;
  const evidence = resolveNeuronEvidence(neurons, couplingNeurons, selected);
  const neuron = evidence?.rank;
  const coupledNeuron = evidence?.coupling;
  const inspectedLayer = evidence?.layer;
  const inspectedNeuron = evidence?.neuron;
  const directSignedImportance = neuron?.importance_mean ?? (coupledNeuron ? coupledNeuron.activation_delta_mean * coupledNeuron.direct_coupling : undefined);
  const residual = rank.measured_delta_mean - rank.predicted_delta_mean;
  return <div className="circuit-stack"><div className="filter-bar"><label>RANKING<select value={objective} onChange={(e) => { setSelectedObjective(e.target.value as RankingObjective); setSelected(null); }}><option value="shared_direction" disabled={!rankedNeurons(rank, "shared_direction").length}>shared direction · paper</option><option value="effect_magnitude" disabled={!rankedNeurons(rank, "effect_magnitude").length}>effect magnitude · RMS</option></select></label><label>SPLIT<select value={split} onChange={(e) => setSplit(e.target.value as typeof split)}><option>all</option><option>discovery</option><option>validation</option><option>heldout</option></select></label><label>DIRECTION<select value={sign} onChange={(e) => setSign(e.target.value as typeof sign)}><option value="all">all</option><option value="target">toward target</option><option value="control">toward control</option></select></label><span>{rank.total_neuron_count.toLocaleString()} scored neurons · verified run {caseValue.stages.find((x) => x.key === "rank")?.run_id}</span></div>
    <section className="panel ranking-objective"><div><h3>{rankingObjectiveLabel(objective)}</h3><p className="muted">Shared direction ranks by |mean importance| across controlled pairs and matches the paper's implemented selection. Effect magnitude ranks by RMS and surfaces strong prompt-conditional or sign-cancelling responses. Both views reuse the same captures. {!dualAvailable && "This older run predates dual views; rerun its immutable spec to compare objectives."}</p></div><dl><dt>Top-{comparison.limit} overlap</dt><dd>{comparison.overlapFraction == null ? "unavailable" : `${(comparison.overlapFraction * 100).toFixed(0)}%`}</dd><dt>Low-coherence RMS candidates</dt><dd>{comparison.cancellationCandidates ?? "unavailable"}</dd></dl></section>
    <FFNDecomposition rank={rank} residual={residual} objective={objective} />
    <LayerAwareCoupling summary={coupling} neurons={couplingNeurons} objective={objective} selected={selected} select={setSelected} />
    <div className="analysis-grid"><FFNLayers summary={rank} objective={objective} /><section className="panel neuron-inspector"><h3>Neuron inspector · score definitions remain separate</h3>{inspectedLayer != null && inspectedNeuron != null && <dl><dt>Neuron</dt><dd>L{inspectedLayer}:n{inspectedNeuron}</dd><dt>Activation Δa</dt><dd>{fmt(coupledNeuron?.activation_delta_mean ?? neuron?.activation_delta_mean, 6)}</dd><dt>Direct coupling</dt><dd>{fmt(coupledNeuron?.direct_coupling ?? neuron?.coupling, 6)}</dd><dt>Direct signed importance</dt><dd>{fmt(directSignedImportance, 6)}</dd><dt>Direct RMS</dt><dd>{fmt(coupledNeuron?.direct_importance_rms ?? neuron?.importance_rms, 6)}</dd><dt>Importance coherence</dt><dd>{neuron?.importance_coherence == null ? "undefined" : `${(neuron.importance_coherence * 100).toFixed(0)}%`}</dd><dt>Native-local coupling</dt><dd>{fmt(coupledNeuron?.native_coupling_mean, 6)}</dd><dt>Native-local RMS</dt><dd>{fmt(coupledNeuron?.native_importance_rms, 6)}</dd><dt>Downstream coupling</dt><dd>{fmt(coupledNeuron?.downstream_coupling_mean, 6)}</dd><dt>Downstream signed importance</dt><dd>{fmt(coupledNeuron?.downstream_importance_mean, 6)}</dd><dt>Downstream RMS</dt><dd>{fmt(coupledNeuron?.downstream_importance_rms, 6)}</dd><dt>Downstream coherence</dt><dd>{coupledNeuron?.downstream_importance_coherence == null ? "undefined" : `${(coupledNeuron.downstream_importance_coherence * 100).toFixed(0)}%`}</dd><dt>Downstream consistency</dt><dd>{coupledNeuron ? `${(coupledNeuron.downstream_sign_consistency * 100).toFixed(0)}%` : "undefined"}</dd><dt>Direct/downstream sign agreement</dt><dd>{coupledNeuron ? `${(coupledNeuron.direct_downstream_sign_agreement * 100).toFixed(0)}%` : "undefined"}</dd></dl>}<div className="mini-neurons">{neurons.slice(0, 20).map((item) => <button className={item.layer === inspectedLayer && item.neuron === inspectedNeuron ? "active" : ""} key={`${item.layer}:${item.neuron}`} onClick={() => setSelected(key(item.layer, item.neuron))}>L{item.layer}:n{item.neuron}<small>{objective === "shared_direction" ? fmt(item.importance_mean, 5) : item.importance_rms.toFixed(5)}</small></button>)}</div></section></div>
    <DosePanel title={intervention?.candidate_score_method === "downstream_endpoint_gradient" ? "Downstream-gradient candidates vs same-layer controls" : intervention?.candidate_score_method === "direct_downstream_overlap" ? "Direct/downstream overlap vs same-layer controls" : "Direct-readout candidates vs same-layer controls"} doses={(intervention?.doses ?? []).filter((item) => split === "all" || item.split === split)} countKey="neuron_count" />
    {intervention && <section className="panel lineage-note"><h3>Candidate provenance</h3><p className="muted">{intervention.candidate_score_method === "downstream_endpoint_gradient" ? "Candidates were ranked by the layer-aware gradient from each FFN output to the final target−control observable." : intervention.candidate_score_method === "direct_downstream_overlap" ? "Candidates are the preregistered intersection of the direct-readout and downstream-gradient top pools, ordered by downstream rank." : "Candidates were ranked by the model's final-norm and LM-head structural readout."} Objective: {rankingObjectiveLabel(intervention.candidate_ranking_objective ?? "effect_magnitude")}. The controlled intervention, qualification gate, and held-out result—not this ranking method—determine claim strength.</p><code>{intervention.parent_run_id}{intervention.rank_run_id && intervention.rank_run_id !== intervention.parent_run_id ? ` → rank ${intervention.rank_run_id}` : ""}</code></section>}
    <section className="panel direction-panel"><div><h3>Residual-direction controllability</h3><p className="muted">This is a separate controllability test. It does not localize the effect to ranked neurons.</p></div>{direction ? <DoseChart doses={direction.doses} countKey="strength" /> : <p className="empty-inline">Direction stage not yet verified.</p>}</section>
  </div>;
}

function TrajectoryWorkspace({ caseValue, summaries, confirmBand, busy }: { caseValue: ResearchCase; summaries: Record<string, EvidenceSummary>; confirmBand: (layers: number[], pairId: string) => Promise<void>; busy: boolean }) {
  const summary = Object.values(summaries).find(isTrajectorySummary);
  const interventions = Object.values(summaries).filter(isInterventionSummary);
  const [selectedPairId, setSelectedPairId] = useState("");
  const [split, setSplit] = useState<"all" | Split>("all");
  const [metric, setMetric] = useState<TrajectoryMetric>("logit_gap");
  const [checkpoint, setCheckpoint] = useState<TrajectoryCheckpointFilter>("all");
  const [bandStart, setBandStart] = useState("");
  const [bandEnd, setBandEnd] = useState("");
  if (!summary) return <Empty>Run the paired trajectory stage to see where the two prompts begin to diverge.</Empty>;
  const visiblePairs = summary.pairs.filter((item) => split === "all" || item.split === split);
  const pairId = visiblePairs.some((item) => item.pair_id === selectedPairId) ? selectedPairId : visiblePairs[0]?.pair_id ?? summary.pairs[0]?.pair_id ?? "";
  const metricView = trajectoryMetricView(summary, pairId, metric, checkpoint);
  const rows = metricView.rows;
  const tickRows = checkpoint === "all" ? rows.filter((item) => item.checkpoint === "post_ffn") : rows;
  const pair = summary.pairs.find((item) => item.pair_id === pairId) ?? summary.pairs[0];
  const transitions = pair?.transitions ?? [];
  const suggestion = suggestedTrajectoryBand(summary, pairId);
  const availableLayers = [...new Set((pair?.checkpoints ?? []).map((item) => item.layer))].sort((a, b) => a - b);
  const lower = bandStart === "" ? suggestion[0] : Number(bandStart);
  const upper = bandEnd === "" ? suggestion.at(-1) : Number(bandEnd);
  const band = availableLayers.filter((layer) => lower != null && upper != null && layer >= lower && layer <= upper);
  const couplingStage = caseValue.stages.find((item) => item.key === "ffn-coupling");
  const couplingLocked = Boolean(couplingStage?.run_id || couplingStage?.job_id);
  const configuredLayers = Array.isArray(caseValue.workflow.ffn_coupling?.layers) ? caseValue.workflow.ffn_coupling.layers as number[] : [];
  const trajectoryRunId = caseValue.stages.find((item) => item.key === "trajectory")?.run_id;
  const overlays = interventions.map((item) => ({ summary: item, selected: interventionTrajectoryRows(item, pairId), controls: matchedControlTrajectoryRows(item, pairId) })).filter((item) => item.selected.length);
  const colors = ["#386cb0", "#c94b78", "#0e9b8d"];
  return <div className="trajectory-stack">
    <div className="filter-bar"><label>SPLIT<select value={split} onChange={(event) => { setSplit(event.target.value as typeof split); setSelectedPairId(""); }}><option>all</option><option>discovery</option><option>validation</option><option>heldout</option></select></label><label>PAIR<select value={pairId} onChange={(event) => { setSelectedPairId(event.target.value); setBandStart(""); setBandEnd(""); }}>{visiblePairs.map((item) => <option key={item.pair_id}>{item.pair_id}</option>)}</select></label><label>METRIC<select value={metric} onChange={(event) => setMetric(event.target.value as TrajectoryMetric)}><option value="logit_gap">target−control gap</option><option value="target_probability">target probability</option><option value="target_rank">target rank</option><option value="entropy">entropy</option><option value="forward_kl">forward KL to final</option><option value="paired_js">paired JS</option><option value="total_variation">total variation</option></select></label><label>CHECKPOINT<select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value as TrajectoryCheckpointFilter)}><option value="all">all</option><option value="block_input">block input</option><option value="post_attention">post attention</option><option value="post_ffn">post FFN</option></select></label></div>
    <section className="panel chart"><h3>Paired prediction trajectory</h3><p className="muted">Every metric uses the same native final-norm and LM-head decoder. Target-rank axes are reversed because lower is better. These curves remain observational until a controlled intervention passes its backend gates.</p><Plot data={metricView.series.map((series, index) => ({ type: series.paired && metric === "logit_gap" ? "bar" as const : "scatter" as const, mode: "lines+markers" as const, name: series.name, x: rows.map((item) => item.x), y: series.values, line: { color: colors[index], dash: series.paired ? "dot" as const : "solid" as const }, marker: { color: colors[index] }, opacity: series.paired ? .55 : 1 }))} layout={{ ...plotLayout, barmode: "overlay", xaxis: { title: "Layer checkpoint", tickmode: "array", tickvals: tickRows.map((item) => item.x), ticktext: tickRows.map((item) => item.label), gridcolor: "#dce5e8" }, yaxis: { title: metricView.yTitle, gridcolor: "#dce5e8", ...(metricView.lowerIsBetter ? { autorange: "reversed" as const } : {}) }, legend: { orientation: "h", y: 1.12 } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 430 }} /></section>
    <div className="analysis-grid"><section className="panel"><h3>Largest transitions · suggested, never causal</h3><div className="trajectory-transitions">{transitions.map((item) => <article key={`${item.rank}-${item.layer}-${item.checkpoint}`}><span>#{item.rank}</span><strong>L{item.layer} · {label(item.checkpoint)}</strong><b>{fmt(item.pair_delta_change, 5)}</b></article>)}</div>{trajectoryRunId && <code className="inspect-command">uv run --locked probe runs transitions {trajectoryRunId} --pair {pairId} --limit 10</code>}</section><section className="panel band-confirmation"><h3>Researcher-confirmed FFN scope</h3><p className="muted">The strongest transition only seeds this editable band. Confirmation changes the unexecuted FFN coupling draft and records the pair and layers in its immutable run provenance.</p><div className="band-fields"><label>START L<input type="number" min={availableLayers[0]} max={availableLayers.at(-1)} value={lower ?? ""} onChange={(event) => setBandStart(event.target.value)} /></label><span>→</span><label>END L<input type="number" min={availableLayers[0]} max={availableLayers.at(-1)} value={upper ?? ""} onChange={(event) => setBandEnd(event.target.value)} /></label></div><div className="band-preview">{band.length ? band.map((layer) => <span key={layer}>L{layer}</span>) : "No valid layers selected"}</div><button className="primary full" disabled={busy || couplingLocked || !band.length} onClick={() => void confirmBand(band, pairId)}>{couplingLocked ? "FFN coupling stage is immutable" : "Confirm band and continue to FFN"}</button>{configuredLayers.length > 0 && <small>Canonical FFN scope: {configuredLayers.map((layer) => `L${layer}`).join(", ")}</small>}</section></div>
    {trajectoryRunId && <section className="panel command-panel"><h3>Exact agent inspection</h3><code>uv run --locked probe runs trajectory {trajectoryRunId} --pair {pairId} --metric {metric} --checkpoint {checkpoint} --limit 500</code></section>}
    {overlays.map(({ summary: intervention, selected: selectedRows, controls }) => <section className="panel chart" key={`${intervention.parent_run_id}-${intervention.candidate_score_method}`}><h3>Intervention effect along the trajectory</h3><p className="muted">{intervention.candidate_score_method === "downstream_endpoint_gradient" ? "Layer-aware downstream-gradient candidates" : intervention.candidate_score_method === "direct_downstream_overlap" ? "Direct/downstream top-pool overlap candidates" : "Direct-readout candidates"} at the widest declared dose. Selected and same-layer random effects use the same native checkpoint decoder. Causal interpretation still follows the intervention's backend claims and gates.</p><Plot data={[{ type: "scatter", mode: "lines+markers", name: "selected Δ gap", x: selectedRows.map((item) => item.x), y: selectedRows.map((item) => item.gap_effect), line: { color: "#0e9b8d", width: 3 } }, { type: "scatter", mode: "lines+markers", name: "matched-random mean", x: controls.map((item) => item.x), y: controls.map((item) => item.gap_effect), line: { color: "#91a7ae", dash: "dot" } }]} layout={{ ...plotLayout, xaxis: { title: "Layer checkpoint", tickmode: "array", tickvals: selectedRows.map((item) => item.x), ticktext: selectedRows.map((item) => item.label), gridcolor: "#dce5e8" }, yaxis: { title: "Intervention − baseline gap", gridcolor: "#dce5e8" }, legend: { orientation: "h", y: 1.12 } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 380 }} /></section>)}
    <section className="panel"><h3>Distribution diagnostics</h3>{rows.length ? <dl><dt>Final paired Δ</dt><dd>{fmt(pair?.final_pair_delta)}</dd><dt>Peak paired JS</dt><dd>{Math.max(...rows.map((item) => item.paired_js)).toFixed(6)}</dd><dt>Peak total variation</dt><dd>{Math.max(...rows.map((item) => item.paired_total_variation)).toFixed(6)}</dd><dt>Original final KL</dt><dd>{rows.at(-1)?.original_forward_kl_to_final.toFixed(8)}</dd><dt>Perturbed final KL</dt><dd>{rows.at(-1)?.perturbed_forward_kl_to_final.toFixed(8)}</dd></dl> : <p className="empty-inline">No checkpoint rows.</p>}</section>
  </div>;
}

function LayerAwareCoupling({ summary, neurons, objective, selected, select }: { summary?: FFNCouplingSummary; neurons: FFNCouplingSummary["neurons"]; objective: RankingObjective; selected: string | null; select: (key: string) => void }) {
  if (!summary) return <section className="panel"><h3>Layer-aware coupling</h3><p className="empty-inline">Run the FFN coupling stage to compare direct readout with downstream endpoint sensitivity.</p></section>;
  const max = Math.max(...neurons.map((item) => Math.max(item.direct_importance_rms, item.downstream_importance_rms)), 1e-8);
  const disagreements = couplingDisagreements({ ...summary, neurons }, 12);
  const layerMass = (item: FFNCouplingSummary["layers"][number]) => objective === "shared_direction" ? (item.downstream_absolute_mean_mass ?? 0) : item.downstream_rms_mass;
  return <section className="panel chart"><h3>Direct readout vs downstream endpoint gradient</h3><p className="muted">Each point is one FFN neuron from the {rankingObjectiveLabel(objective)} view. RMS axes intentionally remain fixed so method disagreement stays comparable; selection follows the active objective and does not establish causality.</p><Plot data={[{ type: "scattergl", mode: "markers", x: neurons.map((item) => item.direct_importance_rms), y: neurons.map((item) => item.downstream_importance_rms), text: neurons.map((item) => `#${item.rank} L${item.layer}:n${item.neuron}`), customdata: neurons.map((item) => [item.activation_delta_mean, item.direct_downstream_sign_agreement, item.native_importance_rms]), marker: { size: neurons.map((item) => 6 + 12 * Math.sqrt(item.downstream_importance_rms / max)), color: neurons.map((item) => item.direct_downstream_sign_agreement), cmin: 0, cmax: 1, colorscale: [[0, "#c94b78"], [.5, "#e6c16a"], [1, "#0e9b8d"]], colorbar: { title: "sign agree" } }, hovertemplate: "%{text}<br>direct RMS=%{x:.6f}<br>downstream RMS=%{y:.6f}<br>Δa=%{customdata[0]:.6f}<br>sign agreement=%{customdata[1]:.0%}<br>native RMS=%{customdata[2]:.6f}<extra></extra>" }, { type: "scatter", mode: "lines", name: "equal sensitivity", x: [0, max], y: [0, max], line: { color: "#91a7ae", dash: "dot" }, hoverinfo: "skip" }]} layout={{ ...plotLayout, xaxis: { title: "Direct importance RMS", gridcolor: "#dce5e8" }, yaxis: { title: "Downstream importance RMS", gridcolor: "#dce5e8" }, legend: { orientation: "h", y: 1.12 } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 420 }} /><div className="layer-strip">{[...summary.layers].sort((a, b) => layerMass(b) - layerMass(a)).slice(0, 8).map((item) => <span key={item.layer}>L{item.layer}<strong>{layerMass(item).toFixed(3)}</strong><small>n{objective === "shared_direction" ? (item.top_shared_neuron ?? item.top_neuron) : item.top_neuron}</small></span>)}</div><div className="disagreement-list"><strong>Largest method disagreements</strong>{disagreements.map((item) => { const neuronKey = `${item.neuron.layer}:${item.neuron.neuron}`; return <button className={selected === neuronKey ? "active" : ""} key={neuronKey} onClick={() => select(neuronKey)}><span>L{item.neuron.layer}:n{item.neuron.neuron}</span><b>{item.direction === "downstream_amplified" ? "downstream" : "direct"} ×{Math.max(item.downstreamToDirectRatio, 1 / item.downstreamToDirectRatio).toFixed(1)}</b><small>{(item.neuron.direct_downstream_sign_agreement * 100).toFixed(0)}% sign agreement</small></button>; })}</div></section>;
}

function FFNDecomposition({ rank, residual, objective }: { rank: RankSummary; residual: number; objective: RankingObjective }) {
  const mass = (item: RankSummary["layers"][number]) => objective === "shared_direction" ? (item.absolute_mean_mass ?? 0) : item.rms_mass;
  const top = [...rank.layers].sort((a, b) => mass(b) - mass(a))[0];
  const share = objective === "shared_direction" ? top?.top_10_mean_share : top?.top_10_rms_share;
  return <section className="panel decomposition"><h3>FFN decomposition</h3><div className="decomp-flow"><div><span>Measured ΔF</span><strong>{fmt(rank.measured_delta_mean)}</strong></div><i>−</i><div><span>Predicted ΣI</span><strong>{fmt(rank.predicted_delta_mean)}</strong></div><i>=</i><div className={Math.abs(residual) > Math.abs(rank.measured_delta_mean) * .5 ? "alert" : ""}><span>Residual / skip</span><strong>{fmt(residual)}</strong></div><i>·</i><div><span>Leading layer</span><strong>{top ? `L${top.layer}` : "—"}</strong><small>{top && share != null ? `${(share * 100).toFixed(1)}% in top 10 · ${objective === "shared_direction" ? "|mean I|" : "RMS"}` : ""}</small></div></div></section>;
}

function AttentionWorkspace({ caseValue, summaries, configurePath }: { caseValue: ResearchCase; summaries: Record<string, EvidenceSummary>; configurePath: (config: { sender: { layer: number; head: number }; receiver: { layer: number; head: number }; pairId: string; operation: "patch" | "restore"; alignment: "identity" | "explicit"; positions: Array<{ original: number; perturbed: number }>; controls: number; seed: number }) => Promise<void> }) {
  const rank = Object.values(summaries).find(isAttentionRankSummary);
  const intervention = Object.values(summaries).find(isAttentionInterventionSummary);
  const tokenTrace = Object.values(summaries).find((item): item is AttentionTraceSummary => isAttentionTraceSummary(item) && item.trace_kind === "token_edges");
  const pathTrace = Object.values(summaries).find((item): item is AttentionTraceSummary => isAttentionTraceSummary(item) && item.trace_kind === "head_paths");
  const [selected, setSelected] = useState<AttentionHead | null>(null);
  const [split, setSplit] = useState<"all" | Split>("all");
  if (!rank) return <Empty>Complete qualification and attention-head ranking to enter the path workspace.</Empty>;
  const head = selected ?? rank.heads[0];
  const allowedPairs = new Set(caseValue.workflow.rank.pairs.filter((item) => split === "all" || item.split === split).map((item) => item.id));
  const visibleTokenTrace = tokenTrace ? { ...tokenTrace, token_edges: tokenTrace.token_edges.filter((item) => allowedPairs.has(item.pair_id)) } : undefined;
  return <div className="attention-stack">
    <div className="filter-bar"><label>SPLIT<select value={split} onChange={(e) => setSplit(e.target.value as typeof split)}><option>all</option><option>discovery</option><option>validation</option><option>heldout</option></select></label><span>Head ranking is aggregated on discovery pairs; interventions and routes can be filtered by declared split.</span></div>
    <div className="analysis-grid"><HeadLandscape summary={rank} select={setSelected} /><HeadInspector head={head} intervention={intervention} /></div>
    <DosePanel title="Selected heads vs matched-random controls" doses={(intervention?.doses ?? []).filter((item) => split === "all" || item.split === split)} countKey="head_count" />
    <TokenRoutes summary={visibleTokenTrace} head={head} />
    <PathBuilder caseValue={caseValue} intervention={intervention} configure={configurePath} />
    <PathResult caseValue={caseValue} summary={pathTrace} />
  </div>;
}

function HeadLandscape({ summary, select }: { summary: AttentionRankSummary; select: (head: AttentionHead) => void }) {
  const max = Math.max(...summary.heads.map((item) => Math.abs(item.direct_effect_mean)), 1e-8);
  return <section className="panel chart"><h3>Head landscape · signed direct effect</h3><Plot data={[{ type: "scattergl", mode: "markers", x: summary.heads.map((item) => item.head), y: summary.heads.map((item) => item.layer), text: summary.heads.map((item) => `#${item.rank} L${item.layer}/H${item.head}`), customdata: summary.heads.map((item) => [item.direct_effect_mean, item.direct_effect_rms, item.sign_consistency]), marker: { symbol: "square", size: summary.heads.map((item) => 12 + 12 * Math.sqrt(item.direct_effect_rms / max)), color: summary.heads.map((item) => item.direct_effect_mean), cmin: -max, cmax: max, colorscale: [[0, "#c94b78"], [.5, "#eef2f2"], [1, "#0e9b8d"]], colorbar: { title: "effect", thickness: 10 } }, hovertemplate: "%{text}<br>mean=%{customdata[0]:.5f}<br>RMS=%{customdata[1]:.5f}<br>consistency=%{customdata[2]:.0%}<extra></extra>" }]} layout={{ ...plotLayout, xaxis: { title: "Head", dtick: 1, gridcolor: "#dce5e8" }, yaxis: { title: "Layer", dtick: 1, gridcolor: "#dce5e8" } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 390 }} /><div className="head-chips">{summary.heads.slice(0, 12).map((item) => <button key={`${item.layer}-${item.head}`} onClick={() => select(item)}>#{item.rank} L{item.layer}/H{item.head}</button>)}</div></section>;
}

function HeadInspector({ head, intervention }: { head: AttentionHead | undefined; intervention?: AttentionInterventionSummary }) {
  if (!head) return <section className="panel"><Empty>No head selected.</Empty></section>;
  const tested = intervention?.selected_heads.some((item) => item.layer === head.layer && item.head === head.head) ?? false;
  return <section className="panel head-inspector"><div className="inspector-title"><span>L{head.layer}</span><strong>H{head.head}</strong></div><span className={`tested ${tested ? "yes" : "no"}`}>{tested ? "intervention tested" : "ranking hypothesis"}</span><dl><dt>Rank</dt><dd>#{head.rank}</dd><dt>Direct effect mean</dt><dd>{fmt(head.direct_effect_mean, 6)}</dd><dt>Direct effect RMS</dt><dd>{head.direct_effect_rms.toFixed(6)}</dd><dt>Sign consistency</dt><dd>{(head.sign_consistency * 100).toFixed(0)}%</dd><dt>Original output norm</dt><dd>{head.original_output_norm_mean.toFixed(5)}</dd><dt>Perturbed output norm</dt><dd>{head.perturbed_output_norm_mean.toFixed(5)}</dd><dt>Output Δ norm</dt><dd>{head.output_delta_norm_mean.toFixed(5)}</dd></dl></section>;
}

function TokenRoutes({ summary, head }: { summary?: AttentionTraceSummary; head?: AttentionHead }) {
  if (!summary) return <section className="panel"><h3>Token routes</h3><p className="empty-inline">Run the token-edge trace after head intervention.</p></section>;
  const edges = summary.token_edges.filter((item) => !head || (item.layer === head.layer && item.head === head.head)).slice(0, 30);
  return <section className="panel token-routes"><div><h3>Source tokens → selected head</h3><p className="muted">Edge width is attention weight; color and sign encode direct observable effect. These are routing hypotheses, not conserved flow.</p></div><div className="token-fan" aria-label="Token route edge fan"><div className="token-column">{edges.map((edge, index) => <span key={`${edge.pair_id}-${edge.condition}-${edge.source_position}-${index}`} style={{ opacity: .35 + .65 * edge.attention_weight, borderColor: edge.direct_effect >= 0 ? "#0e9b8d" : "#c94b78" }}><small>{edge.source_position}</small>{edge.source_token || "∅"}<b>{fmt(edge.direct_effect, 4)}</b></span>)}</div><svg viewBox="0 0 220 300" role="img" aria-label="Token edges converging on an attention head">{edges.slice(0, 18).map((edge, index) => <line key={index} x1="0" y1={12 + index * 16} x2="210" y2="150" stroke={edge.direct_effect >= 0 ? "#0e9b8d" : "#c94b78"} strokeOpacity={.25 + edge.attention_weight * .75} strokeWidth={1 + edge.attention_weight * 6} />)}</svg><div className="route-head">{head ? `L${head.layer}/H${head.head}` : "selected head"}</div></div></section>;
}

function PathBuilder({ caseValue, intervention, configure }: { caseValue: ResearchCase; intervention?: AttentionInterventionSummary; configure: (config: { sender: { layer: number; head: number }; receiver: { layer: number; head: number }; pairId: string; operation: "patch" | "restore"; alignment: "identity" | "explicit"; positions: Array<{ original: number; perturbed: number }>; controls: number; seed: number }) => Promise<void> }) {
  const configured = caseValue.workflow.attention_traces?.find((item) => item.trace_kind === "head_paths") as Record<string, any> | undefined;
  const tested = intervention?.selected_heads ?? [];
  const senderDefault = configured?.senders?.[0] ?? tested.find((item) => tested.some((receiver) => receiver.layer > item.layer)) ?? tested[1] ?? tested[0];
  const receiverDefault = configured?.receivers?.[0] ?? tested.find((item) => senderDefault && item.layer > senderDefault.layer) ?? tested[0];
  const [sender, setSender] = useState(senderDefault ? `${senderDefault.layer}:${senderDefault.head}` : "");
  const [receiver, setReceiver] = useState(receiverDefault ? `${receiverDefault.layer}:${receiverDefault.head}` : "");
  const [pairId, setPairId] = useState(String(configured?.pair_ids?.[0] ?? caseValue.workflow.rank.pairs[0]?.id ?? ""));
  const [operation, setOperation] = useState<"patch" | "restore">((configured?.operation as "patch" | "restore") ?? "patch");
  const [alignment, setAlignment] = useState<"identity" | "explicit">((configured?.alignments?.[0]?.mode as "identity" | "explicit") ?? "identity");
  const [positionText, setPositionText] = useState<string>(String((configured?.alignments?.[0]?.positions ?? []).map((item: { original: number; perturbed: number }) => `${item.original}:${item.perturbed}`).join(", ")));
  const [controls, setControls] = useState(Number(configured?.controls?.samples ?? 5));
  const [seed, setSeed] = useState(Number(configured?.execution?.seed ?? 0));
  const [preview, setPreview] = useState<Record<string, any> | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const pathStage = caseValue.stages.find((item) => item.trace_kind === "head_paths");
  const locked = Boolean(pathStage?.run_id || pathStage?.job_id);
  const parseHead = (value: string) => { const [layer, head] = value.split(":").map(Number); return { layer, head }; };
  const positions = positionText.split(",").map((item) => item.trim()).filter(Boolean).map((item) => { const [original, perturbed] = item.split(":").map(Number); return { original, perturbed }; });
  const apply = async () => {
    setMessage(null);
    const from = parseHead(sender); const to = parseHead(receiver);
    if (from.layer >= to.layer) { setMessage("Sender layer must precede receiver layer."); return; }
    if (alignment === "explicit" && (!positions.length || positions.some((item) => !Number.isInteger(item.original) || !Number.isInteger(item.perturbed)))) { setMessage("Explicit alignment requires comma-separated original:perturbed positions."); return; }
    try { await configure({ sender: from, receiver: to, pairId, operation, alignment, positions, controls, seed }); setMessage("Path configuration saved to the canonical workflow."); }
    catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); }
  };
  const inspectTokens = async () => { try { setPreview(await loadTokenPreview(caseValue.case_id, pairId)); setMessage(null); } catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); } };
  return <section className="panel path-builder"><div className="path-builder-heading"><div><h3>Path builder</h3><p className="muted">Choose intervention-tested endpoints, inspect exact tokenizer output, then record alignment, controls, and seed before execution.</p></div><span className={`tested ${locked ? "yes" : "no"}`}>{locked ? "immutable executed path" : "editable draft"}</span></div><div className="builder-fields">
    <label>SENDER<select value={sender} disabled={locked} onChange={(e) => setSender(e.target.value)}>{tested.map((item) => <option key={`${item.layer}:${item.head}`} value={`${item.layer}:${item.head}`}>L{item.layer}/H{item.head} · rank {item.rank}</option>)}</select></label>
    <label>RECEIVER<select value={receiver} disabled={locked} onChange={(e) => setReceiver(e.target.value)}>{tested.map((item) => <option key={`${item.layer}:${item.head}`} value={`${item.layer}:${item.head}`}>L{item.layer}/H{item.head} · rank {item.rank}</option>)}</select></label>
    <label>PAIR<select value={pairId} disabled={locked} onChange={(e) => { setPairId(e.target.value); setPreview(null); }}>{caseValue.workflow.rank.pairs.map((item) => <option key={item.id}>{item.id}</option>)}</select></label>
    <label>OPERATION<select value={operation} disabled={locked} onChange={(e) => setOperation(e.target.value as typeof operation)}><option>patch</option><option>restore</option></select></label>
    <label>ALIGNMENT<select value={alignment} disabled={locked} onChange={(e) => setAlignment(e.target.value as typeof alignment)}><option>identity</option><option>explicit</option></select></label>
    <label>CONTROL SAMPLES<input type="number" min="1" value={controls} disabled={locked} onChange={(e) => setControls(Number(e.target.value))} /></label>
    <label>SEED<input type="number" value={seed} disabled={locked} onChange={(e) => setSeed(Number(e.target.value))} /></label>
    {alignment === "explicit" && <label className="wide">POSITION MAP · original:perturbed<input value={positionText} disabled={locked} onChange={(e) => setPositionText(e.target.value)} placeholder="0:0, 1:1, 2:3" /></label>}
  </div><div className="builder-actions"><button className="secondary" onClick={inspectTokens}>Preview tokenizer alignment</button><button className="primary" disabled={locked || !tested.length} onClick={apply}>Save path configuration</button></div>{message && <div className="warning">{message}</div>}{preview && <div className="alignment-preview"><TokenSequence title="Original model-input tokens" value={preview.original} /><TokenSequence title="Perturbed model-input tokens" value={preview.perturbed} /><span className={`alignment-status ${preview.identity_eligible ? "yes" : "no"}`}>{preview.identity_eligible ? `Equal ${preview.original.ids.length}-position inputs · identity position alignment eligible` : "Input lengths differ · declare explicit positional alignment"}</span></div>}</section>;
}

function TokenSequence({ title, value }: { title: string; value: { ids: number[]; tokens: string[] } }) { return <div><strong>{title}</strong><div className="token-sequence">{value.tokens.map((token, index) => <span key={`${value.ids[index]}-${index}`}><small>{index} · {value.ids[index]}</small>{token || "∅"}</span>)}</div></div>; }

function PathResult({ caseValue, summary }: { caseValue: ResearchCase; summary?: AttentionTraceSummary }) {
  if (!summary) return <section className="panel path-panel"><h3>Sender → receiver path</h3><p className="empty-inline">Configure endpoints in the canonical YAML, preview token alignment, then run the head-path stage.</p></section>;
  const path = strongestSelectedPath(summary); if (!path) return <section className="panel"><Empty>No selected path observation.</Empty></section>;
  const controls = summary.paths.filter((item) => item.arm === "matched_random_path");
  const randomMean = controls.length ? controls.reduce((sum, item) => sum + Math.abs(item.path_specific_effect), 0) / controls.length : null;
  const causal = isCausalPath(caseValue, summary);
  return <section className={`panel path-panel ${causal ? "causal" : "hypothesis"}`}><div className="path-heading"><div><h3>Sender → receiver path</h3><span className="tested yes">{causal ? "controlled causal path" : "routing hypothesis"}</span></div><div>{path.pair_id} · {path.alignment_mode} alignment</div></div><div className="path-diagram"><div><span>source tokens</span><small>{path.pair_id}</small></div><i>→</i><div><strong>L{path.sender.layer}/H{path.sender.head}</strong><span>sender</span><small>{fmt(path.sender_total_effect)} total</small></div><i>→</i><div><strong>L{path.receiver.layer}/H{path.receiver.head}</strong><span>receiver</span><small>{fmt(path.path_specific_effect)} specific</small></div><i>→</i><div><strong>Δ observable</strong><span>{path.normalized_source_progress == null ? "undefined progress" : `${(path.normalized_source_progress * 100).toFixed(2)}% source progress`}</span></div></div><div className="path-controls"><Metric label="PATH EFFECT" value={fmt(path.path_specific_effect)} /><Metric label="SENDER TOTAL" value={fmt(path.sender_total_effect)} /><Metric label="MATCHED RANDOM |MEAN|" value={fmt(randomMean)} /><Metric label="CONTROLS" value={String(controls.length)} text /></div>{summary.claims.map((claim) => <article className={`claim ${claim.status}`} key={claim.claim_id}><span>{claim.status}</span><strong>{claim.claim_type}</strong><p>{claim.statement}</p></article>)}</section>;
}

function Provenance({ caseValue, setStatus }: { caseValue: ResearchCase; setStatus: (value: string) => void }) {
  const [handoff, setHandoff] = useState<{ prompt: string; ready_stages: string[]; commands: string[] } | null>(null);
  useEffect(() => { void loadHandoff(caseValue.case_id).then(setHandoff).catch(() => setHandoff(null)); }, [caseValue.case_id, caseValue.revision]);
  const packet = async () => { await downloadPacket(caseValue.case_id); setStatus("Downloaded bounded research packet."); };
  const copy = async () => { const value = handoff ?? await loadHandoff(caseValue.case_id); await navigator.clipboard.writeText(value.prompt); setStatus("Copied agent handoff with run IDs and ready stages."); };
  return <div className="provenance-layout"><section className="panel"><h3>Immutable lineage</h3><div className="lineage-list">{caseValue.stages.map((stage) => <article key={stage.key}><span className={`status-dot ${stage.status}`} /><div><strong>{stageTitle(stage.key)}</strong><small>{stage.run_id ?? stage.job_id ?? "not run"}</small></div><span>{label(stage.status)}</span></article>)}</div>{handoff?.commands.length ? <div className="exact-commands"><strong>Exact bounded inspection commands</strong>{handoff.commands.map((command, index) => <code key={`${index}-${command}`}>{command}</code>)}</div> : null}</section><aside className="panel handoff"><h3>Agent handoff</h3><p>Continue through the headless CLI without flooding agent context. The packet contains the canonical driver, bounded evidence, claims, verification, reports, and exact commands—not raw tensors.</p><button className="primary full" onClick={packet}>Download research packet</button><button className="secondary full" onClick={copy}>Copy agent handoff</button><div className="cli-boundary"><strong>CLI-first diagnostics</strong><code>probe compare · stability · sensitivity</code><code>probe replay check · run</code><code>probe harvest · runs export</code></div></aside></div>;
}

function PromptCard({ index, title, value, onChange, perturbed }: { index: string; title: string; value: string; onChange: (value: string) => void; perturbed?: boolean }) { return <article className={`prompt-card ${perturbed ? "perturbed" : "original"}`}><div className="card-heading"><span>{index}</span><h2>{title}</h2></div><textarea value={value} onChange={(e) => onChange(e.target.value)} /></article>; }
function Empty({ children }: { children: string }) { return <div className="empty"><span>∅</span><p>{children}</p></div>; }
function Metric({ label: name, value, text = false }: { label: string; value: string; text?: boolean }) { return <article className="metric"><span>{name}</span><strong className={text ? "text-value" : ""}>{value}</strong></article>; }

function RankOverview({ summary }: { summary: RankSummary | null }) {
  if (!summary) return <Empty>Run or reopen an experiment to inspect its evidence.</Empty>; const pair = summary.pairs[0];
  return <><div className="metric-grid"><Metric label="MEASURED ΔF" value={fmt(summary.measured_delta_mean)} /><Metric label="PREDICTED ΣI" value={fmt(summary.predicted_delta_mean)} /><Metric label="FFN / SKIP" value={fmt(summary.ffn_skip_mean)} /><Metric label="EVIDENCE" value={label(summary.evidence_stage)} text /></div><div className="overview-grid"><article className="panel"><h3>Behavioral movement</h3><dl><dt>Original gap</dt><dd>{fmt(pair.original_gap)}</dd><dt>Perturbed gap</dt><dd>{fmt(pair.perturbed_gap)}</dd><dt>Original next token</dt><dd>{pair.original_prediction}</dd><dt>Perturbed next token</dt><dd>{pair.perturbed_prediction}</dd><dt>Ranked neurons</dt><dd>{summary.total_neuron_count.toLocaleString()}</dd></dl></article><article className="panel"><h3>Circuit diagnostic</h3><p className="regime">{pair.circuit_regime}</p><p className="muted">The ranking is a hypothesis. Causal language requires replication and controlled intervention.</p></article></div>{summary.warnings.map((warning) => <div className="warning" key={warning}>{warning}</div>)}</>;
}

function FFNLayers({ summary, objective: requestedObjective }: { summary: RankSummary | null; objective?: RankingObjective }) {
  if (!summary) return <Empty>Layer ranking is available on rank runs.</Empty>;
  const objective = requestedObjective ?? summary.ranking_objective ?? "effect_magnitude";
  const mass = (item: RankSummary["layers"][number]) => objective === "shared_direction" ? (item.absolute_mean_mass ?? 0) : item.rms_mass;
  const sorted = [...summary.layers].sort((a, b) => mass(b) - mass(a));
  return <section className="panel chart"><h3>Signed contribution by layer</h3><p className="muted">Bars show signed layer sums; the leading-layer strip is ordered by {objective === "shared_direction" ? "absolute signed mean" : "RMS mass"}.</p><Plot data={[{ type: "bar", x: summary.layers.map((x) => x.layer), y: summary.layers.map((x) => x.signed_mean_sum), marker: { color: summary.layers.map((x) => x.signed_mean_sum >= 0 ? "#0e9b8d" : "#c94b78") }, hovertemplate: "Layer %{x}<br>ΣI %{y:.5f}<extra></extra>" }]} layout={{ ...plotLayout, xaxis: { title: "Layer", gridcolor: "#dce5e8" }, yaxis: { title: "Signed ΣI", gridcolor: "#dce5e8" } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 340 }} /><div className="layer-strip">{sorted.slice(0, 8).map((item) => <span key={item.layer}>L{item.layer}<strong>{mass(item).toFixed(3)}</strong></span>)}</div></section>;
}

function FFNNeurons({ summary }: { summary: RankSummary | null }) {
  if (!summary) return <Empty>Neuron ranking is available on rank runs.</Empty>; const max = Math.max(...summary.neurons.map((x) => x.importance_rms), 1e-8);
  return <div className="analysis-grid"><article className="chart panel"><h3>Coupling × perturbation response</h3><Plot data={[{ type: "scattergl", mode: "markers", x: summary.neurons.map((x) => x.coupling), y: summary.neurons.map((x) => x.activation_delta_mean), text: summary.neurons.map((x) => `L${x.layer}:n${x.neuron}`), customdata: summary.neurons.map((x) => x.importance_mean), marker: { size: summary.neurons.map((x) => 7 + 18 * Math.sqrt(x.importance_rms / max)), color: summary.neurons.map((x) => x.importance_mean), colorscale: [[0, "#c94b78"], [.5, "#91a7ae"], [1, "#0e9b8d"]], showscale: true }, hovertemplate: "%{text}<br>c=%{x:.5f}<br>Δa=%{y:.5f}<br>I=%{customdata:.5f}<extra></extra>" }]} layout={plotLayout} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 420 }} /></article><article className="table-panel panel"><h3>Top hypotheses</h3><table><thead><tr><th>Rank</th><th>Neuron</th><th>mean I</th><th>RMS I</th><th>Consistency</th></tr></thead><tbody>{summary.neurons.slice(0, 100).map((x) => <tr key={`${x.layer}:${x.neuron}`}><td>{x.rank}</td><td>L{x.layer}:n{x.neuron}</td><td className={x.importance_mean >= 0 ? "positive" : "negative"}>{fmt(x.importance_mean, 6)}</td><td>{x.importance_rms.toFixed(6)}</td><td>{(x.sign_consistency * 100).toFixed(0)}%</td></tr>)}</tbody></table></article></div>;
}

function DosePanel({ title, doses, countKey }: { title: string; doses: Dose[]; countKey: "neuron_count" | "head_count" }) { return <section className="panel dose-panel"><div><h3>{title}</h3><p className="muted">Controlled effect subtracts the matched-random absolute mean from the selected absolute effect.</p></div>{doses.length ? <DoseChart doses={doses} countKey={countKey} /> : <p className="empty-inline">Controlled intervention stage not yet verified.</p>}</section>; }
function DoseChart({ doses, countKey }: { doses: Dose[]; countKey: "neuron_count" | "head_count" | "strength" }) { const x = doses.map((item) => countKey === "strength" ? item.strength : Number(item[countKey] ?? 0)); return <Plot data={[{ type: "scatter", mode: "lines+markers", name: "selected", x, y: doses.map((item) => item.selected_effect_mean), line: { color: "#0e9b8d" } }, { type: "scatter", mode: "lines+markers", name: "matched random |mean|", x, y: doses.map((item) => item.random_absolute_effect_mean ?? 0), line: { color: "#c94b78", dash: "dot" } }, { type: "bar", name: "controlled |effect|", x, y: doses.map(controlledDose), marker: { color: "#b6dcd7" }, opacity: .45 }]} layout={{ ...plotLayout, barmode: "overlay", xaxis: { title: label(countKey) }, yaxis: { title: "Observable effect", gridcolor: "#dce5e8" }, legend: { orientation: "h", y: 1.12 } }} config={{ displaylogo: false, responsive: true }} useResizeHandler style={{ width: "100%", height: 320 }} />; }

function RunList({ runs, openRun }: { runs: RunManifest[]; openRun: (id: string) => void }) { if (!runs.length) return <Empty>No immutable platform runs yet.</Empty>; return <div className="run-list">{runs.map((run) => <button key={run.run_id} onClick={() => openRun(run.run_id)}><span><strong>{run.run_id}</strong><small>{run.run_kind} · {run.requested_model.id} · {run.pair_count} pair</small></span><span className="run-meta">{label(run.evidence_stage)}<small>{new Date(run.completed_at).toLocaleString()}</small></span></button>)}</div>; }
function formatBytes(value: number) { if (!value) return ""; return value > 1_000_000 ? `${(value / 1_000_000).toFixed(0)} MB` : `${(value / 1_000).toFixed(0)} KB`; }

export default App;
