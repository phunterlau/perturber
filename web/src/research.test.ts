import { describe, expect, it } from "vitest";
import type { AttentionTraceSummary, ResearchCase, ResearchCaseStage } from "./types";
import {
  controlledDose,
  defaultAttentionWorkflow,
  isCausalPath,
  parseWorkflowYaml,
  serializeWorkflowYaml,
  strongestSelectedPath,
} from "./research";

const stage = (key: string, status: ResearchCaseStage["status"], traceKind?: "head_paths"): ResearchCaseStage => ({
  key,
  kind: traceKind ? "attention_trace" : key === "qualification" ? "qualify" : "attention_intervention",
  name: key,
  trace_kind: traceKind,
  status,
  run_id: `${key}-run`,
  parent_run_ids: [],
  verification_failures: [],
  claims: [],
  warnings: [],
});

const pathSummary = (): AttentionTraceSummary => ({
  schema_version: "probe.attention-trace-result/v1",
  parent_run_id: "attention-rank-run",
  parent_intervention_run_id: "attention-intervention-run",
  trace_kind: "head_paths",
  evidence_stage: "attention_causal_paths",
  logical_forward_passes: 5,
  token_edges: [],
  paths: [
    { pair_id: "capital", split: "heldout", arm: "selected_path", sender: { layer: 1, head: 0 }, receiver: { layer: 2, head: 0 }, sender_total_effect: 2, path_specific_effect: .4, normalized_source_progress: .2, alignment_mode: "identity" },
    { pair_id: "capital", split: "heldout", arm: "matched_random_path", sender: { layer: 0, head: 1 }, receiver: { layer: 2, head: 1 }, sender_total_effect: .1, path_specific_effect: .01, normalized_source_progress: .01, alignment_mode: "identity" },
  ],
  claims: [{ claim_id: "path", claim_type: "causal_path", status: "supported", statement: "Selected path beat controls.", limitations: [] }],
  warnings: [],
});

const caseValue = (): ResearchCase => ({
  schema_version: "probe.research-case/v1",
  case_id: "case-1",
  revision: 1,
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-08-10T00:00:00Z",
  intent: { hypothesis: "h", intended_perturbation: "p", invariants: [], falsifying_outcome: "f" },
  workflow: defaultAttentionWorkflow(),
  evidence_label: "locally_causal",
  stages: [stage("qualification", "verified"), stage("attention-intervention-1", "verified"), stage("attention-head-paths-1", "verified", "head_paths")],
  warnings: [],
});

describe("research workbench view models", () => {
  it("round-trips the canonical workflow through YAML", () => {
    const workflow = defaultAttentionWorkflow();
    expect(parseWorkflowYaml(serializeWorkflowYaml(workflow))).toEqual(workflow);
    expect(() => parseWorkflowYaml("schema_version: wrong")).toThrow(/probe.workflow\/v1/);
  });

  it("calculates controlled dose without hiding the matched control", () => {
    expect(controlledDose({ split: "discovery", condition: "original", strength: 1, head_count: 4, selected_effect_mean: 3, selected_absolute_effect_mean: 3, random_absolute_effect_mean: .5, controlled_absolute_effect: null, pair_count: 1, random_observation_count: 5 })).toBe(2.5);
  });

  it("uses the strongest selected path and requires every causal gate", () => {
    const summary = pathSummary();
    expect(strongestSelectedPath(summary)?.path_specific_effect).toBe(.4);
    expect(isCausalPath(caseValue(), summary)).toBe(true);
    const blocked = caseValue();
    blocked.stages[0] = stage("qualification", "gate_failed");
    expect(isCausalPath(blocked, summary)).toBe(false);
    expect(isCausalPath(caseValue(), { ...summary, parent_intervention_run_id: null })).toBe(false);
  });
});
