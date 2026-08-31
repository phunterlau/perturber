import { describe, expect, it } from "vitest";
import type { AttentionTraceSummary, ResearchCase, ResearchCaseStage, TrajectorySummary } from "./types";
import {
  controlledDose,
  defaultAttentionWorkflow,
  isCausalPath,
  parseWorkflowYaml,
  serializeWorkflowYaml,
  strongestSelectedPath,
  trajectoryRows,
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

  it("adds trajectory and layer-aware coupling stages to the canonical template", () => {
    const workflow = defaultAttentionWorkflow();
    expect(workflow.trajectory?.parent_run_id).toBe("$rank");
    expect(workflow.ffn_coupling?.trajectory_run_id).toBe("$trajectory");
  });

  it("prepares ordered native checkpoint rows for a selected pair", () => {
    const checkpoint = (layer: number, name: "block_input" | "post_attention" | "post_ffn", pairDelta: number) => ({ layer, checkpoint: name, original_gap: 1, perturbed_gap: 1 + pairDelta, pair_delta: pairDelta, original_target_probability: .2, perturbed_target_probability: .3, original_control_probability: .1, perturbed_control_probability: .1, original_entropy: 2, perturbed_entropy: 2, original_target_rank: 4, perturbed_target_rank: 3, original_forward_kl_to_final: .1, perturbed_forward_kl_to_final: .1, paired_js: .01, paired_total_variation: .1 });
    const summary = { schema_version: "probe.trajectory-result/v1", parent_run_id: "rank", pair_count: 1, logical_forward_passes: 2, pairs: [{ pair_id: "capital", split: "discovery", checkpoints: [checkpoint(0, "block_input", .1), checkpoint(0, "post_attention", .2), checkpoint(0, "post_ffn", .3)], transitions: [], final_pair_delta: .3, warnings: [] }], evidence_stage: "observational_trajectory", claims: [], warnings: [] } satisfies TrajectorySummary;
    expect(trajectoryRows(summary, "capital").map((item) => item.label)).toEqual(["L0 input", "L0 attention", "L0 FFN"]);
    expect(trajectoryRows(summary, "missing")[2].pair_delta).toBe(.3);
  });
});
