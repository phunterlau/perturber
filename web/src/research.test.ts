import { describe, expect, it } from "vitest";
import type { AttentionTraceSummary, FFNCouplingSummary, InterventionSummary, ResearchCase, ResearchCaseStage, TrajectorySummary } from "./types";
import {
  applyConfirmedTrajectoryBand,
  controlledDose,
  couplingDisagreements,
  defaultAttentionWorkflow,
  isCausalPath,
  interventionTrajectoryRows,
  matchedControlTrajectoryRows,
  parseWorkflowYaml,
  resolveNeuronEvidence,
  serializeWorkflowYaml,
  strongestSelectedPath,
  suggestedTrajectoryBand,
  trajectoryMetricView,
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

  it("switches trajectory metrics and preserves lower-is-better rank semantics", () => {
    const checkpoint = (layer: number, name: "block_input" | "post_attention" | "post_ffn", originalRank: number, perturbedRank: number) => ({ layer, checkpoint: name, original_gap: 1, perturbed_gap: 2, pair_delta: 1, original_target_probability: .2, perturbed_target_probability: .3, original_control_probability: .1, perturbed_control_probability: .1, original_entropy: 2, perturbed_entropy: 1.8, original_target_rank: originalRank, perturbed_target_rank: perturbedRank, original_forward_kl_to_final: .1, perturbed_forward_kl_to_final: .05, paired_js: .01, paired_total_variation: .1 });
    const summary = { schema_version: "probe.trajectory-result/v1", parent_run_id: "rank", pair_count: 1, logical_forward_passes: 2, pairs: [{ pair_id: "capital", split: "discovery", checkpoints: [checkpoint(0, "post_ffn", 20, 10), checkpoint(1, "post_ffn", 8, 2)], transitions: [{ rank: 1, layer: 1, checkpoint: "post_ffn", pair_delta_change: .8, absolute_change: .8, reason: "largest_pair_delta_change" }], final_pair_delta: 1, warnings: [] }], evidence_stage: "observational_trajectory", claims: [], warnings: [] } satisfies TrajectorySummary;
    const view = trajectoryMetricView(summary, "capital", "target_rank", "post_ffn");
    expect(view.lowerIsBetter).toBe(true);
    expect(view.rows.map((item) => item.label)).toEqual(["L0 FFN", "L1 FFN"]);
    expect(view.series[1].values).toEqual([10, 2]);
    expect(suggestedTrajectoryBand(summary, "capital")).toEqual([0, 1]);
  });

  it("records researcher-confirmed trajectory bands in the canonical FFN draft", () => {
    const updated = applyConfirmedTrajectoryBand(defaultAttentionWorkflow(), [21, 19, 20, 20], "capital");
    expect(updated.ffn_coupling?.layers).toEqual([19, 20, 21]);
    expect(updated.ffn_coupling?.tags).toMatchObject({ trajectory_band_confirmation: "researcher", trajectory_band_pair: "capital", trajectory_band_layers: "19,20,21" });
    expect(() => applyConfirmedTrajectoryBand({ ...defaultAttentionWorkflow(), ffn_coupling: null }, [1], "capital")).toThrow(/no FFN coupling/);
  });

  it("orders coupling disagreements without merging score definitions", () => {
    const neuron = (layer: number, index: number, direct: number, downstream: number) => ({ rank: index + 1, layer, neuron: index, activation_delta_mean: 1, direct_coupling: direct, direct_importance_rms: direct, native_coupling_mean: direct, native_importance_mean: direct, native_importance_rms: direct, downstream_coupling_mean: downstream, downstream_importance_mean: downstream, downstream_importance_rms: downstream, downstream_sign_consistency: 1, direct_downstream_sign_agreement: 1 });
    const summary = { schema_version: "probe.ffn-coupling-result/v1", parent_run_id: "rank", trajectory_run_id: "trajectory", pair_count: 1, candidate_pair_ids: ["capital"], logical_forward_passes: 2, logical_backward_passes: 2, methods: ["native_local_readout", "downstream_endpoint_gradient"], pairs: [], layers: [], neurons: [neuron(1, 0, 1, 2), neuron(2, 1, 1, 20)], total_neuron_count: 2, evidence_stage: "observational_ffn_coupling", claims: [], warnings: [] } satisfies FFNCouplingSummary;
    const disagreements = couplingDisagreements(summary);
    expect(disagreements[0].neuron.layer).toBe(2);
    expect(disagreements[0].direction).toBe("downstream_amplified");
    expect(disagreements[0].downstreamToDirectRatio).toBeCloseTo(20);
    const evidence = resolveNeuronEvidence([], summary.neurons, "2:1");
    expect(evidence).toMatchObject({ layer: 2, neuron: 1 });
    expect(evidence?.rank).toBeUndefined();
    expect(evidence?.coupling?.downstream_importance_rms).toBe(20);
  });

  it("prepares the widest selected intervention overlay without mixing controls", () => {
    const row = (arm: "selected" | "matched_random", neuronCount: number, layer: number, checkpoint: "block_input" | "post_attention" | "post_ffn", effect: number) => ({ pair_id: "capital", split: "discovery" as const, arm, control_sample: arm === "selected" ? null : 0, condition: "original" as const, mode: "patch" as const, neuron_count: neuronCount, strength: 1, layer, checkpoint, baseline_gap: 0, intervention_gap: effect, gap_effect: effect, normalized_source_progress: effect });
    const summary = { schema_version: "probe.intervention-result/v1", parent_run_id: "rank", evidence_stage: "causal_intervention", logical_forward_passes: 3, selected_neurons: [], observations: [], doses: [], trajectory_overlays: [row("selected", 1, 0, "post_ffn", .1), row("selected", 4, 0, "post_attention", .2), row("selected", 4, 0, "post_ffn", .3), row("matched_random", 4, 0, "post_ffn", .01)], claims: [], warnings: [] } satisfies InterventionSummary;
    const rows = interventionTrajectoryRows(summary, "capital");
    expect(rows.map((item) => item.label)).toEqual(["L0 attention", "L0 FFN"]);
    expect(rows.map((item) => item.gap_effect)).toEqual([.2, .3]);
    expect(matchedControlTrajectoryRows(summary, "capital").map((item) => item.gap_effect)).toEqual([.01]);
  });
});
