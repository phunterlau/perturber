import type {
  AttentionTraceSummary,
  Dose,
  FFNCouplingNeuron,
  FFNCouplingSummary,
  InterventionSummary,
  NeuronScore,
  PathObservation,
  RankSpec,
  ResearchCase,
  ResearchIntent,
  ResearchWorkflow,
  TrajectoryCheckpoint,
  TrajectorySummary,
} from "./types";
import { parse, stringify } from "yaml";

export const DEFAULT_INTENT: ResearchIntent = {
  hypothesis: "Prompt language changes are routed through a sparse, testable attention pathway.",
  intended_perturbation: "Change only the requested language from English to Chinese while preserving task semantics.",
  invariants: ["same underlying question", "same expected factual answer", "same first-token decision position"],
  falsifying_outcome: "The observable does not track generated language, or selected components do not outperform matched controls.",
};

const execution = (maxForwardPasses: number, maxArtifactBytes: number, seed = 260427401) => ({
  max_forward_passes: maxForwardPasses,
  max_artifact_bytes: maxArtifactBytes,
  allow_download: false as const,
  trust_remote_code: false as const,
  seed,
});

export function defaultAttentionWorkflow(): ResearchWorkflow {
  const rank: RankSpec = {
    schema_version: "probe.rank/v1",
    kind: "rank",
    name: "language-routing-ranking",
    model: {
      id: "Qwen/Qwen3-0.6B",
      revision: null,
      adapter: "qwen3",
      device: "mps",
      dtype: "float16",
      chat_template: true,
      enable_thinking: false,
    },
    pairs: [
      { id: "capital", split: "discovery", original: "What is the capital of France? Answer in English.", perturbed: "法国的首都是哪里？请用中文回答。", metadata: { perturbation_family: "language-routing" } },
      { id: "arithmetic", split: "discovery", original: "What is two plus two? Answer in English.", perturbed: "二加二等于多少？请用中文回答。", metadata: { perturbation_family: "language-routing" } },
      { id: "science", split: "heldout", original: "At what temperature does pure water freeze? Answer in English.", perturbed: "纯水在什么温度结冰？请用中文回答。", metadata: { perturbation_family: "language-routing" } },
    ],
    observable: { name: "language:chinese-minus-english", target_tokens: ["的", "是", "我", "这", "在"], control_tokens: ["The", "It", "This", "A", "I"] },
    ranking: { top_k: 500, pair_aggregation: "rms" },
    execution: execution(6, 100_000_000),
    tags: { paper_case: "EN-vs-ZH", surface: "research-case-web" },
  };
  return {
    schema_version: "probe.workflow/v1",
    name: "qwen3-language-attention-path",
    description: "Qualification-gated FFN and attention-path case with matched controls.",
    rank,
    qualification: {
      schema_version: "probe.qualify/v1", kind: "qualify", name: "language-generation-gate", parent_run_id: "$rank",
      generation: { max_new_tokens: 8, do_sample: false, seed: 260427401 }, evaluator: { kind: "unicode_script", target_values: ["han"], control_values: ["latin"] }, execution: execution(6, 10_000_000),
    },
    trajectory: {
      schema_version: "probe.trajectory/v1", kind: "trajectory", name: "language-native-trajectory", parent_run_id: "$rank",
      checkpoints: ["block_input", "post_attention", "post_ffn"], top_k: 10, transition_limit: 8, execution: execution(6, 50_000_000),
    },
    ffn_coupling: {
      schema_version: "probe.ffn-coupling/v1", kind: "ffn_coupling", name: "language-layer-aware-ffn", parent_run_id: "$rank", trajectory_run_id: "$trajectory",
      methods: ["native_local_readout", "downstream_endpoint_gradient"], top_k: 500, max_backward_passes: 6, execution: execution(6, 100_000_000),
    },
    interventions: [{
      schema_version: "probe.intervention/v1", kind: "intervention", name: "top-neuron-patch", parent_run_id: "$rank", qualification_run_id: "$qualification", trajectory_run_id: "$trajectory",
      selection: { strategy: "ranked_top_k", top_k: 32, sign: "any", min_sign_consistency: 0 }, operation: { mode: "patch", condition: "auto" },
      sweep: { neuron_counts: [1, 4, 16, 32], strengths: [1] }, controls: { samples: 5 }, execution: execution(72, 50_000_000),
    }],
    directions: [{
      schema_version: "probe.direction/v1", kind: "direction", name: "residual-direction-control", parent_run_id: "$rank", qualification_run_id: "$qualification",
      layers: [18, 25], betas: [0.5, 1], condition: "perturbed", normalization: "residual_norm", random_direction_samples: 5, execution: execution(72, 50_000_000),
    }],
    attention_rank: {
      schema_version: "probe.attention-rank/v1", kind: "attention_rank", name: "language-attention-head-ranking", parent_run_id: "$rank", qualification_run_id: "$qualification",
      ranking: { top_k: 64, pair_aggregation: "rms" }, execution: execution(6, 50_000_000),
    },
    attention_interventions: [{
      schema_version: "probe.attention-intervention/v1", kind: "attention_intervention", name: "top-head-patch", parent_run_id: "$attention_rank", qualification_run_id: "$qualification",
      selection: { strategy: "ranked_top_k", top_k: 16, sign: "any", min_sign_consistency: 0 }, operation: { mode: "patch", condition: "auto" },
      sweep: { head_counts: [1, 4, 16], strengths: [1] }, controls: { samples: 5 }, execution: execution(54, 50_000_000),
    }],
    attention_traces: [
      {
        schema_version: "probe.attention-trace/v1", kind: "attention_trace", name: "top-head-token-routes", trace_kind: "token_edges", parent_run_id: "$attention_rank",
        pair_ids: ["capital", "arithmetic", "science"], heads: [{ layer: 25, head: 0 }, { layer: 18, head: 12 }, { layer: 24, head: 0 }, { layer: 24, head: 2 }], max_token_edges: 100, execution: execution(6, 20_000_000),
      },
      {
        schema_version: "probe.attention-trace/v1", kind: "attention_trace", name: "l18h12-to-l25h0-path", trace_kind: "head_paths", parent_run_id: "$attention_rank", parent_intervention_run_id: "$attention_intervention",
        pair_ids: ["capital"], senders: [{ layer: 18, head: 12 }], receivers: [{ layer: 25, head: 0 }], operation: "patch",
        alignments: [{ pair_id: "capital", mode: "identity" }], controls: { samples: 5 }, execution: execution(13, 10_000_000),
      },
    ],
  };
}

export function workflowFromQuick(rank: RankSpec): ResearchWorkflow {
  return { schema_version: "probe.workflow/v1", name: `${rank.name}-case`, description: "Promoted from Quick Probe.", rank };
}

export function parseWorkflowYaml(source: string): ResearchWorkflow {
  const parsed = parse(source) as ResearchWorkflow;
  if (!parsed || parsed.schema_version !== "probe.workflow/v1" || !parsed.rank) {
    throw new Error("Expected a probe.workflow/v1 document with a rank stage.");
  }
  return parsed;
}

export function serializeWorkflowYaml(workflow: ResearchWorkflow): string {
  return stringify(workflow);
}

export const stageTitle = (key: string): string => ({
  rank: "Rank",
  qualification: "Behavioral qualification",
  trajectory: "Paired trajectory",
  "ffn-coupling": "Layer-aware FFN coupling",
  "attention-rank": "Head ranking",
}[key] ?? key.replaceAll("-", " "));

export const checkpointLabel = (checkpoint: TrajectoryCheckpoint["checkpoint"]): string => ({
  block_input: "input",
  post_attention: "attention",
  post_ffn: "FFN",
})[checkpoint];

export type TrajectoryMetric = "logit_gap" | "target_probability" | "target_rank" | "entropy" | "forward_kl" | "paired_js" | "total_variation";
export type TrajectoryCheckpointFilter = "all" | TrajectoryCheckpoint["checkpoint"];

export type TrajectoryMetricView = {
  rows: Array<TrajectoryCheckpoint & { x: number; label: string }>;
  series: Array<{ name: string; values: number[]; paired: boolean }>;
  yTitle: string;
  lowerIsBetter: boolean;
};

export function trajectoryRows(summary: TrajectorySummary, pairId: string): Array<TrajectoryCheckpoint & { x: number; label: string }> {
  const pair = summary.pairs.find((item) => item.pair_id === pairId) ?? summary.pairs[0];
  if (!pair) return [];
  return pair.checkpoints.map((item, index) => ({ ...item, x: index, label: `L${item.layer} ${checkpointLabel(item.checkpoint)}` }));
}

export function trajectoryMetricView(
  summary: TrajectorySummary,
  pairId: string,
  metric: TrajectoryMetric,
  checkpoint: TrajectoryCheckpointFilter = "all",
): TrajectoryMetricView {
  const rows = trajectoryRows(summary, pairId)
    .filter((item) => checkpoint === "all" || item.checkpoint === checkpoint)
    .map((item, index) => ({ ...item, x: index }));
  const paired = (original: number[], perturbed: number[]) => [
    { name: "original", values: original, paired: false },
    { name: "perturbed", values: perturbed, paired: false },
    { name: "perturbed − original", values: perturbed.map((value, index) => value - original[index]), paired: true },
  ];
  if (metric === "logit_gap") return {
    rows,
    series: [
      { name: "original gap", values: rows.map((item) => item.original_gap), paired: false },
      { name: "perturbed gap", values: rows.map((item) => item.perturbed_gap), paired: false },
      { name: "paired Δ", values: rows.map((item) => item.pair_delta), paired: true },
    ],
    yTitle: "Target − control logit gap",
    lowerIsBetter: false,
  };
  if (metric === "target_probability") return { rows, series: paired(rows.map((item) => item.original_target_probability), rows.map((item) => item.perturbed_target_probability)), yTitle: "Target-token probability", lowerIsBetter: false };
  if (metric === "target_rank") return { rows, series: paired(rows.map((item) => item.original_target_rank), rows.map((item) => item.perturbed_target_rank)).slice(0, 2), yTitle: "Target-token rank · lower is better", lowerIsBetter: true };
  if (metric === "entropy") return { rows, series: paired(rows.map((item) => item.original_entropy), rows.map((item) => item.perturbed_entropy)), yTitle: "Vocabulary entropy", lowerIsBetter: false };
  if (metric === "forward_kl") return { rows, series: paired(rows.map((item) => item.original_forward_kl_to_final), rows.map((item) => item.perturbed_forward_kl_to_final)).slice(0, 2), yTitle: "Forward KL to final", lowerIsBetter: true };
  if (metric === "paired_js") return { rows, series: [{ name: "paired JS", values: rows.map((item) => item.paired_js), paired: true }], yTitle: "Paired Jensen−Shannon divergence", lowerIsBetter: false };
  return { rows, series: [{ name: "paired total variation", values: rows.map((item) => item.paired_total_variation), paired: true }], yTitle: "Paired total variation", lowerIsBetter: false };
}

export function suggestedTrajectoryBand(
  summary: TrajectorySummary,
  pairId: string,
  padding = 1,
): number[] {
  const pair = summary.pairs.find((item) => item.pair_id === pairId) ?? summary.pairs[0];
  if (!pair?.checkpoints.length || !pair.transitions.length) return [];
  const layers = pair.checkpoints.map((item) => item.layer);
  const lower = Math.max(Math.min(...layers), pair.transitions[0].layer - padding);
  const upper = Math.min(Math.max(...layers), pair.transitions[0].layer + padding);
  return Array.from({ length: upper - lower + 1 }, (_, index) => lower + index);
}

export function applyConfirmedTrajectoryBand(
  workflow: ResearchWorkflow,
  layers: number[],
  pairId: string,
): ResearchWorkflow {
  if (!workflow.ffn_coupling) throw new Error("The workflow has no FFN coupling stage to scope.");
  const normalized = [...new Set(layers)].sort((a, b) => a - b);
  if (!normalized.length || normalized.some((layer) => !Number.isInteger(layer) || layer < 0)) {
    throw new Error("A confirmed FFN layer band must contain non-negative integer layers.");
  }
  return {
    ...workflow,
    ffn_coupling: {
      ...workflow.ffn_coupling,
      layers: normalized,
      tags: {
        ...((workflow.ffn_coupling.tags as Record<string, string> | undefined) ?? {}),
        trajectory_band_confirmation: "researcher",
        trajectory_band_pair: pairId,
        trajectory_band_layers: normalized.join(","),
      },
    },
  };
}

export type CouplingDisagreement = {
  neuron: FFNCouplingNeuron;
  downstreamToDirectRatio: number;
  log10Ratio: number;
  direction: "downstream_amplified" | "direct_amplified";
};

export function couplingDisagreements(
  summary: FFNCouplingSummary,
  limit = 20,
): CouplingDisagreement[] {
  const epsilon = 1e-12;
  return summary.neurons
    .map((neuron) => {
      const ratio = (neuron.downstream_importance_rms + epsilon) / (neuron.direct_importance_rms + epsilon);
      const log10Ratio = Math.log10(ratio);
      return { neuron, downstreamToDirectRatio: ratio, log10Ratio, direction: log10Ratio >= 0 ? "downstream_amplified" as const : "direct_amplified" as const };
    })
    .sort((left, right) => Math.abs(right.log10Ratio) - Math.abs(left.log10Ratio) || left.neuron.layer - right.neuron.layer || left.neuron.neuron - right.neuron.neuron)
    .slice(0, limit);
}

export function resolveNeuronEvidence(
  rankNeurons: NeuronScore[],
  couplingNeurons: FFNCouplingNeuron[],
  selected: string | null,
): { layer: number; neuron: number; rank?: NeuronScore; coupling?: FFNCouplingNeuron } | null {
  const fallback = couplingNeurons[0] ?? rankNeurons[0];
  if (!fallback) return null;
  const parsed = selected?.split(":").map(Number);
  const layer = parsed?.length === 2 && parsed.every(Number.isInteger) ? parsed[0] : fallback.layer;
  const neuron = parsed?.length === 2 && parsed.every(Number.isInteger) ? parsed[1] : fallback.neuron;
  return {
    layer,
    neuron,
    rank: rankNeurons.find((item) => item.layer === layer && item.neuron === neuron),
    coupling: couplingNeurons.find((item) => item.layer === layer && item.neuron === neuron),
  };
}

export function interventionTrajectoryRows(
  summary: InterventionSummary,
  pairId: string,
): Array<NonNullable<InterventionSummary["trajectory_overlays"]>[number] & { x: number; label: string }> {
  const selected = (summary.trajectory_overlays ?? []).filter(
    (item) => item.pair_id === pairId && item.arm === "selected",
  );
  if (!selected.length) return [];
  const neuronCount = Math.max(...selected.map((item) => item.neuron_count));
  const atWidth = selected.filter((item) => item.neuron_count === neuronCount);
  const strength = [...new Set(atWidth.map((item) => item.strength))].sort(
    (a, b) => Math.abs(b) - Math.abs(a) || b - a,
  )[0];
  return atWidth
    .filter((item) => item.strength === strength)
    .sort((a, b) => a.layer - b.layer || ["block_input", "post_attention", "post_ffn"].indexOf(a.checkpoint) - ["block_input", "post_attention", "post_ffn"].indexOf(b.checkpoint))
    .map((item, index) => ({ ...item, x: index, label: `L${item.layer} ${checkpointLabel(item.checkpoint)}` }));
}

export function matchedControlTrajectoryRows(
  summary: InterventionSummary,
  pairId: string,
): Array<{ x: number; label: string; gap_effect: number }> {
  const selected = interventionTrajectoryRows(summary, pairId);
  if (!selected.length) return [];
  const width = selected[0].neuron_count;
  const strength = selected[0].strength;
  const groups = new Map<string, number[]>();
  for (const item of summary.trajectory_overlays ?? []) {
    if (item.pair_id !== pairId || item.arm !== "matched_random" || item.neuron_count !== width || item.strength !== strength) continue;
    const key = `${item.layer}:${item.checkpoint}`;
    groups.set(key, [...(groups.get(key) ?? []), item.gap_effect]);
  }
  return selected.flatMap((item, index) => {
    const values = groups.get(`${item.layer}:${item.checkpoint}`);
    return values?.length ? [{ x: index, label: item.label, gap_effect: values.reduce((sum, value) => sum + value, 0) / values.length }] : [];
  });
}

export function controlledDose(dose: Dose): number {
  return dose.controlled_absolute_effect ?? dose.selected_absolute_effect_mean - (dose.random_absolute_effect_mean ?? 0);
}

export function strongestSelectedPath(summary: AttentionTraceSummary): PathObservation | null {
  const selected = summary.paths.filter((item) => item.arm === "selected_path");
  return selected.sort((a, b) => Math.abs(b.path_specific_effect) - Math.abs(a.path_specific_effect))[0] ?? null;
}

export function isCausalPath(caseValue: ResearchCase, summary: AttentionTraceSummary): boolean {
  const qualification = caseValue.stages.find((item) => item.key === "qualification");
  const intervention = caseValue.stages.find((item) => item.key.startsWith("attention-intervention-"));
  const pathStage = caseValue.stages.find((item) => item.run_id && item.trace_kind === "head_paths");
  return Boolean(
    qualification?.status === "verified" &&
    intervention?.status === "verified" &&
    pathStage?.status === "verified" &&
    summary.parent_intervention_run_id &&
    summary.paths.some((item) => item.arm === "matched_random_path") &&
    summary.claims.some((item) => item.claim_type === "causal_path" && item.status === "supported"),
  );
}
