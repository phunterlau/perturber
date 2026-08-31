export type Split = "discovery" | "validation" | "heldout";

export type PromptPair = {
  id: string;
  original: string;
  perturbed: string;
  split: Split;
  metadata?: Record<string, string | number | boolean | null>;
};

export type ModelRequest = {
  id: string;
  revision?: string | null;
  adapter: "auto" | "qwen3";
  device: "auto" | "cpu" | "mps" | "cuda";
  dtype: "auto" | "float32" | "float16" | "bfloat16";
  chat_template: boolean;
  enable_thinking: boolean;
};

export type RankSpec = {
  schema_version: "probe.rank/v1";
  kind: "rank";
  name: string;
  model: ModelRequest;
  pairs: PromptPair[];
  observable: {
    name: string;
    target_tokens: string[];
    control_tokens: string[];
    reduction?: "mean_logit_gap";
    decision_position?: 0;
  };
  capture?: { activation: "post_swiglu"; position: number; layers: "all" | number[] };
  ranking: { top_k: number; select_by?: string; pair_aggregation: string };
  execution: Record<string, unknown> & {
    max_forward_passes: number;
    max_artifact_bytes: number;
    allow_download: false;
    trust_remote_code: false;
    seed: number;
  };
  tags?: Record<string, string>;
};

export type ExperimentSpec = Record<string, unknown> & { kind: string; name?: string };

export type ResearchWorkflow = {
  schema_version: "probe.workflow/v1";
  name: string;
  description?: string | null;
  rank: RankSpec;
  qualification?: ExperimentSpec | null;
  trajectory?: ExperimentSpec | null;
  ffn_coupling?: ExperimentSpec | null;
  interventions?: ExperimentSpec[];
  directions?: ExperimentSpec[];
  attention_rank?: ExperimentSpec | null;
  attention_interventions?: ExperimentSpec[];
  attention_traces?: ExperimentSpec[];
};

export type Claim = {
  claim_id: string;
  claim_type: string;
  status: "supported" | "exploratory" | "blocked" | "not_supported";
  statement: string;
  limitations: string[];
};

export type LayerSummary = {
  layer: number;
  signed_mean_sum: number;
  rms_mass: number;
  positive_mean_mass: number;
  negative_mean_mass: number;
  top_10_rms_share: number;
  maximum_rms: number;
  top_neuron: number;
  activation_delta_norm_mean: number;
};

export type NeuronScore = {
  rank: number;
  layer: number;
  neuron: number;
  coupling: number;
  original_activation_mean: number;
  perturbed_activation_mean: number;
  activation_delta_mean: number;
  importance_mean: number;
  importance_rms: number;
  sign_consistency: number;
  observable_effect?: "toward_target" | "toward_control" | "neutral";
};

export type RankSummary = {
  schema_version: "probe.rank-result/v1";
  science_hash: string;
  pair_count: number;
  logical_forward_passes: number;
  model: Record<string, unknown>;
  observable: Record<string, unknown>;
  pairs: Array<{
    pair_id: string;
    split: Split;
    original_gap: number;
    perturbed_gap: number;
    measured_delta: number;
    predicted_delta: number;
    original_prediction: string;
    perturbed_prediction: string;
    ffn_skip_mean: number | null;
    circuit_regime: string;
    elapsed_seconds: number;
    warnings: string[];
  }>;
  layers: LayerSummary[];
  neurons: NeuronScore[];
  total_neuron_count: number;
  measured_delta_mean: number;
  predicted_delta_mean: number;
  ffn_skip_mean: number | null;
  evidence_stage: "exploratory_pair" | "replicated_ranking";
  claims: Claim[];
  warnings: string[];
};

export type QualificationSummary = {
  schema_version: "probe.qualification-result/v1";
  parent_run_id: string;
  evidence_stage: "qualified_observable";
  logical_forward_passes: number;
  aggregate: { informative_pairs: number; weak_pairs: number; invalid_pairs: number; claim_eligible: boolean };
  pairs: Array<{ pair_id: string; split: Split; status: string; reasons: string[] }>;
  claims: Claim[];
  warnings: string[];
};

export type TrajectoryCheckpoint = {
  layer: number;
  checkpoint: "block_input" | "post_attention" | "post_ffn";
  original_gap: number;
  perturbed_gap: number;
  pair_delta: number;
  original_target_probability: number;
  perturbed_target_probability: number;
  original_control_probability: number;
  perturbed_control_probability: number;
  original_entropy: number;
  perturbed_entropy: number;
  original_target_rank: number;
  perturbed_target_rank: number;
  original_forward_kl_to_final: number;
  perturbed_forward_kl_to_final: number;
  paired_js: number;
  paired_total_variation: number;
};

export type TrajectorySummary = {
  schema_version: "probe.trajectory-result/v1";
  parent_run_id: string;
  pair_count: number;
  logical_forward_passes: number;
  pairs: Array<{
    pair_id: string;
    split: Split;
    checkpoints: TrajectoryCheckpoint[];
    transitions: Array<{ rank: number; layer: number; checkpoint: TrajectoryCheckpoint["checkpoint"]; pair_delta_change: number; absolute_change: number }>;
    final_pair_delta: number;
    warnings: string[];
  }>;
  evidence_stage: "observational_trajectory";
  claims: Claim[];
  warnings: string[];
};

export type FFNCouplingNeuron = {
  rank: number;
  layer: number;
  neuron: number;
  activation_delta_mean: number;
  direct_coupling: number;
  direct_importance_rms: number;
  native_coupling_mean: number | null;
  native_importance_mean: number | null;
  native_importance_rms: number | null;
  downstream_coupling_mean: number;
  downstream_importance_mean: number;
  downstream_importance_rms: number;
  downstream_sign_consistency: number;
  direct_downstream_sign_agreement: number;
};

export type FFNCouplingSummary = {
  schema_version: "probe.ffn-coupling-result/v1";
  parent_run_id: string;
  trajectory_run_id: string | null;
  pair_count: number;
  logical_forward_passes: number;
  logical_backward_passes: number;
  methods: string[];
  pairs: Array<{ pair_id: string; split: Split; original_gradient_norm_mean: number; perturbed_gradient_norm_mean: number }>;
  layers: Array<{ layer: number; downstream_rms_mass: number; native_rms_mass: number | null; direct_rms_mass: number; top_neuron: number }>;
  neurons: FFNCouplingNeuron[];
  total_neuron_count: number;
  evidence_stage: "observational_ffn_coupling";
  claims: Claim[];
  warnings: string[];
};

export type Dose = {
  split: Split;
  condition: string;
  strength: number;
  neuron_count?: number;
  head_count?: number;
  selected_effect_mean: number;
  selected_absolute_effect_mean: number;
  random_absolute_effect_mean: number | null;
  controlled_absolute_effect: number | null;
  pair_count: number;
  random_observation_count: number;
};

export type InterventionSummary = {
  schema_version: "probe.intervention-result/v1";
  parent_run_id: string;
  evidence_stage: "causal_intervention";
  logical_forward_passes: number;
  selected_neurons: Array<NeuronScore>;
  observations: Array<Record<string, unknown> & { pair_id: string; split: Split; gap_effect: number }>;
  doses: Dose[];
  claims: Claim[];
  warnings: string[];
};

export type DirectionSummary = {
  schema_version: "probe.direction-result/v1";
  parent_run_id: string;
  evidence_stage: "causal_intervention";
  logical_forward_passes: number;
  doses: Array<Dose & { beta?: number }>;
  claims: Claim[];
  warnings: string[];
};

export type AttentionHead = {
  rank: number;
  layer: number;
  head: number;
  direct_effect_mean: number;
  direct_effect_rms: number;
  sign_consistency: number;
  original_output_norm_mean: number;
  perturbed_output_norm_mean: number;
  output_delta_norm_mean: number;
};

export type AttentionRankSummary = {
  schema_version: "probe.attention-rank-result/v1";
  parent_run_id: string;
  evidence_stage: "attention_hypothesis";
  logical_forward_passes: number;
  output_head_count: number;
  total_head_count: number;
  heads: AttentionHead[];
  layers: Array<{ layer: number; signed_effect_sum: number; rms_mass: number; top_head: number }>;
  pairs: Array<{ pair_id: string; split: Split; measured_delta: number; predicted_attention_delta: number }>;
  claims: Claim[];
  warnings: string[];
};

export type AttentionInterventionSummary = {
  schema_version: "probe.attention-intervention-result/v1";
  parent_run_id: string;
  evidence_stage: "attention_causal_heads";
  logical_forward_passes: number;
  selected_heads: AttentionHead[];
  observations: Array<Record<string, unknown> & { pair_id: string; split: Split; gap_effect: number; normalized_source_progress?: number | null }>;
  doses: Dose[];
  claims: Claim[];
  warnings: string[];
};

export type TokenEdge = {
  pair_id: string;
  condition: "original" | "perturbed";
  layer: number;
  head: number;
  source_position: number;
  source_token: string;
  attention_weight: number;
  direct_effect: number;
  output_norm: number;
};

export type PathObservation = {
  pair_id: string;
  split: Split;
  arm: "selected_path" | "matched_random_path";
  sender: { layer: number; head: number };
  receiver: { layer: number; head: number };
  sender_total_effect: number;
  path_specific_effect: number;
  normalized_source_progress: number | null;
  alignment_mode: "identity" | "explicit";
};

export type AttentionTraceSummary = {
  schema_version: "probe.attention-trace-result/v1";
  parent_run_id: string;
  parent_intervention_run_id: string | null;
  trace_kind: "token_edges" | "head_paths";
  evidence_stage: "attention_hypothesis" | "attention_causal_paths";
  logical_forward_passes: number;
  token_edges: TokenEdge[];
  paths: PathObservation[];
  claims: Claim[];
  warnings: string[];
};

export type EvidenceSummary =
  | RankSummary
  | QualificationSummary
  | TrajectorySummary
  | FFNCouplingSummary
  | InterventionSummary
  | DirectionSummary
  | AttentionRankSummary
  | AttentionInterventionSummary
  | AttentionTraceSummary;

export const isRankSummary = (value: EvidenceSummary): value is RankSummary => value.schema_version === "probe.rank-result/v1";
export const isTrajectorySummary = (value: EvidenceSummary): value is TrajectorySummary => value.schema_version === "probe.trajectory-result/v1";
export const isFFNCouplingSummary = (value: EvidenceSummary): value is FFNCouplingSummary => value.schema_version === "probe.ffn-coupling-result/v1";
export const isInterventionSummary = (value: EvidenceSummary): value is InterventionSummary => value.schema_version === "probe.intervention-result/v1";
export const isDirectionSummary = (value: EvidenceSummary): value is DirectionSummary => value.schema_version === "probe.direction-result/v1";
export const isAttentionRankSummary = (value: EvidenceSummary): value is AttentionRankSummary => value.schema_version === "probe.attention-rank-result/v1";
export const isAttentionInterventionSummary = (value: EvidenceSummary): value is AttentionInterventionSummary => value.schema_version === "probe.attention-intervention-result/v1";
export const isAttentionTraceSummary = (value: EvidenceSummary): value is AttentionTraceSummary => value.schema_version === "probe.attention-trace-result/v1";

export type CaseStageStatus = "not_configured" | "ready" | "running" | "failed" | "gate_failed" | "verified";
export type ResearchCaseStage = {
  key: string;
  kind: string;
  name: string;
  trace_kind?: "token_edges" | "head_paths" | null;
  status: CaseStageStatus;
  job_id?: string | null;
  run_id?: string | null;
  parent_run_ids: string[];
  verification_failures: string[];
  claims: Claim[];
  warnings: string[];
};

export type ResearchIntent = {
  hypothesis: string;
  intended_perturbation: string;
  invariants: string[];
  falsifying_outcome: string;
};

export type ResearchCase = {
  schema_version: "probe.research-case/v1";
  case_id: string;
  revision: number;
  created_at: string;
  updated_at: string;
  intent: ResearchIntent;
  workflow: ResearchWorkflow;
  evidence_label: "observational" | "behaviorally_qualified" | "locally_causal" | "heldout_replicated";
  stages: ResearchCaseStage[];
  warnings: string[];
};

export type ExperimentPlan = {
  kind: string;
  pair_count: number;
  forward_passes: number;
  within_budget: boolean;
  model_cached: boolean;
  resolved_device: string;
  warnings: string[];
};

export type ResearchCasePlan = {
  schema_version: "probe.research-case-plan/v1";
  case_id: string;
  evidence_label: string;
  total_forward_passes: number;
  warnings: string[];
  stages: Array<{ key: string; status: string; plan: ExperimentPlan | null; blocked_reason: string | null }>;
};

export type JobEvent = {
  event: string;
  sequence: number;
  timestamp: string;
  job_id: string;
  request_id: string;
  science_hash: string;
  payload: Record<string, unknown> & { run_id?: string };
};

export type RunManifest = {
  run_id: string;
  completed_at: string;
  evidence_stage: string;
  pair_count: number;
  requested_model: { id: string };
  warnings: string[];
  parent_run_ids: string[];
  artifacts: Array<{ path: string; media_type: string; sha256: string; size_bytes: number }>;
  run_kind: "rank" | "trajectory" | "ffn_coupling" | "qualify" | "intervention" | "direction" | "attention_rank" | "attention_intervention" | "attention_trace";
};
