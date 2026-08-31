# Native Paired Trajectories and Layer-Aware FFN Coupling

Status: implementation plan  
Target repository: Perturber  
Primary acceptance model: `Qwen/Qwen3-0.6B` on macOS MPS  
Scientific boundary: trajectories and coupling scores generate hypotheses;
controlled interventions and replication adjudicate them.

## Purpose

This note plans two related capabilities:

1. a native paired trajectory probe that shows how a controlled prompt
   perturbation becomes decodable, changes, persists, or is erased across model
   depth; and
2. layer-aware FFN coupling that estimates how an observed neuron activation
   change relates to the readout at its own layer and to the actual downstream
   computation remaining after that layer.

They extend the existing evidence chain without redefining its current ranking:

```text
controlled prompt pair
        -> behavioral movement
        -> representation trajectory over depth
        -> candidate FFN writes and downstream sensitivity
        -> matched causal intervention
        -> validation and held-out replication
```

The existing direct structural score remains available and keeps its current
meaning and replay behavior.

## First-principles research value

### The object of study

For a decoder-only transformer, a prompt produces a sequence of residual states:

```text
x -> r0 -> r1 -> ... -> rL -> logits -> behavior
```

A controlled perturbation gives two executions, `control` and `perturbed`, that
are intended to differ in one research factor. A target-minus-control observable
`F` turns a behavioral question into a signed scalar. The research problem is
not merely to find large activations. It is to determine:

1. whether the perturbation changes the chosen behavior;
2. where in computational depth evidence for that change becomes readable;
3. which operations write, preserve, suppress, or route it;
4. whether the model actually uses the candidate state or component; and
5. whether the result survives controls and new prompt pairs.

No single probe answers all five questions. Perturber is valuable when it keeps
their evidence types distinct and connects them through immutable lineage.

### What the paired trajectory contributes

A native trajectory applies the model's own final normalization and unembedding
to intermediate residual checkpoints. It asks:

> If the model had to expose its current residual state through its native
> vocabulary readout here, what target-control evidence would be decodable?

Comparing the two prompts across depth can reveal:

- **onset:** a layer band where the controlled difference first becomes
  decodable;
- **maintenance:** evidence that remains stable across later layers;
- **revision:** sign changes or large changes in the preferred answer;
- **erasure:** a transient difference that downstream computation removes;
- **late construction:** little readable difference until the final layers;
- **component timing:** whether an attention update or an FFN update coincides
  with a trajectory change; and
- **negative evidence:** a behavioral change with no stable native-readout
  transition, suggesting a poor observable, distributed computation, or probe
  mismatch.

This reduces the search space for expensive intervention. It also makes null and
overwrite results visible instead of reporting only the largest final neuron.

The trajectory does **not** show that the model uses the decoded information.
Intermediate decodability can be epiphenomenal, and applying the final readout to
an intermediate state can be miscalibrated because representations drift with
depth.

### What layer-aware coupling contributes

The current FFN structural coupling for neuron `n` at layer `l` is:

```text
c_direct(l,n) = <d_unembed, W_down(l)[:, n]>
I_direct(l,n) = delta_activation(l,n) * c_direct(l,n)
```

This correctly measures whether the neuron's output vector points in the final
target-control unembedding direction. It is fast and model-structural. However,
it does not account for final normalization, the state at that layer, or the
nonlinear downstream computation between layer `l` and the output.

The new implementation should expose three scores rather than replacing one
number with another:

| Score | Question | Interpretation |
|---|---|---|
| Direct structural | Does this neuron write in the final unembedding direction? | Fast geometric hypothesis |
| Native local-readout | Would this neuron move the native readout at this checkpoint and state? | Layer/state-aware decodability hypothesis |
| Downstream-gradient | Is the final observable locally sensitive to this neuron's output here? | First-order downstream-use hypothesis |
| Measured intervention | Does manipulating the neuron change the observable against controls? | Local causal evidence when gates pass |

For prompt `x`, define the native checkpoint observable:

```text
F_native(l, x) = target_control_gap(
    lm_head(final_norm(r_l(x)))
)
```

The native local direction is:

```text
g_native(l, x) = d F_native(l, x) / d r_l
```

The actual downstream direction is:

```text
g_final(l, x) = d F_final(x) / d r_l
```

The corresponding neuron couplings are dot products with the layer's FFN output
column `w(l,n) = W_down(l)[:, n]`. For the first implementation, use the
symmetric endpoint sensitivity across the controlled pair:

```text
c_gradient(l,n) = 0.5 * (
    <g_final(l, control),   w(l,n)> +
    <g_final(l, perturbed), w(l,n)>
)

I_gradient(l,n) = delta_activation(l,n) * c_gradient(l,n)
```

This is still a local linear approximation. Saturation, interactions, and
curvature can make it inaccurate. Integrated gradients or path quadrature can be
added later, but measured intervention remains the adjudicator.

### What becomes scientifically distinguishable

Together, the two capabilities support useful failure and mechanism taxonomies:

| Observed pattern | Plausible interpretation | Required follow-up |
|---|---|---|
| Early trajectory onset; high downstream coupling; successful patch | Early state is available and locally used | Replicate and narrow components |
| Early onset; weak downstream coupling; null patch | Information is decodable but apparently ignored or overwritten | Test erasure layers and alternate positions |
| Late onset; concentrated FFN write | Candidate late construction circuit | Same-layer random controls and held-out pairs |
| Direct coupling high; downstream coupling low | Geometric alignment is suppressed downstream | Do not call the neuron important without intervention |
| Direct coupling low; downstream coupling high | Downstream computation amplifies or rotates the write | Test small-dose intervention and nearby layers |
| Large predicted sum; poor measured delta | Strong nonlinearity, interactions, or incomplete component model | Report decomposition residual; test grouped interventions |
| Trajectory changes without behavioral qualification | Observable-level representation change, not validated behavior | Repair qualification or treat as a negative case |

This is the main research value: the tool helps reject explanations, not only
produce rankings.

## Scientific contract

The following language is mandatory in backend claims, reports, UI, and agent
handoffs:

- A native trajectory is **observational decodability evidence**.
- A direct or gradient coupling is a **candidate influence score**.
- A transition detector emits a **suggested band**, never a causal boundary.
- A residual-direction intervention establishes **controllability**, not neuron
  localization.
- A neuron or group becomes **locally causal** only after behavioral
  qualification, intervention, matched controls, verification, and the declared
  gate all pass.
- A result becomes **held-out replicated** only after the same preregistered
  direction and selection rule succeed on held-out pairs.

The model revision, prompt split, observable, selected token position, decoder
kind, coupling method, seed, dtype, device, and artifact hashes must remain part
of provenance.

## Proposed scientific artifacts

### `probe.trajectory/v1`

Create a new immutable run kind, `trajectory`, normally parented to a rank run.
The specification should contain:

- `parent_run_id` and optional `qualification_run_id`;
- pair/split selection inherited from the immutable parent;
- `position: -1` for the first version;
- checkpoint selection: `block_input`, `post_attention`, and `post_ffn`;
- decoder: `native_logit_lens`;
- scalar metrics: target-control gap, target/control probabilities, entropy,
  target/control ranks, forward KL to final, paired JS, and total variation;
- bounded top-k token storage;
- transition-suggestion parameters such as minimum magnitude, persistence
  window, and maximum suggestions;
- forward-pass and artifact budgets; and
- an explicit seed even though the native forward is deterministic.

The result should retain per-pair checkpoint summaries plus aggregate mean, RMS,
sign consistency, and split summaries. Full vocabulary tensors should not be
stored by default. Compute distribution metrics in memory and store only bounded
top-k values and declared scalar metrics.

Store dense scalar trajectories in `trajectory.safetensors`; keep a bounded
`summary.json`, `checkpoints.jsonl`, and report-friendly CSV.

### `probe.ffn-coupling/v1`

Create a separate immutable run kind, `ffn_coupling`, rather than changing the
meaning of `probe.rank/v1`. The specification should contain:

- `parent_run_id` pointing to an immutable rank run;
- optional `trajectory_run_id` for suggested layer scopes and comparison;
- optional `qualification_run_id` for evidence labeling;
- methods containing `native_local_readout` and/or
  `downstream_endpoint_gradient`;
- layer scope: explicit layers, all layers, or a confirmed trajectory band;
- `position: -1` for v1;
- top-k and aggregation rules;
- maximum forward passes, backward passes, wall time, and artifact bytes; and
- numerical safety thresholds for gradient norms and non-finite values.

For every retained neuron, store:

- original and perturbed activation;
- activation delta;
- existing direct structural coupling and importance;
- original, perturbed, and symmetric native-local coupling;
- original, perturbed, and symmetric downstream-gradient coupling;
- predicted signed effects for each method;
- pair-level values, RMS, sign consistency, and rank;
- cross-method sign agreement and rank displacement; and
- provenance linking the source tensors and adapter algorithm version.

Do not collapse these fields into a single generic `importance` value.

## Native trajectory measurement design

For Qwen3's pre-norm decoder block, capture at the selected token position:

```text
r_pre
  -> input norm -> attention -> residual add = r_post_attention
  -> post-attention norm -> MLP -> residual add = r_post_ffn
```

Apply the same native decoder to all three residual checkpoints:

```text
logits_checkpoint = lm_head(final_norm(r_checkpoint))
```

For the scalar target-control gap only, differences telescope exactly under the
chosen readout:

```text
attention_write = F(r_post_attention) - F(r_pre)
ffn_write       = F(r_post_ffn)       - F(r_post_attention)
block_write     = attention_write + ffn_write
```

The paired write is the perturbed write minus the control write. Probability,
entropy, KL, JS, and rank metrics do not telescope and must not be presented as
additive component contributions.

The final `post_ffn` checkpoint followed by the model final norm and LM head must
match the ordinary model logits within a declared dtype/device tolerance. This
is a hard correctness invariant.

Transition suggestions should be derived from multiple signals:

- large paired target-control gap change;
- stable onset for a configured persistence window;
- sign reversal;
- peak paired distribution divergence; and
- later erasure.

The detector should return its threshold inputs and reasons. It should not hide
the full trajectory or assign a causal claim.

## Layer-aware gradient design

Add an adapter method dedicated to sensitivity capture rather than enabling
gradients inside the existing inference capture. It should:

1. execute under `torch.enable_grad()`;
2. capture block-output residual tensors at the selected position;
3. compute the final target-control observable;
4. call `torch.autograd.grad` with respect to all captured residuals, avoiding
   parameter-gradient accumulation;
5. detach and convert retained gradients to float32 CPU tensors;
6. clear graph references and hooks deterministically; and
7. verify that model parameters and `.grad` fields were not mutated.

For each layer, the FFN down-projection column is added to that block's residual
output. Therefore its dot product with the gradient at the block output is the
appropriate local derivative for a small neuron-output perturbation.

MPS execution needs explicit memory and numerical checks:

- no `torch.inference_mode()` in the gradient path;
- bounded prompt length and batch size one for v1;
- float32 accumulation for dot products and summaries;
- rejection of non-finite gradients;
- reporting of near-zero or extreme gradient norms;
- hook cleanup on success, failure, deadline, and cancellation; and
- peak allocated-memory and elapsed-time metadata when available.

Integrated-gradient coupling is deferred until endpoint gradients are validated
against finite differences. When added, it must declare the interpolation path,
quadrature rule, number of points, and extra call budget.

## Workflow and interface integration

Extend the research workflow with stable optional stages:

```text
Rank -> Behavioral Qualification -> Native Trajectory
                                     -> FFN Coupling
                                     -> FFN Intervention
                                     -> Held-out Validation
```

Suggested symbolic parents:

- `$rank`
- `$qualification`
- `$trajectory`
- `$ffn_coupling`

A trajectory may run after a weak qualification to diagnose a failed case, but
its claim remains observational. Causal intervention stages must continue to
honor the qualification gate.

The existing intervention stage should accept either a rank run or an
`ffn_coupling` run as its candidate source while preserving the immutable rank
lineage and existing v1 behavior.

### Agent-friendly CLI

All new operations should use the existing typed `validate`, `plan`,
`capabilities`, `preflight`, `run`, event, job, verification, report, and replay
machinery. Add bounded inspection commands such as:

```text
probe runs trajectory RUN_ID --pair PAIR_ID --metric logit_gap
probe runs transitions RUN_ID --split validation --limit 10
probe runs ffn-couplings RUN_ID --method downstream_gradient --top 20
probe runs coupling-compare RUN_ID --top 50
```

Machine responses must include schema version, source count, matched count,
returned count, method, filters, parent IDs, and claim label. Exact commands for
the next gated follow-up should appear in reports and research packets.

### WebUI

Add a `Trajectory` workspace between Evidence and component-specific views:

- paired layer trajectory with control, perturbed, and paired-delta views;
- checkpoint toggle for block input, post-attention, and post-FFN;
- metric selector for gap, probability, rank, entropy, KL, JS, and TV;
- pair/split filters and aggregate variability;
- suggested transition bands with explicit reasons and thresholds;
- a researcher-confirmed action to scope FFN or attention analysis to a band;
- baseline-versus-intervened trajectory overlays; and
- observational styling unless the linked intervention claim passes.

Extend the FFN workspace with a comparison view:

```text
x-axis: direct structural importance
y-axis: downstream-gradient importance
color: measured intervention effect when available
```

Disagreement quadrants are a result, not an error. The neuron drawer should show
all score definitions, pair-level values, controls, provenance, and caveats.

## Implementation milestones

### Milestone 0: contracts and frozen scientific fixtures

- Write the two spec/result schemas and example YAML before model code.
- Define claim labels, stage keys, artifact names, and budget accounting.
- Add fake trajectories with known onset, sign reversal, erasure, and no-signal
  cases.
- Add a tiny-Qwen fixture with a known target-control observable.
- Freeze backwards-compatibility tests for existing rank artifacts and replay
  hashes.

Exit criteria: schemas round-trip, invalid budgets/methods fail before model
calls, and existing rank/replay tests remain unchanged.

### Milestone 1: Qwen native trajectory adapter

- Introduce typed checkpoint captures in `adapters/base.py`.
- Implement Qwen hooks for block input, post-attention, and post-FFN states.
- Implement the native decoder from the model final norm and output embedding.
- Compute bounded metrics and explicit transition suggestions.
- Verify hook cleanup and ordinary-final-logit equivalence.

Exit criteria: deterministic fake/tiny-model tests pass; scalar gap increments
telescope; final logits match; no model weights are mutated.

### Milestone 2: immutable trajectory runs

- Add spec parsing, hashing, plan/capability/preflight accounting, service
  dispatch, daemon jobs, cancellation/recovery, artifact commit, verification,
  reporting, queries, and compact execution receipts.
- Add trajectory to workflow lineage and Research Case stage state.
- Add checked-in examples and a replay driver.

Exit criteria: standalone and daemon executions produce equivalent verified
artifacts, request idempotency works, interrupted jobs recover correctly, and
the CLI exposes bounded trajectory results.

### Milestone 3: layer-aware FFN coupling

- Implement native-local and downstream endpoint-gradient directions.
- Reuse immutable parent activation tensors; do not recapture activation deltas
  unless a validation check requires it.
- Compute per-pair and aggregate direct/native/downstream scores separately.
- Add gradient norm diagnostics, cancellation, and MPS cleanup.
- Validate selected couplings against centered finite-difference neuron-output
  perturbations on tiny Qwen.

Exit criteria: finite-difference relative error is within a declared tolerance
for small epsilon; no parameter gradients persist; direct scores remain byte- or
tolerance-compatible with existing artifacts.

### Milestone 4: causal handoff

- Let interventions select candidates from an immutable `ffn_coupling` run.
- Compare top direct, top downstream-gradient, overlap, and same-layer random
  controls at preregistered widths and doses.
- Overlay intervened trajectories to show where restoration begins downstream.
- Preserve behavioral qualification and held-out gates.

Exit criteria: causal styling is impossible without all gates; null controls
remain visible; reports distinguish prediction quality from intervention effect.

### Milestone 5: researcher and agent UI

- Add trajectory charts, band confirmation, coupling comparison, neuron details,
  preflight estimates, and exact agent handoff commands.
- Expand discriminated frontend summary types; do not introduce a generic
  fallback for the new run kinds.
- Add UI tests for metric transformations, transition rendering, score-method
  labels, and claim boundaries.

Exit criteria: a researcher can move from a qualified pair to a trajectory,
confirm a band, inspect coupling disagreements, launch a controlled intervention,
and export the lineage without using raw artifact files.

### Milestone 6: live Qwen3-0.6B acceptance

Run at least:

- the language-routing positive case;
- the CoT protocol negative diagnostic;
- one agreement or factual-entity case; and
- one held-out validation split.

For each case, compare direct and downstream-gradient top-k overlap, sign
agreement, predicted-versus-measured delta, and intervention precision against
same-layer random controls. Do not require reproduction of the paper's neuron
identities on the smaller Qwen model.

Exit criteria: artifacts verify and replay within declared MPS tolerances;
conclusions report both supportive and discrepant results; peak memory and runtime
fit the local-Mac budget.

## Test matrix

### Mathematical and adapter tests

- Final checkpoint native logits equal ordinary logits.
- Target-control gap attention and FFN increments telescope.
- Distribution metrics are finite and bounded where applicable.
- Exact tie ordering is deterministic.
- Transition fixtures produce expected suggestions and reasons.
- Autograd neuron coupling matches centered finite differences.
- Endpoint averaging is symmetric under prompt order with the signed pair delta
  transformed consistently.
- Hooks and graphs are released after success and injected failures.

### Contract and artifact tests

- Strict JSON/YAML parsing and duplicate-key rejection.
- Stable science/request hashes and parent immutability.
- Forward, backward, artifact, and wall-time budget rejection before execution.
- Safetensor shapes, names, digests, and untracked-file verification.
- Backwards-compatible parsing and replay for every existing run kind.

### Service and agent tests

- Fake-engine plan, preflight, run, cancel, recover, and idempotency.
- Bounded query envelopes for trajectories and coupling comparisons.
- Workflow symbolic-parent resolution and failed qualification gates.
- Research packet commands and unresolved-gate recommendations.

### UI tests

- Pair/split/checkpoint/metric filtering.
- Rank-axis handling where lower is better.
- Transition bands cannot render as causal evidence.
- Direct/native/downstream methods are never merged under an unlabeled score.
- Intervened trajectory styling follows backend claim state only.

### Live scientific acceptance

- Qwen3-0.6B MPS/float16 model execution with float32 scoring.
- Two fresh-process replays for at least one positive and one negative case.
- Small-dose finite-difference validation on selected layers.
- Top-ranked versus same-layer random intervention comparison.
- Validation/held-out behavior reported without post-hoc threshold changes.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Native logit lens is miscalibrated at early layers | Label as native decodability; compare with final use and optional Tuned Lens later |
| Gradient is local and unstable | Store endpoint-specific values, norms, agreement, finite-difference checks, and intervention results |
| Gradients increase MPS memory materially | Explicit backward budget, batch size one, scoped layers, cleanup tests, and preflight warning |
| Transition detector encourages cherry-picking | Store full trajectory, thresholds, all suggestions, researcher confirmation, and held-out rules |
| New scores break existing replays | New run kinds and algorithm versions; retain current direct rank semantics |
| Artifact volume grows with vocabulary and tokens | Last decision position and bounded top-k in v1; scalar divergence summaries only |
| Interventions leave the data manifold | Prefer source patch/mix and small doses; compare ablation, norm-matched, and same-layer random controls |
| A single model gives misleading confidence | Use Qwen3-0.6B only for engineering acceptance and state model-specific scientific conclusions |

## Deliberate non-goals for the first increment

- Training or downloading a Tuned Lens artifact.
- Claiming that decodability proves computation or use.
- Arbitrary earlier-token trajectories without explicit alignment and propagation
  semantics.
- Integrated gradients before endpoint-gradient finite-difference validation.
- Full-vocabulary artifact storage.
- Automatic causal claims or automatic intervention submission from a detected
  transition.
- Replacing matched interventions with gradient or lens scores.

## Definition of success

The feature is successful when a researcher or coding agent can answer, with
verified lineage:

1. Where did a controlled prompt difference first become natively decodable?
2. Was that difference written mainly during an attention or FFN update?
3. Which neurons were geometrically aligned, locally decodable, or actually
   downstream-sensitive at that depth?
4. Did those scoring methods agree, and where did they disagree?
5. Did a preregistered neuron or residual intervention restore or remove the
   trajectory more than matched controls?
6. Did the result replicate on validation or held-out prompt pairs?

Anything less should be reported as a bounded hypothesis, not a discovered
mechanism.
