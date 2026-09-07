# Researcher experience and functionality opportunities

Reviewed 2026-09-07 against `eda03dd`.

The current core supports paired ranking, generated-behavior qualification,
native trajectories, layer-aware FFN coupling, neuron and head interventions,
attention paths, immutable artifacts, and case-based WebUI workflows. The next
priority is helping researchers design follow-up experiments, compare competing
explanations, and share inspectable evidence.

## Corrections from this review

These corrections are in scope for the current implementation:

- Require a supported generalization claim and selected held-out observations
  before labeling a case held-out replicated. A supported local claim alone
  is insufficient. Recheck qualification integrity during refresh and require
  an appropriate causal claim type for promotion. Attention stages without an
  explicit supported generalization claim remain locally causal at most.
- Default stability analysis to discovery pairs. Expose `--split discovery`,
  `validation`, `heldout`, and `all`; record the population and pair IDs.
  Explicitly warn when pooling splits or inspecting held-out stability.
- Allow selection of each FFN intervention run, including runs sharing the same
  candidate method. Label the scope of the dose split filter and show an empty
  result when the requested split has no observations.
- Show an empty trajectory view for a split with no pairs, retain the split
  selector, and avoid silently substituting a discovery pair or layer band.
- Return the documented artifact-integrity error and exit code 6 from
  `trajectory-visualize` on failed verification. Preserve an existing export.

Regression coverage should include failed held-out claims, unrelated supported
claims, stale qualification verification, perturbing held-out tensor values,
empty populations, explicit CLI split selection, multiple intervention arms,
missing trajectory pairs, and failed export integrity.

## Feature roadmap

These are proposals for future increments, not functionality implemented by the
correction work above.

### 1. Portable evidence bundles and complete workflow replay

Question: can another researcher inspect and regenerate this result?

Preserve bounded checkpoint data, individual random-control curves, selected
effects, resolved specs, claims, warnings, source hashes, and renderer version.
Provide separate operations for exact figure regeneration from saved data and
new inference compared with declared tolerances. Extend replay beyond rank
runs to complete trajectory/intervention workflows.

The current example contains static HTML/SVG and selected results; the full
curve inputs are in local immutable runs. The checked-in figure is not
interactive. Add pair, width, strength, and checkpoint selectors, plus
observational-only export before interventions exist. Keep raw tensors optional.

Acceptance: a fresh checkout can regenerate figures offline without a model;
inference replay reports each stage's tolerance and lineage result separately.

### 2. Experiment comparison workspace

Question: which candidate method works better under identical conditions?

Align model, observable, pairs, split, operation, width, strength, and ranking
objective before comparing runs. Display direct, downstream-gradient, and
overlap methods together, with individual random controls and pair-level
effects. Estimate uncertainty in the selected-minus-control contrast. Existing
bootstrap intervals describe selected effects, not uncertainty in that contrast.

Acceptance: incompatible runs are visibly identified; missing values remain
missing; filters select the same population in every panel; null and reversed
effects remain visible.

### 3. Observable and perturbation design

Question: does the experiment measure and isolate the intended behavior?

Build on the existing perturbation compiler with rendered chat/token previews,
duplicate checks across splits, template/entity grouping, and observable
sensitivity previews. A useful example crosses input language with requested
output language in four conditions. The present EN/ZH example changes both,
so it cannot isolate their respective contributions.

Extend the fixed first-token observable to teacher-forced answer-sequence
likelihoods, with explicit token-length and normalization semantics. Later
consider generation-step trajectories with properly aligned conditions.

Acceptance: the driver records the factor design, frozen groups, exact tokens,
scoring semantics, and qualification evidence before candidate selection.

### 4. Replication and specificity suite

Question: does the candidate circuit transfer, and what else changes?

Freeze candidate identities and rules, test patching and ablation on new prompt
families, and measure collateral observables and generated behavior where
supported. Make existing additivity, generation, and collateral capabilities
discoverable through workflow templates and visual summaries. Do not treat
patching across generation as already supported: source alignment needs new
semantics. Show transfer matrices and collateral-effect plots.

Acceptance: selection uses discovery only; held-out inspection that informs
new selection is recorded as exploratory; claims identify scope and failures.

### 5. Resumable CLI research cases

Question: can an agent continue or branch a case without reconstructing lineage?

Expose existing case infrastructure through commands such as `case status`,
`case next`, `case run-stage`, and `case fork`. Reuse server-side resolution,
verification, jobs, cancellation, and model reuse. Export concrete continuation
commands with workspace context rather than placeholders.

Acceptance: interruption/reconnect preserves completed runs; changed hypotheses
or selection rules create explicit branches; idempotent continuation does not
repeat a completed stage or silently resubmit a failed one.

### 6. Trajectory-guided competing-explanation tests

Question: is a decodable signal used, overwritten, or merely readable?

Extend largest-jump suggestions with declared persistence, sign reversal, and
erasure criteria. Suggest bounded small-dose tests of gradient predictions,
ablation tests of necessity, and patch/restore experiments around suspected
erasure layers. Link each recommendation to evidence, alternatives, and budgets;
require researcher confirmation for new experiments.

Later, connect attention senders and FFN receivers through new controlled
intervention semantics. Existing head paths and neuron scores cannot by
themselves establish mixed-component causal paths.

Acceptance: suggestions remain observational; controlled outcomes can reject
the proposed explanation; no held-out result silently changes selection.

## Suggested delivery order

1. Correct labels, split handling, UI selection, and failure paths.
2. Deliver portable evidence and aligned experiment comparisons.
3. Make replication, specificity, and agent continuation routine.
4. Extend observable and mixed-component intervention semantics.

Keep model identity, seeds, runtime settings, controls, split membership,
selection objective, claims, and artifact provenance explicit throughout.
