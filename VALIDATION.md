# Validation record

Validated through 2026-08-30 (America/Los_Angeles).

## Environment

- Host: Apple Silicon macOS, `arm64`
- Python: 3.12.12 in `probing/.venv`
- PyTorch: 2.13.0
- Device: MPS, built and available
- Model: `Qwen/Qwen3-0.6B`
- Resolved revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Model dtype: float16
- Captured/scoring dtype: float32
- Layers: 28
- Parameters observed: 596,049,920
- Chat template: enabled, thinking disabled

The model was loaded only from the ignored project-local `.hf-cache/`. No model
download was allowed by either validated experiment spec.

## Automated verification

```bash
uv run --locked pytest
uv run --locked python -m compileall -q src tests
cd web && npm run typecheck && npm run build
```

Current result: `131 passed`; compile, five frontend unit tests, and TypeScript
checks passed. The production React build completed, with a non-fatal Vite size
warning for the 1.43 MB minified Plotly-containing JavaScript bundle (466 KB
gzip). `npm install` reports two transitive audit findings (one moderate and one
critical); no forced dependency upgrade was applied during scientific feature
validation.

The host's older `uv 0.4.20` emits two pathless `Failed to read metadata for
file` warnings during its pre-run editable-project audit. Reinstalling the
editable package does not change them; `uv pip check` reports all 50 installed
packages compatible, commands/tests succeed, and `uv run --no-sync` suppresses
the audit warnings. They are environment noise, not experiment failures.

Coverage includes:

- exact one-token observable resolution and multi-token rejection;
- target-minus-control logit-gap direction;
- signed `I = c * delta_a` ranking and layer aggregation;
- full-tensor multi-pair mean, RMS, and sign-consistency aggregation;
- dual multi-pair FFN ranking views: paper-faithful `|mean(I)|` shared direction
  and `RMS(I)` effect magnitude, plus coherence and objective provenance;
- FFN/Skip computation and empirical regime labels;
- two logical forward passes per pair and hook cleanup;
- strict finite spec constraints, duplicate JSON/YAML key rejection, and budget
  preflight;
- canonical JSONL stdout and versioned machine errors;
- atomic run artifacts, per-pair tensors, and digest verification;
- authenticated FastAPI jobs and terminal NDJSON event streams;
- occupied-port rejection before managed-daemon spawn;
- Textual result rendering and main-thread model preload;
- Transformers 5 chat-template return formats and Qwen post-SwiGLU capture.
- durable failure/cancellation states, including queued cancellation and wall
  deadline classification;
- standalone/daemon error-code and CLI-exit parity;
- deterministic neuron tie-breaking and rejection of non-finite or
  pair-incompatible aggregate tensors;
- artifact traversal defense, corrupt-manifest handling, untracked-file
  detection, and remote verification;
- conservative model download budgets when repository file sizes are unknown;
- immutable-revision pinning between remote size inspection and model download;
- runtime/library metadata in new manifests and last-prompt-position capability
  enforcement;
- exact six-forward execution of the paper-derived capital, arithmetic, and
  science acceptance fixture.
- compact agent completion, versioned overview/query/verification contracts,
  and bounded layer/neuron/file retrieval;
- preflight separation of cached readiness from permitted model acquisition;
- daemon request idempotency, conflicting-key and blank-key rejection, saved
  job specs, warm-engine lifecycle events, and interrupted-job recovery.
- strict parsing and execution budgets for five additional paper-pattern case
  suites, including regression coverage for YAML boolean-token coercion.
- replay-driver path safety, immutable baseline recording, exact fixture replay,
  numeric-drift detection, compact agent results, and exit `9` on mismatch;
- aggregate, per-pair, per-layer, and top-neuron tolerance comparison plus
  ranking overlap, sign agreement, and rank-displacement checks.
- generated-observable qualification and enforced qualification lineage;
- structured role/tool prompts and discovery/validation/held-out separation;
- non-mutating Qwen ablation, amplification, patch/restoration, generation, and
  residual-direction hooks;
- dose/model-call accounting, sign-coherent selection, at-least-three matched
  controls, additivity, causal-width, collateral-observable, and held-out summaries;
- experiment comparison, split-half/bootstrap stability, perturbation-family
  sensitivity, deterministic perturbation compilation, workflow persistence,
  research reports, and their typed CLI paths.
- exact Qwen grouped-query output-head/KV-head mapping and per-head output
  coupling against an actual tiny `Qwen3ForCausalLM` fixture;
- eager attention-value reconstruction, hook cleanup, local head edits without
  weight mutation, deterministic attention artifacts, and typed agent receipts;
- qualification-gated output-head ranking/intervention, conservative weak-pair
  claims, exact call budgets, symbolic attention workflow lineage, explicit
  full-token alignment, endpoint-population enforcement, and matched path controls.
- native block-input/post-attention/post-FFN capture, exact final-checkpoint
  decoder equality, hook cleanup, paired distribution diagnostics, and immutable
  trajectory artifacts;
- native-local and true downstream endpoint gradients without parameter-gradient
  mutation, explicit backward-pass budgets, symbolic trajectory lineage, and a
  finite-difference neuron-injection check on a real tiny Qwen model;
- checked-in trajectory/coupling workflow parsing, including quoted
  boolean-like observable tokens, two exact normalized MPS replays, and verified
  three-stage lineage.

## Live native trajectory and layer-aware FFN coupling

The checked-in capital workflow ran twice with the pinned Qwen3-0.6B revision,
MPS/float16, seed `260427401`, and downloads disabled:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/capital-trajectory-coupling.yaml \
  --events jsonl
```

Primary runs `20260831T031736-698d99413e3d`,
`20260831T031737-cd062577066c`, and
`20260831T031738-fc30a98b5cb8` all passed artifact verification. The replay
created a new immutable lineage but reproduced all normalized summary values
exactly. The largest trajectory changes were L27 post-FFN `+0.265625`, L18
post-FFN `+0.2265625`, and L21 post-FFN `+0.1953125`. Downstream-gradient
candidates began with L21:n1341, L18:n1784, and L27:n840; downstream layer RMS
mass peaked at L17, L18, and L16. These are reproducible observational
hypotheses, not causal localization. Full bounded results and commands are in
[`examples/workflows/capital-trajectory-coupling-reports/`](examples/workflows/capital-trajectory-coupling-reports/).

A wheel and source distribution were built in
`/private/tmp/probe-attention-final-build` using an isolated temporary uv cache.
The wheel contains the CLI, attention/path modules, service, daemon, adapters,
and compiled WebUI assets.

## Live MPS replay bundles

Two pinned Qwen3-0.6B examples were recorded under `examples/replays/` and then
replayed after a daemon restart and fresh model load. Both used model revision
`c1899de289a04d12100db370d81485cdf75e47ca`, MPS/float16 model execution,
float32 capture/scoring, Torch seed 0, evaluation and inference mode, no KV cache,
and no generation or sampling.

| Bundle | Baseline run | Fresh-process replay | Required checks | Numerical checks | Top-100 overlap/sign | Stable hashes |
|---|---|---|---:|---:|---:|---:|
| language routing | `20260809T221605-5f7c5f7b4406` | `20260809T221715-5f7c5f7b4406` | 18/18 | 302/302 | 1.0 / 1.0 | 4/4 |
| CoT protocol | `20260809T221625-6f42038ded72` | `20260809T221721-6f42038ded72` | 18/18 | 302/302 | 1.0 / 1.0 | 4/4 |

For both bundles, maximum numeric difference and mean rank displacement were
zero. Offline `probe replay check` also passed for the fresh-process runs without
starting or loading a model. The full JSON and Markdown reports are checked in
beside their drivers.

Both bundles were run again after the causal-backend implementation. Runs
`20260810T005214-5f7c5f7b4406` and `20260810T005226-6f42038ded72` again passed
18/18 required checks, all 302 numeric/ranking checks, top-100 overlap 1.0, sign
agreement 1.0, and mean rank displacement 0.0. Layer, neuron, and tensor bytes
were exact. Only `spec.json` changed because new default structured-prompt/split
fields are now serialized; the compatibility science/request hashes remained
stable and the driver declares artifact hashes report-only.

The language case is a clean behavioral-routing smoke result: the original
predictions begin in English and all perturbed predictions begin in Chinese. The
CoT case is a reproducible negative diagnostic: both simple and complex prompts
start with `<think>`, and additive estimates are weak, so its stable neuron list
must not be interpreted as complexity-specific or causal.

## Live MPS standalone run

```bash
uv run --locked probe run \
  --spec examples/agreement-replication.yaml \
  --events jsonl
```

The command emitted only versioned lifecycle records on stdout and committed
`20260809T195011-87a9ed263625`. Its manifest records the arm64 host, Python and
library versions, MPS/float16 resolution, and immutable model revision. Local
artifact verification reported no failures.

| Pair | measured delta F | predicted sum I |
|---|---:|---:|
| capital | +0.625000 | +0.288944 |
| arithmetic | +0.312500 | +0.614525 |
| science | +0.359375 | +0.944475 |
| mean | +0.432292 | +0.615981 |

Aggregate FFN/Skip was `0.487289`. The leading RMS neuron was layer 25, neuron
1665 (`RMS I = 0.430854`, sign consistency `2/3`). The run contains 476 named
tensors, including full original, perturbed, delta, and importance vectors for
every pair and layer. All three observables move in the expected direction, but
the next token remains `Yes`; additive estimates differ from measured deltas by
more than 50%, so the tool retains the weak-hypothesis warning.

## Live explicit-daemon run and engine reuse

A foreground daemon was started on free loopback port 8766, then addressed only via
an explicit endpoint:

```bash
uv run --locked probe --endpoint http://127.0.0.1:8766 \
  run --spec examples/agreement-capital.json --events jsonl
```

The daemon loaded Qwen3 on MPS and completed the one-pair request through the
same CLI event contract as standalone mode. A second request reused the loaded
engine: `model.ready` followed acceptance in about 2 ms with no weight-loading
progress, and its numerical result was identical. It committed
`20260809T195039-d18b7434c1df`; remote `runs verify` reported no failures.

The daemon result reproduced capital measured delta F `+0.625000`, predicted
sum I `+0.288944`, FFN/Skip `0.502674`, and `Yes` as both next-token predictions.

The daemon served the packaged HTML, token bootstrap, and health endpoint with
HTTP 200. Shutdown validation confirmed that `.probe/server.json` and
`.probe/server.token` were removed and port 8766 was released. A separate
pre-existing process on port 8765 was left untouched.

## Improved agent-driven CLI workflow

An end-to-end capital experiment was run through a foreground daemon in a fresh
workspace at `/private/tmp/probe-agent-improved-e2e`. The model cache remained
project-local and downloads remained disabled. The core agent flow was:

```bash
probe --endpoint http://127.0.0.1:8766 \
  preflight --spec examples/agreement-capital.json
probe --endpoint http://127.0.0.1:8766 \
  run --spec examples/agreement-capital.json \
  --events none --result compact-json --request-id agent-capital-001
probe --endpoint http://127.0.0.1:8766 \
  runs neurons 20260809T202051-d18b7434c1df --top 5 --sign positive
probe --endpoint http://127.0.0.1:8766 \
  runs verify 20260809T202051-d18b7434c1df
```

Preflight returned `probe.preflight/v1`, resolved MPS/float16, confirmed
`model_ready: true`, `acquisition_required: false`, and exactly two required
forwards. The compact completion returned only one `probe.run-overview/v1`
object. It reproduced measured delta F `+0.625`, predicted sum I `+0.288944`,
and FFN/Skip `0.502674`; both next-token predictions remained `Yes`. Highest
aggregate-mass layers were 27, 25, 26, 22, and 21. The strongest individual
score was layer 18 neuron 1784 (`I = +0.323264`, coupling `0.194262`, activation
delta `+1.6640625`, effect `toward_target`).

Repeating `agent-capital-001` returned the same durable job and run in about one
second, without a new forward. A fresh `agent-capital-002` job emitted
`model.reused` before `model.ready` and reproduced the numerical result. Reusing
`agent-capital-001` with the three-pair spec returned HTTP 409 as the structured
`request_conflict` error and CLI exit 2. The positive-neuron query returned a
`probe.query/v1` envelope with `source_count: 500`, `matched_count: 269`, and
`returned_count: 5`. All seven artifact digests verified through the daemon as
`probe.verification/v1`.

This workflow exposed and fixed a standalone `runs files` serialization defect
for containers of typed artifact records in the earlier pass. The current
acceptance test now exercises the whole compact workflow, including the saved
job spec and each evidence-query contract.

## Qwen3-0.6B paper-case transfer audit

Five additional three-pair suites were run through one reused MPS model instance:
safety BPE swaps, EN-to-ZH routing, factual entity swaps, code versus explanation,
and complex versus simple CoT prompts. This added 30 logical forward passes. All
five runs committed successfully and all artifact digests verified.

Language routing matched the paper qualitatively: 3/3 observable sign flips,
English-to-Chinese first-token changes, FFN/Skip `0.121`, and late-layer
localization. The other cases exposed important transfer and backend limits:

- safety had only 1/3 sign flips and non-refusal baseline first tokens despite a
  high aggregate FFN/Skip value;
- factual FFN/Skip ranged from `0.259` to `0.737` under phrasing changes;
- code was labeled as a high-FFN opposition candidate (`0.946`) although the
  paper reports code neurons as non-causal readouts;
- CoT produced `<think>` in both conditions for all pairs, yet still returned a
  ranking because decision crossing is not currently enforced.

The complete methods, run IDs, numerical table, candidate neurons, and backend
priorities are in
[PAPER_CASE_VALIDATION_QWEN3_0_6B.md](PAPER_CASE_VALIDATION_QWEN3_0_6B.md).

## UI verification boundary

The React source typechecked and built, and the packaged page/API assets were
served successfully. Browser automation was unavailable in this session, so
visual layout and interactive chart behavior were not directly observed. That
is the remaining product QA step; it is not counted as passed here.

The compatibility command and command tree were also checked:

```bash
probe-workbench --help
probe --help
```

The earlier real Textual TTY launch remains the validation evidence for the TUI.

## Scientific boundaries

- A replicated ranking is stronger than one pair but remains observational.
- Ablation, amplification, prompt-position restoration/patching, matched random
  controls, dose response, additivity, and direction injection are implemented
  for the dense-Qwen3 adapter. Their claims remain local to the saved experiment.
- Only dense Qwen3 is capability-gated and live-tested.
- The basic tree-level coupling is used; Gemma's dressed propagator is absent.
- Capture is the last prompt position for a binary first-token observable.
- The FFN/Skip thresholds are diagnostic heuristics, not universal boundaries.
- MPS float16 can perturb the ordering of very small near-zero scores; formula
  and hook tests use deterministic CPU float32 fixtures.
- Supported causal status requires replicated eligible pairs, at least three
  matched-control draws, and a directional bootstrap interval excluding zero;
  this is not a population-level significance test or multiple-testing correction.
- Direction injection demonstrates local controllability, not FFN localization.
- Attention-head scores are direct-logit observational attributions; upstream
  effects and cross-layer dependence prevent additive causal interpretation.
- Token edges are attention-weight times value contributions into a head and
  must reconstruct that head, but remain observational route hypotheses.
- A two-stage path effect is causal only for the declared exact alignment,
  sender/receiver endpoints, prompts, first-token observable, and controls. It
  neither enumerates all mediators nor establishes a unique path.

## Qualified Qwen3-0.6B causal workflow

The seeded workflow in `examples/workflows/language-causal-loop.yaml` was run on
the host Apple MPS backend with the cached, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`. It completed 132/132 planned logical
model calls in 10.7 seconds and committed four integrity-verifiable stages:

- rank `20260810T000018-b5cfefbfd03d`;
- qualification `20260810T000019-35be5cf9a10d`;
- FFN ablation `20260810T000023-c1ec41ec2e7f`;
- direction sweep `20260810T000025-d03e122f0816`.

The sparse first-token token sets crossed their logit-gap boundary for all three
pairs but did not contain both argmax tokens, so the rank-only gate correctly
kept them weak. A predeclared dominant-Unicode-script evaluator then matched the
English/Chinese generated contrast for 3/3 pairs and qualified the causal stages.

For sign-consistent positive neurons, complete ablation reduced the Chinese-minus-
English gap by means of `-0.7667`, `-1.3414`, and `-4.4426` at N=1, 5, and 20.
The corresponding selected-minus-random absolute effects were `0.7631`, `1.3353`,
and `4.4148`; each dose used nine matched-random observations and its across-pair
bootstrap interval excluded zero. Both full and half ablation were monotonic over
the tested N values, with a descriptive 90%-effect width of 20 neurons.

At residual layer 24, norm-aware beta `-0.05/+0.05` moved the gap by
`-3.8919/+3.8458`; controlled absolute effects were `3.8217/3.7585`. Layer 18
effects were about `-0.8388/+0.8387`, also well above three orthogonal-control
draws per pair. This confirms local direction controllability while the parent
FFN/Skip value (`0.1214`) remains in the paper's low-concentration,
readout-compatible range.

The checked-in reports record the important neurons, doses, limitations, and
lineage. This is a representative Qwen transfer smoke study, not a reproduction
of the paper's original model/dataset: the same three discovery pairs selected
and tested the neurons, no held-out split or collateral observable was included,
and interventions were evaluated on the logit gap rather than generated behavior.

## Qwen3-0.6B attention-path MVP

The paper-derived EN/ZH prompts were rerun on the same pinned model using the
attention workflow checked in at
`examples/workflows/language-attention-path.yaml`. All 3/3 pairs passed the
generated Unicode-script gate. The attention adapter resolved 28 layers, 16
output heads, 8 grouped-query KV heads, and head dimension 128.

The leading RMS heads were L25/H0 (`3.7556`), L18/H12 (`2.3003`), L24/H0
(`1.4680`), and L24/H2 (`1.2890`), each sign-consistent across 3/3 pairs.
Patching perturbed head outputs into the original condition gave the following
three-pair controlled dose response:

| Heads | Selected mean gap effect | Random absolute mean | Controlled absolute effect |
|---:|---:|---:|---:|
| 1 | +1.3053 | 0.0548 | 1.2505 |
| 4 | +15.5627 | 0.1886 | 15.3741 |
| 16 | +22.0662 | 0.4276 | 21.6386 |

At 16 heads, all three gaps crossed into the Chinese-target side and each argmax
left its original English lead token. Capital and science became Han tokens;
arithmetic changed from `Two` to `2`, preserving the sparse-observable caveat.
The claim is supported locally because all pairs were generated-behavior
qualified and every dose had five same-layer controls.
By contrast, rerunning the one-pair agreement intervention committed
`20260810T175137-3be988dd9af9` with the same positive controlled effects but an
`exploratory` claim because that pair did not cross the informative gate.

Eager token decomposition reconstructed every selected pre-`o_proj` head output
on real MPS. L18/H12 consistently routed the literal instruction tokens:
`中文` contributed about `+1.24` to `+1.34` on the perturbed prompts, while
` English` contributed about `-0.61` to `-0.91` on the originals. Late L25/H0
was dominated by final formatting/newline routes, distinguishing a semantically
clear language-instruction sender from a later readout/formatting head.

For the exactly aligned 23-token capital pair, patching L18/H12 at all positions
moved the gap by `+14.2070`. Replaying only the resulting L25/H0 state isolated
a `+0.4574` sender-to-receiver path effect. Five matched L18-to-L25 random paths,
excluding the full selected intervention population and unique within the arm,
had absolute effects averaging `0.00570`; the path claim was therefore supported
for this local experiment. The path accounts for only about 1.91% of the full
source-condition movement, so most sender influence remains outside this tested
receiver.

Fresh model loads reproduced the attention ranking, head intervention, and path.
Five stable rank artifacts matched byte-for-byte; intervention summary and
observation JSONL matched exactly; path summary and path JSONL matched exactly.
All original and replay manifests passed digest verification. Full run IDs,
numerical results, replay identities, interpretation, and limitations are in the
checked-in [analysis](examples/workflows/language-attention-path-reports/analysis.md)
and [JSON record](examples/workflows/language-attention-path-reports/results.json).

## Researcher UI trajectory-to-FFN acceptance

The Research Case browser workflow was exercised against the cached pinned
Qwen3-0.6B model on MPS after the researcher/agent UI milestone. Disposable case
`20260902T064311-e47a56cce3` completed and verified:

- rank `20260902T064349-b4153ae9b535`;
- native trajectory `20260902T064413-b08db1888804`; and
- researcher-scoped FFN coupling `20260902T064512-750865dbc8d0`.

The trajectory UI filtered target rank at post-FFN checkpoints with the axis
marked lower-is-better. Its strongest capital-pair suggestion was L27/post-FFN
(`+11.555` paired-gap change); the researcher confirmation control proposed and
then recorded L26-L27 before coupling ran. The coupling artifact retained only
`capital` and `arithmetic` as candidate pairs, excluding the held-out science
pair from ranking.

All four new bounded CLI views ran against the live artifacts and both manifests
verified. The disagreement view exposed examples that the direct readout alone
would obscure: L26:n1704 had downstream/direct RMS ratio `215.9` but a small
absolute downstream RMS (`0.0108`), while L26:n96 was direct-amplified by `22.9`
with direct RMS `0.5302`. This demonstrates why the UI shows absolute scores,
ratio, and sign agreement together rather than turning disagreement into a new
causal ranking.

Browser checks confirmed the metric/checkpoint controls, immutable-band handoff,
coupling detail selection, exact CLI commands, and zero console warnings or
errors. Plotly now loads as a deferred 1.08 MB chart chunk; the initial
application chunk is approximately 351 KB instead of the previous 1.44 MB.
Evidence remained observational because qualification and matched intervention
were deliberately not run in this UI acceptance case.
