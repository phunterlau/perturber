# Perturbation Probing Workbench - Product and Technical Design

Status: architecture record after reviewing the existing Token Explorer and
`ref/2604.27401v1.pdf`. The Qwen3/MPS FFN and attention-path core,
schema-driven CLI, local daemon, WebUI, and immutable artifact store are
implemented; see `README.md` and `VALIDATION.md`.

## Product thesis

Turn a controlled prompt pair into a reproducible, intervention-ready circuit
hypothesis without backpropagation or a separately trained probe.

The useful product is not merely a side-by-side activation heatmap. It should
preserve the complete scientific chain:

1. define a binary behavioral observable;
2. compare original and perturbed prompts;
3. measure the per-neuron activation response;
4. combine it with structural coupling to obtain signed importance;
5. diagnose whether the signal is FFN-concentrated or routed through the skip
   stream;
6. test the resulting hypothesis with ablation, amplification, restoration, or
   direction injection;
7. decompose the residual attention branch into output heads and token-value
   routes, then test selected sender-to-receiver routes with two-stage patching;
8. export enough metadata to reproduce or extend the run.

This makes the tool a hypothesis workbench rather than a claim generator. A
single pair is excellent for exploration, but it is not population-level causal
validation. The UI should say `exploratory: 1 pair` prominently and make it
easy to graduate an interesting pair into a multi-pair experiment.

## Target users and value

### Primary: the paper authors and follow-up research

- Iterate on perturbations and observable token sets in seconds rather than
  editing one-off scripts.
- See whether a surprising neuron is important because it responded strongly,
  because it couples strongly to the behavioral direction, or both.
- Move directly from ranking to controlled intervention and dose response.
- Compare architectures without silently applying a formula that is invalid for
  a model family.

### Secondary: mechanistic-interpretability researchers

- A low-cost entry point: two forward passes for hypothesis generation, no
  gradient memory and no SAE training.
- An inspectable method: every rank decomposes into `c_n`, `delta_a_n`, and
  `I_n`, with a stated observable and token position.
- A reproducible artifact: prompt text and token IDs, model revision, adapter,
  dtype, seed, metric definition, scores, and intervention result are saved
  together.
- A regime warning: the FFN/Skip diagnostic helps distinguish an ablatable FFN
  circuit from a readout/routing signal before the researcher over-interprets a
  neuron list.

## Scientific contract

The v1 computation should implement the paper directly.

Given refusal/target token set `R`, affirmation/control token set `A`, original
prompt `X`, and perturbed prompt `X_tilde`:

```text
F(X)       = mean(logits[R]) - mean(logits[A])
d_F        = mean(W_vocab[R]) - mean(W_vocab[A])
c_(l,n)    = d_F dot W_down[l][:, n]
delta_a    = a_(l,n)(X_tilde) - a_(l,n)(X)
I_(l,n)    = c_(l,n) * delta_a_(l,n)
FFN/Skip   = abs(d_F dot FFN^(L-1)) / abs(d_F dot h^(L-1))
```

The first implementation should capture the post-nonlinearity FFN activation at
the last prompt position, corresponding to the first generated-token decision.
Token-wise and multi-generation-position probing can be added later, but the
position must always be explicit in the run specification.

The interface should expose, not hide, the method's boundaries:

- binary, usually first-token observables;
- perturbation quality is part of experimental design;
- one pair is a noisy hypothesis;
- high activation response without structural coupling is a readout, not
  automatically a causal writer;
- the tree-level coupling is architecture-dependent;
- pre+post-normalization architectures such as Gemma require the dressed
  propagator correction rather than the basic `d_F dot W_down` approximation;
- the empirical FFN/Skip thresholds are diagnostics, not universal constants;
- direction injection requires additional linear-representability and model
  capability conditions.

## Core interaction

### 1. Define

The initial screen contains two multiline editors:

```text
Original / control                         Perturbed / treatment
-----------------------------------        -----------------------------------
Write ... methamphetamine ...              Write ... metahmphetamine ...
```

Below them are the observable controls:

- target/refusal token strings and their resolved token IDs;
- control/affirmation token strings and token IDs;
- decision position, defaulting to the first generated token;
- capture position, defaulting to the last prompt token;
- model, revision, dtype, and device;
- optional preset: safety, agreement, EN/ZH, or custom.

The UI must show tokenizer resolution before analysis. A string that becomes
multiple tokens, differs with leading whitespace, or maps unexpectedly should
be visible rather than silently accepted.

### 2. Analyze

Run the original and perturbed forwards as one logical paired job. The summary
should answer four questions immediately:

```text
Did the observable move?   F(X), F(X_tilde), delta F, sign flip
Where did it move?         ranked layers by signed and absolute contribution
Which neurons explain it?  top |I| with c, delta a, sign, and layer
What kind of circuit?      FFN/Skip plus a qualified intervention suggestion
```

Do not label layers important using the existing entropy heuristic. For this
tool, importance should be tied to the probing quantity. Recommended layer
summaries are:

- `sum(I)` - predicted signed contribution;
- `sum(abs(I))` - total responsive mass;
- positive and negative mass - push/pull structure;
- top-k share - concentration within the layer;
- maximum `abs(I)` and the corresponding neuron;
- FFN activation-delta norm as supporting context.

### 3. Inspect

Selecting a layer filters the neuron table. Selecting a neuron opens a detail
panel with:

- layer and neuron index;
- original activation, perturbed activation, and delta;
- structural coupling `c_n`;
- signed importance `I_n` and global/layer rank;
- gatekeeper/amplifier-style direction label derived from sign, with the exact
  sign convention shown;
- its contribution relative to measured `delta F`;
- warnings if the active model adapter uses an approximation.

The prompt tokenization may be displayed side by side with a visual diff. It is
an aid for experimental design, not an activation alignment assumption. The v1
score uses the configured capture position independently in each sequence.

### 4. Intervene

After analysis, immutable child runs support:

- ablate top N;
- amplify top N by alpha;
- restore or patch activations between the pair;
- inject `d_F` at a selected layer with beta when the diagnostic suggests a
  routing/readout experiment;
- same-layer-count random controls;
- a small dose sweep over N or alpha;
- compare logit gap and, for scale/residual operations, generated continuation
  before/after under a declared behavior evaluator.

Intervention is deliberately a separate action. Ranking results remain immutable
and can be compared with multiple interventions.

### 5. Save and extend

Every analysis produces a run directory containing human-readable JSON plus
tensor/tabular data. Minimum metadata:

- model ID, resolved revision/commit, tokenizer ID, and architecture adapter;
- library versions, device, dtype/quantization, and seed;
- raw prompts and token IDs;
- observable token strings and resolved IDs;
- capture and decision positions;
- formula/algorithm version;
- `F(X)`, `F(X_tilde)`, `delta F`, FFN/Skip, layer summaries, and ranked neurons;
- timings, warnings, and intervention specifications/results.

CSV neuron export and JSON/Markdown research reports make results easy to move
into notebooks and paper experiments. A generated Python reproduction snippet
remains future ergonomics work.

## Legacy TUI information architecture

The Textual surface remains useful for keyboard-first terminal work and as a
compatibility interface. It is no longer the primary place for new visual
analysis features; the WebUI implements the evolving dashboard while the TUI
continues to exercise the same scientific concepts.

Recommended wide-terminal layout:

```text
 Model / revision / dtype / device     Pair: exploratory (N=1)     status
+--------------------------------------+--------------------------------------+
| ORIGINAL prompt                      | PERTURBED prompt                    |
| tokenized preview                    | tokenized preview                   |
+--------------------------------------+--------------------------------------+
| Overview | Layers | Neurons | Interventions | Run log                       |
+----------------------+------------------------------+------------------------+
| layer map / filters  | primary table or heatmap     | selected-item inspector|
| observable settings  |                              | formula decomposition  |
+----------------------+------------------------------+------------------------+
 shortcuts / job progress / warnings / export path
```

Responsive behavior:

- at 160+ columns, use the full three-pane analysis layout;
- around 100-159 columns, collapse the inspector into a toggleable drawer;
- below 100 columns, stack the prompt editors and use tabs for analysis;
- never compress neuron IDs, signs, or numeric columns into unreadability merely
  to preserve a decorative chart.

Recommended views:

1. **Overview** - diagnostic cards, observed vs predicted delta, FFN/Skip, and
   a compact layer contribution sparkline.
2. **Layers** - sortable layer table plus a diverging push/pull heatmap.
3. **Neurons** - virtualized/sortable table with layer, neuron, `c`, `delta a`,
   `I`, sign, and rank; filters for layer, sign, and minimum magnitude.
4. **Interventions** - immutable selection summary, controls, progress, logit
   gap result, continuation diff, and dose-response sparkline.
5. **Run log** - computation stages, timing, memory, warnings, and artifact path.

Use a perceptually simple diverging palette: one hue for positive contribution,
one for negative, and intensity for magnitude. Always print the number and sign;
color cannot be the only encoding.

## Framework and interface decision

The primary interactive surface is a local React WebUI served by FastAPI. It
fits the high-density comparison work better than a terminal: linked plots,
responsive tables, prompt editors, hover inspection, and saved-run browsing can
evolve without fighting terminal cell geometry. The browser remains local and
the Python process owns all model execution.

The headless CLI is the product's stable automation boundary, not a wrapper
around UI actions. Both standalone CLI execution and the daemon call the same
`ResearchService`. The WebUI calls the daemon's versioned HTTP/NDJSON API. The
Textual TUI is retained as a compatibility and low-overhead surface.

```text
JSON/YAML spec
     |
     +--> probe standalone ---------+
     |                              |
     +--> explicit daemon client ---+--> ResearchService --> model adapter
     |                              |          |
     +--> React WebUI --> FastAPI ---+          +--> immutable run repository
```

Key choices:

- FastAPI + Pydantic keep HTTP and file validation on the same models.
- React + Plotly provide the layer/neuron analysis views.
- NDJSON streams ordered, versioned events without a WebSocket-only client.
- One daemon worker serializes MPS jobs and reuses one loaded model safely.
- Explicit `--endpoint` avoids nondeterministic auto-discovery in agent scripts.
- A files-only store keeps runs portable, inspectable, diffable, and easy to
  consume from shell tools or notebooks; SQLite can be added later as an index,
  never as the sole source of truth.
- Exact dependency pins and a lockfile make the local application repeatable.

## What to reuse from Token Explorer

Reuse as implementation reference:

- Hugging Face causal-model and tokenizer loading;
- CUDA/MPS/CPU device selection;
- the token candidate table and append/backtrack mental model;
- token-level cursor navigation and keyboard-first interaction;
- Rich/Textual color rendering techniques;
- hidden-state and residual-stream inspection concepts;
- layer-probability batching as an example of avoiding per-layer Python loops;
- the idea of caching by tokenized prompt rather than raw text.

Do not directly build on these current boundaries:

- `main.py` owns model loading, application state, job queues, view rendering,
  prompt persistence, and actions;
- `Explorer` owns the model, mutable prompt, analysis methods, and several cache
  formats at once;
- `UIDataAdapter` performs expensive model analysis while rendering views;
- manual queues/threads are duplicated and do not provide cancellation or stale
  result protection;
- several display changes synchronously rerun the model and can freeze the UI;
- residual attention/MLP contribution values are currently random placeholders;
- disk cache identity does not fully encode model revision, dtype, observable,
  adapter, or algorithm version;
- `print` output shares the terminal with the TUI;
- no automated tests or lockfile are present.

The existing program is a useful interaction prototype and collection of
analysis experiments. It should remain runnable while `probing/` develops as a
self-contained package.

## Implemented package boundaries

All new files stay under `probing/`:

```text
probing/
  DESIGN.md
  pyproject.toml                 # isolated, pinned app environment
  README.md
  src/probing/
    cli.py                       # Typer command tree and stdout contract
    contracts.py                 # strict versioned Pydantic schemas
    specs.py                     # JSON/YAML parsing and canonical hashes
    service.py                   # frontend-neutral job orchestration
    server.py                    # loopback API, NDJSON stream, static WebUI
    client.py                    # explicit daemon client
    server_process.py            # guarded managed-daemon lifecycle
    domain.py                    # immutable numerical records
    engine.py                    # paired forward and scoring core
    aggregation.py               # full-tensor multi-pair RMS ranking
    attention.py                 # output-head ranking and causal head edits
    attention_trace.py           # eager token routes and two-stage path patches
    observables.py               # token-set resolution and logit-gap metrics
    adapters/
      base.py                    # ModelAdapter protocol and capability report
      qwen.py
    artifacts.py                 # atomic files-only run/job repository
    web_dist/                    # packaged React production build
  web/                           # React/TypeScript source
  examples/                      # canonical JSON/YAML experiments
  tests/
    helpers.py                   # tiny deterministic fake model/results
    test_observables.py
    test_scoring.py
    test_adapters.py
    test_artifacts.py
    test_app.py                  # Textual Pilot interaction tests
```

Key contracts:

- `RankSpec`: model, pairs, observable, capture, ranking, and hard budgets;
- `PreflightReport`: one model-readiness, capability, budget, and hash decision;
- `ObservableSpec`: named positive/negative token sets and reduction;
- `CapabilityReport`: supported activation location, architecture adapter,
  intervention support, and diagnostic limitations;
- `ProbeResult`: immutable measured outputs and warnings;
- `LayerSummary` and `NeuronScore`: view-independent analysis records;
- `RunManifest`: fully versioned serialization envelope.
- `RunOverview` and `QueryEnvelope`: compact agent completion and bounded,
  self-describing evidence queries.
- `QualificationSpec`: generated-behavior validity gate tied to a rank parent;
- `InterventionSpec`: signed neuron selection, operation, dose sweep, matched
  controls, optional generation, collateral observables, and qualification lineage;
- `DirectionInjectionSpec`: norm-aware layer/beta sweep and orthogonal controls;
- `ResearchWorkflowSpec`: seeded symbolic rank-to-causal orchestration without
  hand-edited run IDs.
- `AttentionHeadRankSpec`: direct-logit output-head attribution against an
  immutable rank and optional qualification parent;
- `AttentionHeadInterventionSpec`: head selection, edit mode, dose sweep, and
  same-layer random controls;
- `AttentionTraceSpec`: reconstruction-checked token edges or exact-alignment
  sender-to-receiver path patches whose endpoints were previously tested.

The primary WebUI and CLI consume service records and commands. The compatibility
TUI still calls the shared engine directly; if it receives major new features,
it should first become a thin service consumer. No widget triggers a forward pass
from `render()`.

## Engine and performance design

- In standalone/TUI mode, initialize PyTorch/MPS on the macOS main thread. In
  daemon mode, a single long-lived executor owns model work and the service
  caches one engine by model/revision/device/dtype.
- Queue daemon jobs FIFO. Cancellation is cooperative between model operations;
  PyTorch forwards are not preempted mid-kernel.
- Execute original and perturbed forwards sequentially to bound peak memory.
- Capture only requested positions. Do not retain every layer x token x FFN
  activation when the method needs the final prompt position.
- Move reduced activation vectors to CPU layer by layer when GPU memory is the
  constraint.
- Compute structural coupling from the resolved observable and the loaded model;
  a separate coupling cache is a future optimization.
- Science hashes exclude operational metadata; run fingerprints additionally
  include the resolved model and algorithm version.
- Use explicit hook handles in context managers and remove all hooks in `finally`.
- Treat unsupported architectures as capability errors with an explanation, not
  as an invitation to guess module paths.

## Roadmap after the implemented causal-loop foundation

### Completed - contracts, ranking, and local interfaces

- Strict rank, event, error, job, result, and run schemas.
- Standalone and explicit-daemon CLI paths with canonical JSON/JSONL output.
- Compact run overviews, query envelopes, preflight, request idempotency, saved
  job specs, and interrupted-job reconciliation for agent automation.
- Qwen3 post-SwiGLU ranking on CPU/MPS plus multi-pair RMS aggregation.
- Local WebUI, retained Textual TUI, and complete immutable tensor artifacts.
- Deterministic fake-adapter tests and live Qwen3/MPS smoke validation.
- Portable replay drivers with pinned execution records, immutable baselines,
  tolerance/ranking policies, agent-compact outcomes, and full saved reports.

### Completed - qualification and causal loop

- First-token and generated-behavior validity gates, including task-declared
  token, text, regex, exact, and Unicode-script evaluators.
- Top-N ablation/amplification and prompt-position patch/restoration through
  non-mutating Qwen hooks.
- Dose sweeps, at least-three-draw support threshold, same-layer random controls,
  bootstrap intervals, additivity residuals, collateral gaps, and descriptive
  causal-width estimates.
- Residual-direction layer/beta sweeps with norm-aware injection and orthogonal
  random-direction controls.
- Discovery-only ranking, validation/held-out intervention summaries, run
  comparison, stability, perturbation sensitivity, reports, and seeded workflow
  drivers.

### Completed - attention routing and path MVP

- Qwen3 output-head metadata with explicit grouped-query output-to-KV mapping.
- Direct-logit output-head attribution from pre-`o_proj` head outputs and
  per-head output-projection coupling.
- Non-mutating head ablation, amplification, patch, and restoration with exact
  model-call budgets, dose sweeps, and same-layer controls.
- Eager attention-weight times value decomposition with mandatory reconstruction
  of each captured pre-output-projection head vector.
- Two-stage sender patch then receiver-only replay under declared full-token
  alignment, restricted to endpoints in a parent intervention population.
- Qualification inheritance, conservative weak-pair claims, symbolic workflow
  handoffs, typed agent receipts, immutable attention artifacts, and seeded
  Qwen3/MPS replay examples.

### Next - stronger inference and breadth

- Cross-stage replay baselines for causal observations, not only configuration
  replay and existing rank-result tolerance checks.
- More random-control inference, permutation tests, multiple-comparison policy,
  and explicit minimum-effect thresholds.
- Behavioral generation interventions for every operation where source-state
  semantics are well defined, plus evaluator calibration sets.
- Llama adapter and architecture-specific hook reconstruction tests.
- Gemma only with the dressed-propagator correction and explicit compute cost.
- Multi-token/sequence observables and generation-level attention interventions.
- Notebook-ready columnar exports and richer WebUI experiment authoring.

## Most important product risks

1. **False confidence from one pair.** Use explicit exploratory labeling,
   measured/predicted delta comparison, and a path to multi-pair stability.
2. **Wrong hook semantics.** Test the captured FFN activation by reconstructing
   the module output for each supported architecture.
3. **Architecture over-generalization.** Capability-gate each formula and
   intervention; never treat a module-name match as scientific validation.
4. **Observable mistakes.** Make whitespace-sensitive tokenization and multi-token
   strings visible; save resolved IDs.
5. **UI stalls and stale results.** Separate engine workers from rendering and
   version every request/result.
6. **Visualization outrunning evidence.** Default to signed numbers and sortable
   tables; use color and heatmaps as navigation aids.
7. **Irreproducible local state.** Pin dependencies and serialize model revision,
   adapter version, algorithm version, and all experimental inputs.

## Implemented value boundary

The shipped foundation takes a bounded experiment set from observational FFN and
attention ranking through behavioral qualification, controlled head/neuron
interventions, token-route hypotheses, and exact-alignment path tests. Its
strongest value is the shared contract: a person, WebUI, notebook, or coding
agent can run the same experiment, observe the same lifecycle, and query the same
immutable evidence. It does not turn a selected small prompt set into a universal
circuit claim: held-out replication, evaluator validity, architecture coverage,
path completeness, and stronger statistical controls remain explicit research
responsibilities.
