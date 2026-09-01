# Multi-stage research workflows

These drivers make the causal loop replayable without copying generated run IDs
between files. `$rank`, `$qualification`, `$attention_rank`, and
`$attention_intervention` are symbolic references resolved by `probe workflow`;
every committed child spec contains the actual immutable run IDs. The workflow
directory under `.probe/workflows/` retains both the original symbolic driver
and the completed stage map.

The smallest native-trajectory acceptance workflow is:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/capital-trajectory-coupling.yaml \
  --events jsonl
```

It performs two rank forwards, two native-trajectory forwards, then two
forward/backward sensitivity passes. The trajectory is observational, and the
direct/native/downstream coupling comparison remains a candidate ranking until
controlled intervention.

Measured Qwen3-0.6B MPS results and exact replay conclusions are checked in at
[`capital-trajectory-coupling-reports/`](capital-trajectory-coupling-reports/).

The trajectory-guided causal follow-up compares direct-readout,
downstream-gradient, and preregistered top-24-pool-overlap candidates under the
same widths, patch dose, generated-behavior gate, discovery/held-out split, and
three same-layer random controls:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-trajectory-causal-followup.yaml \
  --events jsonl
```

Its intervention runs link the baseline trajectory and emit checkpoint-level
selected-versus-control overlays without adding model calls. Candidate ranking
and trajectory localization remain hypotheses; the backend intervention claims
depend on qualification, matched controls, and split-specific results.

The verified MPS [analysis](language-trajectory-causal-followup-reports/analysis.md)
and [machine-readable results](language-trajectory-causal-followup-reports/results.json)
show that downstream-gradient candidates beat direct-readout candidates at both
tested widths on discovery and one frozen held-out prompt.

The language driver pins the Qwen3-0.6B model revision, MPS/float16 execution,
all generation settings, and separate seeds for ranking, qualification,
ablation controls, and random residual directions:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-causal-loop.yaml \
  --events jsonl
```

The driver performs 132 bounded logical model calls when all three pairs clear
the generated-behavior gate. A failed gate stops causal stages before their
forward passes. Use `probe report RUN_ID` for a conservative JSON and Markdown
interpretation of any resulting stage, and `probe runs verify RUN_ID` before
using its evidence.

This is a representative transfer smoke case, not a reproduction of the
paper's original benchmark prompts or model. Direction effects demonstrate
controllability; they do not identify an FFN circuit. Ranked-neuron effects must
beat the same-layer random controls and generalize before receiving a strong
causal interpretation.

The attention driver runs a separate 91-call evidence chain:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-attention-path.yaml \
  --events jsonl
```

It adds qualification-gated head ranking, a top-head patch dose sweep, eager
token-to-head routes, and an exactly aligned L18/H12-to-L25/H0 two-stage path
test. The checked-in [analysis](language-attention-path-reports/analysis.md) and
[machine-readable results](language-attention-path-reports/results.json) record
the live Qwen3-0.6B/MPS outcome and deterministic replay checks.
