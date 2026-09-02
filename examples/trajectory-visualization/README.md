# Trajectory visualization example

This is a complete, local-first example of the trajectory-guided research loop:

```text
controlled prompts → generated-behavior gate → native paired trajectory
                   → layer-aware FFN candidates → matched patch controls
                   → portable evidence visualization
```

Open [`trajectory.html`](trajectory.html) to inspect the checked-in result. It
is a self-contained light-theme HTML/SVG file: it makes no network requests and
does not need a plotting runtime. [`analysis.md`](analysis.md) records the
scientific interpretation, while [`results.json`](results.json) records the
exact run lineage, key values, verification state, and figure hash.

## Run from a clean checkout

Run commands from the repository root. Install the locked environment and
confirm that the pinned model snapshot is already local:

```bash
uv sync --extra dev --locked
uv run --locked probe model inspect Qwen/Qwen3-0.6B
```

The driver never downloads a model. If `cached` is false, follow the explicit
acquisition instructions in the root README before continuing.

Execute the 96-forward-pass workflow on Apple MPS:

```bash
uv run --locked probe workflow \
  --driver examples/trajectory-visualization/driver.yaml \
  --events none
```

The final JSON object contains `trajectory_run_id` and three values in
`intervention_run_ids`. Verify those four immutable runs before plotting:

```bash
uv run --locked probe runs verify TRAJECTORY_RUN_ID
uv run --locked probe runs verify DIRECT_PATCH_RUN_ID
uv run --locked probe runs verify DOWNSTREAM_PATCH_RUN_ID
uv run --locked probe runs verify OVERLAP_PATCH_RUN_ID
```

Generate the same view for the `capital` pair. The renderer verifies every
source again and refuses an intervention that does not descend from the given
trajectory run:

```bash
uv run --locked probe runs trajectory-visualize \
  TRAJECTORY_RUN_ID \
  DIRECT_PATCH_RUN_ID \
  DOWNSTREAM_PATCH_RUN_ID \
  OVERLAP_PATCH_RUN_ID \
  --pair capital \
  --output examples/trajectory-visualization/trajectory.html
```

Open the result directly on macOS:

```bash
open examples/trajectory-visualization/trajectory.html
```

Use `--pair arithmetic` or `--pair science` to render another declared pair.
The output receipt includes the selected pair, source verification map, SHA-256
hash, and byte size, making it suitable for an agent workflow.

## How to read it

- The first chart decodes the original prompt, perturbed prompt, and paired
  difference through the model's native final norm and LM head. It is
  observational evidence.
- The transition list ranks abrupt changes only as inspection suggestions.
- The second chart shows selected-neuron patch effects as solid lines and
  same-layer matched-random means as dashed lines. Claim badges are copied from
  the immutable backend summaries.
- The dose table compares selected absolute effects with matched controls on
  discovery and held-out splits.
- A downstream trajectory is not conserved signal flow. Later blocks may
  amplify, attenuate, or reverse what the decoder can read.

The driver fixes the model revision, MPS/float16 runtime, deterministic
generation, ranking objective (`shared_direction`), prompt splits, budgets,
control count, candidate pool, patch widths, strengths, and every random seed.
Exact floating-point equality across macOS/PyTorch releases is not promised;
retain the run manifests and verification results when comparing replays.
