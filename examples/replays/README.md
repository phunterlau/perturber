# Replay examples

Each directory is a self-contained, reviewable experiment bundle:

```text
<example>/
  driver.yaml       replay identity, runtime record, and acceptance policy
  spec.yaml         complete executable experiment request
  baseline.json     recorded result summary and stable artifact hashes
  analysis.md       researcher interpretation and scientific limitations
  reports/          machine JSON and concise Markdown replay checks
```

The driver has two jobs. `reproducibility` states what must be held fixed: model
commit, adapter, device/dtype, Torch seed, inference settings, and selected host
versions. `comparison` states what counts as the same result: scalar tolerances,
exact first-token predictions, top-neuron overlap, sign agreement, and whether
stable artifact hashes are report-only or required.

Run IDs, timestamps, event logs, elapsed time, and report-generation time are
operational records, not scientific equality conditions. The current probes do
no sampling or generation, so only `torch_seed` is active; Python and NumPy seeds
are explicitly `null`. The driver still records `model_eval`, `inference_mode`,
and `use_cache` because those are implementation controls worth preserving as
generation and interventions are added.

Inspect and replay a bundle:

```bash
uv run --locked probe replay inspect \
  examples/replays/language-routing/driver.yaml

uv run --locked probe replay run \
  examples/replays/language-routing/driver.yaml
```

For repeated MPS work, point the same command at an explicit daemon to avoid
reloading weights:

```bash
uv run --locked probe --endpoint http://127.0.0.1:8765 replay run \
  examples/replays/language-routing/driver.yaml \
  --request-id my-language-replay-001
```

To create a new bundle, execute its pinned `spec.yaml`, then record the immutable
run as the baseline:

```bash
uv run --locked probe replay record path/to/driver.yaml --run-id RUN_ID
```

An existing run can be checked without loading a model:

```bash
uv run --locked probe replay check path/to/driver.yaml --run-id RUN_ID
```

`replay run` and `replay check` emit one compact `probe.replay-outcome/v1` JSON
object and exit `9` when any required comparison fails. Pass `--output full` to
emit the complete `probe.replay-report/v1` instead. This gives a coding agent a
bounded, stable pass/fail interface while retaining detailed evidence in the
report files.

The checked-in Qwen3-0.6B profiles intentionally target the recorded Apple
Silicon/MPS environment. Replaying on CPU, CUDA, another model revision, or an
unmatched package stack should use a separately reviewed profile and baseline,
not silently loosen this one.
