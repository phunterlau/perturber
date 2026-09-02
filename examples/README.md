# Examples

The examples are executable research inputs, not screenshots or mock data. They
cover quick ranking probes, multi-stage causal workflows, and strict replay
checks. Run every command below from the repository root.

## Before running a model

Install the locked environment and inspect the local model cache:

```bash
uv sync --extra dev --locked
uv run --locked probe model inspect Qwen/Qwen3-0.6B
```

If the model is missing, acquire it only with an explicit download budget:

```bash
uv run --locked probe model fetch Qwen/Qwen3-0.6B \
  --max-download-bytes 2000000000
```

The examples use the project-local `.hf-cache/` and `.probe/` directories. They
do not use a global Hugging Face cache.

## Small prompt-pair probes

Start with `agreement-capital.json`. It is a bounded, one-pair smoke case that
is useful for checking an installation and learning the artifact interface.

```bash
uv run --locked probe validate --spec examples/agreement-capital.json
uv run --locked probe plan --spec examples/agreement-capital.json
uv run --locked probe capabilities --spec examples/agreement-capital.json
uv run --locked probe preflight --spec examples/agreement-capital.json
uv run --locked probe run \
  --spec examples/agreement-capital.json \
  --events jsonl
```

For a compact response suitable for a coding agent, replace `--events jsonl`
with:

```bash
--events none --result compact-json
```

The other standalone probes are:

| File | Research pattern |
| --- | --- |
| `agreement-replication.yaml` | Multiple agreement pairs and split-aware ranking |
| `paper-safety-bpe.yaml` | Refusal versus compliance perturbation |
| `paper-language-en-zh.yaml` | English versus Chinese response routing |
| `paper-factual-entity.yaml` | Factual entity substitution |
| `paper-code-vs-explain.yaml` | Code generation versus explanation |
| `paper-cot-complex-simple.yaml` | Complex versus simplified reasoning prompt |

These are small transfer cases inspired by the paper's perturbation patterns;
they are not exact reproductions of its original models or benchmark runs.

Use the same four preparation commands before executing any of them:

```bash
uv run --locked probe validate --spec examples/paper-language-en-zh.yaml
uv run --locked probe plan --spec examples/paper-language-en-zh.yaml
uv run --locked probe capabilities --spec examples/paper-language-en-zh.yaml
uv run --locked probe preflight --spec examples/paper-language-en-zh.yaml
```

## Multi-stage workflows

The workflow drivers resolve symbolic parents such as `$rank` and
`$qualification`, reuse one loaded model, and retain their resolved immutable
lineage under `.probe/workflows/`.

Run the FFN causal loop:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-causal-loop.yaml \
  --events jsonl
```

Run the direct-versus-layer-aware trajectory causal follow-up:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-trajectory-causal-followup.yaml \
  --events jsonl
```

Run the attention-path workflow:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-attention-path.yaml \
  --events jsonl
```

Workflow orchestration is standalone and intentionally rejects `--endpoint`.
The FFN driver uses up to 132 logical model calls; the attention driver uses 91.
Qualification gates stop downstream causal stages before their forward passes
when the declared behavior does not qualify.

See [workflows/README.md](workflows/README.md) for stage-by-stage interpretation.
The adjacent report directories contain checked-in Qwen3-0.6B/MPS results and
scientific caveats.

## Replay bundles

Replay bundles pin the executable spec, model revision, device and dtype, Torch
seed, runtime settings, numeric tolerances, and neuron-ranking acceptance
thresholds.

Inspect a bundle without loading the model:

```bash
uv run --locked probe replay inspect \
  examples/replays/language-routing/driver.yaml
```

Execute and compare it with its recorded baseline:

```bash
uv run --locked probe replay run \
  examples/replays/language-routing/driver.yaml
```

Check an already completed compatible run without another model execution:

```bash
uv run --locked probe replay check \
  examples/replays/language-routing/driver.yaml --run-id RUN_ID
```

Available bundles:

- `replays/language-routing/` — English/Chinese routing probe
- `replays/cot-protocol/` — reasoning-protocol perturbation probe

See [replays/README.md](replays/README.md) before recording or replacing a
baseline. Never loosen replay tolerances silently to make a run pass.

## Inspecting results

Every successful run returns an immutable run ID. Use bounded queries first:

```bash
uv run --locked probe runs overview RUN_ID
uv run --locked probe runs layers RUN_ID --top 10
uv run --locked probe runs neurons RUN_ID --ranking-objective shared_direction --top 20
uv run --locked probe runs neurons RUN_ID --ranking-objective effect_magnitude --top 20
uv run --locked probe claims RUN_ID
uv run --locked probe runs verify RUN_ID
uv run --locked probe report RUN_ID
```

New multi-pair examples use signed-mean aggregation (`shared_direction`) for
the primary paper-faithful FFN ranking while retaining an RMS view in the same
run. Replay specifications under `examples/replays/` deliberately preserve the
objective recorded by their existing baselines; do not edit them merely to
match the new example default.

For attention results:

```bash
uv run --locked probe attention heads ATTENTION_RANK_RUN_ID --top 20
uv run --locked probe attention paths ATTENTION_PATH_RUN_ID --limit 20
```

Researchers and coding agents should follow [`RESEARCH.md`](../RESEARCH.md) for
the complete evidence gates, failure handling, interpretation rules, and final
report format.
