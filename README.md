# Perturber

Perturber is a local-first workbench for studying how a controlled prompt
change moves through a language model. Give it an original prompt and a
perturbed prompt, then follow the evidence from behavior to candidate FFN
neurons, attention heads, token routes, and controlled interventions.

The tool implements the perturbation-probing method described in
[Perturbation Probing: A New Window into LLM Internal States](https://arxiv.org/abs/2604.27401).
It is built for mechanistic-interpretability researchers and for coding agents
that need a reproducible, machine-readable research workflow.

> Perturber is research software. Rankings identify hypotheses; causal claims
> require behavioral qualification, interventions, matched controls, and
> verified artifacts.

## What you can do

- Compare controlled prompt pairs across discovery, validation, and held-out
  splits.
- Rank FFN layers and neurons using signed perturbation importance.
- Decode native paired residual trajectories to find where a prediction
  difference emerges across attention and FFN checkpoints.
- Re-rank FFN candidates with native local and downstream endpoint gradients,
  rather than assuming one layer-independent output coupling.
- Test neurons with ablation, amplification, patching, dose sweeps, and random
  controls.
- Rank attention heads and inspect token-to-head routes.
- Test sender-to-receiver attention paths with explicit token alignment and
  matched controls.
- Save every run as an immutable, verifiable artifact with its exact spec,
  parents, seed, model revision, and evidence claims.
- Use the same scientific backend from the WebUI, CLI, or a coding agent.

The current reference model is `Qwen/Qwen3-0.6B`. Apple Silicon macOS with
PyTorch MPS is the primary development target; CPU and CUDA are capability
gated by the same model adapter.

## Quick start

Requirements:

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- macOS on Apple Silicon recommended

From the repository root:

```bash
uv sync --extra dev --locked
```

Download the reference model once. Perturber stores it in the ignored local
`.hf-cache/` directory and refuses unbudgeted model downloads during research
runs.

```bash
uv run --locked probe model fetch Qwen/Qwen3-0.6B \
  --max-download-bytes 2000000000
```

Start the local server:

```bash
uv run --locked probe server start
uv run --locked probe server status
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). The server binds only to
loopback; research state stays in the ignored `.probe/` workspace.

When finished:

```bash
uv run --locked probe server stop
```

See [UI_HOWTO.md](UI_HOWTO.md) for the complete UI launch guide and port
troubleshooting.

## Two ways to investigate

**Quick Probe** is the shortest path from two prompts to an observational FFN
ranking. A useful run can be promoted into a research case without rerunning it.

**Research Cases** guide a checkpointed evidence pipeline:

```text
Rank -> Behavioral Qualification -> Candidate Circuit
                                    |-> FFN Intervention -> Direction Control
                                    `-> Head Ranking -> Head Intervention
                                                     |-> Token Routes
                                                     `-> Sender -> Receiver Path
```

Each stage shows its parents, estimated model calls, artifact budget, seed,
controls, and gate status before it runs. The UI labels evidence as
observational, behaviorally qualified, locally causal, or held-out replicated
from backend results—not from browser-side inference.

## Headless CLI and agent workflow

The CLI emits versioned JSON and JSONL contracts, uses stable exit categories,
and supports bounded result queries. This makes it suitable for notebooks,
shell pipelines, and autonomous coding agents.

Try the small checked-in example:

```bash
uv run --locked probe validate --spec examples/agreement-capital.json
uv run --locked probe plan --spec examples/agreement-capital.json
uv run --locked probe preflight --spec examples/agreement-capital.json
uv run --locked probe run \
  --spec examples/agreement-capital.json \
  --events jsonl
```

Inspect a completed run without loading the model again:

```bash
uv run --locked probe runs list
uv run --locked probe runs overview RUN_ID
uv run --locked probe runs neurons RUN_ID --top 20
uv run --locked probe claims RUN_ID
uv run --locked probe runs verify RUN_ID
uv run --locked probe report RUN_ID
```

For a multi-stage, model-reusing workflow with symbolic parent references:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-attention-path.yaml \
  --events jsonl
```

For the smallest trajectory-guided FFN workflow and its checked-in result:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/capital-trajectory-coupling.yaml \
  --events jsonl
```

See [the capital trajectory/coupling report](examples/workflows/capital-trajectory-coupling-reports/analysis.md).

Agents should read [RESEARCH.md](RESEARCH.md) before operating the tool or
interpreting results. A Research Case can also export a bounded research packet
with canonical YAML, immutable run lineage, reports, exact continuation
commands, warnings, and unresolved gates.

## Reproducibility

Successful runs are immutable and content-verifiable. Their manifests record
the model and revision, device and dtype, random seed, inference controls,
parent runs, artifact hashes, and evidence stage. Portable replay bundles under
[`examples/replays/`](examples/replays/) add numeric tolerances and ranking
acceptance thresholds for cross-run checks.

Runtime state is intentionally not committed:

- `.hf-cache/` — model snapshots
- `.probe/` — runs, cases, jobs, reports, and server credentials
- `.venv/` — the project environment
- `web/node_modules/` — frontend dependencies

## Development

Run the backend suite:

```bash
uv run --locked pytest
uv run --locked python -m compileall -q src tests
```

Run and build the frontend:

```bash
cd web
npm ci
npm test
npm run typecheck
npm run build
```

The production React bundle is written to `src/probing/web_dist/` and packaged
with the Python application.

## Documentation

- [UI_HOWTO.md](UI_HOWTO.md) — start and use the WebUI or compatibility TUI
- [RESEARCH.md](RESEARCH.md) — researcher and coding-agent operating workflow
- [DESIGN.md](DESIGN.md) — architecture, contracts, and evidence boundaries
- [VALIDATION.md](VALIDATION.md) — verified behavior and scientific caveats
- [Qwen3-0.6B paper-case audit](PAPER_CASE_VALIDATION_QWEN3_0_6B.md) — transfer
  results across safety, language, factual, code, and chain-of-thought cases
- [Examples guide](examples/README.md) — executable specs, workflows, replays,
  and checked-in reports

## License

MIT. See [LICENSE](LICENSE).
