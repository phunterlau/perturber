# Perturbation Probing: Agent Research Workflow

This file is the operating guide for a coding or research agent using the
`probe` CLI. Work from the repository root. Treat the CLI, schemas, and
immutable artifacts as the source of truth; the WebUI is an optional view over
the same backend.

## Research contract

- Use controlled prompt pairs that differ only in the intended perturbation.
- Pin the model revision, device, dtype, seed, observable, generation settings,
  budgets, and controls in every reusable experiment.
- Never describe a neuron or attention head as causal from ranking alone.
- A single pair is exploratory. Replication, generated-behavior qualification,
  interventions, matched controls, and held-out transfer strengthen evidence in
  separate stages.
- Direction injection establishes controllability, not neuron localization.
- Token-to-head edges are routing hypotheses. A head path is causal evidence
  only when qualification, endpoint interventions, exact token alignment, and
  matched path controls all pass.
- Verify an immutable run before relying on it. Report discrepancies and null
  results; do not tune thresholds after seeing results without recording a new
  specification.

## 1. Establish the environment

```bash
uv sync --extra dev --locked
uv run --locked probe model inspect Qwen/Qwen3-0.6B
uv run --locked probe server status
```

The default local stores are `.hf-cache/` for models and `.probe/` for jobs,
runs, workflows, and reports. Do not read or write a global Hugging Face cache.

If the model is absent, stop and report that acquisition is required. Fetch it
only when the user has authorized the download and an explicit byte budget:

```bash
uv run --locked probe model fetch Qwen/Qwen3-0.6B \
  --max-download-bytes 2000000000
```

On Apple Silicon, prefer MPS with `float16`. A healthy daemon avoids repeatedly
loading the model for individual stage commands. Never infer or auto-discover a
daemon: select it explicitly with `--endpoint http://127.0.0.1:8765`. The
multi-stage `probe workflow` orchestrator is intentionally standalone and
rejects `--endpoint`; it reuses one model within its own process.

## 2. Choose or create a specification

Start from the smallest relevant checked-in example:

- `examples/agreement-capital.json`: one-pair smoke analysis.
- `examples/agreement-replication.yaml`: replicated basic ranking.
- `examples/paper-*.yaml`: representative paper-pattern cases.
- `examples/workflows/language-causal-loop.yaml`: qualification, FFN
  intervention, and residual-direction controls.
- `examples/workflows/language-trajectory-causal-followup.yaml`: native
  trajectory, discovery-only layer-aware FFN coupling, and matched direct,
  downstream, and overlap interventions.
- `examples/workflows/language-attention-path.yaml`: qualification, head
  ranking/intervention, token routes, and sender-receiver path probing.

Discover contracts instead of guessing fields:

```bash
uv run --locked probe schema list
uv run --locked probe schema show rank
uv run --locked probe examples list
uv run --locked probe examples show agreement-capital
```

For a new question, state before execution:

1. the intended perturbation and invariants;
2. the target-minus-control observable and why it measures the question;
3. discovery, validation, and held-out pairs;
4. what result would count against the hypothesis;
5. the planned qualification, intervention, and matched controls.

## 3. Validate before spending model calls

Run all four checks. Do not continue when validation fails, preflight is not
executable, the requested adapter is unsupported, or the forward/artifact
budget is insufficient.

```bash
SPEC=examples/agreement-capital.json
uv run --locked probe validate --spec "$SPEC"
uv run --locked probe plan --spec "$SPEC"
uv run --locked probe capabilities --spec "$SPEC"
uv run --locked probe preflight --spec "$SPEC"
```

`model_ready: false` means the snapshot is not locally executable.
`acquisition_required: true` is permission encoded by the spec, not evidence
that a download has occurred.

## 4. Execute through the machine interface

For a bounded agent response, use compact JSON:

```bash
uv run --locked probe run \
  --spec "$SPEC" \
  --events none \
  --result compact-json
```

When a healthy daemon is already running, reuse it explicitly and supply a
unique, stable idempotency key:

```bash
uv run --locked probe \
  --endpoint http://127.0.0.1:8765 \
  run --spec "$SPEC" \
  --events none --result compact-json \
  --request-id agent-QUESTION-RUN-001
```

Use `--events jsonl` when monitoring progress. Stdout then contains only
versioned JSON objects. Parse the final object rather than scraping human text.
Preserve the returned run ID. If execution is interrupted, inspect the durable
job instead of blindly resubmitting:

```bash
uv run --locked probe jobs status JOB_ID
uv run --locked probe jobs spec JOB_ID
uv run --locked probe jobs watch JOB_ID
```

Exit codes are stable: `2` invalid spec, `3` capability/budget, `4` model
policy, `5` runtime, `6` artifact integrity, `7` cancellation, `8` endpoint,
and `9` replay mismatch.

## 5. Inspect evidence without flooding agent context

Query compact slices first:

```bash
uv run --locked probe runs overview RUN_ID
uv run --locked probe runs layers RUN_ID --top 10
uv run --locked probe runs neurons RUN_ID --top 20 --sign positive
uv run --locked probe runs neurons RUN_ID --top 20 --sign negative
uv run --locked probe claims RUN_ID
uv run --locked probe runs files RUN_ID
uv run --locked probe runs verify RUN_ID
uv run --locked probe report RUN_ID
```

For each important neuron, retain at least its layer, neuron index, signed
importance, absolute/RMS importance, sign consistency, and
`observable_effect`. Do not select neurons solely by one pair's absolute score.
Compare measured delta-F with the additive neuron prediction and report large
residual/skip disagreement as model inadequacy, not as noise to ignore.

Useful follow-up diagnostics are:

```bash
uv run --locked probe stability RUN_ID --top-n 50 --seed 0
uv run --locked probe sensitivity RUN_ID --metadata-key perturbation_family
uv run --locked probe compare REFERENCE_RUN CANDIDATE_RUN --top-n 50
```

## 6. Escalate an observational ranking to a causal workflow

Prefer a seeded workflow driver so parent run IDs are resolved and retained
automatically. Run workflow orchestration standalone—do not add `--endpoint`:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-causal-loop.yaml \
  --events jsonl
```

To compare candidate scoring methods under one frozen split and dose schedule:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-trajectory-causal-followup.yaml \
  --events jsonl
```

In this workflow, `ffn_coupling.candidate_pair_ids` must contain discovery pairs
only. Treat any held-out pair appearing there as leakage and reject the run.
`candidate_method: direct_downstream_overlap` requires an explicit
`overlap_pool_size`; report that pool as well as the intervention width. Compare
controlled absolute effects at identical widths and splits. The intervened
trajectory's first non-zero checkpoint describes effect propagation, while the
matched intervention claim—not the trajectory—determines causal status.

Interpret the stages separately:

1. **Rank:** locate FFN hypotheses from paired activation and coupling changes.
2. **Qualify:** confirm the observable tracks generated behavior. Stop causal
   interpretation when this gate fails.
3. **Intervene:** compare ranked-neuron ablation, amplification, or patching
   against same-layer random controls across widths and strengths.
4. **Inject:** test residual-direction controllability against norm-matched
   orthogonal random directions.
5. **Validate:** repeat on validation and held-out pairs and report effect
   stability, sign consistency, collateral changes, and failures.

For attention routing:

```bash
uv run --locked probe workflow \
  --driver examples/workflows/language-attention-path.yaml \
  --events jsonl
```

Require intervention-tested heads before interpreting token routes. Require
exact alignment and matched controls before interpreting sender-to-receiver path
patching. Inspect bounded results with:

```bash
uv run --locked probe attention heads ATTENTION_RANK_RUN_ID --top 20
uv run --locked probe attention paths ATTENTION_PATH_RUN_ID --limit 20
```

## 7. Replay and hand off reproducibly

Use an existing replay bundle before modifying its baseline:

```bash
uv run --locked probe replay inspect \
  examples/replays/language-routing/driver.yaml
uv run --locked probe replay run \
  examples/replays/language-routing/driver.yaml
```

For an already completed run, use the offline check:

```bash
uv run --locked probe replay check \
  examples/replays/language-routing/driver.yaml --run-id RUN_ID
```

Use only a run produced from that replay driver's pinned spec. Exact numeric and
neuron agreement does not override a seed, science-hash, request-hash, or model
identity mismatch; strict replay should fail such a handoff with exit code `9`.

Record or overwrite a baseline only when explicitly requested. A new bundle
must include `driver.yaml`, the complete executable spec, `baseline.json`, an
`analysis.md`, and generated reports. Pin the exact model revision and record
seeds, device/dtype, inference settings, relevant runtime versions, numeric
tolerances, and neuron-overlap/sign acceptance thresholds.

## 8. Write the conclusion

The final research note should contain:

- the question, model revision, device/dtype, seed, and prompt splits;
- the observable and generated-behavior qualification result;
- measured behavioral movement and FFN/skip agreement;
- leading layers and neurons with direction and stability;
- intervention effect versus matched controls and dose/width behavior;
- attention or residual-direction evidence, if run;
- collateral effects, discrepancies, and falsifying/null results;
- exact run IDs, verification outcome, and replay status;
- one calibrated conclusion labeled observational, qualified, locally causal,
  or held-out replicated.

Never generalize beyond the tested model revision, prompts, evaluator, split,
intervention, and controls. Prefer “candidate neuron,” “supports,” and “did not
replicate” over unqualified claims about what a neuron represents.
