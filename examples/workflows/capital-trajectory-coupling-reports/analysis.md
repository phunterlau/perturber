# Capital agreement: trajectory-guided FFN follow-up

This report was produced by running
[`../capital-trajectory-coupling.yaml`](../capital-trajectory-coupling.yaml)
twice with the pinned Qwen3-0.6B revision on MPS/float16. Downloads were
disabled. The bounded machine-readable results are in [`results.json`](results.json).

## Run it

```bash
uv run --locked probe workflow \
  --driver examples/workflows/capital-trajectory-coupling.yaml \
  --events jsonl
```

Use the run IDs printed in the outcome to inspect and verify each new lineage:

```bash
uv run --locked probe runs show TRAJECTORY_RUN_ID
uv run --locked probe runs show FFN_COUPLING_RUN_ID
uv run --locked probe runs verify RANK_RUN_ID
uv run --locked probe runs verify TRAJECTORY_RUN_ID
uv run --locked probe runs verify FFN_COUPLING_RUN_ID
```

## Measured result

The controlled prompt change moved the `No − Yes` first-token observable by
`+0.625`. The original fixed-readout FFN sum predicted `+0.288944`, leaving the
existing `0.502674` FFN/skip diagnostic and its weak-additivity warning.

The native paired trajectory reached the same final `+0.625` delta. Its largest
single checkpoint changes were:

| Rank | Checkpoint | Paired-gap change |
|---:|---|---:|
| 1 | L27 post-FFN | +0.265625 |
| 2 | L18 post-FFN | +0.226562 |
| 3 | L21 post-FFN | +0.195312 |
| 4 | L17 post-attention | +0.175781 |
| 5 | L23 post-FFN | -0.156250 |

These transitions suggest where to spend intervention budget. They do not show
that a layer caused the behavioral change.

Layer-aware endpoint gradients changed the prioritization. The strongest
downstream candidates were L21:n1341 (`+0.080000`), L18:n1784 (`+0.073525`),
L27:n840 (`-0.052597`), L19:n826 (`+0.041652`), and L25:n1665
(`-0.039611`). Downstream RMS mass peaked at layers 17, 18, and 16, rather than
simply concentrating in the latest layers favored by a direct decoder.

The top ten downstream candidates all agreed in sign with the direct score, but
agreement fell to `0.700` in the top 100 and `0.652` across the stored top 500.
This is the practical value of layer-aware coupling: it preserves strong shared
candidates while exposing candidates whose apparent direct-logit effect is
reshaped by the remaining transformer blocks.

## Replay and interpretation

All three primary run manifests passed digest verification. A second complete
workflow produced new immutable lineage IDs, as expected, but its normalized
rank, trajectory, and coupling summaries were exactly equal. Only lineage/hash
fields and rank wall time were excluded from that equality check.

The conclusion is deliberately limited: this one-pair case yields a reproducible
trajectory-guided candidate set, not a localized causal circuit. The next
research step is a matched-control intervention dose sweep over the shared
L21:n1341/L18:n1784 candidates and trajectory-specific L17/L16 candidates,
followed by validation and held-out pairs.
