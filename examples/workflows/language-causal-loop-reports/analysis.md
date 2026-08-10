# Qwen3-0.6B language-routing causal-loop example

This checked-in example records a complete run of
[`language-causal-loop.yaml`](../language-causal-loop.yaml) on Apple MPS. The
model revision, dtype, generation controls, logical-call budgets, and separate
random seeds are all in the driver. The workflow completed 132/132 planned
logical model calls in 10.7 seconds.

## Immutable run map

| Stage | Run ID | Calls | Evidence |
|---|---|---:|---|
| Ranking | `20260810T000018-b5cfefbfd03d` | 6 | replicated observational ranking |
| Generated qualification | `20260810T000019-35be5cf9a10d` | 6 | qualified observable |
| Positive-neuron ablation | `20260810T000023-c1ec41ec2e7f` | 72 | local causal intervention |
| Residual direction sweep | `20260810T000025-d03e122f0816` | 48 | local controllability |

Both causal manifests reference the rank and qualification runs. All four run
directories passed artifact-integrity verification at completion.

## Qualification result

The Chinese-minus-English gap crossed from control to target on all three pairs,
with a mean movement of `+23.8568`. The sparse observable did not contain every
actual argmax token (`The`/`法国`, `Two`/`二`, and `The`/`纯`), so rank-only
qualification correctly remained weak.

The predeclared dominant-Unicode-script evaluator then agreed with the gap in all
six generated conditions:

- `The capital of France is Paris.` versus `法国的首都是巴黎。`;
- `Two plus two is four.` versus `二加二等于四。`;
- `The temperature at which pure water freezes is` versus `纯水在**0°C（3`.

That makes all 3/3 pairs eligible for the causal stages. The last Chinese sample
is truncated at eight generated tokens, but its dominant script is unambiguous.

## Important neurons

The aggregate FFN/Skip value is `0.1214`, which is the paper's low-concentration,
readout-compatible regime. Highest layer RMS masses occur at layers 27, 26, 25,
24, and 23. The leading positive, sign-consistent neurons used in ablation are:

| Selected order | Rank | Layer | Neuron | Mean importance | RMS importance |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 23 | 59 | 2.6100 | 2.6201 |
| 2 | 2 | 27 | 52 | 1.9600 | 1.9688 |
| 3 | 3 | 27 | 22 | 1.7555 | 1.7747 |
| 4 | 6 | 27 | 53 | 1.3071 | 1.3841 |
| 5 | 7 | 25 | 163 | 1.2765 | 1.3737 |
| 6 | 8 | 23 | 179 | 1.0707 | 1.2144 |
| 7 | 9 | 27 | 30 | 1.0462 | 1.1094 |
| 8 | 10 | 23 | 561 | 1.0538 | 1.1014 |
| 9 | 12 | 27 | 39 | 0.9949 | 0.9973 |
| 10 | 13 | 21 | 230 | 0.9174 | 0.9483 |

All shown neurons have sign consistency `1.0` across the three pairs. The full
20-unit population and exact scores are stored in the intervention summary.

## Controlled ablation

The intervention scales positive post-SwiGLU activations on the perturbed prompt.
Each row aggregates three prompt pairs and nine same-layer matched-random
observations (three random draws per pair).

| N | Activation scale | Mean gap effect | Random absolute mean | Controlled absolute effect | Bootstrap interval |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | -0.7667 | 0.0037 | 0.7631 | [-0.8642, -0.7117] |
| 5 | 0.0 | -1.3414 | 0.0061 | 1.3353 | [-1.7855, -0.9738] |
| 20 | 0.0 | -4.4426 | 0.0278 | 4.4148 | [-5.9917, -3.4367] |
| 1 | 0.5 | -0.3657 | 0.0046 | 0.3610 | [-0.4111, -0.3410] |
| 5 | 0.5 | -0.6036 | 0.0044 | 0.5992 | [-0.7931, -0.4434] |
| 20 | 0.5 | -1.9594 | 0.0292 | 1.9302 | [-2.6039, -1.5242] |

The sign, strength response, and neuron-count response are coherent: reducing
the selected positive units reduces the Chinese-oriented gap, and complete
ablation is stronger than half ablation. The descriptive 90%-effect width is 20
for both strengths. This supports local necessity evidence for this selected
population; it does not establish that 20 is a universal circuit size.

## Residual-direction control

The behavioral direction is injected with residual-norm scaling and compared
with three norm-matched orthogonal directions per pair.

| Layer | Beta | Mean gap effect | Random absolute mean | Controlled absolute effect | Bootstrap interval |
|---:|---:|---:|---:|---:|---:|
| 18 | -0.05 | -0.8388 | 0.0554 | 0.7833 | [-0.9355, -0.7508] |
| 18 | +0.05 | +0.8387 | 0.0621 | 0.7766 | [+0.7334, +0.9551] |
| 24 | -0.05 | -3.8919 | 0.0702 | 3.8217 | [-4.2500, -3.5570] |
| 24 | +0.05 | +3.8458 | 0.0873 | 3.7585 | [+3.5271, +4.1165] |

Layer 24 is much more controllable than layer 18 at the tested dose, and the
effect reverses with beta. This is a clean local controllability result, not
evidence that the direction is localized to the ranked FFN neurons.

## Conclusion and limits

For these three Qwen3-0.6B prompts, the tool finds a reproducible language gap,
qualifies it against generated language, identifies sign-consistent late-layer
FFN units, and shows that ablating those units changes the gap far more than
same-layer random units. It also finds strong bidirectional residual control at
layer 24. This is useful evidence that the ranked neurons play a local causal
role in the chosen first-token observable.

It is not yet a paper-level circuit result. Selection and testing use the same
three discovery pairs; there is no held-out perturbation family, generated-text
intervention evaluator, collateral observable, permutation test, or
multiple-comparison correction. The paper's original model and benchmark prompts
are not reproduced. The next decisive experiment is to freeze the selected
neurons and doses, then test them on a held-out EN/ZH prompt set while recording
both language behavior and unrelated capabilities.
