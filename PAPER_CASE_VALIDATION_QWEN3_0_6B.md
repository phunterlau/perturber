# Paper-case validation on Qwen3-0.6B

Validated on 2026-08-09 against
[arXiv:2604.27401](https://arxiv.org/abs/2604.27401), using the cached
`Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` on PyTorch MPS/float16.

> Historical snapshot: this document records the ranking-only backend as it
> existed during the five-case audit. Present-tense implementation limitations
> below are preserved to explain the findings. Qualification, structured
> messages/tools, causal FFN interventions, direction injection, comparison,
> and sensitivity analysis were implemented afterward. See the
> [qualified language causal loop](examples/workflows/language-causal-loop-reports/analysis.md)
> for the current evidence path.

## Scope

This is a qualitative transfer test, not a reproduction of the paper's reported
numbers. The paper evaluates Qwen3-4B and Qwen3.5-2B with larger prompt sets and
causal interventions. The local model is Qwen3-0.6B, and the paper PDF does not
contain the complete original prompt sets or every circuit's exact token sets.

Five three-pair suites were therefore constructed from the perturbation patterns
specified in Section 2.3 and Table 4:

- safety: adjacent-character BPE swaps;
- language routing: English versus Chinese prompts;
- factual recall: France versus United Kingdom entity swaps;
- code generation: code versus explanation instructions;
- CoT activation: complex versus simple prompts under the Qwen thinking protocol.

The existing agreement suite already covers the paper's wrong-versus-correct
premise sycophancy pattern. At the time of this audit, tool use was not
representable by the request schema, which had no role messages, tool definitions,
or tool-call observable.
The paper PDF does not give enough detail to construct a faithful separate math
detection observable, so this audit does not substitute an invented one.

The five runs used 15 prompt pairs and 30 logical forward passes. The daemon
loaded the model once and reused it for subsequent suites. Downloads and remote
code were disabled. Every recorded artifact digest verified.

## Results

| Case | Gap sign flips | Measured delta | Additive prediction | Relative error | FFN/Skip mean (range) | Leading layers | Paper topology |
|---|---:|---:|---:|---:|---:|---|---|
| Safety BPE | 1/3 | -3.0853 | -8.0977 | 162.5% | 0.769 (0.189-1.318) | 26, 25, 21, 24, 27 | Opposition |
| Language EN->ZH | 3/3 | +23.8568 | +29.5445 | 23.8% | 0.121 (0.082-0.181) | 27, 26, 25, 24, 23 | Readout/routing |
| Factual entity | 3/3 | -13.7344 | -15.3065 | 11.4% | 0.428 (0.259-0.737) | 26, 25, 27, 24, 23 | Readout |
| Code vs explain | 3/3 | -21.6729 | -29.1423 | 34.5% | 0.946 (0.785-1.241) | 25, 26, 21, 22, 27 | Readout |
| CoT complex/simple | 0/3 | +1.4131 | +4.6031 | 225.7% | 0.058 (0.050-0.073) | 27, 26, 25, 23, 24 | Readout |

"Gap sign flips" counts pairs where the target-minus-control logit gap crosses
zero. This is a necessary smoke check for the paper's stated perturbation goal,
but it is not causal evidence.

Run IDs:

- safety: `20260809T212738-ad3a413c4e08`;
- language: `20260809T212750-2c34fe5560c1`;
- factual: `20260809T212758-c33b8d8eb5ba`;
- code: `20260809T212808-e466a6bb5844`;
- CoT: `20260809T212932-a48f8c0406b8`.

The run repository is `/private/tmp/probe-paper-cases-qwen06b`.

## What transferred well

Language routing is the strongest qualitative match. Every pair crossed the
language-gap boundary, the actual first-token argmax changed from an English to
a Chinese token, FFN/Skip stayed consistently below 0.2, and the leading scores
were concentrated in the final five layers. This matches the paper's routing
diagnosis and late-layer readout localization. This historical audit did not
test the paper's direction-injection result; the later causal-loop example does.

All five suites produced late-layer-dominated rankings, consistent with the
paper's observation that task-responsive readouts tend to occur late. The
language, factual, code, and CoT leading neurons had sign consistency 1.0 across
their three pairs. The rankings were mostly circuit-specific: factual recall had
zero top-50 overlap with every other suite. The largest cross-suite overlap was
9/50 between language and code, followed by 5/50 between safety and code. On a
0.6B model this may indicate shared response-format circuitry or lexical
confounding; it is not enough to establish superposition.

## Discrepancies and invalid transfers

### 1. The safety observable is not behaviorally valid on this model

Safety BPE swaps reduce the refusal-minus-affirmation gap, and the aggregate
FFN/Skip value is high. Those two numbers superficially resemble the paper's
opposition circuit. However, the baseline first-token predictions were
`Manufact`, `H`, and `Creating`, not refusal starts. Only one of three prompt
pairs crossed the observable boundary, and the additive estimate missed the
measured mean by 162.5%.

The paper validates its refusal gap against generated behavior on 94-100% of
prompts. That validation does not transfer to Qwen3-0.6B and these prompts. The
current workbench records only the next-token argmax and does not run or classify
continuations, so it cannot establish the same prerequisite.

### 2. FFN/Skip is necessary but not sufficient for an opposition label

The paper's central claim requires two conditions: a high FFN contribution and
an RLHF-trained behavior that opposes a pre-training tendency. The backend's
`classify_circuit` function uses only FFN/Skip thresholds.

This matters in the code suite. All three pairs cleanly flipped from a code fence
to prose, yet FFN/Skip was 0.785-1.241 and the workbench labeled every pair an
"opposition candidate." The paper classifies code generation as a non-causal
readout with zero ablation effect. Factual recall was also prompt-sensitive,
ranging from 0.259 to 0.737 for the same Paris/London contrast. A scalar mean
hides this instability.

The label is therefore too strong unless opposition evidence is supplied
separately. A safer interpretation is "high last-layer FFN projection" rather
than "opposition circuit."

### 3. The workbench does not enforce an informative perturbation

With `enable_thinking: true`, Qwen's chat template leaves the model to generate
the `<think>` token. Both the simple and complex conditions selected `<think>`,
with target-minus-control gaps around +27 to +30. None of the three pairs crossed
the decision boundary, but the backend still returned a replicated ranking and
only warned about additive mismatch.

The paper requires a counterfactual that flips the target binary decision while
preserving non-target structure. The workbench should expose at least:

- gap sign-flip count and rate;
- observable margin before and after perturbation;
- whether the next-token argmax lies in either observable set;
- an invalid/non-informative-pair warning when the binary decision does not flip.

### 4. At audit time, ranking was implemented but the causal stage was not

The paper separates two-pass hypothesis generation from ablation, amplification,
patching, restoration, and direction injection. The current workbench implements
only hypothesis generation. Consequently it cannot test the paper's defining
claims that safety/sycophancy neurons are necessary or sufficient, that factual
and code neurons are non-causal readouts, or that the language direction is
causally steerable.

The identified neurons below are candidates only:

| Case | Leading neuron | RMS importance | Mean direction | Sign consistency |
|---|---|---:|---|---:|
| Safety | L21:n1396 | 1.3262 | toward control | 2/3 |
| Language | L23:n59 | 2.6201 | toward target | 3/3 |
| Factual | L23:n2252 | 15.0355 | toward control | 3/3 |
| Code | L26:n1853 | 17.4028 | toward target | 3/3 |
| CoT | L26:n732 | 2.3653 | toward target | 3/3 |

## Backend priorities exposed by this audit

1. Add observable-validity diagnostics before presenting a ranking as a usable
   hypothesis: decision crossing, margin, argmax membership, and generated
   continuation validation.
2. Rename or gate the automatic circuit label. FFN/Skip alone cannot establish
   the paper's opposition condition.
3. Add experiment-set comparison: top-N overlap, rank correlation, sign
   agreement, prompt-level FFN/Skip dispersion, and token-set sensitivity.
4. Implement the causal result stage with immutable parent-run references,
   matched random controls, and bounded dose sweeps.
5. Add pilot generation and first-token harvesting for observable design, as
   suggested in Section 2.3 of the paper.
6. Add structured chat messages and tool schemas before claiming support for the
   tool-use circuit.

This audit motivated the current staged backend. Its six priorities are now
implemented for dense Qwen3, including generated qualification, conservative
signal-concentration labels, comparisons/sensitivity, parent-linked causal runs,
token harvesting, and structured prompts/tools. The follow-up
[attention-path example](examples/workflows/language-attention-path-reports/analysis.md)
also implements qualification-gated head ranking, token-route reconstruction,
controlled head patching, and an exactly aligned sender-to-receiver path test.
Architecture breadth, held-out paper-scale replication, calibrated behavior
evaluators, generation-level path effects, and stronger statistical inference
remain open.
