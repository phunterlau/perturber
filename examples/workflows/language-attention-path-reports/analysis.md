# Qwen3-0.6B language attention-path example

This is a checked-in analysis for
[`language-attention-path.yaml`](../language-attention-path.yaml). It uses the
paper-derived English/Chinese routing pattern, a pinned Qwen3-0.6B revision,
Apple MPS/float16, deterministic decoding, and explicit seeds. The six stages
require 91 logical forward passes.

## Evidence chain

| Stage | Run ID | Calls | Evidence |
|---|---|---:|---|
| FFN rank | `20260810T170133-2c34fe5560c1` | 6 | replicated observational ranking |
| Generated qualification | `20260810T170237-2c8bff05deb5` | 6 | 3/3 informative pairs |
| Attention-head rank | `20260810T170329-5539a70b30aa` | 6 | direct-logit hypothesis |
| Head patch sweep | `20260810T171018-99e985b6dd5c` | 54 | controlled causal heads |
| Token routes | `20260810T173044-b708d29e58b5` | 6 | observational route hypotheses |
| Two-stage path | `20260810T175635-f16c842a5393` | 13 | controlled causal path |

The generated Unicode-script evaluator qualified all three pairs. Attention
trace stages inherit that immutable qualification lineage; they do not fall
back to the weaker sparse-token argmax test.

## Important attention heads

The leading replicated heads were L25/H0 (`RMS=3.7556`), L18/H12 (`2.3003`),
L24/H0 (`1.4680`), and L24/H2 (`1.2890`). All four had the same effect sign on
3/3 pairs. Their roles differ:

- L18/H12 directly attends to the instruction-language token. Across all three
  perturbed prompts, `中文` carried a positive direct effect of roughly
  `+1.24` to `+1.34`; the corresponding ` English` token carried roughly
  `-0.61` to `-0.91` in the original prompts.
- L25/H0 is a late readout/routing head. Its largest token contributions often
  come from the final formatting/newline tokens and change sign with language.
- L24/H0 and L24/H2 also concentrate on end-of-prompt and thinking-boundary
  tokens. They are strong causal candidates, but their token semantics are less
  clean than L18/H12.

The token-edge decomposition reconstructs each selected pre-output-projection
head vector before any edge is reported. These edges remain observational.

## Controlled head patching

Patching Chinese-condition head outputs into the English condition produced a
clear dose response, each compared with five same-layer random draws per pair:

| Heads | Mean selected gap effect | Random absolute mean | Controlled absolute effect |
|---:|---:|---:|---:|
| 1 | +1.3053 | 0.0548 | 1.2505 |
| 4 | +15.5627 | 0.1886 | 15.3741 |
| 16 | +22.0662 | 0.4276 | 21.6386 |

At 16 heads, every English-condition observable gap crossed from negative to
positive and every argmax left its original English lead token. Capital and
science changed to Han tokens (`法国` and `纯`); arithmetic changed from `Two`
to `2`, so the sparse language gap still needs the generated-behavior caveat.
This supports a local causal role for the selected population on these prompts.
It does not show that all 16 heads are independently necessary or that the set
generalizes.

## Sender-to-receiver path

The capital pair has exact 23-to-23 token identity alignment. Patching L18/H12
at every aligned position moved the gap by `+14.2070`. Replaying only the
resulting L25/H0 receiver state at the decision position isolated a `+0.4574`
path effect, or about 1.91% of the full source-condition gap. Five matched
L18-to-L25 random paths, sampled outside the entire top-16 intervention
population, were unique within the control arm, and had absolute effects
averaging about `0.00570`.

The path therefore supports the narrow claim that some L18/H12 influence reaches
the declared first-token observable through L25/H0 in this aligned prompt pair.
It does not imply that L25/H0 is the only mediator; most of the sender effect is
carried elsewhere.

## Replay result

Fresh model loads reproduced the attention rank, intervention, and path stages.
The attention rank matched all five stable artifacts byte-for-byte (summary,
pair JSONL, layer CSV, head CSV, and safetensors). The intervention summary and
observation JSONL were exact, as were the path summary and path JSONL. Run IDs,
timestamps, jobs, and event streams are intentionally not scientific equality
targets.

## Limits and next experiment

Selection and intervention use the same three discovery pairs, while the path
claim uses one aligned pair. The observable is still a sparse first-token gap,
the model is much smaller than the paper models, and random-control comparison
is descriptive rather than a corrected significance test. The next useful
study is to freeze the head set and sender/receiver endpoints, add held-out
English/Chinese prompts with predeclared exact alignments, and measure both the
gap and generated-language behavior.
