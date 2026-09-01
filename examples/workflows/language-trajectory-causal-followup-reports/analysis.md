# Qwen3-0.6B trajectory-guided FFN causal follow-up

This is the live Apple MPS acceptance record for
[`language-trajectory-causal-followup.yaml`](../language-trajectory-causal-followup.yaml).
It compares direct structural readout, downstream endpoint-gradient, and a
predeclared direct/downstream top-24 overlap under identical patch doses and
same-layer random controls.

## Evidence chain

| Stage | Run | Calls | Status |
|---|---|---:|---|
| Rank | `20260901T195601-ee007a7444f0` | 6 | observational, discovery-ranked |
| Generated qualification | `20260901T195602-5dcaeecb4319` | 6 | 3/3 informative |
| Native trajectory | `20260901T195605-8d983fa08b99` | 6 | observational decodability |
| Layer-aware coupling | `20260901T195607-237ecd6c3c64` | 6 forward + 6 backward | discovery-ranked hypothesis |
| Direct patch | `20260901T195616-c5c6d61bf243` | 24 | controlled local sufficiency |
| Downstream patch | `20260901T195625-99882b5900b1` | 24 | controlled local sufficiency |
| Top-pool overlap patch | `20260901T195635-b8ea6d0af19c` | 24 | controlled local sufficiency |

All seven runs pass artifact-integrity verification. Candidate aggregation uses
only `capital` and `arithmetic`; `science` is retained as a held-out evaluation
pair and does not contribute to direct or downstream selection.

## Candidate comparison

The direct and downstream top-16 sets share 12 neurons (75%). Their ordering and
remaining members differ enough to affect intervention precision. The leading
neuron under both methods is `L23:n59`; other stable shared candidates include
`L22:n349`, `L27:n52`, `L27:n22`, `L27:n39`, `L27:n30`, `L27:n53`,
`L25:n163`, `L23:n179`, `L27:n165`, `L23:n561`, and `L21:n230`.

The overlap arm is not simply the top-16 intersection. It freezes the first 16
shared identities within each method's top-24 discovery pool, then preserves
downstream rank order. The checked-in `results.json` records all 16 identities.

## Controlled patch result

The values below are selected absolute effects minus the mean absolute effect
of three same-layer random draws. Positive values favor the selected set.

| Candidate method | Width | Discovery | Held-out science |
|---|---:|---:|---:|
| Direct structural | 4 | +3.5819 | +3.0014 |
| Downstream gradient | 4 | +5.8121 | +5.1214 |
| Top-24 overlap | 4 | +5.8031 | +5.1138 |
| Direct structural | 16 | +8.9806 | +9.5857 |
| Downstream gradient | 16 | +11.1450 | +11.4689 |
| Top-24 overlap | 16 | +10.5500 | +11.0229 |

At both widths, the layer-aware downstream set is more intervention-efficient
than the direct set on discovery and the untouched held-out prompt. At width 16,
the downstream advantage is +2.1645 discovery logit-gap units and +1.8832 on the
held-out pair. The overlap arm retains most of that gain, which suggests that
agreement between the two rankings is useful but not sufficient to recover the
full downstream set.

## Where the controlled effect appears

The intervention trajectories add no logical model calls: the same patched
forward yields block-input, post-attention, and post-FFN residual checkpoints.
At width 16, the first non-zero discovery-mean effects are:

| Candidate method | First affected checkpoint | Effect there | Final effect |
|---|---|---:|---:|
| Direct structural | L21 post-FFN | +0.7485 | +9.0390 |
| Downstream gradient | L19 post-FFN | +0.8420 | +11.1700 |
| Top-24 overlap | L20 post-FFN | +0.5495 | +10.5803 |

This is a useful mechanistic distinction. The downstream-gradient set includes
an earlier causal contributor (`L19:n305`), while the direct set does not begin
changing the decoded observable until layer 21. Effects are subsequently
transformed—sometimes attenuated—before the large final-layer FFN write. The
trajectory therefore shows propagation, not a conserved flow.

## Acceptance review

The first live audit detected that the initial coupling implementation included
held-out pairs in aggregate candidate scores. Those runs were rejected as
acceptance evidence. Commit `17d9f0a` adds a regression that multiplies a
held-out activation delta by 10,000 and proves discovery candidates are
unchanged. A second pilot used an overlap pool of 250 and showed that 244
candidates overlapped, making the arm uninformative; the final driver freezes a
top-24 pool, where exactly 16 candidates overlap.

## Calibrated conclusion

For these Qwen3-0.6B prompts and this Chinese-minus-English first-token
observable, the layer-aware gradient improves causal intervention efficiency
over the fixed direct readout, and its effect starts two layers earlier. All
three selected sets strongly outperform same-layer random controls and preserve
their effect on one predeclared held-out prompt.

This is local sufficiency evidence, not a universal language circuit. There are
only two discovery pairs and one held-out pair, intervention generation was not
evaluated, and no unrelated-capability collateral observable was included. The
next scientific step is a larger frozen held-out family plus generated-language
and collateral-behavior evaluation—not a larger candidate search on these same
three prompts.
