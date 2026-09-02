# Qwen3-0.6B paired trajectory visualization

This example asks when an English-versus-Chinese prompt perturbation becomes
decodable and whether predeclared neuron patches cause a corresponding change.
It was run on Apple MPS with the pinned Qwen3-0.6B revision and the seeds in
[`driver.yaml`](driver.yaml). All immutable runs used to generate the figure
passed artifact verification.

## What the visualization shows

The upper chart is observational. For the `capital` discovery pair, the final
Chinese-minus-English logit-gap difference is `+23.9425`. The largest single
change is the layer 27 FFN (`+11.5550`); a large earlier change appears after
layer 18 attention (`+7.1408`). These checkpoints suggest where to inspect, but
do not identify the components that caused the behavior.

The lower chart is interventional. It traces the effect of the widest declared
patch at every residual checkpoint and compares it with the mean of three
same-layer random controls. On the `capital` pair, the first non-zero decoded
effects are:

| Candidate method | First affected checkpoint | Effect there | Final effect |
| --- | --- | ---: | ---: |
| Direct readout | L21 post-FFN | +0.9763 | +10.1191 |
| Downstream gradient | L19 post-FFN | +0.8038 | +12.2131 |
| Direct/downstream overlap | L20 post-FFN | +0.4963 | +11.4613 |

Across the two discovery pairs, width-16 controlled effects are `+8.9806`,
`+11.2762`, and `+10.4539`, respectively. On the one predeclared held-out
science pair they are `+9.5857`, `+11.7954`, and `+10.3705`. All three
intervention summaries record a supported local-sufficiency claim.

## Calibrated conclusion

For these prompts and this first-token observable, layer-aware downstream
ranking finds a patch set whose effect begins two layers earlier and ends larger
than the direct-readout set. The overlap set retains most, but not all, of that
effect. This is controlled local sufficiency evidence with one held-out pair;
it is not a universal language circuit or population-level generalization.

The plot should be read as a sequence of transformations, not a conserved
flow. A residual checkpoint can make an effect more or less decodable even
when no newly patched neuron appears at that checkpoint.
