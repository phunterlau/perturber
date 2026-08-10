# CoT-protocol analysis

This example is deliberately useful as both a replay success and an observable
failure case. The complex-minus-simple `<think>` gap changed by a mean **1.4131
logits**, with a mean FFN/Skip ratio of **0.0582**. However, all original and
perturbed prompts predicted `<think>` as the first token. The Qwen thinking chat
protocol therefore dominates the tested semantic contrast, and the additive
neuron estimate also differs from the measured change by more than 50% for each
pair. The ranking is reproducible, but it is weak evidence for complexity-specific
reasoning circuitry.

The aggregate RMS mass is concentrated in layers 27, 26, 25, 23, and 24. The
strongest replicated candidates are:

| Rank | Layer:neuron | Mean importance | RMS importance | Sign consistency | Direction |
|---:|---:|---:|---:|---:|---|
| 1 | 26:732 | 2.3583 | 2.3653 | 1.00 | toward `<think>` target |
| 2 | 27:84 | 1.4643 | 1.7035 | 1.00 | toward `<think>` target |
| 3 | 26:315 | -1.5225 | 1.5590 | 1.00 | toward direct-prose control |
| 4 | 27:47 | 0.9737 | 1.2092 | 0.67 | toward `<think>` target |
| 5 | 26:507 | 0.9646 | 1.0417 | 1.00 | toward `<think>` target |

After restarting the daemon, the replay passed all 18 required checks. All 102
aggregate, pair, and layer metrics and all 200 top-neuron importance metrics were
exactly equal. Top-100 overlap and sign agreement were 1.0, mean rank displacement
was 0, and all four stable artifact hashes matched. See the
[machine report](reports/replay-20260809T221721-6f42038ded72.json) or
[researcher report](reports/replay-20260809T221721-6f42038ded72.md).

The right follow-up is not to interpret these neurons causally. First redesign
the observable so the chat template does not force `<think>` on both sides—for
example, compare a downstream answer-token or reasoning-state score—then rerun
ranking and intervention validation.
