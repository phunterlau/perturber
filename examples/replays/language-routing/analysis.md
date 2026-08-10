# Language-routing analysis

This is the stronger of the two checked-in replay examples. Across three
English-to-Chinese prompt perturbations, the Chinese-minus-English first-token
gap changed by a mean **23.8568 logits**. The original predictions were English
tokens (`The`, `Two`, `The`); the perturbed predictions began with Chinese tokens
(`法国`, `二`, `纯`). The mean FFN/Skip ratio was **0.1214**, consistent with a
routing/readout hypothesis rather than a claim that the last-layer FFN alone
causes the behavioral shift.

The aggregate RMS mass is concentrated late in the network: layers 27, 26, 25,
24, and 23 lead the ranking. The strongest replicated neuron candidates are:

| Rank | Layer:neuron | Mean importance | RMS importance | Sign consistency | Direction |
|---:|---:|---:|---:|---:|---|
| 1 | 23:59 | 2.6100 | 2.6201 | 1.00 | toward Chinese target |
| 2 | 27:52 | 1.9600 | 1.9688 | 1.00 | toward Chinese target |
| 3 | 27:22 | 1.7555 | 1.7747 | 1.00 | toward Chinese target |
| 4 | 27:248 | -1.5621 | 1.7045 | 1.00 | toward English control |
| 5 | 27:31 | -1.6232 | 1.6630 | 1.00 | toward English control |

After restarting the daemon and reloading the model, the replay passed all 18
required checks. All 102 aggregate, pair, and layer metrics and all 200 top-neuron
importance metrics were exactly equal. Top-100 overlap and sign agreement were
1.0, mean rank displacement was 0, and all four stable artifact hashes matched.
See the [machine report](reports/replay-20260809T221715-5f7c5f7b4406.json)
or [researcher report](reports/replay-20260809T221715-5f7c5f7b4406.md).

These are replicated observational candidates, not localized causal neurons.
The next scientific step is activation intervention on the leading positive and
negative candidates, followed by held-out prompt families and paraphrases.
