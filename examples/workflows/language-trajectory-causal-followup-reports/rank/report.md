# Replicated observational neuron ranking

- Run: `20260901T195601-ee007a7444f0`
- Kind: `rank`
- Evidence stage: `replicated_ranking`

## Key results

- Observable movement mean: +23.7845.
- FFN additive prediction mean: +26.7362; FFN/Skip mean: +0.131399.
- First-token qualification: 0/3 informative pairs.
- Highest aggregate FFN mass is layer 27 (RMS mass 40.0432).
- Leading ranked unit is L23/N59 (importance RMS 2.60582, sign consistency 1.000).

## Claims

- **blocked / observable_validity:** The first-token observable was checked for binary decision crossing and argmax membership.
  - Limitation: Generated continuation behavior has not yet been evaluated.
- **blocked / replicated_ranking:** The signed ranking was aggregated across prompt pairs.
  - Limitation: Replication is observational until controlled intervention.
  - Limitation: Only informative pairs support the aggregate claim.

## Limitations

- Replicated ranking from 2 discovery prompt pairs remains observational; causal claims require intervention and broader validation.
- Neuron ranking uses discovery pairs only; validation and held-out pairs are retained for separate evaluation.

## Recommended next steps

- Run generated-behavior qualification with a predeclared evaluator.
- Test ranked neurons with dose sweeps and matched random controls.
