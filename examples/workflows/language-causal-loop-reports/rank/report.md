# Replicated observational neuron ranking

- Run: `20260810T000018-b5cfefbfd03d`
- Kind: `rank`
- Evidence stage: `replicated_ranking`

## Key results

- Observable movement mean: +23.8568.
- FFN additive prediction mean: +29.5445; FFN/Skip mean: +0.121397.
- First-token qualification: 0/3 informative pairs.
- Highest aggregate FFN mass is layer 27 (RMS mass 43.7732).
- Leading ranked unit is L23/N59 (importance RMS 2.62011, sign consistency 1.000).

## Claims

- **blocked / observable_validity:** The first-token observable was checked for binary decision crossing and argmax membership.
  - Limitation: Generated continuation behavior has not yet been evaluated.
- **blocked / replicated_ranking:** The signed ranking was aggregated across prompt pairs.
  - Limitation: Replication is observational until controlled intervention.
  - Limitation: Only informative pairs support the aggregate claim.

## Limitations

- Replicated ranking from 3 discovery prompt pairs remains observational; causal claims require intervention and broader validation.

## Recommended next steps

- Run generated-behavior qualification with a predeclared evaluator.
- Test ranked neurons with dose sweeps and matched random controls.
