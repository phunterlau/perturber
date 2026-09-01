# Controlled FFN intervention evidence

- Run: `20260901T195616-c5c6d61bf243`
- Kind: `intervention`
- Evidence stage: `causal_intervention`
- Parent runs: `20260901T195601-ee007a7444f0`, `20260901T195602-5dcaeecb4319`, `20260901T195605-8d983fa08b99`

## Key results

- Tested 16 neurons selected by direct structural readout with patch over 4 split/condition/dose summaries.
- Captured 2016 native checkpoint overlays against trajectory run 20260901T195605-8d983fa08b99, including selected and matched-random arms.
- Largest selected-minus-random absolute effect was +9.58566 at N=16, strength=1, split=heldout, condition=original.
- Estimated 90% causal widths: discovery/original/strength=1:N16, heldout/original/strength=1:N16.

## Claims

- **supported / sufficiency:** The patch sweep measured selected-neuron effects against same-layer matched-random controls.
  - Limitation: Causal interpretation is local to the declared prompts, observable, model revision, and doses.
  - Limitation: Generated behavior is evidence only when generation was enabled and evaluated separately.
  - Limitation: Supported status requires at least three random-control draws and a non-zero directional bootstrap interval; it is not a population-level significance test.
- **supported / generalization:** The declared intervention was evaluated on held-out prompt pairs.
  - Limitation: Held-out evidence is limited to the declared experiment set and evaluator.

## Limitations

- Intervened trajectory overlays localize where a controlled effect becomes decodable; claim status still follows qualification, matched controls, and replication.

## Recommended next steps

- Replicate the controlled effect on held-out perturbations.
- Compare direct-readout and downstream-gradient candidate sets under identical doses and controls.
- Inspect collateral observables and additivity before assigning a narrow circuit claim.
