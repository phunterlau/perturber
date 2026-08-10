# Controlled FFN intervention evidence

- Run: `20260810T000023-c1ec41ec2e7f`
- Kind: `intervention`
- Evidence stage: `causal_intervention`
- Parent runs: `20260810T000018-b5cfefbfd03d`, `20260810T000019-35be5cf9a10d`

## Key results

- Tested 20 ranked neurons with ablate over 6 split/condition/dose summaries.
- Largest selected-minus-random absolute effect was +4.41477 at N=20, strength=0, split=discovery, condition=perturbed.
- Estimated 90% causal widths: discovery/perturbed/strength=0:N20, discovery/perturbed/strength=0.5:N20.

## Claims

- **supported / necessity:** The ablate sweep measured selected-neuron effects against same-layer matched-random controls.
  - Limitation: Causal interpretation is local to the declared prompts, observable, model revision, and doses.
  - Limitation: Generated behavior is evidence only when generation was enabled and evaluated separately.
  - Limitation: Supported status requires at least three random-control draws and a non-zero directional bootstrap interval; it is not a population-level significance test.

## Limitations

- No additional run warning was recorded; method-level limits still apply.

## Recommended next steps

- Replicate the controlled effect on held-out perturbations.
- Inspect collateral observables and additivity before assigning a narrow circuit claim.
