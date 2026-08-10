# Residual-direction controllability evidence

- Run: `20260810T000025-d03e122f0816`
- Kind: `direction`
- Evidence stage: `causal_intervention`
- Parent runs: `20260810T000018-b5cfefbfd03d`, `20260810T000019-35be5cf9a10d`

## Key results

- Swept 2 layers and 2 beta values with direction norm 0.601142.
- Largest direction-minus-random absolute effect was +3.82167 at layer 24, beta=-0.05, split=discovery, condition=perturbed.

## Claims

- **supported / causal_effect:** Behavioral-direction injection was compared with norm-matched orthogonal random directions.
  - Limitation: A logit-gap response does not by itself establish generated-behavior control.
  - Limitation: Direction effects can alter unrelated behaviors and require collateral evaluation.
  - Limitation: Supported status requires at least three random-direction draws and a non-zero directional bootstrap interval; it is not a population-level significance test.

## Limitations

- Direction injection tests linear controllability, not localization to an FFN circuit.
- Parent FFN/Skip is in the paper's low-concentration readout-compatible range; successful injection remains a separate empirical question.

## Recommended next steps

- Compare the layer sweep with FFN intervention effects.
- Do not infer neuron localization from direction controllability alone.
