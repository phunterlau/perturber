# Observational layer-aware FFN coupling

- Run: `20260901T195607-237ecd6c3c64`
- Kind: `ffn_coupling`
- Evidence stage: `observational_ffn_coupling`
- Parent runs: `20260901T195601-ee007a7444f0`, `20260901T195605-8d983fa08b99`

## Key results

- Compared direct, native-local, and downstream-gradient coupling for 86016 neurons.
- Leading downstream-sensitive unit is L23/N59 (gradient importance RMS 1.23576, direct/gradient sign agreement 1.000).

## Claims

- No formal claim was emitted.

## Limitations

- Layer-aware coupling is a local first-order influence hypothesis; controlled intervention remains required.
- Candidate aggregation uses discovery pairs only; validation and held-out gradients are retained for separate diagnostics.

## Recommended next steps

- Compare top gradient-ranked neurons with same-layer random controls.
- Treat direct-versus-gradient disagreement as a hypothesis about downstream transformation.
