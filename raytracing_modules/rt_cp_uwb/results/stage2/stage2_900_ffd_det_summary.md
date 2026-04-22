# Stage 2 Full Sweep (Deterministic)

- cases: 900
- elapsed_s: 5737.874149
- elapsed_min: 95.631236
- per_case_ms: 6375.415721
- failed: 0
- checkpoint_every: 100
- deterministic_seed_rule: case_id -> rng + injectSnr local stream

| room | n_total | n_valid | n_failed | n_los | n_nlos |
|---|---:|---:|---:|---:|---:|
| A | 300 | 300 | 0 | 28 | 272 |
| B | 300 | 300 | 0 | 25 | 275 |
| C | 300 | 300 | 0 | 26 | 274 |
