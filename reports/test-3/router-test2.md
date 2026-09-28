# Per-domain router: validation on test-2

Pre-registration: docs/METHODOLOGY.md (router validation). 156 labeled test-2 cases. Times: RX 7600, measured on test-3.

| configuration | correct /156 | /138 without numeric | s/question (test-2 mix) | s/question (test-3 mix) |
|---|---:|---:|---:|---:|
| E | 144 | 132 | 4.94 | 5.34 |
| V | 138 | 126 | 1.34 | 1.41 |
| 8b-q8-fast | 132 | 126 | 0.48 | 0.48 |
| 4b-q8-fast | 123 | 115 | 0.29 | 0.29 |
| 8b-q8-short | 144 | 133 | 11.62 | 11.62 |
| 4b-q8-think | 141 | 129 | 8.26 | 8.3 |
| jev | 144 | 135 | - | - |

| comparison | delta (points) | 95 % bootstrap CI | only A / only B | McNemar p |
|---|---:|---|---|---:|
| E - jev | +0.0 | [-3.8; 3.8] | 5 / 5 | 1.0 |
| V - jev | -3.8 | [-9.0; 1.3] | 5 / 11 | 0.2101 |
| E - 8b-q8-fast | +7.7 | [3.2; 12.2] | 13 / 1 | 0.0018 |
| E - 4b-q8-fast | +13.5 | [7.1; 19.9] | 26 / 5 | 0.0002 |
| E - 8b-q8-short | +0.0 | [-3.8; 3.8] | 4 / 4 | 1.0 |
| E - 4b-q8-think | +1.9 | [-1.9; 6.4] | 7 / 4 | 0.5488 |
| V - 8b-q8-fast | +3.8 | [0.6; 7.7] | 7 / 1 | 0.0703 |

| domain | E | V | Jev | 8b-q8-fast | 4b-q8-fast | 8b-q8-short | 4b-q8-think |
|---|---:|---:|---:|---:|---:|---:|---:|
| factual | 16/18 | 14/18 | 17/18 | 14/18 | 17/18 | 16/18 | 17/18 |
| noul_refund | 23/24 | 23/24 | 23/24 | 23/24 | 22/24 | 22/24 | 23/24 |
| numeric | 12/18 | 12/18 | 9/18 | 6/18 | 8/18 | 11/18 | 12/18 |
| robotic_style | 15/18 | 15/18 | 17/18 | 15/18 | 14/18 | 17/18 | 17/18 |
| score_urgency | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 23/24 |
| sentence | 18/18 | 14/18 | 18/18 | 14/18 | 17/18 | 18/18 | 18/18 |
| sentiment | 18/18 | 18/18 | 18/18 | 18/18 | 14/18 | 18/18 | 18/18 |
| subjective_tone | 18/18 | 18/18 | 18/18 | 18/18 | 7/18 | 18/18 | 13/18 |

E = router, V = cheap variant (see scripts/test3_router.py).
