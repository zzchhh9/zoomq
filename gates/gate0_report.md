# Gate 0 — per-round residual calibration (ZoomQ dyadic codec)

- Data: `/mnt/workspace/zoomq/runs/A/demo_buffer` — 54 episodes, key `action` (198, 15) float32, D=15
- Chunks: 7709 of 7709 valid non-padded starts (subsampled=False, seed=0)
- Action range observed: global [-1.0000, 1.0000]; per-dim min/max all inside [-1, 1] (see JSON)
- Discrete dims (<= 2 distinct values in the whole dataset): [13, 14]; constant dims: [14]
- RMSE definition: sqrt(mean over T and D of (recon - a)^2), per chunk (== zoomq.py skeleton_rmse)
- Chunk indexing: ep_len = len(action)-1; starts idx in [1, ep_len-T+1]; chunk=action[idx:idx+T] (replicates `scripts/analyze_q_offpath.py::sample_states`)
- Reproduce: `python gate0_residuals.py`; validate with `python gate0_residuals.py --self-test --zoomq-src <repo>/bigym_src/zoomq.py` (schedule identical to the training code to float32 precision; DP verified equal to brute force over all knot subsets)

## 1. Per-round residuals `e = a[t] - p_t`

Pooled over chunks x knots-of-round x action dims.

| round | knots | count | mean\|e\| | RMSE | p50 | p90 | p99 | p99.9 | max | w_r = 1.5·p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | [0, 15] | — | — | — | — | — | — | — | — | 1.0 (full range) |
| 1 | [7] | 115635 | 0.0449 | 0.1109 | 0.0182 | 0.1021 | 0.3863 | 1.0667 | 1.0667 | **0.5794** |
| 2 | [3, 11] | 231270 | 0.0158 | 0.0684 | 0.0054 | 0.0307 | 0.1083 | 1.0000 | 1.1429 | **0.1624** |
| 3 | [1, 5, 9, 13] | 462540 | 0.0062 | 0.0464 | 0.0018 | 0.0103 | 0.0397 | 1.0000 | 1.3333 | **0.0596** |
| 4 | [2, 4, 6, 8, 10, 12, 14] | 809445 | 0.0039 | 0.0342 | 0.0010 | 0.0071 | 0.0263 | 1.0000 | 1.0000 | **0.0395** |

**Recommended `w_schedule` = `[1.0000, 0.5794, 0.1624, 0.0596, 0.0395]`**

Same pooling but restricted to the continuous dims [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] (the binary/dead dims removed):

| round | count | mean\|e\| | RMSE | p50 | p90 | p99 | p99.9 | max | w_r = 1.5·p99 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 100217 | 0.0423 | 0.0691 | 0.0229 | 0.1055 | 0.2592 | 0.4343 | 0.9572 | **0.3887** |
| 2 | 200434 | 0.0135 | 0.0248 | 0.0070 | 0.0326 | 0.0902 | 0.2130 | 0.8521 | **0.1353** |
| 3 | 400868 | 0.0048 | 0.0107 | 0.0023 | 0.0111 | 0.0378 | 0.1128 | 0.4713 | **0.0568** |
| 4 | 701519 | 0.0032 | 0.0070 | 0.0013 | 0.0078 | 0.0265 | 0.0692 | 0.3312 | **0.0398** |

Continuous-dims-only `w_schedule` = `[1.0000, 0.3887, 0.1353, 0.0568, 0.0398]`

### Round 1 (t=7) per action dimension

| dim | mean\|e\| | RMSE | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| 0 | 0.0467 | 0.0643 | 0.0341 | 0.1056 | 0.1958 | 0.3567 | 0.4180 |
| 1 | 0.0896 | 0.1349 | 0.0451 | 0.2455 | 0.4079 | 0.5783 | 0.7133 |
| 2 | 0.0292 | 0.0516 | 0.0166 | 0.0677 | 0.1786 | 0.4764 | 0.8454 |
| 3 | 0.0325 | 0.0431 | 0.0248 | 0.0723 | 0.1245 | 0.1677 | 0.1843 |
| 4 | 0.0511 | 0.0698 | 0.0368 | 0.1150 | 0.2127 | 0.2639 | 0.3175 |
| 5 | 0.0646 | 0.0924 | 0.0412 | 0.1616 | 0.2687 | 0.3682 | 0.5551 |
| 6 | 0.0525 | 0.0726 | 0.0352 | 0.1260 | 0.2122 | 0.2669 | 0.3061 |
| 7 | 0.0442 | 0.0679 | 0.0280 | 0.1005 | 0.2448 | 0.4701 | 0.5989 |
| 8 | 0.0172 | 0.0255 | 0.0121 | 0.0363 | 0.0948 | 0.1881 | 0.2291 |
| 9 | 0.0393 | 0.0613 | 0.0206 | 0.0986 | 0.2132 | 0.2939 | 0.3581 |
| 10 | 0.0404 | 0.0596 | 0.0224 | 0.1058 | 0.1843 | 0.2220 | 0.2449 |
| 11 | 0.0212 | 0.0339 | 0.0125 | 0.0492 | 0.1340 | 0.2107 | 0.2976 |
| 12 | 0.0222 | 0.0535 | 0.0100 | 0.0487 | 0.1939 | 0.7019 | 0.9572 |
| 13 | 0.1235 | 0.3496 | 0.0000 | 0.9333 | 1.0667 | 1.0667 | 1.0667 |
| 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 2. Reconstruction error vs number of knots j

| j | round | dyadic RMSE mean | median | p90 | max | dyadic max-abs (mean / max) | optimal RMSE mean | median | p90 | max | optimal max-abs (mean / max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 0.0610 | 0.0386 | 0.1596 | 0.2844 | 0.3224 / 1.8667 | 0.0610 | 0.0386 | 0.1596 | 0.2844 | 0.3224 / 1.8667 |
| 3 | 1 | 0.0303 | 0.0136 | 0.1093 | 0.1917 | 0.2382 / 1.7500 | 0.0219 | 0.0120 | 0.0619 | 0.1419 | 0.1565 / 1.1111 |
| 5 | 2 | 0.0176 | 0.0056 | 0.0791 | 0.1213 | 0.1912 / 1.5000 | 0.0048 | 0.0040 | 0.0081 | 0.0488 | 0.0277 / 0.3251 |
| 9 | 3 | 0.0106 | 0.0030 | 0.0646 | 0.0649 | 0.1416 / 1.0000 | 0.0020 | 0.0016 | 0.0036 | 0.0173 | 0.0123 / 0.1318 |
| 16 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 / 0.0000 |

- **j\* (dyadic, mean RMSE <= 0.05) = 3**
- **j\* (optimal DP, mean RMSE <= 0.05) = 3**

Off-schedule budgets (the dyadic schedule has no such round; measured only to compare against the proposal's four-knot figure):

| j | knot set | mean RMSE | median | p90 | max | max-abs (mean / max) |
|---|---|---|---|---|---|---|
| 4 uniform | [0, 5, 10, 15] | 0.0215 | 0.0076 | 0.1000 | 0.1433 | 0.2065 / 1.6000 |
| 4 optimal DP | per-chunk | 0.0074 | 0.0062 | 0.0126 | 0.0624 | 0.0396 / 0.4062 |

## 3. Smoothness envelope

| series | mean\|Δa\| | p99\|Δa\| | mean\|Δ²a\| | p99\|Δ²a\| |
|---|---|---|---|---|
| demo chunks | 0.0129 | 0.0845 | 0.0078 | 0.0528 |
| dyadic j=2 | 0.0111 | 0.0815 | 0.0000 | 0.0000 |
| dyadic j=3 | 0.0121 | 0.0836 | 0.0009 | 0.0216 |
| dyadic j=5 | 0.0125 | 0.0826 | 0.0018 | 0.0253 |
| dyadic j=9 | 0.0127 | 0.0824 | 0.0033 | 0.0277 |
| dyadic j=16 | 0.0129 | 0.0845 | 0.0078 | 0.0528 |
| optimal j=2 | 0.0111 | 0.0815 | 0.0000 | 0.0000 |
| optimal j=3 | 0.0121 | 0.0839 | 0.0011 | 0.0232 |
| optimal j=5 | 0.0126 | 0.0808 | 0.0038 | 0.0289 |
| optimal j=9 | 0.0128 | 0.0835 | 0.0057 | 0.0474 |
| optimal j=16 | 0.0129 | 0.0845 | 0.0078 | 0.0528 |

## 4. Exec-match fractions (dyadic RMSE <= exec_match_tol = 0.05)

| round r | cumulative knots j | all chunks | no binary toggle | binary toggle |
|---|---|---|---|---|
| 0 | 2 | 0.6273 | 0.7172 | 0.0000 |
| 1 | 3 | 0.8638 | 0.9875 | 0.0000 |
| 2 | 5 | 0.8734 | 0.9985 | 0.0000 |
| 3 | 9 | 0.8817 | 1.0000 | 0.0559 |
| 4 | 16 | 1.0000 | 1.0000 | 1.0000 |

A binary dim toggles inside 0.1253 of chunks; no interpolation through knots can represent such a chunk before the toggling step itself is a knot.

Predicted `exec_n_r` = `[0.6273, 0.8638, 0.8734, 0.8817, 1.0000]`

## 5. Verdict

Measured on real demonstrations, the residual windows the dyadic codec actually needs are `[1.0000, 0.5794, 0.1624, 0.0596, 0.0395]` for rounds 0-4, i.e. 3.86x, 2.03x, 1.49x, 1.97x the proposal's illustrative w = (0.15, 0.08, 0.04, 0.02) for rounds 1-4. Dropping the binary/dead dims [13, 14] gives `[1.0000, 0.3887, 0.1353, 0.0568, 0.0398]` (2.59x, 1.69x, 1.42x, 1.99x the proposal), so the discrepancy at the coarse rounds is driven by the binary gripper dimension, which no interpolation window can shrink. Reconstruction error falls from 0.0610 (j=2) to 0.0303 (j=3), 0.0176 (j=5) and 0.0106 (j=9) under the fixed dyadic schedule; the optimal DP placement reaches 0.0219 / 0.0048 / 0.0020 at the same budgets, so the cost of the fixed schedule is +0.0129 RMSE at j=5. j* = 3 (dyadic) and 3 (optimal DP) at the 0.05 threshold, against the proposal's claim of RMSE 0.0252 at four knots (j*=4); the dyadic schedule has no 4-knot member, and its nearest budgets bracket that claim at 0.0303 (j=3) and 0.0176 (j=5). At exactly four knots the fixed uniform set [0, 5, 10, 15] gives 0.0215 and the optimal DP placement gives 0.0074, versus the proposal's 0.0252. The exec-match predicate at tol=0.05 admits 0.627, 0.864, 0.873, 0.882, 1.000 of demo chunks at rounds 0-4, which is what the `exec_n_r` counters should log.
