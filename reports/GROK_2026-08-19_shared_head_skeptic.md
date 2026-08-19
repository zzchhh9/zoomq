# Grok skeptic — shared-head story (fifth mechanism)

2026-08-19 ~01:40 BST. Claude accepted the lattice rebuttal and promoted
"one L2 head, 69% r3+r4 cells → constant that r0 inherits." Line B and C
have already returned. Line A has finished the pre-registered pair.

## p12 meters, corrected

`5ef2153` added `scripts/p12_gate.sh` only. `spread_r0_L2` / `binacc_r0_L2`
are **not** in `train.csv`. `save_eval_snapshot=false`; there is a live
`snapshot.pt` (mtime 01:35) so `critic_localize.py` can still be pointed at it.

`depth_share_0 = 0.7384` at 1200 frames is **not a p12 effect**. It is identical
to `p9_mp_zqEA_s1` at the same frame, same `eps=0.287`. At 25401 frames p9 is
0.8876 because `eps` annealed to 0.05. Comparing those two numbers as if they
were treatment vs control is the same class of error as the last four.

Clamp at **matched** 1200 frames (s1):

| | c1 | c3 | c4 |
|---|---|---|---|
| p9 zqEA | 0.0410 | 0.0081 | 0.0155 |
| p12 zqEAw | 0.0229 | 0.0033 | 0.0017 |

c4 is ~9× lower, not 16× (the 0.0269 was p9 at 25k, after clamp had drifted
up). Treatment reached r3/r4. That was never in doubt once hydra said `0.2`.

p12 s1 has a success at episode 11. Do not read it. First-success noise,
n=1, zqEA's first is usually 20–21.

## The shared-head story as stated is already dead. A weaker cousin is not.

### Architecture does not force a constant

`cond = [mids(15) | p(15) | w(15) | onehot_t(16) | onehot_r(5)]` (`zoomq.py:660`).
Round index is in the trunk input three times. `eye_l` is level-only; CQN-AS
feeds only `(low+high)/2`. Line B measured it:

- change only the round signal → L2 Q moves **25.13 atoms** (identical-cond
  control = 0.0)
- on-manifold kill of `onehot_r` and `w` → L2 stays dead (0.001787 → 0.001325),
  L0 stays healthy (16.95 → 15.96)
- GRU has not collapsed (93.6% of L2 adv-feature variance is within-round)
- isolated `step(..., h0=None)`, one position: r0 L2 spread 0.0037 vs L0 13.84.
  Recurrence across five rounds is not required to produce the flatness.

A head that *can* tell rounds apart, and does, cannot be "emitting one
constant because it cannot tell them apart." zqE L1 already showed the same
thing: 0.0025 at r0 vs 15.00 at r3, same Linear.

### Targets do not churn

Line C, 61,440 (state, cell) pairs, 12 zqEA + 10 CQN-AS snapshots: L2 target
bin change = **exactly 0.0** at every round and level. Teacher-forced
`parent_prediction(DEMO)` (`zoomq.py:733-736`). The "neighbour moves → cell
target moves" channel is inference-only (r4 parent moves 111 L2 bins / 2100
frames) and never enters the gradient.

r0 L2 target is **bit-identical** to CQN-AS on 7680/7680 pairs. Same labels:
CQN-AS L2 spread 6.79 / binacc 0.928; ZoomQ 0.0023 / 0.467. The target is not
the unlearnable object.

### What Line A actually shows (this is the part that survives)

Harness has power: CQN-AS fresh, all cells, 8000 steps → L2 0 → **3.84**
atoms, binacc **0.910**.

| run | r0 L2 spread | r0 L2 binacc |
|---|---|---|
| zqEA **fresh / all** | 0.00042 | 0.338 |
| zqEA **fresh / r0-only** | **3.49** | **0.968** |
| zqEA trained / all | 0.004 → 0.411 | 0.495 → 0.651 |
| zqEA trained / r0-only | 0.003 → **5.09** | 0.491 → 1.00 (holdout acc 0.54, holdout spread **4.50**) |
| zqEA trained / r34-only | 0.003 → 0.10 | 0.488 → **0.368** |

Pre-registered A: *r0-only opens and all-rounds does not → pollution*.
On a **fresh** critic that is exactly what happens. r34-only does not rescue
r0 and slightly hurts r0 accuracy. So the r3+r4 **loss terms** suppress L2
*magnitude*, even though the head is not round-blind and the target does not
move.

That is a different claim from the one in the table:

| said | measured |
|---|---|
| same head → same constant | head moves Q 25 atoms by round; L1 already specialises |
| CQN-AS has no rounds | also true, but CQN-AS shares the same Linear across *levels* and L2 opens; the extra ZoomQ ingredient is the **mixture of L2 losses**, not the absence of a round id |
| L0 healthy because coarse targets do not churn | L0 and L2 both have zero target churn (Line C). L0 is healthy in r3/r4 too (15.6–17.8). So "churning deep-round targets" is not why L0 survives |
| constant target opens L2 in 25 steps | consistent with an escapable fixed point. Line B's actual locus: at L2 the five bins emit the same atom distribution (`||p_j-p̄||_1 = 2.6e-4` vs 0.288 CQN-AS); residual logit contrast sits on dead-tail atoms |

Two more Line A caveats before anyone launches an arm:

1. r0-only train binacc 0.97–1.00 with holdout 0.54 is **memorising 30 cells**.
   Use holdout spread (4.5, real) not train binacc.
2. r0-only also opens *untrained* rounds' spread (fresh r3/r4 L2 → 8.8 atoms)
   while those rounds' binacc stays ~0.31. Sharing transmits **magnitude**,
   not a constant, and not accuracy. The "inherits a constant" cartoon is
   still wrong even after A.

## What I would do with this

Do not launch a "add a round signal" arm. It is already there three times
(Line B). Do not launch target-EMA / freeze (Line C). Do not read p12's
scoreboard as a test of pollution (r0 still `w=1.0`; no live `spread_r0_L2`).

If the next arm is about this at all, it is **stop sending r1–r4 L2 into the
shared `adv_head` gradient** (mask in the shipped loss, or a per-level L2
head). Line A already says a fresh critic then opens r0 L2. That is a
training intervention; one 2-seed wave, readout = `spread_r0_L2` and
`binacc_r0_L2` at 45 ep plus the 11/90 success cut. Not another `w_schedule`.

p12 stays up. Gate script should call `critic_localize.py` on `snapshot.pt`
instead of claiming columns that are not logged.
