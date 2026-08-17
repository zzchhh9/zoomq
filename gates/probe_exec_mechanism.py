#!/usr/bin/env python3
"""Is ZoomQ's stopping mechanism alive? Read it off a snapshot, on CPU, in minutes.

    python gates/probe_exec_mechanism.py --runs p7_mp_zqA_s1 p7_mp_zqS_s1 ...
    python gates/probe_exec_mechanism.py --glob 'p7_mp_*'

WHY THIS EXISTS. ZoomQ's contribution is a stopping rule that reads Q^exec_r, "the
value of stopping at round r and executing that skeleton". Shipped, that head could
not see the skeleton: dQ^exec_0/d(action) measured EXACTLY 0.0, because at round 0
`parent_prediction` is multiplied by a constant zero (root_mask) and the only other
route from the action into the conditioning is `torch.floor` in
`windowed_midpoints`, which has no gradient. `zoomq.exec_cond_skeleton=true` feeds
the skeleton in. This script decides whether that changed anything, WITHOUT waiting
for a success rate: every number below is a property of the critic, readable from
the first snapshot an arm writes.

The thresholds are not invented here. They come from measurements on 40K-frame
snapshots of the shipped build and its CQN-AS baseline (C51 atom spacing
dz = 4/50 = 0.08):

    quantity                              shipped   per-cell head   baseline   alive
    six-action sensitivity of Q^exec      0.064 dz     3.039 dz     7.218 dz   > 1 dz
    Q^exec demo-vs-random AUC             0.50-0.54       --           --      > 0.60
    across-depth spread of Q^exec         0.015        --           --      > 1 dz

The last one is what the stopping rule actually gates on: `refine iff
Delta_r > kappa*u_r`, and `u_r` is the critic's own C51 spread, whose FLOOR is one
atom (measured 0.084). A depth spread below one atom therefore cannot fire the rule
no matter what kappa is, which is why the shipped p90 of Delta/(kappa*u) never
exceeded 0.103 across 45 arms.

READ-ONLY. Loads snapshots, touches nothing. Runs on CPU by design -- the box is
shared and this needs no accelerator.
"""

import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
import torch

RUNS_ROOT = "/mnt/workspace/zoomq/runs"
DZ = 4.0 / 50  # C51 atom spacing on the [-2, 2] support


def newest_snapshot(run_dir):
    snaps = sorted(
        globmod.glob(os.path.join(run_dir, "eval_snapshots", "*.pt")),
        key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0),
    )
    return snaps[-1] if snaps else None


def load_agent(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["agent"] if isinstance(obj, dict) and "agent" in obj else obj


def demo_chunks(run_dir, T, D, n_want=192, seed=0):
    """Real demo chunks, indexed the way ReplayBuffer._sample does.

    The first transition of an episode is a dummy, so ep_len = len(action) - 1 and
    the valid, never-padded chunk starts are idx in [1, ep_len - T + 1].
    """
    rng = np.random.default_rng(seed)
    out = []
    files = sorted(globmod.glob(os.path.join(run_dir, "demo_buffer", "*.npz")))
    rng.shuffle(files)
    for f in files:
        try:
            a = np.load(f)["action"].astype(np.float32)
        except Exception:
            continue
        if a.ndim != 2 or a.shape[1] != D:
            continue
        n = len(a) - 1
        for s in range(1, n - T + 2, max(1, (n - T) // 6 or 1)):
            out.append(a[s : s + T])
            if len(out) >= n_want:
                return np.stack(out)
    return np.stack(out) if out else None


def probe(run, n_states=24, seed=0):
    run_dir = os.path.join(RUNS_ROOT, run)
    snap = newest_snapshot(run_dir)
    if snap is None:
        return {"run": run, "error": "no snapshot yet"}
    torch.manual_seed(seed)
    ag = load_agent(snap)
    cr = ag.critic.float().cpu().eval()
    net = cr.network
    T, D = cr.chunk_len, cr.act_dim
    sup = cr.support.view(1, 1, -1)
    flag = bool(getattr(cr, "exec_cond_skeleton", False))

    rgb_dim = net.value_rgb_encoder[0].in_features
    low_dim = net.value_low_dim_encoder[0].in_features

    # Synthetic observations. The absolute Q level depends on the observation, but
    # every quantity here is a CONTRAST at a fixed observation, so synthetic ones
    # are sound and avoid needing the pixel cache.
    rgbs = torch.randn(n_states, 1, rgb_dim)
    lows = torch.randn(n_states, 1, low_dim)

    demos = demo_chunks(run_dir, T, D, n_want=n_states)
    have_demos = demos is not None and len(demos) >= n_states

    sens, sens0, spread, aucs = [], [], [], []
    with torch.no_grad():
        for i in range(n_states):
            rgb, low = rgbs[i], lows[i]

            # 1. sensitivity: how much does Q^exec move across very different chunks?
            cand = [
                torch.zeros(1, T, D),
                torch.ones(1, T, D),
                -torch.ones(1, T, D),
                torch.rand(1, T, D) * 2 - 1,
                torch.rand(1, T, D) * 2 - 1,
                torch.sign(torch.randn(1, T, D)),
            ]
            qs = torch.stack(
                [(cr.forward_zoomq(rgb, low, a)["exec_probs"] * sup).sum(-1)[0]
                 for a in cand]
            )  # [6, R]
            # Per-round, then reported BOTH ways. Taking only the max over rounds
            # -- which this probe originally did -- hides the one round that
            # matters: eval executes ROUND 0 essentially always (the stopping rule
            # never fires), and under the shipped config dQ^exec_0/d(action) is
            # EXACTLY zero while rounds 1-4 do have a gradient. So a max-over-rounds
            # number rises with training on an arm whose round-0 head is still
            # structurally blind, and reads as a fix that is not there.
            per_round = (qs.max(0).values - qs.min(0).values)  # [R]
            sens.append(per_round.max().item())
            sens0.append(per_round[0].item())

            # 2. across-depth spread, on a plausible (demo) chunk when available
            base = torch.as_tensor(demos[i])[None] if have_demos else cand[3]
            qd = (cr.forward_zoomq(rgb, low, base)["exec_probs"] * sup).sum(-1)[0]
            spread.append((qd.max() - qd.min()).item())

            # 3. demo-vs-random AUC at the deepest round (a ranking, not a scale)
            if have_demos:
                q_demo = qd[-1].item()
                negs = [
                    (cr.forward_zoomq(rgb, low, torch.rand(1, T, D) * 2 - 1)
                     ["exec_probs"] * sup).sum(-1)[0, -1].item()
                    for _ in range(8)
                ]
                aucs.append(float(np.mean([q_demo > x for x in negs])))

    res = {
        "run": run,
        "snapshot": os.path.basename(snap),
        "exec_cond_skeleton": flag,
        "exec_head_in": int(net.exec_head[0].in_features),
        "n_states": n_states,
        "sensitivity_atoms": float(np.mean(sens) / DZ),
        "sensitivity_round0_atoms": float(np.mean(sens0) / DZ),
        "depth_spread_atoms": float(np.mean(spread) / DZ),
        "demo_vs_random_auc": float(np.mean(aucs)) if aucs else None,
    }
    res["verdict"] = verdict(res)
    return res


def verdict(r):
    """Three independent gates; all three must clear for the mechanism to be usable."""
    if r.get("sensitivity_atoms") is None:
        return "no data"
    fails = []
    # ROUND 0 is the gate that matters: it is the depth eval executes essentially
    # always, and it is the one the shipped conditioning leaves with an exactly
    # zero action-gradient. A healthy max-over-rounds number next to a dead round-0
    # number means rounds 1-4 carry all the sensitivity and the executed skeleton
    # is still unscored.
    if r["sensitivity_round0_atoms"] <= 1.0:
        fails.append("round-0 Q^exec barely moves with the action "
                     "(the depth eval actually executes)")
    if r["depth_spread_atoms"] <= 1.0:
        # Below one atom the C51-spread u_r floor makes the rule unfireable at ANY
        # kappa, so this is the binding one.
        fails.append("depth spread under one C51 atom: the rule cannot fire")
    if r.get("demo_vs_random_auc") is not None and r["demo_vs_random_auc"] <= 0.60:
        fails.append("cannot rank a demo above a random chunk")
    return "ALIVE" if not fails else "inert (" + "; ".join(fails) + ")"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--glob", default=None, help="e.g. 'p7_mp_*'")
    ap.add_argument("--states", type=int, default=24)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = list(args.runs)
    if args.glob:
        runs += [os.path.basename(p)
                 for p in sorted(globmod.glob(os.path.join(RUNS_ROOT, args.glob)))
                 if os.path.isdir(p)]
    runs = sorted(set(runs))
    if not runs:
        print("no runs given", file=sys.stderr)
        return 2

    torch.set_num_threads(4)
    out = []
    print(f"{'run':<18} {'skel':>5} {'sens_r0':>8} {'sens_max':>9} {'depth':>7} {'AUC':>6}  verdict")
    print(f"{'':<18} {'':>5} {'atoms':>8} {'atoms':>9} {'atoms':>7} {'':>6}")
    print("-" * 104)
    for run in runs:
        r = probe(run, n_states=args.states)
        out.append(r)
        if "error" in r:
            print(f"{run:<18} {'-':>5} {'-':>8} {'-':>9} {'-':>7} {'-':>6}  {r['error']}")
            continue
        auc = "-" if r["demo_vs_random_auc"] is None else f"{r['demo_vs_random_auc']:.3f}"
        print(f"{run:<18} {str(r['exec_cond_skeleton']):>5} "
              f"{r['sensitivity_round0_atoms']:>8.3f} {r['sensitivity_atoms']:>9.3f} "
              f"{r['depth_spread_atoms']:>7.3f} {auc:>6}  {r['verdict']}")

    print("\nsens_r0 is the load-bearing one: round 0 is what eval executes, and the")
    print("shipped conditioning leaves its action-gradient EXACTLY zero while rounds")
    print("1-4 do have one -- so sens_max can rise with training on an arm whose")
    print("executed skeleton is still unscored.")
    print("\nthresholds: sensitivity > 1 atom (shipped 0.064, per-cell head 3.039,")
    print("            CQN-AS baseline 7.218) | depth spread > 1 atom (shipped 0.015;")
    print("            below one atom the u_r floor makes the rule unfireable at any")
    print("            kappa) | demo-vs-random AUC > 0.60 (shipped 0.50-0.54)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
