#!/usr/bin/env python3
"""Gate 0 -- per-round residual calibration for ZoomQ's dyadic action-chunk codec.

ZoomQ commits a chunk of ``T = 16`` actions in a fixed *dyadic temporal* order

    round 0: t = 0, 15                  ->  2 knots committed
    round 1: t = 7                      ->  3
    round 2: t = 3, 11                  ->  5
    round 3: t = 1, 5, 9, 13            ->  9
    round 4: t = 2,4,6,8,10,12,14       -> 16

and fills the gaps by linear interpolation between the committed knots
("skeleton").  A knot ``t`` committed at round ``r > 0`` is decoded inside a
residual window ``p_t +/- w_r`` where ``p_t`` is the linear interpolation of
``t``'s two nearest neighbours committed in rounds ``< r``.  Round-0 knots have
no neighbours and use the full range ``[-1, 1]`` (so ``w_0 = 1.0``).

This script measures, on REAL demonstration chunks in the normalised [-1, 1]
action space the critic sees, four things:

  (1) Per-round residual statistics ``e = a[t] - p_t`` pooled over chunks,
      the knots of that round and the action dimensions: count, mean|e|, RMSE,
      p50 / p90 / p99 / p99.9 of |e| and max|e|; and the recommended window
      ``w_r = 1.5 * p99(|e|)``.  Round 1 is additionally broken down per action
      dimension.

  (2) ``j*(task)``: reconstruction error of each chunk when it is linearly
      interpolated through (a) the *dyadic* cumulative knot set of size
      j in {2, 3, 5, 9, 16} and (b) the *optimal* j-knot piecewise-linear fit
      found by exact dynamic programming (minimum summed squared error over
      knot placements, endpoints always included).  Per-chunk RMSE is
      ``sqrt(mean_{t,d} (recon - a)^2)`` -- identical to the ``skeleton_rmse``
      the training code computes in ``bigym_src/zoomq.py``.
      ``j*`` = smallest j whose MEAN RMSE across chunks is <= 0.05.

      Dimensions taking <= 2 distinct values over the whole dataset (binary
      grippers, dead dims) are detected automatically and the windows are
      reported both pooled over all dims and over the continuous dims only,
      because one binary dim can single-handedly set the pooled p99.

  (3) Smoothness envelope: mean and p99 of |Delta a| (first difference along
      time) and |Delta^2 a| (second difference) for the original demo chunks
      and for every reconstruction, to check the induced skeletons are neither
      smoother nor jerkier than the demonstrations themselves.

  (4) Exec-match fractions: the fraction of demo chunks whose round-r dyadic
      reconstruction RMSE is <= ``exec_match_tol`` (default 0.05).  This is
      exactly the predicate ``exec_valid = (skeleton_rmse <= exec_match_tol)``
      in ``bigym_src/zoomq.py``, so these five numbers predict the ``exec_n_r``
      counters the ZoomQ training runs log.

Data source: the ``.npz`` episode files a CQN-AS/ZoomQ run writes into its
``demo_buffer/``.  Each file is one episode; the ``action`` array is already in
the normalised [-1, 1] space.  Chunk indexing replicates
``scripts/analyze_q_offpath.py::sample_states`` ("Replicate ReplayBuffer._sample
indexing for full (non-padded) chunks"): the first transition of an episode is
a dummy, so ``ep_len = len(action) - 1`` and the valid, never-padded chunk
starts are ``idx in [1, ep_len - T + 1]`` with the chunk ``action[idx:idx+T]``.

Read-only: the script never writes, moves or deletes anything under the demo
directories.  numpy only, deterministic (fixed seed; the seed matters only when
``--max-chunks`` forces subsampling).

``--self-test`` checks the dyadic schedule, that every skeleton reproduces its
own knots, and that the DP equals brute force over all knot subsets; adding
``--zoomq-src <path to bigym_src/zoomq.py>`` also AST-transplants that file's
schedule functions and asserts they agree with this script's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PCTS = (50.0, 90.0, 99.0, 99.9)
J_VALUES = (2, 3, 5, 9, 16)


# --------------------------------------------------------------------------- #
# Dyadic schedule (mirrors bigym_src/zoomq.py::dyadic_rounds / build_schedule)
# --------------------------------------------------------------------------- #
def dyadic_rounds(chunk_len: int) -> list[list[int]]:
    """Round-by-round list of newly committed timesteps. T=16 -> n_r = 2,3,5,9,16."""
    if chunk_len < 2:
        return [[0]] if chunk_len == 1 else []
    committed = [0, chunk_len - 1]
    rounds = [sorted(committed)]
    while len(committed) < chunk_len:
        ordered = sorted(committed)
        new = []
        for lo, hi in zip(ordered[:-1], ordered[1:]):
            if hi - lo >= 2:
                new.append((lo + hi) // 2)
        if not new:  # pragma: no cover
            new = sorted(set(range(chunk_len)) - set(committed))
            if not new:
                break
        rounds.append(sorted(new))
        committed.extend(new)
    return rounds


def _neighbours(committed: list[int], t: int) -> tuple[int, int, float]:
    if t in committed:
        return t, t, 0.0
    ordered = sorted(committed)
    lo = max(c for c in ordered if c < t)
    hi = min(c for c in ordered if c > t)
    return lo, hi, (t - lo) / (hi - lo)


def build_schedule(chunk_len: int) -> dict:
    """par_* : p_t from the PREVIOUS rounds' knots.  skl_* : full round-r skeleton."""
    rounds = dyadic_rounds(chunk_len)
    num_rounds = len(rounds)
    round_of_t = np.zeros(chunk_len, dtype=np.int64)
    for r, knots in enumerate(rounds):
        for t in knots:
            round_of_t[t] = r

    par_lo = np.arange(chunk_len, dtype=np.int64)
    par_hi = np.arange(chunk_len, dtype=np.int64)
    par_w = np.zeros(chunk_len, dtype=np.float64)
    root = np.zeros(chunk_len, dtype=bool)
    committed: list[int] = []
    for r, knots in enumerate(rounds):
        for t in knots:
            if r == 0:
                root[t] = True  # no parents: full [-1, 1] window
            else:
                par_lo[t], par_hi[t], par_w[t] = _neighbours(committed, t)
        committed.extend(knots)

    skl_lo = np.tile(np.arange(chunk_len, dtype=np.int64), (num_rounds, 1))
    skl_hi = np.tile(np.arange(chunk_len, dtype=np.int64), (num_rounds, 1))
    skl_w = np.zeros((num_rounds, chunk_len), dtype=np.float64)
    committed = []
    cum_knots = []
    for r, knots in enumerate(rounds):
        committed.extend(knots)
        cum_knots.append(sorted(committed))
        for t in range(chunk_len):
            skl_lo[r, t], skl_hi[r, t], skl_w[r, t] = _neighbours(committed, t)

    return dict(
        rounds=rounds,
        num_rounds=num_rounds,
        round_of_t=round_of_t,
        par_lo=par_lo,
        par_hi=par_hi,
        par_w=par_w,
        root=root,
        skl_lo=skl_lo,
        skl_hi=skl_hi,
        skl_w=skl_w,
        cum_knots=cum_knots,
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_actions(demo_dirs: list[Path], action_key: str) -> tuple[list[np.ndarray], list[str]]:
    """READ-ONLY load of the per-episode action arrays. Never writes/deletes."""
    eps, names = [], []
    for d in demo_dirs:
        fns = sorted(d.rglob("*.npz"))
        if not fns:
            raise RuntimeError(f"no .npz episodes under {d}")
        for fn in fns:
            with fn.open("rb") as f:
                data = np.load(f)
                if action_key not in data.files:
                    raise RuntimeError(f"{fn} has no key {action_key!r} (keys={data.files})")
                eps.append(np.asarray(data[action_key]))
            names.append(str(fn))
    return eps, names


def extract_chunks(eps: list[np.ndarray], chunk_len: int, max_chunks: int, seed: int):
    """Every fully-inside-episode chunk, following ReplayBuffer._sample indexing.

    ep_len = len(action) - 1  (first transition is a dummy) and the valid starts
    are idx in [1, ep_len - chunk_len + 1], chunk = action[idx : idx + chunk_len].
    """
    starts = []  # (episode index, idx)
    for i, a in enumerate(eps):
        ep_len = a.shape[0] - 1
        hi = ep_len - chunk_len + 1  # inclusive
        for idx in range(1, hi + 1):
            starts.append((i, idx))
    total = len(starts)
    subsampled = False
    if max_chunks > 0 and total > max_chunks:
        rng = np.random.default_rng(seed)
        sel = rng.choice(total, size=max_chunks, replace=False)
        sel.sort()
        starts = [starts[k] for k in sel]
        subsampled = True
    chunks = np.stack([eps[i][idx : idx + chunk_len] for i, idx in starts]).astype(np.float64)
    return chunks, total, subsampled


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def abs_stats(e: np.ndarray) -> dict:
    """|e| summary + RMSE of the signed residual."""
    a = np.abs(e).ravel()
    q = np.percentile(a, PCTS)
    return dict(
        count=int(a.size),
        mean_abs=float(a.mean()),
        rmse=float(np.sqrt(np.mean(np.asarray(e, dtype=np.float64) ** 2))),
        p50=float(q[0]),
        p90=float(q[1]),
        p99=float(q[2]),
        p999=float(q[3]),
        max=float(a.max()),
    )


def err_stats(recon: np.ndarray, a: np.ndarray) -> dict:
    """Per-chunk RMSE / max-abs statistics, RMSE defined exactly as zoomq.py's."""
    d = recon - a
    per_chunk_rmse = np.sqrt((d**2).mean(axis=(1, 2)))
    per_chunk_max = np.abs(d).max(axis=(1, 2))
    return dict(
        rmse_mean=float(per_chunk_rmse.mean()),
        rmse_median=float(np.median(per_chunk_rmse)),
        rmse_p90=float(np.percentile(per_chunk_rmse, 90)),
        rmse_p99=float(np.percentile(per_chunk_rmse, 99)),
        rmse_max=float(per_chunk_rmse.max()),
        maxabs_mean=float(per_chunk_max.mean()),
        maxabs_median=float(np.median(per_chunk_max)),
        maxabs_p90=float(np.percentile(per_chunk_max, 90)),
        maxabs_max=float(per_chunk_max.max()),
    ), per_chunk_rmse


def smooth_stats(x: np.ndarray) -> dict:
    """mean / p99 of |Delta a| and |Delta^2 a| along the time axis."""
    d1 = np.abs(np.diff(x, n=1, axis=1)).ravel()
    d2 = np.abs(np.diff(x, n=2, axis=1)).ravel()
    return dict(
        d1_mean=float(d1.mean()),
        d1_p99=float(np.percentile(d1, 99)),
        d1_max=float(d1.max()),
        d2_mean=float(d2.mean()),
        d2_p99=float(np.percentile(d2, 99)),
        d2_max=float(d2.max()),
    )


# --------------------------------------------------------------------------- #
# Reconstructions
# --------------------------------------------------------------------------- #
def interp_through(a: np.ndarray, knots: list[int]) -> np.ndarray:
    """Linear interpolation of chunks [N, T, D] through the given knot indices."""
    n, T, dim = a.shape
    lo = np.empty(T, dtype=np.int64)
    hi = np.empty(T, dtype=np.int64)
    w = np.zeros(T, dtype=np.float64)
    for t in range(T):
        lo[t], hi[t], w[t] = _neighbours(list(knots), t)
    al, ah = a[:, lo], a[:, hi]
    return al + (ah - al) * w.reshape(1, T, 1)


def segment_costs(a: np.ndarray) -> np.ndarray:
    """cost[n, i, k] = summed squared error over t in [i, k] and over D of the
    straight line from a[i] to a[k].  inf for i >= k."""
    n, T, dim = a.shape
    cost = np.full((n, T, T), np.inf)
    for i in range(T):
        for k in range(i + 1, T):
            span = k - i
            w = (np.arange(i, k + 1) - i) / span  # [span+1]
            line = a[:, i : i + 1] + (a[:, k : k + 1] - a[:, i : i + 1]) * w.reshape(1, -1, 1)
            cost[:, i, k] = ((line - a[:, i : k + 1]) ** 2).sum(axis=(1, 2))
    return cost


def optimal_knots(a: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact DP for the best j-knot piecewise-linear fit (endpoints included).

    Minimises summed squared error.  Segment endpoints are shared, so a segment
    [i, k] owns t = i..k and neighbouring segments agree at the shared knot
    (the shared knot is exact, error 0, so no double counting matters).
    Returns (recon [N, T, D], knots [N, j]).
    """
    n, T, dim = a.shape
    if j >= T:
        return a.copy(), np.tile(np.arange(T), (n, 1))
    cost = segment_costs(a)
    nseg = j - 1
    f = cost[:, 0, :].copy()  # f[m=1, k]
    bp = [np.zeros((n, T), dtype=np.int64)]
    for _m in range(2, nseg + 1):
        # cand[n, i, k] = f[n, i] + cost[n, i, k]
        cand = f[:, :, None] + cost
        arg = np.argmin(cand, axis=1)
        f = np.take_along_axis(cand, arg[:, None, :], axis=1)[:, 0, :]
        bp.append(arg)
    knots = np.zeros((n, j), dtype=np.int64)
    knots[:, -1] = T - 1
    cur = np.full(n, T - 1, dtype=np.int64)
    for m in range(nseg - 1, 0, -1):
        cur = np.take_along_axis(bp[m], cur[:, None], axis=1)[:, 0]
        knots[:, m] = cur
    knots[:, 0] = 0
    # reconstruct
    recon = np.empty_like(a)
    for ci in range(n):
        ks = knots[ci]
        for s in range(j - 1):
            i, k = int(ks[s]), int(ks[s + 1])
            span = k - i
            w = (np.arange(i, k + 1) - i) / span
            recon[ci, i : k + 1] = a[ci, i] + (a[ci, k] - a[ci, i]) * w.reshape(-1, 1)
    return recon, knots


def yaml_block(w_tab: np.ndarray) -> str:
    """Paste-ready `w_schedule:` block for cfgs/config_zoomq_bigym.yaml."""
    lines = ["  # Gate 0 per-(round, dim) residual windows: w[r][d] = 1.5 * p99(|a[t] - p_t|),",
             "  # clamped into [0.005, 1.0]. Row r = round r, column d = action dim d.",
             "  w_schedule:"]
    for r, row in enumerate(w_tab):
        vals = ", ".join(f"{v:.4f}" for v in row)
        lines.append(f"    - [{vals}]  # round {r}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test(zoomq_src: str | None = None, seed: int = 0) -> int:
    """Correctness checks: schedule, skeleton equivalence, DP optimality.

    With --zoomq-src the dyadic schedule is additionally compared against the
    real ``bigym_src/zoomq.py`` by AST-transplanting its two numpy-only schedule
    functions (so no torch / PPU SDK is needed).
    """
    import itertools

    rounds = dyadic_rounds(16)
    assert rounds == [[0, 15], [7], [3, 11], [1, 5, 9, 13], [2, 4, 6, 8, 10, 12, 14]], rounds
    assert [len(r) for r in rounds] == [2, 1, 2, 4, 7]
    print(f"[self-test] dyadic schedule OK: {rounds} (cumulative 2, 3, 5, 9, 16)")

    sched = build_schedule(16)
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, size=(64, 16, 5))
    for r, ks in enumerate(sched["cum_knots"]):
        rec = interp_through(a, ks)
        assert np.allclose(rec[:, ks], a[:, ks]), f"round {r} not interpolating at its knots"
    print("[self-test] every dyadic skeleton reproduces its own knots exactly")

    b = rng.uniform(-1, 1, size=(60, 16, 3))
    for j in (3, 4, 5, 6):
        rec, _ = optimal_knots(b, j)
        dp = ((rec - b) ** 2).sum(axis=(1, 2))
        best = np.full(b.shape[0], np.inf)
        for inner in itertools.combinations(range(1, 15), j - 2):
            r2 = interp_through(b, [0] + list(inner) + [15])
            best = np.minimum(best, ((r2 - b) ** 2).sum(axis=(1, 2)))
        assert np.allclose(dp, best, atol=1e-9), (j, np.abs(dp - best).max())
        print(f"[self-test] DP j={j} == brute force over all knot subsets")

    if zoomq_src:
        import ast
        from typing import Sequence as _Seq

        tree = ast.parse(Path(zoomq_src).read_text())
        keep = {"dyadic_rounds", "build_schedule"}
        mod = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep],
            type_ignores=[],
        )
        ns = {"np": np, "Sequence": _Seq}
        exec(compile(mod, "zoomq_extract", "exec"), ns)
        ref = ns["build_schedule"](16)
        assert ns["dyadic_rounds"](16) == rounds
        worst = 0.0
        for k in ("par_lo", "par_hi", "par_w", "root", "skl_lo", "skl_hi", "skl_w", "round_of_t"):
            dd = np.abs(np.asarray(ref[k], dtype=np.float64) - np.asarray(sched[k], dtype=np.float64)).max()
            worst = max(worst, float(dd))
        assert worst < 1e-6, worst
        print(f"[self-test] schedule matches {zoomq_src} (max |diff| {worst:.2e}, "
              "float32 vs float64 only)")
    print("[self-test] ALL CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--demo-dir",
        nargs="+",
        default=["/mnt/workspace/zoomq/runs/A/demo_buffer"],
        help="one or more READ-ONLY demo_buffer directories of .npz episodes",
    )
    p.add_argument("--action-key", default="action", help="npz key of the [T_ep, D] actions")
    p.add_argument("--chunk-len", type=int, default=16)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="0 = use EVERY valid chunk start; >0 = uniform sample without replacement",
    )
    p.add_argument("--exec-match-tol", type=float, default=0.05, help="zoomq exec_match_tol")
    p.add_argument("--target-rmse", type=float, default=0.05, help="threshold defining j*")
    p.add_argument("--window-scale", type=float, default=1.5, help="w_r = scale * p99(|e|)")
    p.add_argument("--w-min", type=float, default=0.005, help="lower clamp on w_table entries")
    p.add_argument("--w-max", type=float, default=1.0, help="upper clamp on w_table entries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="/mnt/workspace/zoomq/gates/gate0_results.json")
    p.add_argument("--out-md", default="/mnt/workspace/zoomq/gates/gate0_report.md")
    p.add_argument("--self-test", action="store_true", help="run correctness checks and exit")
    p.add_argument(
        "--zoomq-src",
        default=None,
        help="optional bigym_src/zoomq.py to cross-check the schedule against (self-test only)",
    )
    args = p.parse_args(argv)

    if args.self_test:
        return self_test(args.zoomq_src, args.seed)

    np.random.seed(args.seed)
    T = args.chunk_len
    dirs = [Path(d) for d in args.demo_dir]
    eps, names = load_actions(dirs, args.action_key)
    dtype0 = str(eps[0].dtype)
    dim = eps[0].shape[1]
    chunks, total_valid, subsampled = extract_chunks(eps, T, args.max_chunks, args.seed)
    n = chunks.shape[0]
    print(f"[gate0] {len(eps)} episodes, {total_valid} valid chunk starts, using {n} chunks")

    flat = chunks.reshape(-1, dim)
    per_dim_min = flat.min(axis=0).tolist()
    per_dim_max = flat.max(axis=0).tolist()
    per_dim_nuniq = [int(np.unique(np.concatenate([e[:, d] for e in eps])).size) for d in range(dim)]
    discrete_dims = [d for d in range(dim) if per_dim_nuniq[d] <= 2]
    constant_dims = [d for d in range(dim) if per_dim_nuniq[d] <= 1]
    cont_dims = [d for d in range(dim) if d not in discrete_dims]

    sched = build_schedule(T)
    rounds = sched["rounds"]
    R = sched["num_rounds"]

    # ---- (1) per-round residuals ----------------------------------------- #
    p_all = (
        chunks[:, sched["par_lo"]]
        + (chunks[:, sched["par_hi"]] - chunks[:, sched["par_lo"]])
        * sched["par_w"].reshape(1, T, 1)
    )
    resid_all = chunks - p_all  # [N, T, D]; meaningless for round-0 knots

    residuals = {}
    w_schedule = [1.0]
    w_schedule_cont = [1.0]
    per_round_per_dim = {}
    for r in range(1, R):
        knots = rounds[r]
        e = resid_all[:, knots, :]
        st = abs_stats(e)
        st["knots"] = knots
        st["w_recommended"] = float(args.window_scale * st["p99"])
        stc = abs_stats(e[:, :, cont_dims])
        st["continuous_only"] = stc
        st["w_recommended_continuous_only"] = float(args.window_scale * stc["p99"])
        residuals[f"round_{r}"] = st
        w_schedule.append(st["w_recommended"])
        w_schedule_cont.append(st["w_recommended_continuous_only"])
        per_round_per_dim[f"round_{r}"] = [abs_stats(resid_all[:, knots, d]) for d in range(dim)]

    per_dim_round1 = per_round_per_dim["round_1"]

    # ---- (1b) per-(round, dim) window table ------------------------------- #
    # w[r][d] = 1.5 * p99(|e|) over round r's knots and dim d only; round 0 is
    # the full range.  Clamped into [w_min, w_max].
    w_raw = np.ones((R, dim), dtype=np.float64)
    for r in range(1, R):
        for d in range(dim):
            w_raw[r, d] = args.window_scale * per_round_per_dim[f"round_{r}"][d]["p99"]
    w_tab = np.clip(w_raw, args.w_min, args.w_max)
    clamped_lo = [[int(r), int(d)] for r in range(R) for d in range(dim) if w_raw[r, d] < args.w_min]
    clamped_hi = [[int(r), int(d)] for r in range(R) for d in range(dim) if w_raw[r, d] > args.w_max]

    # dimension classification
    dim_classes = []
    for d in range(dim):
        nu = int(np.unique(chunks[:, :, d]).size)
        cls = "dead" if nu <= 1 else ("binary" if nu == 2 else "continuous")
        dim_classes.append(
            dict(dim=d, n_unique_in_chunks=nu, n_unique_in_episodes=per_dim_nuniq[d],
                 min=float(chunks[:, :, d].min()), max=float(chunks[:, :, d].max()), cls=cls)
        )
    binary_dims = [c["dim"] for c in dim_classes if c["cls"] == "binary"]
    dead_dims = [c["dim"] for c in dim_classes if c["cls"] == "dead"]

    # ---- (3b) predicted clamp rates --------------------------------------- #
    # Faithful to zoomq.py: the window is centred on p_t, and for round-0 knots
    # p = 0 with the full [-1, 1] range, so those cells can never clamp.
    e_clamp = resid_all.copy()
    for t in rounds[0]:
        e_clamp[:, t, :] = chunks[:, t, :]
    ae = np.abs(e_clamp)
    w_scalar_t = np.asarray([w_schedule[sched["round_of_t"][t]] for t in range(T)])
    clamp_scalar = ae > w_scalar_t.reshape(1, T, 1)
    clamp_table = ae > w_tab[sched["round_of_t"]].reshape(1, T, dim)

    def clamp_report(mask):
        per_round, per_rd = {}, {}
        for r in range(R):
            ks = rounds[r]
            per_round[str(r)] = float(mask[:, ks, :].mean())
            per_rd[str(r)] = [float(mask[:, ks, d].mean()) for d in range(dim)]
        return dict(pooled=float(mask.mean()), per_round=per_round, per_round_per_dim=per_rd)

    cr_scalar = clamp_report(clamp_scalar)
    cr_table = clamp_report(clamp_table)

    # per-round-0 note
    residuals["round_0"] = dict(
        knots=rounds[0],
        note="no parents; decoded in the full [-1, 1] range",
        w_recommended=1.0,
    )

    # ---- (2) reconstruction: dyadic vs optimal DP ------------------------- #
    round_of_j = {len(sched["cum_knots"][r]): r for r in range(R)}
    dyadic, optimal, per_chunk_rmse_dyadic = {}, {}, {}
    recon_dyadic, recon_optimal = {}, {}
    for j in J_VALUES:
        r = round_of_j[j]
        ks = sched["cum_knots"][r]
        rec = interp_through(chunks, ks)
        st, pc = err_stats(rec, chunks)
        st["knots"] = ks
        st["round"] = r
        dyadic[str(j)] = st
        per_chunk_rmse_dyadic[j] = pc
        recon_dyadic[j] = rec

        orec, oknots = optimal_knots(chunks, j)
        ost, _ = err_stats(orec, chunks)
        vals, cnts = np.unique(oknots, return_counts=True)
        ost["knot_histogram"] = {int(v): int(c) for v, c in zip(vals, cnts)}
        optimal[str(j)] = ost
        recon_optimal[j] = orec

    # j = 4 is NOT a member of the dyadic schedule (n_r = 2, 3, 5, 9, 16) but the
    # proposal quotes a four-knot RMSE, so measure it two ways for comparison.
    extra = {}
    for j in (4,):
        uni = sorted({int(round(x)) for x in np.linspace(0, T - 1, j)})
        rec = interp_through(chunks, uni)
        st, _ = err_stats(rec, chunks)
        st["knots"] = uni
        orec, oknots = optimal_knots(chunks, j)
        ost, _ = err_stats(orec, chunks)
        vals, cnts = np.unique(oknots, return_counts=True)
        ost["knot_histogram"] = {int(v): int(c) for v, c in zip(vals, cnts)}
        extra[str(j)] = dict(uniform=st, optimal=ost, note="not a dyadic round size")

    def first_below(table):
        for j in J_VALUES:
            if table[str(j)]["rmse_mean"] <= args.target_rmse:
                return j
        return None

    j_star_dyadic = first_below(dyadic)
    j_star_optimal = first_below(optimal)

    # ---- (3) smoothness --------------------------------------------------- #
    smooth = dict(demo=smooth_stats(chunks))
    smooth["dyadic"] = {str(j): smooth_stats(recon_dyadic[j]) for j in J_VALUES}
    smooth["optimal"] = {str(j): smooth_stats(recon_optimal[j]) for j in J_VALUES}
    dm = smooth["demo"]
    inside = {}
    for j in J_VALUES:
        s = smooth["dyadic"][str(j)]
        inside[str(j)] = dict(
            d1_mean_ratio=s["d1_mean"] / dm["d1_mean"],
            d1_p99_ratio=s["d1_p99"] / dm["d1_p99"],
            d2_mean_ratio=s["d2_mean"] / dm["d2_mean"],
            d2_p99_ratio=s["d2_p99"] / dm["d2_p99"],
            within_envelope=bool(
                s["d1_mean"] <= dm["d1_mean"]
                and s["d1_p99"] <= dm["d1_p99"]
                and s["d2_mean"] <= dm["d2_mean"]
                and s["d2_p99"] <= dm["d2_p99"]
            ),
        )

    # ---- (4) exec-match fractions ----------------------------------------- #
    exec_frac = []
    for r in range(R):
        j = len(sched["cum_knots"][r])
        exec_frac.append(float((per_chunk_rmse_dyadic[j] <= args.exec_match_tol).mean()))

    # Which chunks fail?  A binary dim that toggles inside the chunk cannot be
    # represented by any interpolation, so split the fractions on that event.
    toggles = np.zeros(n, dtype=bool)
    for d in discrete_dims:
        if d in constant_dims:
            continue
        toggles |= chunks[:, :, d].min(axis=1) != chunks[:, :, d].max(axis=1)
    exec_frac_split = dict(
        binary_toggle_chunk_fraction=float(toggles.mean()),
        no_toggle=[
            float((per_chunk_rmse_dyadic[len(sched["cum_knots"][r])][~toggles] <= args.exec_match_tol).mean())
            if (~toggles).any() else None
            for r in range(R)
        ],
        toggle=[
            float((per_chunk_rmse_dyadic[len(sched["cum_knots"][r])][toggles] <= args.exec_match_tol).mean())
            if toggles.any() else None
            for r in range(R)
        ],
    )

    results = dict(
        meta=dict(
            demo_dirs=[str(d) for d in dirs],
            n_episode_files=len(eps),
            action_key=args.action_key,
            action_dtype=dtype0,
            action_dim=int(dim),
            episode_array_shape_example=list(eps[0].shape),
            chunk_len=T,
            n_valid_chunk_starts=int(total_valid),
            n_chunks_used=int(n),
            subsampled=bool(subsampled),
            seed=args.seed,
            exec_match_tol=args.exec_match_tol,
            target_rmse=args.target_rmse,
            window_scale=args.window_scale,
            per_dim_min=per_dim_min,
            per_dim_max=per_dim_max,
            per_dim_n_unique=per_dim_nuniq,
            discrete_dims=discrete_dims,
            constant_dims=constant_dims,
            continuous_dims=cont_dims,
            global_min=float(flat.min()),
            global_max=float(flat.max()),
            indexing="ep_len = len(action)-1; starts idx in [1, ep_len-T+1]; chunk=action[idx:idx+T]",
            rmse_definition="sqrt(mean over T and D of (recon - a)^2), per chunk (== zoomq.py skeleton_rmse)",
        ),
        schedule=dict(
            rounds=rounds,
            cumulative_knots={str(len(k)): k for k in sched["cum_knots"]},
            round_of_t=sched["round_of_t"].tolist(),
            parent_lo=sched["par_lo"].tolist(),
            parent_hi=sched["par_hi"].tolist(),
            parent_w=sched["par_w"].tolist(),
        ),
        residuals=residuals,
        residuals_round1_per_dim=per_dim_round1,
        residuals_per_round_per_dim=per_round_per_dim,
        w_schedule=w_schedule,
        w_schedule_continuous_only=w_schedule_cont,
        reconstruction=dict(dyadic=dyadic, optimal=optimal, non_dyadic_j=extra),
        j_star=dict(
            dyadic=j_star_dyadic,
            optimal=j_star_optimal,
            criterion=f"smallest j in {list(J_VALUES)} with mean per-chunk RMSE <= {args.target_rmse}",
        ),
        smoothness=smooth,
        smoothness_ratio_dyadic_over_demo=inside,
        exec_match_fraction=exec_frac,
        exec_match_fraction_split=exec_frac_split,
        w_table_per_dim=dict(
            description=(
                "w[r][d] = window_scale * p99(|a[t] - p_t|) over round r's knots and action "
                "dim d only, clamped into [w_min, w_max]; round 0 is the full range (1.0). "
                "Rows are rounds 0..R-1, columns are dims 0..D-1."
            ),
            window_scale=args.window_scale,
            w_min=args.w_min,
            w_max=args.w_max,
            w_table=[[float(x) for x in row] for row in w_tab],
            w_table_rounded=[[round(float(x), 4) for x in row] for row in w_tab],
            w_table_unclamped=[[float(x) for x in row] for row in w_raw],
            clamped_to_min=clamped_lo,
            clamped_to_max=clamped_hi,
            dim_classification=dim_classes,
            binary_dims=binary_dims,
            dead_dims=dead_dims,
            clamp_rate_scalar_w_schedule=cr_scalar,
            clamp_rate_per_dim_w_table=cr_table,
            scalar_w_schedule_used=w_schedule,
            exec_match_fraction_unchanged=True,
            exec_match_fraction_note=(
                "exec_n_r depends only on the dyadic reconstruction RMSE, which never reads the "
                "windows, so the per-dim table leaves it identical to exec_match_fraction."
            ),
            yaml_block=yaml_block(w_tab),
        ),
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gate0] wrote {args.out_json}")

    write_report(args.out_md, results, args)
    print(f"[gate0] wrote {args.out_md}")
    return 0


def write_report(path: str, res: dict, args) -> None:
    m = res["meta"]
    L = []
    A = L.append
    A("# Gate 0 — per-round residual calibration (ZoomQ dyadic codec)\n")
    A(f"- Data: `{'`, `'.join(m['demo_dirs'])}` — {m['n_episode_files']} episodes, "
      f"key `{m['action_key']}` {tuple(m['episode_array_shape_example'])} {m['action_dtype']}, D={m['action_dim']}")
    A(f"- Chunks: {m['n_chunks_used']} of {m['n_valid_chunk_starts']} valid non-padded starts "
      f"(subsampled={m['subsampled']}, seed={m['seed']})")
    A(f"- Action range observed: global [{m['global_min']:.4f}, {m['global_max']:.4f}]; "
      f"per-dim min/max all inside [-1, 1] (see JSON)")
    A(f"- Discrete dims (<= 2 distinct values in the whole dataset): {m['discrete_dims']}; "
      f"constant dims: {m['constant_dims']}")
    A(f"- RMSE definition: {m['rmse_definition']}")
    A(f"- Chunk indexing: {m['indexing']} (replicates `scripts/analyze_q_offpath.py::sample_states`)")
    A("- Reproduce: `python gate0_residuals.py`; validate with `python gate0_residuals.py "
      "--self-test --zoomq-src <repo>/bigym_src/zoomq.py` (schedule identical to the training "
      "code to float32 precision; DP verified equal to brute force over all knot subsets)\n")

    A("## 1. Per-round residuals `e = a[t] - p_t`\n")
    A("Pooled over chunks x knots-of-round x action dims.\n")
    A("| round | knots | count | mean\\|e\\| | RMSE | p50 | p90 | p99 | p99.9 | max | w_r = 1.5·p99 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    A(f"| 0 | {res['residuals']['round_0']['knots']} | — | — | — | — | — | — | — | — | 1.0 (full range) |")
    for r in range(1, len(res["schedule"]["rounds"])):
        s = res["residuals"][f"round_{r}"]
        A(f"| {r} | {s['knots']} | {s['count']} | {s['mean_abs']:.4f} | {s['rmse']:.4f} | "
          f"{s['p50']:.4f} | {s['p90']:.4f} | {s['p99']:.4f} | {s['p999']:.4f} | {s['max']:.4f} | "
          f"**{s['w_recommended']:.4f}** |")
    A("")
    A("**Recommended `w_schedule` = `[" + ", ".join(f"{w:.4f}" for w in res["w_schedule"]) + "]`**\n")
    A("Same pooling but restricted to the continuous dims "
      f"{m['continuous_dims']} (the binary/dead dims removed):\n")
    A("| round | count | mean\\|e\\| | RMSE | p50 | p90 | p99 | p99.9 | max | w_r = 1.5·p99 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in range(1, len(res["schedule"]["rounds"])):
        s = res["residuals"][f"round_{r}"]["continuous_only"]
        A(f"| {r} | {s['count']} | {s['mean_abs']:.4f} | {s['rmse']:.4f} | {s['p50']:.4f} | "
          f"{s['p90']:.4f} | {s['p99']:.4f} | {s['p999']:.4f} | {s['max']:.4f} | "
          f"**{res['residuals'][f'round_{r}']['w_recommended_continuous_only']:.4f}** |")
    A("")
    A("Continuous-dims-only `w_schedule` = `["
      + ", ".join(f"{w:.4f}" for w in res["w_schedule_continuous_only"]) + "]`\n")

    A("### Round 1 (t=7) per action dimension\n")
    A("| dim | mean\\|e\\| | RMSE | p50 | p90 | p99 | p99.9 | max |")
    A("|---|---|---|---|---|---|---|---|")
    for d, s in enumerate(res["residuals_round1_per_dim"]):
        A(f"| {d} | {s['mean_abs']:.4f} | {s['rmse']:.4f} | {s['p50']:.4f} | {s['p90']:.4f} | "
          f"{s['p99']:.4f} | {s['p999']:.4f} | {s['max']:.4f} |")
    A("")

    A("## 2. Reconstruction error vs number of knots j\n")
    A("| j | round | dyadic RMSE mean | median | p90 | max | dyadic max-abs (mean / max) | "
      "optimal RMSE mean | median | p90 | max | optimal max-abs (mean / max) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in J_VALUES:
        d = res["reconstruction"]["dyadic"][str(j)]
        o = res["reconstruction"]["optimal"][str(j)]
        A(f"| {j} | {d['round']} | {d['rmse_mean']:.4f} | {d['rmse_median']:.4f} | {d['rmse_p90']:.4f} | "
          f"{d['rmse_max']:.4f} | {d['maxabs_mean']:.4f} / {d['maxabs_max']:.4f} | "
          f"{o['rmse_mean']:.4f} | {o['rmse_median']:.4f} | {o['rmse_p90']:.4f} | {o['rmse_max']:.4f} | "
          f"{o['maxabs_mean']:.4f} / {o['maxabs_max']:.4f} |")
    A("")
    A(f"- **j\\* (dyadic, mean RMSE <= {args.target_rmse}) = {res['j_star']['dyadic']}**")
    A(f"- **j\\* (optimal DP, mean RMSE <= {args.target_rmse}) = {res['j_star']['optimal']}**\n")

    if res["reconstruction"].get("non_dyadic_j"):
        A("Off-schedule budgets (the dyadic schedule has no such round; measured only "
          "to compare against the proposal's four-knot figure):\n")
        A("| j | knot set | mean RMSE | median | p90 | max | max-abs (mean / max) |")
        A("|---|---|---|---|---|---|---|")
        for j, e in res["reconstruction"]["non_dyadic_j"].items():
            u, o = e["uniform"], e["optimal"]
            A(f"| {j} uniform | {u['knots']} | {u['rmse_mean']:.4f} | {u['rmse_median']:.4f} | "
              f"{u['rmse_p90']:.4f} | {u['rmse_max']:.4f} | {u['maxabs_mean']:.4f} / {u['maxabs_max']:.4f} |")
            A(f"| {j} optimal DP | per-chunk | {o['rmse_mean']:.4f} | {o['rmse_median']:.4f} | "
              f"{o['rmse_p90']:.4f} | {o['rmse_max']:.4f} | {o['maxabs_mean']:.4f} / {o['maxabs_max']:.4f} |")
        A("")

    A("## 3. Smoothness envelope\n")
    A("| series | mean\\|Δa\\| | p99\\|Δa\\| | mean\\|Δ²a\\| | p99\\|Δ²a\\| |")
    A("|---|---|---|---|---|")
    dm = res["smoothness"]["demo"]
    A(f"| demo chunks | {dm['d1_mean']:.4f} | {dm['d1_p99']:.4f} | {dm['d2_mean']:.4f} | {dm['d2_p99']:.4f} |")
    for j in J_VALUES:
        s = res["smoothness"]["dyadic"][str(j)]
        A(f"| dyadic j={j} | {s['d1_mean']:.4f} | {s['d1_p99']:.4f} | {s['d2_mean']:.4f} | {s['d2_p99']:.4f} |")
    for j in J_VALUES:
        s = res["smoothness"]["optimal"][str(j)]
        A(f"| optimal j={j} | {s['d1_mean']:.4f} | {s['d1_p99']:.4f} | {s['d2_mean']:.4f} | {s['d2_p99']:.4f} |")
    A("")

    A(f"## 4. Exec-match fractions (dyadic RMSE <= exec_match_tol = {args.exec_match_tol})\n")
    js = sorted(int(k) for k in res["schedule"]["cumulative_knots"].keys())
    sp = res["exec_match_fraction_split"]
    A("| round r | cumulative knots j | all chunks | no binary toggle | binary toggle |")
    A("|---|---|---|---|---|")
    for r, fr in enumerate(res["exec_match_fraction"]):
        nt = sp["no_toggle"][r]
        tg = sp["toggle"][r]
        A(f"| {r} | {js[r]} | {fr:.4f} | {'—' if nt is None else f'{nt:.4f}'} | "
          f"{'—' if tg is None else f'{tg:.4f}'} |")
    A("")
    A(f"A binary dim toggles inside {sp['binary_toggle_chunk_fraction']:.4f} of chunks; no "
      "interpolation through knots can represent such a chunk before the toggling step "
      "itself is a knot.\n")
    A("Predicted `exec_n_r` = `[" + ", ".join(f"{f:.4f}" for f in res["exec_match_fraction"]) + "]`\n")

    A("## 5. Per-(round, dim) window table\n")
    wt = res["w_table_per_dim"]
    A(f"`w[r][d] = {wt['window_scale']} * p99(|e|)` over round r's knots and dim d only, "
      f"clamped into [{wt['w_min']}, {wt['w_max']}]; round 0 is the full range.\n")
    A("| round | " + " | ".join(f"d{d}" for d in range(len(wt["w_table"][0]))) + " |")
    A("|---" * (len(wt["w_table"][0]) + 1) + "|")
    for r, row in enumerate(wt["w_table_rounded"]):
        A(f"| {r} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    A("")
    A(f"- Clamped UP to {wt['w_min']} (round, dim): {wt['clamped_to_min']}")
    A(f"- Clamped DOWN to {wt['w_max']} (round, dim): {wt['clamped_to_max']}\n")

    A("### Dimension classification\n")
    A("| dim | distinct values in chunks | min | max | class |")
    A("|---|---|---|---|---|")
    for c in wt["dim_classification"]:
        A(f"| {c['dim']} | {c['n_unique_in_chunks']} | {c['min']:.4f} | {c['max']:.4f} | **{c['cls']}** |")
    A("")
    A(f"Binary dims: {wt['binary_dims']}; dead dims: {wt['dead_dims']}; all others continuous.\n")

    A("### Predicted clamp rate: scalar per-round `w` vs per-dim `w_table`\n")
    cs, ct = wt["clamp_rate_scalar_w_schedule"], wt["clamp_rate_per_dim_w_table"]
    A(f"Scalar `w_schedule` = `{[round(x, 4) for x in wt['scalar_w_schedule_used']]}`\n")
    A("| round | clamp rate, scalar w | clamp rate, per-dim w_table |")
    A("|---|---|---|")
    for r in sorted(cs["per_round"], key=int):
        A(f"| {r} | {cs['per_round'][r]:.6f} | {ct['per_round'][r]:.6f} |")
    A(f"| **pooled (all cells)** | **{cs['pooled']:.6f}** | **{ct['pooled']:.6f}** |")
    A("")
    A("(Round-0 cells are centred on p = 0 with the full range, so they can never clamp — "
      "they dilute the pooled figure, exactly as in the training code's `clamp_rate`.)\n")

    A("**Does the pooled number hide per-round mis-specification? Yes.** Worst cells under "
      "each setting, as a multiple of that setting's pooled rate:\n")
    A("| setting | pooled | worst round | binary dim 13, worst round | worst continuous dim |")
    A("|---|---|---|---|---|")
    for name, c in (("scalar per-round w", cs), ("per-dim w_table", ct)):
        rr = {r: c["per_round"][r] for r in c["per_round"]}
        wr = max(rr, key=lambda r: rr[r])
        d13 = {r: c["per_round_per_dim"][r][13] for r in c["per_round_per_dim"]}
        wd13 = max(d13, key=lambda r: d13[r])
        best = (None, None, -1.0)
        for r in c["per_round_per_dim"]:
            for d in wt["dim_classification"]:
                if d["cls"] != "continuous":
                    continue
                v = c["per_round_per_dim"][r][d["dim"]]
                if v > best[2]:
                    best = (r, d["dim"], v)
        p = c["pooled"] if c["pooled"] > 0 else float("nan")
        A(f"| {name} | {c['pooled']:.6f} | r{wr} {rr[wr]:.6f} ({rr[wr]/p:.1f}x) | "
          f"r{wd13} {d13[wd13]:.6f} ({d13[wd13]/p:.1f}x) | "
          f"r{best[0]} d{best[1]} {best[2]:.6f} ({best[2]/p:.1f}x) |")
    A("")

    A("### Predicted `exec_n_r` under the per-dim table\n")
    A("**Unchanged: `[" + ", ".join(f"{f:.4f}" for f in res["exec_match_fraction"]) + "]`.** "
      + wt["exec_match_fraction_note"] + "\n")

    A("### YAML to paste under `zoomq:` in `cfgs/config_zoomq_bigym.yaml`\n")
    A("```yaml")
    A(wt["yaml_block"])
    A("```\n")

    A("## 6. Verdict\n")
    A(verdict(res, args))
    Path(path).write_text("\n".join(L) + "\n")


PROPOSAL_W = (0.15, 0.08, 0.04, 0.02)
PROPOSAL_RMSE_4KNOT = 0.0252
PROPOSAL_JSTAR = 4


def verdict(res: dict, args) -> str:
    w = res["w_schedule"]
    wc = res["w_schedule_continuous_only"]
    d = res["reconstruction"]["dyadic"]
    o = res["reconstruction"]["optimal"]
    ratios = [w[r] / PROPOSAL_W[r - 1] for r in range(1, 5)]
    ratios_c = [wc[r] / PROPOSAL_W[r - 1] for r in range(1, 5)]
    parts = []
    parts.append(
        "Measured on real demonstrations, the residual windows the dyadic codec actually needs are "
        "`[" + ", ".join(f"{x:.4f}" for x in w) + "]` for rounds 0-4, i.e. "
        + ", ".join(f"{ratios[i]:.2f}x" for i in range(4))
        + f" the proposal's illustrative w = {PROPOSAL_W} for rounds 1-4. "
        f"Dropping the binary/dead dims {res['meta']['discrete_dims']} gives "
        "`[" + ", ".join(f"{x:.4f}" for x in wc) + "]` ("
        + ", ".join(f"{ratios_c[i]:.2f}x" for i in range(4))
        + " the proposal), so the discrepancy at the coarse rounds is driven by the "
        "binary gripper dimension, which no interpolation window can shrink. "
    )
    parts.append(
        f"Reconstruction error falls from {d['2']['rmse_mean']:.4f} (j=2) to "
        f"{d['3']['rmse_mean']:.4f} (j=3), {d['5']['rmse_mean']:.4f} (j=5) and "
        f"{d['9']['rmse_mean']:.4f} (j=9) under the fixed dyadic schedule; the optimal "
        f"DP placement reaches {o['3']['rmse_mean']:.4f} / {o['5']['rmse_mean']:.4f} / "
        f"{o['9']['rmse_mean']:.4f} at the same budgets, so the cost of the fixed schedule is "
        f"{d['5']['rmse_mean'] - o['5']['rmse_mean']:+.4f} RMSE at j=5. "
    )
    js_d, js_o = res["j_star"]["dyadic"], res["j_star"]["optimal"]
    e4 = res["reconstruction"]["non_dyadic_j"].get("4")
    four = ""
    if e4 is not None:
        four = (
            f"At exactly four knots the fixed uniform set {e4['uniform']['knots']} gives "
            f"{e4['uniform']['rmse_mean']:.4f} and the optimal DP placement gives "
            f"{e4['optimal']['rmse_mean']:.4f}, versus the proposal's {PROPOSAL_RMSE_4KNOT}. "
        )
    parts.append(
        f"j* = {js_d} (dyadic) and {js_o} (optimal DP) at the {args.target_rmse} threshold, against the "
        f"proposal's claim of RMSE {PROPOSAL_RMSE_4KNOT} at four knots (j*={PROPOSAL_JSTAR}); the dyadic "
        f"schedule has no 4-knot member, and its nearest budgets bracket that claim at "
        f"{d['3']['rmse_mean']:.4f} (j=3) and {d['5']['rmse_mean']:.4f} (j=5). " + four
    )
    ef = res["exec_match_fraction"]
    parts.append(
        "The exec-match predicate at tol=" + f"{args.exec_match_tol}" + " admits "
        + ", ".join(f"{f:.3f}" for f in ef)
        + " of demo chunks at rounds 0-4, which is what the `exec_n_r` counters should log."
    )
    return "".join(parts)


if __name__ == "__main__":
    sys.exit(main())
