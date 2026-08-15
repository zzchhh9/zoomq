# ZoomQ

Three-layer workspace for the ZoomQ project (coarse-to-fine *temporal* refinement of
action chunks with a learned stopping rule).

| Layer | Where | What lives there |
|---|---|---|
| Paper | `~/anchor_q` (**not** in this repo) | `zoomq.pdf`, `main.tex` — the proposal being tested. |
| Algorithm | `third_party/CQN-AS-G1` (submodule, branch `zoomq`) | `bigym_src/zoomq.py` + `cfgs/config_zoomq_bigym.yaml`. Nothing else may change: `git diff fba52f7..zoomq` is the "everything else identical" evidence. |
| Experiments | this repo — `gates/ configs/ scripts/ reports/` | Falsification gates, launch scripts, run configs, nightly reports. |

Baseline SHA of the fork: **`fba52f7b38fed6e12ee3983ae456a1f0e657bf6f`**
(= `Nagi-ovo/CQN-AS-G1@main` as of 2026-08-15).

## Submodule discipline

`git submodule update --init` leaves a **detached HEAD**; commits made there are lost at the
next pointer bump. Before editing algorithm code, always:

```bash
git -C third_party/CQN-AS-G1 checkout zoomq
```

Commit algorithm changes on the submodule's `zoomq` branch, push it, then bump the pointer
here and push this repo too.

## Machines

- **ali** (`ssh ali`) — 16x PPU-ZW810E, 184 cores. Training + Gate 0. Needs
  `source /usr/local/PPU_SDK/envsetup.sh` before any accelerator work; python is
  `/mnt/workspace/anchorq/.venv/bin/python`. No tmux — detach with
  `setsid nohup ... > log 2>&1 < /dev/null &`.
- **SWIRL03** — local dev box, 64 cores. Repo bootstrap, unit tests, Gate 1 (if
  `bigym-loco` imports).
