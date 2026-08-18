"""Self-test the three tie-break modes without building a critic.

_pick_bin only touches `self.zq_tie_break` and the tensor, so a stand-in with
that attribute exercises the identical code path. Verified against the shipped
baseline helper, imported LAZILY (a module-scope import of cqn_utils before the
first encoder forward segfaults this interpreter).
"""
import sys, types, torch
sys.path.insert(0, "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
sys.path.insert(0, "/mnt/workspace/zoomq/third_party/CQN-AS-G1/bigym_src")

import ast, textwrap
src = open("/mnt/workspace/zoomq/third_party/CQN-AS-G1/bigym_src/zoomq.py").read()
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_pick_bin")
mod = ast.Module(body=[fn], type_ignores=[])
ns = {"torch": torch, "getattr": getattr}
exec(compile(mod, "<pick>", "exec"), ns)
pick = ns["_pick_bin"]

class Stub:
    pass

torch.manual_seed(0)
B, n, D, bins = 4, 5, 15, 5
qs = torch.randn(B, n, D, bins)
# force a realistic tie population: flatten 40% of the cells
flat = torch.rand(B, n, D) < 0.40
qs = torch.where(flat.unsqueeze(-1), qs.mean(-1, keepdim=True).expand_as(qs), qs)
tied = (qs.max(-1).values - qs.min(-1).values) < 1e-4
print("tied cells: %d / %d (%.1f%%)" % (tied.sum(), tied.numel(), 100.0*tied.float().mean()))

ok = True
def chk(c, m):
    global ok
    print(("  OK   " if c else "  FAIL ") + m); ok = ok and bool(c)

# 1) argmax must equal the pre-change expression, everywhere
s = Stub(); s.zq_tie_break = "argmax"
chk(torch.equal(pick(s, qs), qs.max(-1)[1]),
    "mode=argmax is bit-identical to the pre-change qs.max(-1)[1]")

# 2) random must equal the baseline helper under the same RNG draw
from cqn_utils import random_action_if_within_delta
s = Stub(); s.zq_tie_break = "random"
torch.manual_seed(123); mine = pick(s, qs)
torch.manual_seed(123); theirs = random_action_if_within_delta(qs)
chk(torch.equal(mine, theirs),
    "mode=random is bit-identical to cqn_utils.random_action_if_within_delta")
chk(torch.equal(mine[~tied], qs.max(-1)[1][~tied]),
    "mode=random leaves every UNtied cell on its argmax")

# 3) middle
s = Stub(); s.zq_tie_break = "middle"
mid = pick(s, qs)
chk(bool((mid[tied] == 2).all()), "mode=middle sends every tied cell to bin 2 of 5")
chk(torch.equal(mid[~tied], qs.max(-1)[1][~tied]),
    "mode=middle leaves every UNtied cell on its argmax")

# 4) the bias this is all about: what does the shipped rule do on ties?
s = Stub(); s.zq_tie_break = "argmax"
a = pick(s, qs)
print("  tied cells' chosen bin -- argmax: mean %.3f (bin 0 share %.1f%%)"
      % (a[tied].float().mean(), 100.0*(a[tied] == 0).float().mean()))
torch.manual_seed(7); s.zq_tie_break = "random"; r = pick(s, qs)
print("                          random: mean %.3f (bin 0 share %.1f%%)"
      % (r[tied].float().mean(), 100.0*(r[tied] == 0).float().mean()))

# 5) bad mode raises
s = Stub(); s.zq_tie_break = "nonsense"
try:
    pick(s, qs); chk(False, "an invalid mode should raise")
except ValueError:
    chk(True, "an invalid mode raises ValueError")

print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
