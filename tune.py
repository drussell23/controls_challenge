"""Decisive controller comparison on a fixed segment set (apples-to-apples).

Compares the stock PID against tuned feed-forward and a future-aware
feed-forward that anticipates the upcoming trajectory via future_plan.
"""
import sys
import numpy as np
from pathlib import Path
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import BaseController


class PIDReal(BaseController):
  def __init__(self):
    self.p, self.i, self.d = 0.195, 0.100, -0.053
    self.ei = 0.0
    self.pe = 0.0

  def update(self, target, current, state, future_plan):
    e = target - current
    self.ei += e
    de = e - self.pe
    self.pe = e
    return self.p * e + self.i * self.ei + self.d * de


class FFPID(BaseController):
  """Feed-forward on (optionally future-averaged) target + PID on error."""
  def __init__(self, ff=0.4, p=0.1, i=0.05, d=-0.05, lookahead=0):
    self.ff, self.p, self.i, self.d, self.lookahead = ff, p, i, d, lookahead
    self.ei = 0.0
    self.pe = 0.0

  def update(self, target, current, state, future_plan):
    ff_target = target
    if self.lookahead and getattr(future_plan, "lataccel", None):
      window = future_plan.lataccel[:self.lookahead]
      if window:
        ff_target = (target + float(np.mean(window))) / 2.0
    e = target - current
    self.ei += e
    de = e - self.pe
    self.pe = e
    return self.ff * ff_target + self.p * e + self.i * self.ei + self.d * de


N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
segs = sorted(Path("./data/SYNTHETIC").glob("*.csv"))[:N]


def evalc(make):
  cs = []
  for s in segs:
    sim = TinyPhysicsSimulator(model, str(s), controller=make(), debug=False)
    cs.append(sim.rollout()['total_cost'])
  return float(np.mean(cs))


candidates = {
  "pid_real         ": lambda: PIDReal(),
  "ff0.4 + weak pid  ": lambda: FFPID(ff=0.4, p=0.1, i=0.05, d=-0.05),
  "ff0.4 + look10    ": lambda: FFPID(ff=0.4, p=0.1, i=0.05, d=-0.05, lookahead=10),
  "ff0.4 + look20    ": lambda: FFPID(ff=0.4, p=0.1, i=0.05, d=-0.05, lookahead=20),
}
print(f"segments: {len(segs)}")
results = {name: evalc(make) for name, make in candidates.items()}
for name, c in sorted(results.items(), key=lambda kv: kv[1]):
  print(f"{name}  mean_total_cost={c:.2f}", flush=True)
