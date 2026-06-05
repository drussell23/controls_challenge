"""Representative benchmark on the OFFICIAL 0-4999 range (every 50th seg = 100).

Usage: python bench.py            # compares pid vs inv (default gains)
All tuning must use THIS sample, never the first 12-20 segs.
"""
import numpy as np
from pathlib import Path
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PID
from controllers.inv import Controller as INV

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
REP = sorted(Path("./data/SYNTHETIC").glob("*.csv"))[:5000][::50]


def evalc(make):
  out = []
  for s in REP:
    sim = TinyPhysicsSimulator(model, str(s), controller=make(), debug=False)
    out.append(sim.rollout())
  lat = np.mean([o['lataccel_cost'] for o in out])
  jrk = np.mean([o['jerk_cost'] for o in out])
  tot = np.mean([o['total_cost'] for o in out])
  return lat, jrk, tot


print(f"REP sample: {len(REP)} segs from official 0-4999 range")
print("%-26s %8s %8s %9s" % ("controller", "lat", "jerk", "TOTAL"))
print("%-26s %8.3f %8.3f %9.2f" % ("pid (baseline)", *evalc(lambda: PID())))
for cfg in [
  dict(p=0.00, i=0.00, d=0.00),                 # pure inverse FF
  dict(p=0.05, i=0.01, d=-0.01),
  dict(p=0.10, i=0.05, d=-0.05),
  dict(p=0.15, i=0.05, d=-0.08),
  dict(p=0.20, i=0.08, d=-0.10),
]:
  lat, jrk, tot = evalc(lambda cfg=cfg: INV(**cfg))
  print("%-26s %8.3f %8.3f %9.2f" % (f"inv {cfg}", lat, jrk, tot), flush=True)
