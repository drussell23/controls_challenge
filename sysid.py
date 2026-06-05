"""System identification of the tinyphysics plant for linear MPC.

Hypothesis: lataccel_{t+1} ~= a*lataccel_t + b*steer_t + c*roll_t + d
Fit from real ONNX rollouts (PID controller) and report R^2. If the linear
surrogate predicts the black box well, a fast QP-based MPC is justified.
"""
import sys
import numpy as np
from pathlib import Path
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
from controllers.pid import Controller as PID

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
segs = sorted(Path("./data/SYNTHETIC").glob("*.csv"))[:N]

rows_X, rows_y = [], []
for s in segs:
  sim = TinyPhysicsSimulator(model, str(s), controller=PID(), debug=False)
  sim.rollout()
  lat = np.asarray(sim.current_lataccel_history, dtype=float)
  act = np.asarray(sim.action_history, dtype=float)
  roll = np.asarray([st[0] for st in sim.state_history], dtype=float)  # State.roll_lataccel
  n = min(len(lat), len(act), len(roll))
  for t in range(CONTROL_START_IDX, n - 1):
    rows_X.append([lat[t], act[t], roll[t], 1.0])
    rows_y.append(lat[t + 1])

X = np.asarray(rows_X)
y = np.asarray(rows_y)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
ss_res = float(np.sum((y - pred) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
a, b, c, d = coef

print(f"samples: {len(y)}  segments: {len(segs)}")
print(f"lataccel_(t+1) = {a:.4f}*lataccel + {b:.4f}*steer + {c:.4f}*roll + {d:.4f}")
print(f"R^2 = {r2:.5f}   RMSE = {rmse:.5f}")
print(f"implied steady-state steer->lataccel gain = b/(1-a) = {b/(1-a):.3f}  (earlier ff~0.4 implied ~2.5)")
