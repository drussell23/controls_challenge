import numpy as np
from . import BaseController

# Identified linear plant (sysid.py, R^2=0.988):
#   lat_{t+1} = A*lat_t + B*steer_t + C*roll_t + D
A, B, C, D = 0.9474, 0.0755, 0.0526, -0.0103
DEL_T = 0.1

# Cost weights matched to the scoring engine:
#   total = 50*mean((target-lat)^2)*100 + mean((dlat/DEL_T)^2)*100
# -> per-step relative weights: 50 on (target-lat)^2, (1/DEL_T)^2 on (lat_t - lat_{t-1})^2
W_LAT = np.sqrt(50.0)
W_JERK = 1.0 / DEL_T


class Controller(BaseController):
  """Receding-horizon MPC on an identified linear plant, solved as a QP.

  The horizon's linear dynamics make the predicted trajectory affine in the
  steer sequence, so the cost is quadratic -> the optimal sequence is a linear
  least-squares solve. All structure is constant, so we precompute the
  pseudo-inverse once and each step is one matrix-vector product.
  """

  def __init__(self, horizon=30, steer_limit=2.5, R=2.0):
    self.N = horizon
    self.steer_limit = steer_limit
    self.R = R          # weight pulling steer toward the in-distribution feed-forward
    N = self.N

    Apow = np.array([A ** i for i in range(N + 1)])      # A^0 .. A^N
    # lat_k (k=1..N) = Apow[k]*lat0 + sum_{j<k} Apow[k-1-j]*(B*u_j + C*roll_j + D)
    G = np.zeros((N, N))                                  # lat[1..N] = G@u + h
    for k in range(1, N + 1):
      for j in range(k):
        G[k - 1, j] = Apow[k - 1 - j] * B
    M = G / B                                             # decay kernel for the roll/D free response
    self.Apow_lat0 = Apow[1:]                             # contribution of lat0 to h
    self.M = M
    self.Mones = M @ np.ones(N)
    self.G = G

    # lataccel-jerk action matrix: row0 = lat_1 - lat0 ; row k = lat_k - lat_{k-1}
    Dj = np.zeros((N, N))
    Dj[0] = G[0]
    for k in range(1, N):
      Dj[k] = G[k] - G[k - 1]
    self.Dj = Dj

    # regularize each u_j toward its steady-state feed-forward (in-distribution, stable)
    Ast = np.vstack([W_LAT * G, W_JERK * Dj, np.sqrt(R) * np.eye(N)])
    self.pinv = np.linalg.pinv(Ast)                                 # precomputed once

  def _pad(self, seq, n, fallback):
    seq = list(seq) if seq else []
    if not seq:
      return np.full(n, fallback)
    if len(seq) < n:
      seq = seq + [seq[-1]] * (n - len(seq))
    return np.asarray(seq[:n], dtype=float)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    N = self.N
    fut_t = getattr(future_plan, "lataccel", None)
    if not fut_t:
      # no horizon left: static inversion to hold the target
      return float(np.clip((target_lataccel - C * state.roll_lataccel - D) / B,
                           -self.steer_limit, self.steer_limit))

    targ = self._pad(fut_t, N, target_lataccel)
    roll = self._pad(getattr(future_plan, "roll_lataccel", None), N, state.roll_lataccel)

    lat0 = current_lataccel
    h = self.Apow_lat0 * lat0 + C * (self.M @ roll) + D * self.Mones
    dj = np.empty(N)
    dj[0] = h[0] - lat0
    dj[1:] = h[1:] - h[:-1]

    u_ff = (targ * (1 - A) - C * roll - D) / B           # steady-state feed-forward reference
    bst = np.concatenate([W_LAT * (targ - h), W_JERK * (-dj), np.sqrt(self.R) * u_ff])
    u = self.pinv @ bst
    return float(np.clip(u[0], -self.steer_limit, self.steer_limit))
