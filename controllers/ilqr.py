import numpy as np
from pathlib import Path
import onnxruntime as ort
from . import BaseController
from tinyphysics import (CONTEXT_LENGTH as CTX, VOCAB_SIZE as VOCAB, LATACCEL_RANGE,
                         MAX_ACC_DELTA, STEER_RANGE, DEL_T, LataccelTokenizer)

# analytic Jacobians for the *correction* QP (local linearization; sysid R^2=0.99)
A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102
TEMP = 0.8
BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB)
W_LAT = np.sqrt(50.0)
W_JERK = 1.0 / DEL_T


class Controller(BaseController):
  """Iterative LQR / receding-horizon control against the TRUE ONNX plant.

  Root-cause fix vs. linear MPC: the QP's nominal (free-response) trajectory is
  obtained by rolling the *real* model (no multi-step drift), then we linearize
  locally (analytic Jacobians A, B(v)) and solve a small QP for the steer
  *correction* that best balances tracking vs. lataccel-jerk under steer bounds,
  iterating a few times. CPU-feasible: ~N model calls/step (not MPPI's K*N).
  """

  def __init__(self, N=25, iters=3, r_ctrl=0.4):
    self.N, self.iters, self.r = N, iters, r_ctrl
    self.tok = LataccelTokenizer()
    p = Path(__file__).resolve().parent.parent / "models" / "tinyphysics.onnx"
    so = ort.SessionOptions(); so.log_severity_level = 3
    self.sess = ort.InferenceSession(p.read_bytes(), so, ["CPUExecutionProvider"])
    self.S, self.L, self.Acts = [], [], []
    self.nominal = None
    self.ei = 0.0; self.pe = 0.0

  def _predict(self, states, tokens):                       # (M,CTX,4),(M,CTX) -> (M,)
    logits = self.sess.run(None, {'states': states, 'tokens': tokens})[0][:, -1, :]
    z = logits / TEMP; z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p @ BINS

  def _rollout(self, U, exo, x0_prev):
    """Roll M control sequences U (M,N) through the true plant.
    exo = (roll,v,a) arrays length N+1 (index 0 = current step). Returns lat (M,N)."""
    M, N = U.shape
    roll, v, a = exo
    ba = np.asarray(self.Acts[-(CTX - 1):], dtype=np.float32)
    bs = np.asarray(self.S[-CTX:], dtype=np.float32)
    bl = np.asarray(self.L[-CTX:], dtype=np.float32)
    W_act = np.tile(np.concatenate([ba, [0.0]]), (M, 1)).astype(np.float32)
    W_state = np.tile(bs, (M, 1, 1)).astype(np.float32)
    W_lat = np.tile(bl, (M, 1)).astype(np.float32)
    out = np.zeros((M, N)); prev = np.full(M, x0_prev)
    for h in range(N):
      W_act[:, -1] = U[:, h]
      states_in = np.concatenate([W_act[:, :, None], W_state], axis=2).astype(np.float32)
      lat = self._predict(states_in, self.tok.encode(W_lat).astype(np.int64))
      lat = np.clip(lat, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA)
      out[:, h] = lat; prev = lat
      ns = np.tile([roll[h + 1], v[h + 1], a[h + 1]], (M, 1)).astype(np.float32)
      W_act = np.roll(W_act, -1, axis=1)
      W_state = np.concatenate([W_state[:, 1:, :], ns[:, None, :]], axis=1)
      W_lat = np.concatenate([W_lat[:, 1:], lat[:, None]], axis=1).astype(np.float32)
    return out

  def _fallback(self, t, c, st, fp):
    fut = getattr(fp, "lataccel", None)
    ft = fut[min(2, len(fut) - 1)] if fut else t
    B = B0 + B1 * st.v_ego
    e = t - c; self.ei += e; de = e - self.pe; self.pe = e
    return (ft * (1 - A) - C * st.roll_lataccel - D) / B + 0.2 * e + 0.05 * self.ei - 0.1 * de

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.S.append([state.roll_lataccel, state.v_ego, state.a_ego]); self.L.append(current_lataccel)
    if len(self.L) < CTX or len(self.Acts) < CTX - 1:
      u = float(np.clip(self._fallback(target_lataccel, current_lataccel, state, future_plan),
                        STEER_RANGE[0], STEER_RANGE[1])); self.Acts.append(u); return u

    N = self.N
    def seq(cur, lst):
      o = [cur] + (list(lst) if lst else []); o += [o[-1]] * (N + 1 - len(o)); return np.asarray(o[:N + 1])
    roll = seq(state.roll_lataccel, getattr(future_plan, "roll_lataccel", None))
    v    = seq(state.v_ego,        getattr(future_plan, "v_ego", None))
    a    = seq(state.a_ego,        getattr(future_plan, "a_ego", None))
    tgt  = seq(target_lataccel,    getattr(future_plan, "lataccel", None))[1:]   # targets for x_1..x_N
    exo = (roll, v, a)

    # warm start: stable half-gain feed-forward (the FFB family that gives ~72)
    if self.nominal is None:
      Bv0 = B0 + B1 * v[:N]
      u = 0.6 * (tgt * (1 - A) - C * roll[:N] - D) / Bv0
      self.nominal = np.clip(u, STEER_RANGE[0], STEER_RANGE[1])

    def hcost(x):                       # true horizon cost (scoring weights), x (N,)
      err = x - tgt
      dx = np.empty(N); dx[0] = x[0] - current_lataccel; dx[1:] = x[1:] - x[:-1]
      return 50.0 * np.sum(err ** 2) + np.sum((dx / DEL_T) ** 2)

    # analytic sensitivity G: x_k = sum_{j<=k} A^{k-j} B_j du_j   (G[k,j], k,j in 0..N-1)
    Bv = (B0 + B1 * v[:N]).astype(float)
    G = np.zeros((N, N))
    for k in range(N):
      for j in range(k + 1):
        G[k, j] = (A ** (k - j)) * Bv[j]
    # lataccel-jerk difference operator on [x_0..x_{N-1}] with x_{-1}=current
    Diff = np.eye(N) - np.eye(N, k=-1)
    DG = Diff @ G

    u = self.nominal.copy()
    xbar = self._rollout(u[None, :], exo, current_lataccel)[0]              # TRUE nominal (N,)
    cur_cost = hcost(xbar)
    alphas = np.array([1.0, 0.5, 0.25, 0.1])
    for _ in range(self.iters):
      jerk0 = np.empty(N); jerk0[0] = xbar[0] - current_lataccel; jerk0[1:] = xbar[1:] - xbar[:-1]
      Astack = np.vstack([W_LAT * G, W_JERK * DG, np.sqrt(self.r) * np.eye(N)])
      bstack = np.concatenate([W_LAT * (tgt - xbar), -W_JERK * jerk0, np.zeros(N)])
      du, *_ = np.linalg.lstsq(Astack, bstack, rcond=None)
      # batched backtracking line search: accept the alpha that most reduces TRUE cost
      Ucand = np.clip(u[None, :] + alphas[:, None] * du[None, :], STEER_RANGE[0], STEER_RANGE[1])
      Xc = self._rollout(Ucand, exo, current_lataccel)                     # (len(alphas), N)
      costs = np.array([hcost(Xc[m]) for m in range(len(alphas))])
      m = int(costs.argmin())
      if costs[m] < cur_cost - 1e-6:
        u, xbar, cur_cost = Ucand[m], Xc[m], costs[m]
      else:
        break                                                             # converged

    u0 = float(u[0])
    self.nominal = np.concatenate([u[1:], u[-1:]])
    self.Acts.append(u0)
    return u0
