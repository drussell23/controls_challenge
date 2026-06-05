import numpy as np
from pathlib import Path
import onnxruntime as ort
from . import BaseController
from tinyphysics import (CONTEXT_LENGTH as CTX, VOCAB_SIZE as VOCAB, LATACCEL_RANGE,
                         MAX_ACC_DELTA, STEER_RANGE, DEL_T, LataccelTokenizer)

# linear plant for the feed-forward warm-start / fallback (sysid, speed-scheduled)
A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102
LAT_W = 50.0                       # matches scoring: total = 50*lat_cost + jerk_cost
TEMP = 0.8                         # plant sampling temperature
BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB)


class Controller(BaseController):
  """Model Predictive Path Integral (MPPI) control against the TRUE ONNX plant.

  Each step: sample K time-correlated steer-perturbation sequences over horizon
  N, roll all K through a *batched* forward pass of the real tinyphysics model
  (expected lataccel = prob-weighted bin mean, so we plan against the plant's
  mean dynamics), score each path on the official cost, and combine them with
  the path-integral softmax weighting. Mirrors the sim's exact bookkeeping.
  """

  def __init__(self, K=256, N=20, sigma=0.25, lam=2.0, beta_corr=0.6):
    self.K, self.N, self.sigma, self.lam, self.beta = K, N, sigma, lam, beta_corr
    self.tok = LataccelTokenizer()
    path = Path(__file__).resolve().parent.parent / "models" / "tinyphysics.onnx"
    so = ort.SessionOptions(); so.log_severity_level = 3
    self.sess = ort.InferenceSession(path.read_bytes(), so, ["CPUExecutionProvider"])
    # accumulated true history (controller sees one step at a time)
    self.S, self.L, self.Acts = [], [], []      # states[roll,v,a], lataccels, my actions
    self.nominal = np.zeros(N)                   # warm-started steer plan
    self.ei = 0.0; self.pe = 0.0

  # ---- batched true-plant one-step prediction: expected next lataccel ----
  def _predict(self, states, tokens):           # states (K,CTX,4)f32, tokens (K,CTX)i64
    logits = self.sess.run(None, {'states': states, 'tokens': tokens})[0][:, -1, :]
    z = logits / TEMP
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return p @ BINS                              # (K,) expected lataccel

  def _fallback(self, target, current, state, future_plan):
    fut = getattr(future_plan, "lataccel", None)
    ft = fut[min(2, len(fut) - 1)] if fut else target
    B = B0 + B1 * state.v_ego
    u_ff = (ft * (1 - A) - C * state.roll_lataccel - D) / B
    e = target - current; self.ei += e; de = e - self.pe; self.pe = e
    return u_ff + 0.2 * e + 0.05 * self.ei - 0.1 * de

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.S.append([state.roll_lataccel, state.v_ego, state.a_ego])
    self.L.append(current_lataccel)

    # need CTX preds + CTX-1 past actions to seed the model window
    if len(self.L) < CTX or len(self.Acts) < CTX - 1:
      u = self._fallback(target_lataccel, current_lataccel, state, future_plan)
      u = float(np.clip(u, STEER_RANGE[0], STEER_RANGE[1])); self.Acts.append(u); return u

    K, N = self.K, self.N
    # future state sequence: index 0 = current step, 1..N = from future_plan (held if short)
    fl = future_plan
    def seq(cur, lst):
      out = [cur] + (list(lst) if lst else [])
      out += [out[-1]] * (N + 1 - len(out))
      return np.asarray(out[:N + 1], dtype=np.float32)
    roll_f = seq(state.roll_lataccel, getattr(fl, "roll_lataccel", None))
    v_f    = seq(state.v_ego,        getattr(fl, "v_ego", None))
    a_f    = seq(state.a_ego,        getattr(fl, "a_ego", None))
    tgt    = seq(target_lataccel,    getattr(fl, "lataccel", None))

    # time-correlated noise (AR(1) smoothing) around the warm-started nominal
    eps = np.random.randn(K, N) * self.sigma
    for t in range(1, N):
      eps[:, t] = self.beta * eps[:, t - 1] + (1 - self.beta) * eps[:, t]
    U = np.clip(self.nominal[None, :] + eps, STEER_RANGE[0], STEER_RANGE[1])   # (K,N)

    # seed rolling windows (broadcast shared true history across K candidates)
    base_act = np.asarray(self.Acts[-(CTX - 1):], dtype=np.float32)            # (CTX-1,)
    base_state = np.asarray(self.S[-CTX:], dtype=np.float32)                   # (CTX,3)
    base_lat = np.asarray(self.L[-CTX:], dtype=np.float32)                     # (CTX,)
    W_act = np.tile(np.concatenate([base_act, [0.0]]), (K, 1))                 # (K,CTX) last filled per step
    W_state = np.tile(base_state, (K, 1, 1)).astype(np.float32)                # (K,CTX,3)
    W_lat = np.tile(base_lat, (K, 1)).astype(np.float32)                       # (K,CTX)

    cost = np.zeros(K); prev_lat = np.full(K, current_lataccel, dtype=np.float64)
    for h in range(N):
      W_act[:, -1] = U[:, h]
      states_in = np.concatenate([W_act[:, :, None], W_state], axis=2).astype(np.float32)  # (K,CTX,4)
      tokens_in = self.tok.encode(W_lat).astype(np.int64)
      lat = self._predict(states_in, tokens_in)
      lat = np.clip(lat, prev_lat - MAX_ACC_DELTA, prev_lat + MAX_ACC_DELTA)
      cost += LAT_W * (lat - tgt[h]) ** 2 + ((lat - prev_lat) / DEL_T) ** 2
      prev_lat = lat
      # advance windows by one (drop oldest, append step h's outcome / next state)
      ns = np.tile([roll_f[h + 1], v_f[h + 1], a_f[h + 1]], (K, 1)).astype(np.float32)
      W_act = np.roll(W_act, -1, axis=1)
      W_state = np.concatenate([W_state[:, 1:, :], ns[:, None, :]], axis=1)
      W_lat = np.concatenate([W_lat[:, 1:], lat[:, None]], axis=1).astype(np.float32)

    w = np.exp(-(cost - cost.min()) / self.lam); w /= w.sum()
    u_seq = self.nominal + (w[:, None] * eps).sum(axis=0)
    u0 = float(np.clip(u_seq[0], STEER_RANGE[0], STEER_RANGE[1]))
    self.nominal = np.concatenate([u_seq[1:], u_seq[-1:]])     # receding-horizon shift
    self.Acts.append(u0)
    return u0
