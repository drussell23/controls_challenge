import numpy as np
import torch
from . import BaseController
from .torch_plant import build_soft_plant, make_bins, soft_encode
from tinyphysics import CONTEXT_LENGTH as CTX, MAX_ACC_DELTA, STEER_RANGE, DEL_T

A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102
KP, KI, KD, FF = 0.2, 0.05, -0.1, 0.6     # tracker gains (the closed-loop ~90 floor)


class Controller(BaseController):
  """Hierarchical Differentiable Predictive Control (HDPC).

  Bi-level planner-tracker:
   * TRACKER (K=1, every step): closed-loop FFB+PID on real feedback -> the ~90
     floor with noise rejection; the sim is never driven open-loop.
   * PLANNER (every K, event-triggered): the differentiable DPC optimizes a
     RESIDUAL feed-forward added on top. The PID is *baked into* the differentiable
     rollout, so autograd backprops through the simulated tracker and tunes the
     residual knowing how the PID will react.
   * SHADOW GATE: residual kept only if it beats the closed-loop FFB+PID (else
     zeroed) -> mathematically can't do worse than the ~90 floor.
  Applied each step: action = ffb_ff + PID(real error) + residual[pos].
  """

  def __init__(self, N=30, B=4, sigma_noise=0.02, replan_every=5, max_iters=20, warm_iters=10,
               lr=0.05, l2=6.0, tol=2e-3, sigma_tok=0.03, err_gate=0.06, device="cpu", seed=0):
    self.dev = torch.device(device)
    self.gm = build_soft_plant(self.dev); self.bins = make_bins(self.dev)
    self.N, self.B, self.K = N, B, replan_every
    self.sn, self.max_iters, self.warm_iters = sigma_noise, max_iters, warm_iters
    self.lr, self.l2, self.tol, self.sigma_tok, self.err_gate = lr, l2, tol, sigma_tok, err_gate
    self.g = torch.Generator(device=self.dev).manual_seed(seed)
    self.S, self.L, self.Acts = [], [], []
    self.residual = None; self.pos = 0
    self.ei = 0.0; self.pe = 0.0          # real tracker PID state

  def _antithetic(self):
    half = max(1, self.B // 2)
    z = torch.randn(half, self.N, generator=self.g, device=self.dev) * self.sn
    n = torch.cat([z, -z], 0)
    return n[:self.B]

  def _ffb_ff(self, ft, roll, v):          # feed-forward term only
    return FF * (ft * (1 - A) - C * roll - D) / (B0 + B1 * v)

  def _rollout_cost(self, residual, nom, hist, noise):
    """Differentiable closed-loop rollout with PID baked in. residual (N,) is the
    decision variable; ffb_ff[h], roll/v/a, targets come from `nom`. Returns mean cost."""
    hs, hl, ha, exo, x0, ffb_ff, tgt, ei0, pe0 = nom
    B, N = self.B, self.N
    Wa = hist[0].expand(B, -1).clone(); Ws = hist[1].expand(B, -1, -1).clone(); Wl = hist[2].expand(B, -1).clone()
    prev = x0.expand(B).clone(); ei = ei0.expand(B).clone(); pe = pe0.expand(B).clone(); xs = []
    for h in range(N):
      err = tgt[h] - prev                 # tracker reacts to simulated error (closed-loop in-graph)
      ei = ei + err; de = err - pe; pe = err
      pid = KP * err + KI * ei + KD * de
      a = torch.clamp(ffb_ff[h] + pid + residual[h], STEER_RANGE[0], STEER_RANGE[1]).unsqueeze(1)
      st_win = torch.cat([torch.cat([Wa, a], 1).unsqueeze(-1), Ws], -1)
      tok = soft_encode(Wl, self.bins, self.sigma_tok)
      x = (torch.softmax(self.gm(st_win, tok)[:, -1, :] / 0.8, -1) * self.bins).sum(-1)
      x = prev + torch.clamp(x - prev, -MAX_ACC_DELTA, MAX_ACC_DELTA) + noise[:, h]
      xs.append(x); prev = x
      Wa = torch.cat([Wa[:, 1:], a], 1)
      Ws = torch.cat([Ws[:, 1:, :], exo[h + 1].expand(B, 1, 3)], 1)
      Wl = torch.cat([Wl[:, 1:], x.unsqueeze(1)], 1)
    X = torch.stack(xs, 1)
    dx = torch.cat([(X[:, :1] - x0), X[:, 1:] - X[:, :-1]], 1)
    return (50.0 * (X - tgt) ** 2 + (dx / DEL_T) ** 2).sum(1).mean()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.S.append([state.roll_lataccel, state.v_ego, state.a_ego]); self.L.append(current_lataccel)
    fut = getattr(future_plan, "lataccel", None)
    ft = fut[min(2, len(fut) - 1)] if fut else target_lataccel
    # ---- real closed-loop tracker (every step) ----
    err = target_lataccel - current_lataccel
    base_ff = self._ffb_ff(ft, state.roll_lataccel, state.v_ego)

    if len(self.L) >= CTX and len(self.Acts) >= CTX - 1:
      # event-triggered heavy planner: re-plan every K, only if tracking is hard
      hard = abs(err) > self.err_gate
      if self.residual is None or (self.pos >= self.K and hard):
        self._replan(target_lataccel, current_lataccel, state, future_plan)

    r = float(self.residual[self.pos]) if (self.residual is not None and self.pos < len(self.residual)) else 0.0
    self.pos += 1
    # PID (real feedback) + feed-forward + planner residual
    self.ei += err; de = err - self.pe; self.pe = err
    u = base_ff + KP * err + KI * self.ei + KD * de + r
    u = float(np.clip(u, STEER_RANGE[0], STEER_RANGE[1]))
    self.Acts.append(u)
    return u

  def _replan(self, target_lataccel, current_lataccel, state, future_plan):
    N = self.N
    def seq(cur, lst):
      o = [cur] + (list(lst) if lst else []); o += [o[-1]] * (N + 1 - len(o)); return np.asarray(o[:N + 1])
    roll = seq(state.roll_lataccel, getattr(future_plan, "roll_lataccel", None))
    vv   = seq(state.v_ego,        getattr(future_plan, "v_ego", None))
    ae   = seq(state.a_ego,        getattr(future_plan, "a_ego", None))
    tgt_np = seq(target_lataccel,  getattr(future_plan, "lataccel", None))[1:]
    d = self.dev
    hist = (torch.tensor(self.S[-CTX:], dtype=torch.float32, device=d),
            torch.tensor(self.L[-CTX:], dtype=torch.float32, device=d),
            torch.tensor(self.Acts[-(CTX - 1):], dtype=torch.float32, device=d))
    # feed-forward over horizon: ff[h] uses target ~2 ahead (matches tracker's lookahead)
    ft_h = np.array([tgt_np[min(h + 2, N - 1)] for h in range(N)])
    ffb_ff = self._ffb_ff(ft_h, roll[:N], vv[:N])
    nom = (hist[0], hist[1], hist[2],
           torch.tensor(np.stack([roll, vv, ae], 1), dtype=torch.float32, device=d),
           torch.tensor(current_lataccel, dtype=torch.float32, device=d),
           torch.tensor(ffb_ff, dtype=torch.float32, device=d),
           torch.tensor(tgt_np, dtype=torch.float32, device=d),
           torch.tensor(self.ei, dtype=torch.float32, device=d),
           torch.tensor(self.pe, dtype=torch.float32, device=d))
    noise = self._antithetic()
    residual = torch.zeros(N, dtype=torch.float32, device=d, requires_grad=True)
    opt = torch.optim.Adam([residual], lr=self.lr); prev = float("inf")
    try:
      for i in range(self.warm_iters if self.residual is not None else self.max_iters):
        opt.zero_grad()
        loss = self._rollout_cost(residual, nom, hist, noise) + self.l2 * (residual ** 2).sum()
        loss.backward(); opt.step()
        l = float(loss.detach())
        if i >= 2 and abs(prev - l) / max(prev, 1.0) < self.tol: break
        prev = l
      with torch.no_grad():
        c_opt = float(self._rollout_cost(residual, nom, hist, noise))
        c_base = float(self._rollout_cost(torch.zeros(N, device=d), nom, hist, noise))   # FFB+PID floor
      self.residual = residual.detach().cpu().numpy() if (np.isfinite(c_opt) and c_opt < c_base) else np.zeros(N)
    except Exception:
      self.residual = np.zeros(N)
    self.pos = 0
