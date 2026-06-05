import numpy as np
import torch
from . import BaseController
from .torch_plant import build_soft_plant, make_bins, soft_encode
from tinyphysics import CONTEXT_LENGTH as CTX, MAX_ACC_DELTA, STEER_RANGE, DEL_T

# linear plant for the FFB warm-start / fallback (sysid, speed-scheduled)
A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102


class Controller(BaseController):
  """Differentiable Predictive Control, executed efficiently on the M1.

  Same exact-gradient method, but: (1) MPS-resident (M1 GPU), (2) re-plan only
  every K steps (causal receding-horizon, not every micro-step), (3) warm-start
  each re-plan from the shifted previous solution with dynamic early-stop. This
  turns the ~40 min/seg brute-force loop into something M1-feasible without a
  cloud GPU, cheating (no whole-segment peeking), or dumbing down the model.
  """

  def __init__(self, N=40, replan_every=5, max_iters=30, warm_iters=14,
               lr=0.08, tol=2e-3, sigma=0.03, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    self.dev = torch.device(device)
    self.gm = build_soft_plant(self.dev)
    self.bins = make_bins(self.dev)
    self.N, self.K = N, replan_every
    self.max_iters, self.warm_iters, self.lr, self.tol, self.sigma = max_iters, warm_iters, lr, tol, sigma
    self.S, self.L, self.Acts = [], [], []
    self.prev_plan = None         # last optimized full-horizon plan (np, len N)
    self.plan = None              # active plan being consumed
    self.pos = 0
    self.ei = 0.0; self.pe = 0.0

  def _ffb(self, t, c, st, fp):
    fut = getattr(fp, "lataccel", None)
    ft = fut[min(2, len(fut) - 1)] if fut else t
    B = B0 + B1 * st.v_ego
    e = t - c; self.ei += e; de = e - self.pe; self.pe = e
    return 0.6 * (ft * (1 - A) - C * st.roll_lataccel - D) / B + 0.2 * e + 0.05 * self.ei - 0.1 * de

  def _rollout_cost(self, U, hist_state, hist_lat, hist_act, exo_state, tgt, x0prev):
    Wa, Ws, Wl = hist_act, hist_state, hist_lat
    prev = x0prev; xs = []
    for h in range(self.N):
      act_win = torch.cat([Wa, U[h:h + 1]])
      st_win = torch.cat([act_win.unsqueeze(-1), Ws], dim=-1).unsqueeze(0)
      tok_win = soft_encode(Wl, self.bins, self.sigma).unsqueeze(0)
      logits = self.gm(st_win, tok_win)[:, -1, :]
      x = (torch.softmax(logits / 0.8, -1) * self.bins).sum(-1).squeeze(0)
      x = prev + torch.clamp(x - prev, -MAX_ACC_DELTA, MAX_ACC_DELTA)
      xs.append(x); prev = x
      Wa = torch.cat([Wa[1:], U[h:h + 1]])
      Ws = torch.cat([Ws[1:], exo_state[h + 1:h + 2]], dim=0)
      Wl = torch.cat([Wl[1:], x.reshape(1)], dim=0)
    X = torch.stack(xs)
    dx = torch.cat([(X[0] - x0prev).reshape(1), X[1:] - X[:-1]])
    return 50.0 * ((X - tgt) ** 2).sum() + ((dx / DEL_T) ** 2).sum()

  def _optimize(self, warm, packs, iters):
    U = torch.tensor(warm, dtype=torch.float32, device=self.dev, requires_grad=True)
    opt = torch.optim.Adam([U], lr=self.lr)
    prev = float("inf")
    for i in range(iters):
      opt.zero_grad()
      loss = self._rollout_cost(U, *packs)
      loss.backward(); opt.step()
      with torch.no_grad(): U.clamp_(STEER_RANGE[0], STEER_RANGE[1])
      l = float(loss.detach())
      if i >= 3 and abs(prev - l) / max(prev, 1.0) < self.tol:
        break                                              # dynamic early-stop on plateau
      prev = l
    return U.detach().cpu().numpy()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.S.append([state.roll_lataccel, state.v_ego, state.a_ego]); self.L.append(current_lataccel)
    if len(self.L) < CTX or len(self.Acts) < CTX - 1:
      u = float(np.clip(self._ffb(target_lataccel, current_lataccel, state, future_plan),
                        STEER_RANGE[0], STEER_RANGE[1])); self.Acts.append(u); return u

    if self.plan is None or self.pos >= self.K:            # re-plan every K steps
      N = self.N
      def seq(cur, lst):
        o = [cur] + (list(lst) if lst else []); o += [o[-1]] * (N + 1 - len(o)); return np.asarray(o[:N + 1])
      roll = seq(state.roll_lataccel, getattr(future_plan, "roll_lataccel", None))
      vv   = seq(state.v_ego,        getattr(future_plan, "v_ego", None))
      ae   = seq(state.a_ego,        getattr(future_plan, "a_ego", None))
      tgt_np = seq(target_lataccel,  getattr(future_plan, "lataccel", None))[1:]
      dev = self.dev
      packs = (
        torch.tensor(self.S[-CTX:], dtype=torch.float32, device=dev),
        torch.tensor(self.L[-CTX:], dtype=torch.float32, device=dev),
        torch.tensor(self.Acts[-(CTX - 1):], dtype=torch.float32, device=dev),
        torch.tensor(np.stack([roll, vv, ae], 1), dtype=torch.float32, device=dev),
        torch.tensor(tgt_np, dtype=torch.float32, device=dev),
        torch.tensor(current_lataccel, dtype=torch.float32, device=dev),
      )
      if self.prev_plan is not None:                       # warm-start from shifted previous (few iters)
        warm = np.concatenate([self.prev_plan[self.K:], np.repeat(self.prev_plan[-1], self.K)])[:N]
        iters = self.warm_iters
      else:
        Bv = B0 + B1 * vv[:N]
        warm = np.clip(0.6 * (tgt_np * (1 - A) - C * roll[:N] - D) / Bv, STEER_RANGE[0], STEER_RANGE[1])
        iters = self.max_iters
      try:
        self.plan = self._optimize(warm, packs, iters); self.prev_plan = self.plan
      except Exception:
        self.plan = warm; self.prev_plan = warm           # graceful fallback
      self.pos = 0

    u = float(np.clip(self.plan[self.pos], STEER_RANGE[0], STEER_RANGE[1]))
    self.pos += 1
    self.Acts.append(u)
    return u
