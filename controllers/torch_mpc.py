import numpy as np
import torch
from . import BaseController
from .torch_plant import build_soft_plant, make_bins, soft_encode
from tinyphysics import CONTEXT_LENGTH as CTX, MAX_ACC_DELTA, STEER_RANGE, DEL_T

# linear plant for FFB warm-start / fallback (sysid, speed-scheduled)
A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102
torch.set_num_threads(max(1, torch.get_num_threads()))


class Controller(BaseController):
  """Stochastic Differentiable Predictive Control (CPU).

  Causal receding-horizon trajectory optimization against the exact, fully
  differentiable PyTorch plant. Per (re)plan: roll an ANTITHETIC batch of B
  noise-perturbed trajectories (Monte-Carlo robustness against the stochastic
  plant, variance-reduced via +/-z pairs), minimize EXPECTED cost with Adam +
  exact gradients, dynamic early-stop (0 iters if already converged), warm-start
  from the shifted previous plan. CPU (7.5x faster than MPS for this sequential
  graph). No whole-segment peeking, no action-hold (1-step control).
  """

  def __init__(self, N=30, B=8, sigma_noise=0.03, replan_every=4,
               max_iters=25, warm_iters=10, lr=0.08, tol=2e-3, conv_loss=0.5,
               sigma_tok=0.03, device="cpu", seed=0):
    self.dev = torch.device(device)
    self.gm = build_soft_plant(self.dev)
    self.bins = make_bins(self.dev)
    self.N, self.B, self.K = N, B, replan_every
    self.sn, self.max_iters, self.warm_iters = sigma_noise, max_iters, warm_iters
    self.lr, self.tol, self.conv_loss, self.sigma_tok = lr, tol, conv_loss, sigma_tok
    self.g = torch.Generator(device=self.dev).manual_seed(seed)
    self.S, self.L, self.Acts = [], [], []
    self.prev_plan = None; self.plan = None; self.pos = 0
    self.ei = 0.0; self.pe = 0.0

  def _ffb(self, t, c, st, fp):
    fut = getattr(fp, "lataccel", None)
    ft = fut[min(2, len(fut) - 1)] if fut else t
    Bv = B0 + B1 * st.v_ego
    e = t - c; self.ei += e; de = e - self.pe; self.pe = e
    return 0.6 * (ft * (1 - A) - C * st.roll_lataccel - D) / Bv + 0.2 * e + 0.05 * self.ei - 0.1 * de

  def _antithetic_noise(self):
    half = self.B // 2
    z = torch.randn(half, self.N, generator=self.g, device=self.dev) * self.sn
    return torch.cat([z, -z], 0) if 2 * half == self.B else torch.cat([z, -z, z[:1] * 0], 0)

  def _rollout_cost(self, U, hist, noise):
    hs, hl, ha, exo, tgt, x0 = hist
    B, N = self.B, self.N
    Wa = ha.expand(B, -1).clone()                    # (B,CTX-1)
    Ws = hs.expand(B, -1, -1).clone()                # (B,CTX,3)
    Wl = hl.expand(B, -1).clone()                    # (B,CTX)
    prev = x0.expand(B).clone(); xs = []
    for h in range(N):
      uh = U[h].expand(B, 1)
      st_win = torch.cat([torch.cat([Wa, uh], 1).unsqueeze(-1), Ws], -1)  # (B,CTX,4)
      tok = soft_encode(Wl, self.bins, self.sigma_tok)                    # (B,CTX,V)
      logits = self.gm(st_win, tok)[:, -1, :]
      x = (torch.softmax(logits / 0.8, -1) * self.bins).sum(-1)
      x = prev + torch.clamp(x - prev, -MAX_ACC_DELTA, MAX_ACC_DELTA)
      x = x + noise[:, h]                                                 # injected plant stochasticity
      xs.append(x); prev = x
      Wa = torch.cat([Wa[:, 1:], uh], 1)
      Ws = torch.cat([Ws[:, 1:, :], exo[h + 1].expand(B, 1, 3)], 1)
      Wl = torch.cat([Wl[:, 1:], x.unsqueeze(1)], 1)
    X = torch.stack(xs, 1)                                                # (B,N)
    dx = torch.cat([(X[:, :1] - x0), X[:, 1:] - X[:, :-1]], 1)
    return (50.0 * (X - tgt) ** 2 + (dx / DEL_T) ** 2).sum(1).mean()      # expected cost

  def _optimize(self, warm, hist, iters):
    U = torch.tensor(warm, dtype=torch.float32, device=self.dev, requires_grad=True)
    noise = self._antithetic_noise()
    opt = torch.optim.Adam([U], lr=self.lr)
    init = float(self._rollout_cost(U, hist, noise).detach())
    if init < self.conv_loss:                                            # already converged -> 0 steps
      return warm
    prev = float("inf")
    for i in range(iters):
      opt.zero_grad()
      loss = self._rollout_cost(U, hist, noise); loss.backward(); opt.step()
      with torch.no_grad(): U.clamp_(STEER_RANGE[0], STEER_RANGE[1])
      l = float(loss.detach())
      if i >= 2 and abs(prev - l) / max(prev, 1.0) < self.tol: break
      prev = l
    return U.detach().cpu().numpy()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.S.append([state.roll_lataccel, state.v_ego, state.a_ego]); self.L.append(current_lataccel)
    if len(self.L) < CTX or len(self.Acts) < CTX - 1:
      u = float(np.clip(self._ffb(target_lataccel, current_lataccel, state, future_plan),
                        STEER_RANGE[0], STEER_RANGE[1])); self.Acts.append(u); return u

    if self.plan is None or self.pos >= self.K:
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
              torch.tensor(self.Acts[-(CTX - 1):], dtype=torch.float32, device=d),
              torch.tensor(np.stack([roll, vv, ae], 1), dtype=torch.float32, device=d),
              torch.tensor(tgt_np, dtype=torch.float32, device=d),
              torch.tensor(current_lataccel, dtype=torch.float32, device=d))
      if self.prev_plan is not None:
        warm = np.concatenate([self.prev_plan[self.K:], np.repeat(self.prev_plan[-1], self.K)])[:N]; iters = self.warm_iters
      else:
        Bv = B0 + B1 * vv[:N]
        warm = np.clip(0.6 * (tgt_np * (1 - A) - C * roll[:N] - D) / Bv, STEER_RANGE[0], STEER_RANGE[1]); iters = self.max_iters
      try:
        self.plan = self._optimize(warm, hist, iters); self.prev_plan = self.plan
      except Exception:
        self.plan = warm; self.prev_plan = warm
      self.pos = 0

    u = float(np.clip(self.plan[self.pos], STEER_RANGE[0], STEER_RANGE[1])); self.pos += 1
    self.Acts.append(u)
    return u
