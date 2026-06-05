from . import BaseController

# Speed-scheduled plant (sysid on official 0-4999 range, R^2=0.990):
#   lat_{t+1} = A*lat_t + B(v)*steer_t + C*roll_t + D,   B(v) = B0 + B1*v_ego
A, B0, B1, C, D = 0.9416, 0.0830, 0.00044, 0.0628, -0.0102


class Controller(BaseController):
  """Inverse-model feed-forward + PID feedback.

  Feed-forward inverts the (speed-scheduled) plant on the KNOWN next target:
      u_t = (target_{t+1} - A*target_t - C*roll_t - D) / B(v)
  i.e. the exact steer that lands lataccel on the next target in one step,
  anticipating target changes and compensating the slow pole. PID corrects
  only the residual model error. No horizon -> no multi-step compounding.
  """

  def __init__(self, p=0.05, i=0.01, d=-0.01, ff=1.0, steer_limit=2.5):
    self.p, self.i, self.d, self.ff, self.steer_limit = p, i, d, ff, steer_limit
    self.ei = 0.0
    self.pe = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    fut = getattr(future_plan, "lataccel", None)
    tgt_next = fut[0] if fut else target_lataccel
    v = state.v_ego
    roll = state.roll_lataccel
    B = B0 + B1 * v
    u_ff = self.ff * (tgt_next - A * target_lataccel - C * roll - D) / B

    e = target_lataccel - current_lataccel
    self.ei += e
    de = e - self.pe
    self.pe = e
    u = u_ff + self.p * e + self.i * self.ei + self.d * de
    return float(min(max(u, -self.steer_limit), self.steer_limit))
