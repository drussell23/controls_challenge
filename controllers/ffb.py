from . import BaseController

# Identified linear plant (sysid.py, R^2=0.988): lat_{t+1}=A*lat+B*steer+C*roll+D
A, B, C, D = 0.9474, 0.0755, 0.0526, -0.0103


class Controller(BaseController):
  """Model-based feed-forward + PID feedback.

  Feed-forward inverts the identified plant for the steady-state steer that
  holds the target (compensating road roll and the slow pole). It also looks
  one step ahead in the future plan to pre-empt the lag. PID corrects the
  residual model error each step (which is why FF+FB beats open-loop FF).
  """

  def __init__(self, p=0.10, i=0.05, d=-0.05, ff=1.0, lookahead=3):
    self.p, self.i, self.d, self.ff, self.lookahead = p, i, d, ff, lookahead
    self.ei = 0.0
    self.pe = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    fut = getattr(future_plan, "lataccel", None)
    ff_target = target_lataccel
    if self.lookahead and fut:
      idx = min(self.lookahead - 1, len(fut) - 1)
      ff_target = fut[idx]
    roll = state.roll_lataccel
    u_ff = self.ff * (ff_target * (1 - A) - C * roll - D) / B

    e = target_lataccel - current_lataccel
    self.ei += e
    de = e - self.pe
    self.pe = e
    return u_ff + self.p * e + self.i * self.ei + self.d * de
