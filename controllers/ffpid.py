from . import BaseController


class Controller(BaseController):
  """
  Feed-forward + PID controller.

  The provided baseline PID is purely reactive: it only acts on the tracking
  error and ignores both the target itself and the known road-roll disturbance.
  In this simulator the steer->lataccel gain is ~1, so we can feed the desired
  lataccel forward directly and let the PID handle only the residual error.

  - feed-forward: target_lataccel (the bulk of the command)
  - PID: corrects the residual tracking error
  """

  def __init__(self):
    self.ff = 1.0          # steer per unit target lataccel (sim gain ~1)
    self.p = 0.30
    self.i = 0.05
    self.d = -0.10
    self.error_integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    feedforward = self.ff * target_lataccel
    feedback = self.p * error + self.i * self.error_integral + self.d * error_diff
    return feedforward + feedback
