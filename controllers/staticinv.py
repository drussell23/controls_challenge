from . import BaseController
A, B, C, D = 0.9474, 0.0755, 0.0526, -0.0103
class Controller(BaseController):
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return (target_lataccel * (1 - A) - C * state.roll_lataccel - D) / B
