class Controller:
    """Negative control: a timed throttle schedule, right on paper, no feedback at all."""

    def __init__(self, cfg):
        pass

    def step(self, t, s):
        if t < 3.0:
            duty = 0.0
        elif t < 5.0:
            duty = 0.65
        elif t < 13.0:
            duty = 0.59
        elif t < 16.0:
            duty = 0.55
        else:
            duty = 0.0
        return {"duty": duty, "vanes": [0.0, 0.0, 0.0, 0.0]}
