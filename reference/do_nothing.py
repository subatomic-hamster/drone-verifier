class Controller:
    """Negative control: never spins the fan."""

    def __init__(self, cfg):
        pass

    def step(self, t, s):
        return {"duty": 0.0, "vanes": [0.0, 0.0, 0.0, 0.0]}
