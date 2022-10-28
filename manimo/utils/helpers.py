import time

HOMES = {
    "pour": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "scoop": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "zip": [-0.1337, 0.3634, -0.1395, -2.3153, 0.1478, 2.7733, -1.1784],
    "insertion": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
}

class Rate:
    """
    Maintains constant control rate for the environment loop
    """

    def __init__(self, frequency):
        self._period = 1.0 / frequency
        self._last = time.time()

    def sleep(self):
        current_delta = time.time() - self._last
        sleep_time = max(0, self._period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self._last = time.time()