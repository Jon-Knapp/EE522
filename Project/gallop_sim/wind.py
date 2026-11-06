"""
wind.py - Wind speed time series that drives the simulated gallop.

The mean wind follows the piecewise-linear WIND_SCHEDULE from config.py, and
turbulent gusts are added as an Ornstein-Uhlenbeck (exponentially correlated)
random process. The Den Hartog model in physics.py uses this speed to set the
aerodynamic forcing, so gallop starts when the wind rises past the critical speed
and dies away when it drops. Precipitation drift and camera shake in degrade.py
also scale with it, tying the visual nuisances to the same weather.
"""

import math

import numpy as np


class WindModel:
    """Mean wind schedule plus correlated gusts, advanced in small time steps by the simulator."""

    def __init__(self, schedule, gust_std_ms, gust_timescale_s, rng):
        """Store the (time_s, speed_ms) schedule, gust statistics, and random generator; start at t = 0."""
        pairs = np.asarray(schedule, dtype=float)
        self._times = pairs[:, 0]
        self._speeds = pairs[:, 1]
        self.gust_std = float(gust_std_ms)
        self.gust_timescale = max(float(gust_timescale_s), 1e-6)
        self.rng = rng
        self.gust = 0.0
        self.current = self.mean_speed(0.0)

    def mean_speed(self, t):
        """Return the scheduled mean wind speed (m/s) at time t, holding the end values outside the schedule."""
        return float(np.interp(t, self._times, self._speeds))

    def step(self, t, dt):
        """Advance the gust process by dt and return the total wind speed (m/s, never negative) at time t + dt.

        Uses the exact discrete update of an Ornstein-Uhlenbeck process, so the gust statistics
        do not depend on the step size.
        """
        decay = math.exp(-dt / self.gust_timescale)
        self.gust = self.gust * decay + self.gust_std * math.sqrt(1.0 - decay ** 2) * self.rng.standard_normal()
        self.current = max(0.0, self.mean_speed(t + dt) + self.gust)
        return self.current

    def max_mean_speed(self):
        """Return the largest mean speed in the schedule, used for the physics summary."""
        return float(self._speeds.max())
