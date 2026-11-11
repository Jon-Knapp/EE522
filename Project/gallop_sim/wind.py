"""
wind.py - Wind speed time series that drives the simulated gallop.

The mean wind follows the piecewise-linear WIND_SCHEDULE from config.py, specified
at the 10 m meteorological reference height, and turbulent gusts are added as an
Ornstein-Uhlenbeck (exponentially correlated) random process. When
WIND_GUST_STD_MS is None the gust strength follows the terrain: the standard
deviation is the turbulence intensity Iu = 1 / ln(z / z0) times the mean speed
(Orange Book Eq. 2.2-7, Table 2.2-6).

Each conductor hangs at a different height, so wind_at() scales the reference speed
with the power-law profile V(z) = V_ref (z / 10 m) ** alpha (Orange Book Eqs. 2.2-6
and 7.5-5) and keeps only the component normal to the line, which is what drives
galloping and blowout (Orange Book Sec. 4.3.3, Fig. 4.3-33, and Eq. 7.5-4). The Den Hartog model
in physics.py uses that per-conductor speed, so gallop starts when the wind rises
past the critical speed and dies away when it drops. Precipitation drift and camera
shake in degrade.py scale with the reference speed.
"""

import math

import numpy as np

from . import reference


class WindModel:
    """Mean wind schedule plus correlated gusts at the reference height, advanced in small steps by the simulator."""

    def __init__(self, schedule, gust_std_ms, gust_timescale_s, rng, terrain_category=1,
                 wind_angle_deg=0.0, reference_height_m=reference.WIND_REFERENCE_HEIGHT_M):
        """Store the (time_s, speed_ms) schedule, gust statistics, terrain, and random generator; start at t = 0."""
        pairs = np.asarray(schedule, dtype=float)
        self._times = pairs[:, 0]
        self._speeds = pairs[:, 1]
        self.gust_std = None if gust_std_ms is None else float(gust_std_ms)
        self.gust_timescale = max(float(gust_timescale_s), 1e-6)
        self.rng = rng
        self.terrain = reference.TERRAIN_CATEGORIES[int(terrain_category)]
        self.reference_height = float(reference_height_m)
        self.normal_factor = abs(math.cos(math.radians(float(wind_angle_deg))))
        self.intensity = reference.turbulence_intensity(self.reference_height, self.terrain["z0_m"])
        self._unit_gust = 0.0
        self.gust = 0.0
        self.current = self.mean_speed(0.0)

    def mean_speed(self, t):
        """Return the scheduled mean wind speed (m/s) at time t, holding the end values outside the schedule."""
        return float(np.interp(t, self._times, self._speeds))

    def gust_std_at(self, t):
        """Return the gust standard deviation (m/s) at time t: the fixed setting, or Iu times the mean speed."""
        if self.gust_std is not None:
            return self.gust_std
        return self.intensity * self.mean_speed(t)

    def step(self, t, dt):
        """Advance the gust process by dt and return the total reference-height wind (m/s, never negative) at t + dt.

        Uses the exact discrete update of a unit-variance Ornstein-Uhlenbeck process, scaled by the
        current gust standard deviation, so the gust statistics do not depend on the step size.
        """
        decay = math.exp(-dt / self.gust_timescale)
        self._unit_gust = self._unit_gust * decay + math.sqrt(1.0 - decay ** 2) * self.rng.standard_normal()
        self.gust = self.gust_std_at(t + dt) * self._unit_gust
        self.current = max(0.0, self.mean_speed(t + dt) + self.gust)
        return self.current

    def height_factor(self, height_m):
        """Return V(z) / V_ref for a conductor at height_m from the terrain's power-law exponent."""
        return reference.power_law_factor(height_m, self.terrain["alpha"], self.reference_height)

    def wind_at(self, height_m, reference_speed_ms=None):
        """Return the line-normal wind speed (m/s) at height_m for a reference-height speed (default: current)."""
        speed = self.current if reference_speed_ms is None else reference_speed_ms
        return speed * self.height_factor(height_m) * self.normal_factor

    def max_mean_speed(self):
        """Return the largest mean speed in the schedule, used for the physics summary."""
        return float(self._speeds.max())
