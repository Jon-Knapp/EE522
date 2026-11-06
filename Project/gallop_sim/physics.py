"""
physics.py - Conductor mechanics and gallop dynamics.

A span is modelled as a taut string hanging in a catenary. Gallop displacement is
a sum of standing-wave modes,

    z(x, t) = sum_n q_n(t) * sin(n * pi * x / L),

whose modal coordinates q_n(t) come from one of two models:

* DenHartogModes - quasi-steady aerodynamics of an iced conductor. Each mode obeys

      m q'' + c q' + m w^2 q = 0.5 * rho * D * (a1 * U * q' - 0.75 * a3 * q'^3 / U)

  where a1 = -(dCL/dalpha + CD). When a1 > 0 (the Den Hartog criterion) the
  aerodynamic term acts as negative damping that grows with wind speed U; once it
  exceeds the mechanical damping c the mode grows, and the cubic term saturates it
  onto a limit cycle. Gallop therefore starts, grows, and decays naturally as the
  wind changes, which is what an alerting system must handle.
* KinematicModes - prescribed sinusoids with a scripted amplitude envelope, for
  controlled tests where gallop must reach an exact amplitude at an exact time.

Transverse sway is derived from the vertical motion with a fixed ratio and phase
lead, giving the elliptical orbits seen on galloping lines. scene.py converts
these displacements to 3D wire positions; simulator.py advances the dynamics.
"""

import math

import numpy as np

MIN_EFFECTIVE_WIND_MS = 0.5   # Keeps the cubic term (which divides by U) finite in calm air
HARMONIC_BALANCE_FACTOR = 9.0 / 16.0  # (3/4 modal projection of q'^3) x (3/4 fundamental of sin^3)


def catenary_sag(x, span_m, tension_n, mass_per_m, gravity=9.81):
    """Return the sag (m, positive downward) below the chord at positions x for a level span.

    Uses the exact catenary for horizontal tension `tension_n` and weight `mass_per_m * gravity`
    per metre; sag is zero at both towers and largest at midspan.
    """
    a = tension_n / (mass_per_m * gravity)
    x = np.asarray(x, dtype=float)
    return a * (np.cosh(span_m / (2.0 * a)) - np.cosh((x - span_m / 2.0) / a))


def modal_frequencies_hz(mode_numbers, span_m, tension_n, mass_per_m):
    """Return taut-string natural frequencies f_n = (n / 2L) * sqrt(T / mu) in hertz.

    Sag and bending stiffness are neglected, a reasonable first approximation for the
    low-order modes involved in gallop.
    """
    wave_speed = math.sqrt(tension_n / mass_per_m)
    return np.asarray(mode_numbers, dtype=float) * wave_speed / (2.0 * span_m)


def mode_shapes(x, span_m, mode_numbers):
    """Return an (n_modes, len(x)) array of mode shapes sin(n * pi * x / L).

    Each row is one standing wave with nodes at the towers; mode n has n loops.
    """
    n = np.asarray(mode_numbers, dtype=float)[:, None]
    return np.sin(n * np.pi * np.atleast_1d(np.asarray(x, dtype=float))[None, :] / span_m)


def damping_coefficient(damping_ratio, omega, mass_per_m):
    """Return the viscous damping coefficient c = 2 * zeta * omega * m per unit length.

    This is the mechanical damping that aerodynamic negative damping must overcome for gallop to start.
    """
    return 2.0 * damping_ratio * np.asarray(omega, dtype=float) * mass_per_m


def critical_wind_speed(omega, mass_per_m, damping_ratio, air_density, aero_diameter_m, a1):
    """Return, per mode, the wind speed (m/s) above which the mode becomes Den Hartog unstable.

    Instability begins when 0.5 * rho * D * a1 * U exceeds the mechanical damping c; modes with
    a1 <= 0 are aerodynamically stable and return infinity.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), omega.shape)
    c = damping_coefficient(damping_ratio, omega, mass_per_m)
    denom = 0.5 * air_density * aero_diameter_m * a1
    result = np.full(omega.shape, np.inf)
    positive = denom > 0
    result[positive] = c[positive] / denom[positive]
    return result


def cubic_coefficient_for_target(target_amplitude_m, omega, mass_per_m, damping_ratio,
                                 air_density, aero_diameter_m, a1, reference_wind_ms):
    """Return the cubic aerodynamic coefficient a3 that puts a mode's limit cycle at a target amplitude.

    Derived from harmonic balance of the modal equation at the reference wind; raises ValueError
    if that wind is below the mode's critical speed, because no limit cycle exists there.
    """
    c = 2.0 * damping_ratio * omega * mass_per_m
    net = 0.5 * air_density * aero_diameter_m * a1 * reference_wind_ms - c
    if net <= 0:
        u_crit = c / (0.5 * air_density * aero_diameter_m * a1) if a1 > 0 else float("inf")
        raise ValueError(
            f"REFERENCE_WIND_MS ({reference_wind_ms} m/s) is below the critical wind speed "
            f"({u_crit:.1f} m/s) of the reference mode; raise REFERENCE_WIND_MS or DEN_HARTOG_A1, "
            f"or lower DAMPING_RATIO")
    return net * reference_wind_ms / (
        0.5 * air_density * aero_diameter_m * HARMONIC_BALANCE_FACTOR * (target_amplitude_m * omega) ** 2)


def predicted_amplitude(omega, mass_per_m, damping_ratio, air_density, aero_diameter_m, a1, a3, wind_ms):
    """Return the harmonic-balance limit-cycle amplitude (m) of each mode at a steady wind speed.

    Modes below their critical wind speed return 0. Handy for sanity-checking config values
    before rendering a long video.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), omega.shape)
    u = max(float(wind_ms), MIN_EFFECTIVE_WIND_MS)
    net = 0.5 * air_density * aero_diameter_m * a1 * u - damping_coefficient(damping_ratio, omega, mass_per_m)
    amplitude = np.zeros(omega.shape)
    ok = (net > 0) & (a3 > 0)
    amplitude[ok] = np.sqrt(net[ok] * u / (0.5 * air_density * aero_diameter_m
                                             * HARMONIC_BALANCE_FACTOR * a3)) / omega[ok]
    return amplitude


class DenHartogModes:
    """Modal state of one conductor driven by quasi-steady Den Hartog aerodynamics (see module docstring)."""

    def __init__(self, omegas, mass_per_m, damping_ratio, air_density, aero_diameter_m,
                 a1, a3, initial_q, initial_v):
        """Store per-mode parameters and the initial modal displacement and velocity."""
        self.omegas = np.asarray(omegas, dtype=float)
        self.mass = float(mass_per_m)
        self.a1 = np.asarray(a1, dtype=float)
        self.a3 = float(a3)
        self._c = damping_coefficient(damping_ratio, self.omegas, self.mass)
        self._k_aero = 0.5 * air_density * aero_diameter_m
        self.q = np.array(initial_q, dtype=float)
        self.v = np.array(initial_v, dtype=float)

    def _acceleration(self, q, v, wind_ms):
        """Return modal accelerations q'' for state (q, v) at the given wind speed."""
        u = max(wind_ms, MIN_EFFECTIVE_WIND_MS)
        aero = self._k_aero * (self.a1 * u * v - 0.75 * self.a3 * v ** 3 / u)
        return (aero - self._c * v) / self.mass - self.omegas ** 2 * q

    def step(self, dt, wind_ms):
        """Advance the modal state by dt seconds using one classical fourth-order Runge-Kutta step.

        Wind is held constant over the step, which is accurate because steps are much shorter
        than both the gallop period and the gust timescale.
        """
        q, v = self.q, self.v
        k1q, k1v = v, self._acceleration(q, v, wind_ms)
        k2q, k2v = v + 0.5 * dt * k1v, self._acceleration(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, wind_ms)
        k3q, k3v = v + 0.5 * dt * k2v, self._acceleration(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, wind_ms)
        k4q, k4v = v + dt * k3v, self._acceleration(q + dt * k3q, v + dt * k3v, wind_ms)
        self.q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        self.v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)

    def envelope(self):
        """Return each mode's instantaneous amplitude envelope sqrt(q^2 + (q'/w)^2) in metres."""
        return np.sqrt(self.q ** 2 + (self.v / self.omegas) ** 2)


class KinematicModes:
    """Prescribed modal motion q_n(t) = A_n * s(t) * cos(w_n t + phi_n) with a scripted envelope s(t)."""

    def __init__(self, omegas, amplitudes_m, envelope_schedule, phases_rad):
        """Store mode frequencies, peak amplitudes, the (time_s, scale) envelope, and phases; start at t = 0."""
        self.omegas = np.asarray(omegas, dtype=float)
        self.amplitudes = np.asarray(amplitudes_m, dtype=float)
        pairs = np.asarray(envelope_schedule, dtype=float)
        self._env_t, self._env_s = pairs[:, 0], pairs[:, 1]
        self.phases = np.asarray(phases_rad, dtype=float)
        self.time = 0.0
        self.update(0.0)

    def scale(self, t):
        """Return the envelope multiplier s(t) by linear interpolation of the schedule."""
        return float(np.interp(t, self._env_t, self._env_s))

    def update(self, t):
        """Set the modal displacement q and velocity q' for absolute time t."""
        self.time = t
        s = self.scale(t)
        angle = self.omegas * t + self.phases
        self.q = self.amplitudes * s * np.cos(angle)
        self.v = -self.amplitudes * s * self.omegas * np.sin(angle)

    def step(self, dt, wind_ms=None):
        """Advance the internal clock by dt and update the state; wind is ignored by this model."""
        self.update(self.time + dt)

    def envelope(self):
        """Return each mode's current amplitude envelope A_n * s(t) in metres."""
        return np.abs(self.amplitudes * self.scale(self.time))


def displacement_field(shapes, q, v, omegas, sway_ratio, sway_phase_rad):
    """Return vertical (dz) and transverse (dy) displacement at the x positions represented by `shapes`.

    Vertical motion is the modal sum; transverse sway reuses each mode's motion scaled by
    `sway_ratio` and phase-shifted by `sway_phase_rad`, which traces elliptical orbits.
    """
    dz = q @ shapes
    q_sway = sway_ratio * (math.cos(sway_phase_rad) * q + math.sin(sway_phase_rad) * v / omegas)
    dy = q_sway @ shapes
    return dz, dy


def amplitude_envelope(envelopes, shapes):
    """Return the largest vertical displacement envelope (m) anywhere along the span.

    Weights each mode's envelope by |mode shape| and takes the maximum over x: exact for one
    active mode and a conservative upper bound when several modes are active.
    """
    return float(np.max(np.abs(shapes).T @ envelopes))
