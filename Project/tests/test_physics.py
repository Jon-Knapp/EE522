"""
test_physics.py - Checks the conductor mechanics and gallop dynamics in gallop_sim.physics.

These tests confirm the simulator's motion is trustworthy as ground truth: the
catenary and modal frequencies match textbook formulas, Den Hartog modes grow to
the predicted limit-cycle amplitude above the critical wind and decay below it,
and the kinematic model follows its scripted envelope.
"""

import math

import numpy as np

from gallop_sim import physics

SPAN, TENSION, MASS, G = 250.0, 35_000.0, 1.8, 9.81
ZETA, RHO, DIAM = 0.04, 1.25, 0.05


def _single_mode(a1=2.0, target=1.5, u_ref=15.0, q0=0.05):
    """Build a one-mode (mode 2) Den Hartog model calibrated to `target` metres at `u_ref` m/s."""
    omega = 2 * math.pi * physics.modal_frequencies_hz([2], SPAN, TENSION, MASS)
    a3 = physics.cubic_coefficient_for_target(target, omega[0], MASS, ZETA, RHO, DIAM, a1, u_ref)
    model = physics.DenHartogModes(omega, MASS, ZETA, RHO, DIAM, [a1], a3, [q0], [0.0])
    return model, omega, a3


def _simulate(model, wind_ms, seconds, dt=1 / 600):
    """Advance `model` at constant wind for `seconds` and return the final envelope of its first mode."""
    for _ in range(int(seconds / dt)):
        model.step(dt, wind_ms)
    return float(model.envelope()[0])


def test_catenary_sag_matches_parabolic_approximation():
    """Midspan sag should be within 1% of w L^2 / (8 H) for a taut span, and zero at the towers."""
    sag = physics.catenary_sag([0.0, SPAN / 2, SPAN], SPAN, TENSION, MASS, G)
    expected = MASS * G * SPAN ** 2 / (8 * TENSION)
    assert abs(sag[1] - expected) / expected < 0.01
    assert abs(sag[0]) < 1e-9 and abs(sag[2]) < 1e-9


def test_modal_frequencies_follow_string_formula():
    """f_n should equal (n / 2L) * sqrt(T / mu) and scale linearly with mode number."""
    f = physics.modal_frequencies_hz([1, 2, 3], SPAN, TENSION, MASS)
    f1 = math.sqrt(TENSION / MASS) / (2 * SPAN)
    assert np.allclose(f, [f1, 2 * f1, 3 * f1])


def test_mode_shapes_have_nodes_at_towers():
    """Every mode shape should be zero at both ends of the span, and mode 2 zero at midspan."""
    shapes = physics.mode_shapes([0.0, SPAN / 2, SPAN], SPAN, [1, 2, 3])
    assert np.allclose(shapes[:, 0], 0) and np.allclose(shapes[:, 2], 0, atol=1e-12)
    assert abs(shapes[1, 1]) < 1e-12


def test_limit_cycle_reaches_target_amplitude():
    """Above the critical wind, a mode should settle onto the amplitude predicted by harmonic balance."""
    model, omega, a3 = _single_mode()
    final = _simulate(model, 15.0, 240.0)
    assert abs(final - 1.5) / 1.5 < 0.05
    predicted = physics.predicted_amplitude(omega, MASS, ZETA, RHO, DIAM, [2.0], a3, 15.0)[0]
    assert abs(predicted - 1.5) < 1e-6


def test_motion_decays_below_critical_wind():
    """Below the critical wind speed, an initial disturbance should die away."""
    model, omega, _ = _single_mode(q0=0.5)
    u_crit = physics.critical_wind_speed(omega, MASS, ZETA, RHO, DIAM, [2.0])[0]
    assert _simulate(model, 0.5 * u_crit, 60.0) < 0.05


def test_stable_section_never_gallops():
    """With a1 <= 0 (Den Hartog criterion not met) the critical wind should be infinite."""
    omega = 2 * math.pi * physics.modal_frequencies_hz([1, 2], SPAN, TENSION, MASS)
    assert np.all(np.isinf(physics.critical_wind_speed(omega, MASS, ZETA, RHO, DIAM, [0.0, -1.0])))


def test_kinematic_envelope_follows_schedule():
    """The kinematic model's envelope should equal amplitude times the scripted scale at each time."""
    omega = np.array([3.5])
    model = physics.KinematicModes(omega, [2.0], [(0, 0.0), (10, 1.0)], [0.0])
    model.step(5.0)
    assert math.isclose(model.envelope()[0], 1.0)
    model.step(10.0)
    assert math.isclose(model.envelope()[0], 2.0)


def test_sway_traces_ellipse():
    """Transverse sway should have the configured amplitude ratio relative to vertical motion."""
    omega = np.array([2.0])
    shapes = np.array([[1.0]])
    model = physics.KinematicModes(omega, [1.0], [(0, 1.0)], [0.0])
    dz, dy = [], []
    for _ in range(400):
        model.step(0.01)
        z, y = physics.displacement_field(shapes, model.q, model.v, omega, 0.3, math.radians(60))
        dz.append(z[0]); dy.append(y[0])
    assert abs(np.ptp(dy) / np.ptp(dz) - 0.3) < 0.01
