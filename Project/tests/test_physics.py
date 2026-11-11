"""
test_physics.py - Checks the conductor mechanics and gallop dynamics in gallop_sim.physics.

These tests confirm the simulator's motion is trustworthy as ground truth: the
catenary and modal frequencies match textbook formulas, Den Hartog modes grow to
the predicted limit-cycle amplitude above the critical wind and decay below it,
coupled modes lock each other out as the Orange Book describes, the orbit has the
filmed shape, and the kinematic model follows its scripted envelope.
"""

import math

import numpy as np

from gallop_sim import physics, reference

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


def test_fundamental_matches_book_sag_relation():
    """f1 from tension/mass should match the Orange Book's sag-only form 0.5 sqrt(g / 8 sag)."""
    sag = physics.catenary_sag([SPAN / 2], SPAN, TENSION, MASS, G)[0]
    f1 = physics.modal_frequencies_hz([1], SPAN, TENSION, MASS)[0]
    assert abs(reference.fundamental_frequency_from_sag(sag, G) - f1) / f1 < 0.01


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


def test_coupled_projection_reduces_to_single_mode_case():
    """With one active mode, the coupled cubic projection must equal the classic 0.75 v^3."""
    omega = 2 * math.pi * physics.modal_frequencies_hz([1, 2, 3], SPAN, TENSION, MASS)
    model = physics.DenHartogModes(omega, MASS, ZETA, RHO, DIAM, [1, 1, 1], 1.0, [0, 0, 0], [0, 0, 0],
                                   mode_numbers=(1, 2, 3))
    v = np.array([0.0, 0.7, 0.0])
    assert np.allclose(model.cubic_projection(v), [0.0, 0.75 * 0.7 ** 3, 0.0], atol=1e-12)


def test_coupled_limit_cycle_still_hits_target():
    """Calibration must stay exact for a single mode when coupling is switched on."""
    omega = 2 * math.pi * physics.modal_frequencies_hz([2], SPAN, TENSION, MASS)
    a3 = physics.cubic_coefficient_for_target(1.5, omega[0], MASS, ZETA, RHO, DIAM, 2.0, 15.0)
    model = physics.DenHartogModes(omega, MASS, ZETA, RHO, DIAM, [2.0], a3, [0.05], [0.0], mode_numbers=(2,))
    assert abs(_simulate(model, 15.0, 240.0) - 1.5) / 1.5 < 0.05


def test_first_mode_to_saturate_locks_out_the_other():
    """Two unstable modes: uncoupled both persist; coupled, exactly one survives (Orange Book Sec. 4.3.5)."""
    omega = 2 * math.pi * physics.modal_frequencies_hz([1, 2], SPAN, TENSION, MASS)
    a1 = [1.6, 2.0]   # both unstable at 15 m/s
    a3 = physics.cubic_coefficient_for_target(1.5, omega[1], MASS, ZETA, RHO, DIAM, 2.0, 15.0)
    results = {}
    for coupled in (False, True):
        model = physics.DenHartogModes(omega, MASS, ZETA, RHO, DIAM, a1, a3, [0.05, 0.05], [0.0, 0.0],
                                       mode_numbers=(1, 2) if coupled else None)
        for _ in range(int(400 / (1 / 200))):
            model.step(1 / 200, 15.0)
        results[coupled] = model.envelope()
    assert results[False][0] > 0.5 and results[False][1] > 0.5   # independent: both modes coexist
    winner, loser = max(results[True]), min(results[True])
    assert winner > 1.0 and loser < 0.01 * winner                # coupled: one mode locks the other out


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


def _trace_orbit(sway_ratio, sway_phase_deg, upward_bias=0.0):
    """Return (dz, dy) samples over one steady kinematic cycle at a point with unit mode shape."""
    omega = np.array([2.0])
    shapes = np.array([[1.0]])
    model = physics.KinematicModes(omega, [1.0], [(0, 1.0)], [0.0])
    dz, dy = [], []
    for _ in range(400):
        model.step(0.01)
        z, y = physics.displacement_field(shapes, model.q, model.v, omega, sway_ratio,
                                          math.radians(sway_phase_deg), upward_bias, model.envelope())
        dz.append(z[0]); dy.append(y[0])
    return np.array(dz), np.array(dy)


def test_sway_traces_ellipse():
    """Transverse sway should have the configured amplitude ratio relative to vertical motion."""
    dz, dy = _trace_orbit(0.3, 60.0)
    assert abs(np.ptp(dy) / np.ptp(dz) - 0.3) < 0.01


def test_book_orbit_shape_is_reproduced():
    """Sway derived from the filmed orbit (10 % wide, 5 deg tilt) should trace exactly that ellipse."""
    ratio, phase = reference.sway_from_orbit(0.10, 5.0)
    width, tilt = reference.orbit_from_sway(ratio, phase)
    assert abs(width - 0.10) < 1e-6 and abs(tilt - 5.0) < 1e-6
    dz, dy = _trace_orbit(ratio, phase)
    assert np.ptp(dy) / np.ptp(dz) < 0.2   # horizontal < 1/5 of vertical, as filmed


def test_upward_bias_puts_rest_at_lower_quartile():
    """With bias 0.5 the conductor rises three times as far as it falls (Orange Book Sec. 4.5.4)."""
    dz, _ = _trace_orbit(0.1, 50.0, reference.upward_bias_from_quantile(0.25))
    assert abs(dz.max() / -dz.min() - 3.0) < 0.05
    assert abs(np.ptp(dz) - 2.0) < 0.01   # peak-to-peak unchanged


def test_modal_projection_of_sag_change():
    """A parabolic sag change projects mostly onto mode 1 (coefficient 32/pi^3) and not onto even modes."""
    drop = 2.0
    coeffs = physics.modal_projection(lambda x: drop * 4 * x * (SPAN - x) / SPAN ** 2, SPAN, (1, 2, 3))
    assert abs(coeffs[0] - 32 * drop / math.pi ** 3) < 1e-3
    assert abs(coeffs[1]) < 1e-9 and abs(coeffs[2] - 32 * drop / (27 * math.pi ** 3)) < 1e-3
