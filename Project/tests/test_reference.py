"""
test_reference.py - Checks the Orange Book relations in gallop_sim.reference against the book's own examples.

Each test reproduces a number printed in EPRI's Transmission Line Reference Book:
Wind-Induced Conductor Motion, so a transcription error would show up here.
"""

import math

from gallop_sim import reference


def test_hunt_richards_matches_book_example():
    """Sec. 4.3.5: 10 m/s wind, 5 m sag -> one-loop f = 0.25 Hz, Ymax = 10.4 m; two-loop gives 5.2 m."""
    f1 = reference.fundamental_frequency_from_sag(5.0)
    assert abs(f1 - 0.25) < 0.005
    assert abs(reference.hunt_richards_pkpk(10.0, 0.25) - 10.4) < 1e-9
    assert abs(reference.hunt_richards_pkpk(10.0, 0.50) - 5.2) < 1e-9


def test_lilien_havard_single_exceeds_bundle():
    """Eqs. 4.5-16/17: single conductors gallop up to ~2.5x more (in diameters) than bundles for the same sag."""
    sag, d = 8.0, 0.03
    single = reference.lilien_havard_max_pkpk(sag, d, bundle=False)
    bundle = reference.lilien_havard_max_pkpk(sag, d, bundle=True)
    assert single > bundle > 0
    assert 0 < single / d <= 500   # reduced amplitude range seen in the database


def test_turbulence_intensity_matches_eurocode_form():
    """Eq. 2.2-7: Iu = 1 / ln(z / z0); open terrain at 10 m gives about 14.5 %."""
    assert abs(reference.turbulence_intensity(10.0, 0.01) - 1 / math.log(1000)) < 1e-12


def test_power_law_profile():
    """Eq. 7.5-5 with alpha = 0.16 (the book's example) scales 10 m wind to 30 m by 3 ** 0.16."""
    assert abs(reference.power_law_factor(30.0, 0.16) - 3 ** 0.16) < 1e-12


def test_blowout_angle_balances_drag_and_weight():
    """Eq. 7.5-4: tan(theta) = drag / weight; 45 degrees when they are equal."""
    rho, cd, d = 1.25, 1.0, 0.03
    wind = 15.0
    drag = 0.5 * rho * wind ** 2 * cd * d
    assert abs(reference.blowout_angle_rad(wind, rho, cd, d, drag) - math.pi / 4) < 1e-12
    assert reference.blowout_angle_rad(0.0, rho, cd, d, 10.0) == 0.0


def test_drake_preset_matches_book():
    """Drake ACSR 795 kcmil: 28.143 mm, 1.6281 kg/m; 28,024 N is ~20 % RTS (Sec. 2.2.5, 2.5.2)."""
    drake = reference.CONDUCTOR_PRESETS["drake_acsr"]
    assert drake["diameter_m"] == 0.028143 and drake["mass_per_m"] == 1.6281
    assert abs(28_024 / drake["rated_strength_n"] - 0.20) < 0.005


def test_book_checks_flag_out_of_range_cases():
    """A 5 Hz 'gallop' at 25 m/s onset should be flagged; a typical case should not."""
    typical = reference.book_checks("ok", [0.5], 5.0, 0.028, [3.0], 16.0, 10.0)
    assert typical == []
    odd = reference.book_checks("odd", [5.0], 5.0, 0.028, [12.0], 16.0, 25.0)
    assert len(odd) >= 3
