"""
reference.py - Empirical data and relations from the EPRI "Orange Book".

Source: EPRI, *Transmission Line Reference Book: Wind-Induced Conductor Motion*,
2nd ed. (Palo Alto, CA: EPRI, 2006), cited below as [OB]. Section, equation,
table and figure numbers refer to that book so each value can be checked.

Everything book-derived lives here so the rest of the simulator reads as plain
physics and the provenance of each number is in one place:

* conductor properties from the Appendix 1 tables (presets for config.py),
* typical galloping ranges from Table 1.3-1 (used for plausibility checks),
* wind profile and turbulence by terrain category (Ch. 2, Table 2.2-6),
* field-data amplitude envelopes (Lilien & Havard, Hunt & Richards; Ch. 4),
* the galloping orbit shape measured from films (Pon & Havard; Sec. 4.5),
* the mean blowout (swing-out) angle of a conductor in wind (Ch. 7, Eq. 7.5-4),
* observed number-of-loops statistics (Table 4.2-1).

These are empirical design aids, not laws: the book itself stresses that field
data are sparse and scattered. They are used to choose realistic defaults and to
flag configurations that fall outside what has been observed on real lines.
"""

import math

import numpy as np

# -----------------------------------------------------------------------------
# Conductor presets [OB Appendix 1 and Ch. 2]
# -----------------------------------------------------------------------------
# diameter_m / mass_per_m are bare-conductor values; rated_strength_n is RTS.
CONDUCTOR_PRESETS = {
    # 795 kcmil 26/7 "Drake" ACSR: d = 28.143 mm, m = 1.6281 kg/m [OB Sec. 2.2.5 example];
    # 28,024 N quoted as ~20 % RTS [OB Sec. 2.5.2], i.e. RTS ~ 140 kN.
    "drake_acsr": {"diameter_m": 0.028143, "mass_per_m": 1.6281, "rated_strength_n": 140_100.0},
    # 3/8 in. 7-wire galvanized steel strand, Extra-High-Strength grade, a common shield wire
    # [OB Table A1-11]: d = 9.144 mm, 406.3 kg/km, RTS 68.50 kN.
    "ehs_3_8_steel": {"diameter_m": 0.009144, "mass_per_m": 0.4063, "rated_strength_n": 68_500.0},
    # 795 kcmil-equivalent "Greeley" AAAC 6201-T81 [OB Table A1-7]: d = 28.143 mm, 1295.3 kg/km, 135.67 kN.
    "greeley_aaac": {"diameter_m": 0.028143, "mass_per_m": 1.2953, "rated_strength_n": 135_670.0},
}

GLAZE_ICE_DENSITY_KGM3 = 900.0   # Freezing-rain (glaze) ice; galloping ice is usually thin glaze [OB Sec. 4.3.1]

# -----------------------------------------------------------------------------
# Typical ranges for galloping [OB Table 1.3-1]
# -----------------------------------------------------------------------------
GALLOP_FREQUENCY_RANGE_HZ = (0.08, 3.0)
GALLOP_PKPK_RANGE_DIAMETERS = (5.0, 300.0)      # peak-to-peak amplitude in conductor diameters
GALLOP_WIND_RANGE_MS = (7.0, 18.0)              # steady crosswind

# Observed number of loops per span (cases reported) [OB Table 4.2-1].
LOOPS_OBSERVED = {"phase": {1: 42, 2: 26, 3: 34, 4: 2}, "ground_wire": {1: 2, 2: 3, 3: 6, 4: 1}}

# -----------------------------------------------------------------------------
# Wind profile and turbulence [OB Sec. 2.2.5, Table 2.2-6, Eqs. 2.2-6, 2.2-7; Sec. 7.5, Eq. 7.5-5]
# -----------------------------------------------------------------------------
# Terrain category -> power-law exponent alpha and roughness length z0 (m).
TERRAIN_CATEGORIES = {
    1: {"name": "open terrain", "alpha": 0.12, "z0_m": 0.01},
    2: {"name": "farmland", "alpha": 0.16, "z0_m": 0.05},
    3: {"name": "suburban", "alpha": 0.22, "z0_m": 0.30},
    4: {"name": "urban", "alpha": 0.30, "z0_m": 1.00},
}
WIND_REFERENCE_HEIGHT_M = 10.0   # Meteorological standard height [OB Sec. 2.2.5]


def power_law_factor(height_m, alpha, reference_height_m=WIND_REFERENCE_HEIGHT_M):
    """Return V(z) / V_ref = (z / z_ref) ** alpha, the mean-wind height correction [OB Eqs. 2.2-6, 7.5-5]."""
    return (max(float(height_m), 0.5) / reference_height_m) ** float(alpha)


def turbulence_intensity(height_m, z0_m):
    """Return the longitudinal turbulence intensity Iu = 1 / ln(z / z0) for flat terrain [OB Eq. 2.2-7]."""
    return 1.0 / math.log(max(float(height_m), 1.01 * z0_m) / z0_m)


# -----------------------------------------------------------------------------
# Amplitude envelopes from field data [OB Sec. 4.5.4 and Appendix 4.5]
# -----------------------------------------------------------------------------
HUNT_RICHARDS_RATIO_MAX = 0.26           # f * Ymax / V upper guide, exceeded by ~9 % of cases [OB Eq. A4.5-9]
HUNT_RICHARDS_RATIO_RANGE = (0.05, 0.30)  # Spread of observed f * Ymax / V [OB Fig. A4.5-2]
MAX_PKPK_OVER_SAG_FILMED = {"single": 1.7, "bundle": 0.93}   # From 44 scaled films [OB Sec. 4.5.4]


def hunt_richards_pkpk(wind_ms, frequency_hz, ratio=HUNT_RICHARDS_RATIO_MAX):
    """Return the peak-to-peak amplitude (m) implied by f * Ymax / V = ratio [OB Eq. A4.5-9].

    Ymax is peak-to-peak; the book's example (10 m/s, 0.25 Hz one-loop) gives 10.4 m.
    """
    return float(ratio) * float(wind_ms) / float(frequency_hz)


def lilien_havard_max_pkpk(sag_m, diameter_m, bundle=False):
    """Return the field-data envelope of maximum peak-to-peak amplitude (m) [OB Eqs. 4.5-16, 4.5-17].

    Single conductors:  A/phi = 80  * ln(8 f / (50 phi))
    Bundles:            A/phi = 170 * ln(8 f / (500 phi))
    with f the sag and phi the conductor diameter (Lilien and Havard 2000). Returns 0 where the
    fit would be negative (very small sag relative to diameter, outside the data).
    """
    coefficient, divisor = (170.0, 500.0) if bundle else (80.0, 50.0)
    reduced = coefficient * math.log(8.0 * sag_m / (divisor * diameter_m))
    return max(reduced, 0.0) * diameter_m


def conductor_span_parameter(sag_m, diameter_m):
    """Return the dimensionless conductor span parameter 100 * phi / f used in [OB Eq. 4.5-15]."""
    return 100.0 * diameter_m / sag_m


def fundamental_frequency_from_sag(sag_m, gravity=9.81):
    """Return the one-loop frequency (Hz) of a level span from its sag: f1 = 0.5 * sqrt(g / (8 sag)).

    Equivalent to f1 = sqrt(T/m) / 2L with sag = m g L^2 / 8T; used in [OB Sec. 4.3.5] (5 m sag -> 0.25 Hz).
    """
    return 0.5 * math.sqrt(gravity / (8.0 * sag_m))


# -----------------------------------------------------------------------------
# Galloping orbit shape [OB Sec. 4.5.4, "Orbit Shape and Orientation", Fig. 4.5-31]
# -----------------------------------------------------------------------------
# Films of 44 events (Pon and Havard 1994): horizontal motion < 1/5 of vertical; proposed envelope
# is an ellipse whose width is 10 % of its height, tilted 5 degrees from vertical, with the static
# conductor position at the lower quartile of the motion (upward excursion ~3x downward).
ORBIT_WIDTH_RATIO = 0.10
ORBIT_TILT_DEG = 5.0
ORBIT_STATIC_POSITION_QUANTILE = 0.25


def sway_from_orbit(width_ratio, tilt_deg):
    """Return (sway_ratio, sway_phase_deg) that make the simulator trace a given orbit ellipse.

    The simulator moves a point as z = A cos(wt), y = r A cos(wt + phi). This inverts that for an
    ellipse with minor/major axis ratio `width_ratio` whose major axis is tilted `tilt_deg` from
    vertical (positive tilt leans the top of the orbit downwind, +Y).
    """
    w = float(width_ratio)
    a = math.radians(float(tilt_deg))
    z_amp = math.sqrt(math.cos(a) ** 2 + (w * math.sin(a)) ** 2)
    y_amp = math.sqrt(math.sin(a) ** 2 + (w * math.cos(a)) ** 2)
    cos_phi = (1.0 - w * w) * math.sin(a) * math.cos(a) / (z_amp * y_amp)
    return y_amp / z_amp, math.degrees(math.acos(max(-1.0, min(1.0, cos_phi))))


def orbit_from_sway(sway_ratio, sway_phase_deg):
    """Return (width_ratio, tilt_deg) of the orbit ellipse traced for a given sway ratio and phase."""
    r = float(sway_ratio)
    c = r * math.cos(math.radians(sway_phase_deg))
    cov = np.array([[1.0, c], [c, r * r]]) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    tilt = math.degrees(math.atan2(major[1], major[0]))
    if tilt > 90.0:
        tilt -= 180.0
    elif tilt < -90.0:
        tilt += 180.0
    return math.sqrt(max(eigvals[0], 0.0) / eigvals[1]), tilt


def upward_bias_from_quantile(quantile=ORBIT_STATIC_POSITION_QUANTILE):
    """Return the vertical offset of the orbit centre, as a fraction of the semi-amplitude.

    If the static position sits at quantile q of a motion of peak-to-peak 2A, the centre lies
    (1 - 2q) A above it: q = 0.25 gives 0.5 A (up 1.5 A, down 0.5 A, i.e. up = 3x down).
    """
    return 1.0 - 2.0 * float(quantile)


# -----------------------------------------------------------------------------
# Mean blowout (swing-out) angle [OB Sec. 7.5.5, Eq. 7.5-4]
# -----------------------------------------------------------------------------
BLOWOUT_DRAG_COEFFICIENT = 1.0   # Value assumed in [OB Eq. 7.5-4]


def blowout_angle_rad(wind_normal_ms, air_density, drag_coefficient, diameter_m, weight_n_per_m):
    """Return the mean swing angle of a conductor about its chord, tan(theta) = 0.5 rho V^2 Cd d / w.

    Per-unit-length form of [OB Eq. 7.5-4] with the insulator weight neglected. The book uses a wind
    averaged over half the swing period (Tsujimoto et al. 1982, [OB Sec. 7.5.7]); scene.py applies that filter.
    """
    drag = 0.5 * air_density * wind_normal_ms ** 2 * drag_coefficient * diameter_m
    return math.atan2(drag, weight_n_per_m)


def ice_mass_per_m(diameter_m, radial_thickness_m, density=GLAZE_ICE_DENSITY_KGM3):
    """Return the mass per metre (kg/m) of a uniform radial ice layer on a conductor."""
    t = float(radial_thickness_m)
    return density * math.pi * t * (float(diameter_m) + t)


# -----------------------------------------------------------------------------
# Plausibility report
# -----------------------------------------------------------------------------
def book_checks(name, frequencies_hz, sag_m, diameter_m, predicted_pkpk_m, wind_normal_ms,
                critical_wind_ref_ms=None, bundle=False):
    """Return a list of human-readable warnings where a conductor's settings fall outside field experience.

    Compares modal frequencies, critical wind, and predicted peak-to-peak amplitude against
    [OB Table 1.3-1], the Lilien-Havard envelope [OB Eq. 4.5-16/17], the filmed amplitude/sag maximum
    [OB Sec. 4.5.4], and the Hunt-Richards ratio [OB Eq. A4.5-9].
    """
    notes = []
    f_lo, f_hi = GALLOP_FREQUENCY_RANGE_HZ
    for f in np.atleast_1d(frequencies_hz):
        if not f_lo <= f <= f_hi:
            notes.append(f"{name}: mode frequency {f:.2f} Hz is outside the {f_lo}-{f_hi} Hz galloping range")
    if critical_wind_ref_ms is not None and math.isfinite(critical_wind_ref_ms):
        w_lo, w_hi = GALLOP_WIND_RANGE_MS
        if not w_lo <= critical_wind_ref_ms <= w_hi:
            notes.append(f"{name}: onset wind {critical_wind_ref_ms:.1f} m/s is outside the {w_lo}-{w_hi} m/s "
                         f"range where galloping is usually seen")
    for f, pkpk in zip(np.atleast_1d(frequencies_hz), np.atleast_1d(predicted_pkpk_m)):
        if pkpk <= 0:
            continue
        d_lo, d_hi = GALLOP_PKPK_RANGE_DIAMETERS
        if not d_lo <= pkpk / diameter_m <= d_hi:
            notes.append(f"{name}: {pkpk:.2f} m peak-to-peak is {pkpk / diameter_m:.0f} diameters "
                         f"(observed range {d_lo:.0f}-{d_hi:.0f})")
        envelope = lilien_havard_max_pkpk(sag_m, diameter_m, bundle)
        if pkpk > envelope:
            notes.append(f"{name}: {pkpk:.2f} m peak-to-peak exceeds the field-data envelope {envelope:.2f} m")
        limit = MAX_PKPK_OVER_SAG_FILMED["bundle" if bundle else "single"]
        if pkpk / sag_m > limit:
            notes.append(f"{name}: peak-to-peak / sag = {pkpk / sag_m:.2f} exceeds the filmed maximum {limit}")
        if wind_normal_ms > 0:
            ratio = f * pkpk / wind_normal_ms
            lo, hi = HUNT_RICHARDS_RATIO_RANGE
            if not lo <= ratio <= hi:
                notes.append(f"{name}: f*Ymax/V = {ratio:.3f} at {f:.2f} Hz is outside the observed {lo}-{hi}")
    return notes
