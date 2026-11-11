"""
scene.py - 3D geometry of the simulated span: conductors and lattice towers.

build_conductors() turns each entry of config.CONDUCTORS into a Conductor that
owns its rest catenary, mode shapes, aerodynamic coefficients, and a gallop
dynamics object from physics.py. Conductor properties can come from presets taken
from the Orange Book conductor tables (reference.CONDUCTOR_PRESETS), with the ice
mass added from the radial ice thickness.

Each conductor also carries two slower, book-derived effects that a camera sees:

* blowout - the whole catenary swings about its chord under the mean wind drag,
  tan(theta) = 0.5 rho V^2 Cd d / w (Orange Book Eq. 7.5-4), with the wind averaged
  over half the swing period (Sec. 7.5.7). Gallop motion is centred on this
  blown-out position (Sec. 4.5.4), so a CV system must not mistake swing for gallop;
* ice shedding - when the ice drops off, the sag shrinks and the conductor jumps up
  and rings down at the higher bare-wire frequency (Orange Book Sec. 7.4). Without
  ice the cross-section is round and the Den Hartog driving term disappears.

tower_segments() produces the line segments of a simple lattice tower with
cross-arms and insulator strings. render.py projects and draws this geometry;
simulator.py advances each conductor's dynamics and logs its displacement as
ground truth.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import physics, reference


@dataclass
class Conductor:
    """One conductor (phase or shield wire) on the span, including its current gallop, swing, and ice state."""

    name: str
    attach_y_m: float
    attach_z_m: float
    insulator_m: float
    tension_n: float
    bare_mass_per_m: float
    mass_per_m: float
    diameter_m: float
    ice_thickness_m: float
    aero_diameter_m: float
    rated_strength_n: float
    span_m: float
    gravity: float
    mode_numbers: tuple
    x_grid: np.ndarray
    rest_sag: np.ndarray
    shapes: np.ndarray
    omegas: np.ndarray
    a1: np.ndarray
    a3: float
    sway_ratio: float
    sway_phase_rad: float
    upward_bias: float
    wind_factor: float
    dynamics: object
    blowout_rad: float = 0.0
    wind_ms: float = 0.0
    filtered_wind_ms: float = 0.0
    iced: bool = True

    # ----------------------------------------------------------------- statics
    @property
    def frequencies_hz(self):
        """Return the modal natural frequencies in hertz."""
        return self.omegas / (2.0 * math.pi)

    @property
    def one_loop_frequency_hz(self):
        """Return the fundamental (one-loop) frequency, whether or not mode 1 is among the simulated modes."""
        return float(self.omegas[0] / (2.0 * math.pi * self.mode_numbers[0]))

    @property
    def sag_m(self):
        """Return the static midspan sag in metres (in the plane of the conductor, before blowout)."""
        return float(self.rest_sag.max())

    @property
    def rest_z(self):
        """Return the at-rest heights along the drawing grid with no blowout (for reference and tests)."""
        return self.attach_z_m - self.rest_sag

    @property
    def mean_height_m(self):
        """Return the span-averaged conductor height (attachment minus two-thirds of the sag) used for the wind."""
        return self.attach_z_m - 2.0 * self.sag_m / 3.0

    def sag_at(self, xs):
        """Return the static sag below the chord at positions xs for the current (iced or bare) mass."""
        return physics.catenary_sag(xs, self.span_m, self.tension_n, self.mass_per_m, self.gravity)

    # ----------------------------------------------------------------- motion
    def displacement(self, shapes=None):
        """Return (dz, dy) gallop displacement at the x positions of `shapes` (default: the drawing grid)."""
        shapes = self.shapes if shapes is None else shapes
        return physics.displacement_field(shapes, self.dynamics.q, self.dynamics.v, self.omegas,
                                          self.sway_ratio, self.sway_phase_rad, self.upward_bias,
                                          self.dynamics.envelope())

    def _positions(self, xs, sag, dz, dy):
        """Combine the blown-out catenary and the gallop displacement into (N, 3) world points."""
        s, c = math.sin(self.blowout_rad), math.cos(self.blowout_rad)
        return np.column_stack([xs, self.attach_y_m + sag * s + dy, self.attach_z_m - sag * c + dz])

    def world_points(self):
        """Return an (N, 3) array of the conductor's current 3D points along the span."""
        dz, dy = self.displacement()
        return self._positions(self.x_grid, self.rest_sag, dz, dy)

    def sample_points(self, fractions):
        """Return (points, dz, dy) at fractional positions along the span (0 = left tower, 1 = right tower).

        Used for ground truth: `points` is (K, 3) world coordinates; dz and dy are the gallop
        displacements from the (blown-out) rest position at those positions.
        """
        xs = np.asarray(fractions, dtype=float) * self.span_m
        shapes = physics.mode_shapes(xs, self.span_m, self.mode_numbers)
        dz, dy = self.displacement(shapes)
        return self._positions(xs, self.sag_at(xs), dz, dy), dz, dy

    def amplitude(self):
        """Return the current vertical gallop amplitude envelope (m, semi-amplitude), the quantity alerts use."""
        return physics.amplitude_envelope(self.dynamics.envelope(), self.shapes)

    # ----------------------------------------------------------------- wind effects
    def update_wind(self, reference_wind_ms, dt, cfg):
        """Set this conductor's line-normal wind and advance its blowout angle by dt seconds.

        Blowout follows the wind averaged over half the swing period (first-order filter with that
        time constant), as recommended for swing-out estimates (Orange Book Sec. 7.5.7).
        """
        self.wind_ms = reference_wind_ms * self.wind_factor
        tau = 0.5 / self.one_loop_frequency_hz   # half the swing (one-loop) period
        self.filtered_wind_ms += (self.wind_ms - self.filtered_wind_ms) * (1.0 - math.exp(-dt / tau))
        if cfg.BLOWOUT_ENABLED:
            self.blowout_rad = reference.blowout_angle_rad(
                self.filtered_wind_ms, cfg.AIR_DENSITY_KGM3, cfg.BLOWOUT_DRAG_COEFF,
                self.aero_diameter_m, self.mass_per_m * self.gravity)
        return self.wind_ms

    def shed_ice(self):
        """Drop all ice: lighter, rounder conductor, smaller sag, higher frequencies, and an upward jump.

        The change of sag is projected onto the simulated modes as an initial displacement, so the
        span rises toward its new bare-wire position and rings down with mechanical damping. This is
        the linear version of the energy picture in Orange Book Sec. 7.4.3 (real shedding is nonlinear
        and often happens in chunks rather than over the whole span at once).
        """
        if not self.iced:
            return
        old_mass = self.mass_per_m
        self.mass_per_m = self.bare_mass_per_m
        self.aero_diameter_m = self.diameter_m
        self.ice_thickness_m = 0.0
        self.iced = False

        def sag_drop(x):
            """Return how far the conductor sits below its new rest position: -(old sag - new sag)."""
            old = physics.catenary_sag(x, self.span_m, self.tension_n, old_mass, self.gravity)
            return -(old - self.sag_at(x))

        self.rest_sag = self.sag_at(self.x_grid)
        self.omegas = 2.0 * math.pi * physics.modal_frequencies_hz(self.mode_numbers, self.span_m,
                                                                   self.tension_n, self.mass_per_m)
        self.a1 = np.zeros_like(self.a1)   # bare round conductor: no Den Hartog driving
        self.upward_bias = 0.0             # the filmed up/down asymmetry describes gallop, not the ice-jump ring-down
        if hasattr(self.dynamics, "set_section"):
            self.dynamics.set_section(self.omegas, self.mass_per_m, self.aero_diameter_m, self.a1)
            self.dynamics.q = self.dynamics.q + physics.modal_projection(sag_drop, self.span_m, self.mode_numbers)


def resolve_conductor_properties(spec, cfg):
    """Return (diameter_m, bare_mass_per_m, rated_strength_n, tension_n) for one CONDUCTORS entry.

    Precedence: explicit values in the entry, then the entry's preset, then CONDUCTOR_PRESET,
    then the global CONDUCTOR_* defaults. Tension may be given as tension_pct_rts (percent of the
    preset's rated strength).
    """
    preset_name = spec.get("preset", cfg.CONDUCTOR_PRESET)
    preset = reference.CONDUCTOR_PRESETS.get(preset_name, {}) if preset_name else {}
    diameter = float(spec.get("diameter_m", preset.get("diameter_m", cfg.CONDUCTOR_DIAMETER_M)))
    mass = float(spec.get("mass_per_m", preset.get("mass_per_m", cfg.CONDUCTOR_MASS_PER_M)))
    rts = float(spec.get("rated_strength_n", preset.get("rated_strength_n", float("nan"))))
    if "tension_n" in spec:
        tension = float(spec["tension_n"])
    elif "tension_pct_rts" in spec:
        if not math.isfinite(rts):
            raise ValueError(f"Conductor '{spec['name']}' uses tension_pct_rts but has no rated strength")
        tension = rts * float(spec["tension_pct_rts"]) / 100.0
    else:
        tension = float(cfg.CONDUCTOR_TENSION_N)
    return diameter, mass, rts, tension


def orbit_settings(cfg):
    """Return (sway_ratio, sway_phase_deg): explicit SWAY_* values, or those that trace the ORBIT_* ellipse."""
    derived_ratio, derived_phase = reference.sway_from_orbit(cfg.ORBIT_WIDTH_RATIO, cfg.ORBIT_TILT_DEG)
    ratio = derived_ratio if cfg.SWAY_RATIO is None else float(cfg.SWAY_RATIO)
    phase = derived_phase if cfg.SWAY_PHASE_DEG is None else float(cfg.SWAY_PHASE_DEG)
    return ratio, phase


def wind_factor_for_height(cfg, height_m):
    """Return line-normal wind at height_m divided by the 10 m reference wind (power law x wind angle)."""
    alpha = reference.TERRAIN_CATEGORIES[int(cfg.TERRAIN_CATEGORY)]["alpha"]
    return reference.power_law_factor(height_m, alpha) * abs(math.cos(math.radians(cfg.WIND_ANGLE_DEG)))


def build_conductors(cfg, rng):
    """Create a Conductor for every entry in cfg.CONDUCTORS, filling unspecified values from global defaults.

    Computes iced mass, sag, modal frequencies, the wind at each conductor's height, aerodynamic
    coefficients, and the cubic saturation term, then attaches Den Hartog or kinematic dynamics with
    randomized initial conditions from `rng`.
    """
    span = float(cfg.SPAN_LENGTH_M)
    modes = tuple(int(n) for n in cfg.MODE_NUMBERS)
    weights = np.asarray(cfg.MODE_AERO_WEIGHTS, dtype=float)
    ref_mode = int(np.argmax(weights))
    x_grid = np.linspace(0.0, span, int(cfg.CONDUCTOR_POINTS))
    shapes = physics.mode_shapes(x_grid, span, modes)
    sway_ratio_default, sway_phase_deg = orbit_settings(cfg)
    upward_bias = reference.upward_bias_from_quantile(cfg.ORBIT_STATIC_QUANTILE)
    conductors = []

    for spec in cfg.CONDUCTORS:
        diameter, bare_mass, rts, tension = resolve_conductor_properties(spec, cfg)
        ice = float(spec.get("ice_thickness_m", cfg.ICE_THICKNESS_M))
        mass = bare_mass + reference.ice_mass_per_m(diameter, ice, cfg.ICE_DENSITY_KGM3)
        aero_d = diameter + 2.0 * ice
        aero_scale = float(spec.get("aero_scale", 1.0))
        attach_z = float(spec["attach_z_m"])

        omegas = 2.0 * math.pi * physics.modal_frequencies_hz(modes, span, tension, mass)
        rest_sag = physics.catenary_sag(x_grid, span, tension, mass, cfg.GRAVITY_MS2)
        wind_factor = wind_factor_for_height(cfg, attach_z - 2.0 * rest_sag.max() / 3.0)
        a1_section = cfg.DEN_HARTOG_A1 * weights
        a1 = a1_section * aero_scale

        if cfg.MOTION_MODEL == "den_hartog":
            u_ref = cfg.REFERENCE_WIND_MS * wind_factor
            if cfg.AMPLITUDE_CALIBRATION == "hunt_richards":
                f_ref = omegas[ref_mode] / (2.0 * math.pi)
                target = 0.5 * reference.hunt_richards_pkpk(u_ref, f_ref, cfg.HUNT_RICHARDS_RATIO)
            else:
                target = float(cfg.TARGET_AMPLITUDE_M)
            # a3 describes the iced cross-section, so it is set from the unscaled a1 of the reference mode.
            a3 = physics.cubic_coefficient_for_target(
                target, omegas[ref_mode], mass, cfg.DAMPING_RATIO,
                cfg.AIR_DENSITY_KGM3, aero_d, a1_section[ref_mode], u_ref)
            q0 = rng.normal(0.0, cfg.INITIAL_PERTURBATION_M, len(modes))
            v0 = rng.normal(0.0, cfg.INITIAL_PERTURBATION_M, len(modes)) * omegas
            dynamics = physics.DenHartogModes(omegas, mass, cfg.DAMPING_RATIO, cfg.AIR_DENSITY_KGM3,
                                              aero_d, a1, a3, q0, v0,
                                              mode_numbers=modes if cfg.MODE_COUPLING else None)
        else:
            a3 = float("nan")
            amplitudes = np.asarray(cfg.KINEMATIC_AMPLITUDES_M, dtype=float) * aero_scale
            phases = rng.uniform(0.0, 2.0 * math.pi, len(modes))
            dynamics = physics.KinematicModes(omegas, amplitudes, cfg.KINEMATIC_ENVELOPE, phases)

        conductor = Conductor(
            name=spec["name"], attach_y_m=float(spec["attach_y_m"]), attach_z_m=attach_z,
            insulator_m=float(spec.get("insulator_m", 0.0)), tension_n=tension, bare_mass_per_m=bare_mass,
            mass_per_m=mass, diameter_m=diameter, ice_thickness_m=ice, aero_diameter_m=aero_d,
            rated_strength_n=rts, span_m=span, gravity=cfg.GRAVITY_MS2, mode_numbers=modes, x_grid=x_grid,
            rest_sag=rest_sag, shapes=shapes, omegas=omegas, a1=a1, a3=a3,
            sway_ratio=float(spec.get("sway_ratio", sway_ratio_default)),
            sway_phase_rad=math.radians(sway_phase_deg), upward_bias=upward_bias,
            wind_factor=wind_factor, dynamics=dynamics, iced=ice > 0)
        conductors.append(conductor)
    return conductors


def settle_wind(conductors, reference_wind_ms, cfg):
    """Start every conductor already in equilibrium with the initial wind (filtered wind and blowout settled)."""
    for c in conductors:
        c.filtered_wind_ms = reference_wind_ms * c.wind_factor
        c.update_wind(reference_wind_ms, 1e-6, cfg)
    return conductors


def tower_segments(x_m, conductors, cfg):
    """Return a list of (start, end) 3D point pairs describing a lattice tower at span position x_m.

    The tower has four tapered legs with horizontal and X-bracing, a triangular cross-arm and
    insulator string for each insulated conductor, and a small peak for each shield wire.
    """
    arm_heights = [c.attach_z_m + c.insulator_m for c in conductors if c.insulator_m > 0]
    body_top = (max(arm_heights) if arm_heights else max(c.attach_z_m for c in conductors) - 1.0) \
        + cfg.TOWER_PEAK_ABOVE_M
    base, top = cfg.TOWER_BASE_HALF_WIDTH_M, cfg.TOWER_TOP_HALF_WIDTH_M

    def half_width(z):
        """Return the tower's half-width at height z (linear taper from base to top)."""
        return base + (top - base) * min(max(z / body_top, 0.0), 1.0)

    def corners(z):
        """Return the four leg positions at height z, ordered around the tower."""
        hw = half_width(z)
        return [np.array([x_m + sx * hw, sy * hw, z]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    n_levels = max(2, int(math.ceil(body_top / cfg.TOWER_BRACE_SPACING_M)) + 1)
    levels = [corners(z) for z in np.linspace(0.0, body_top, n_levels)]
    segments = []
    for lower, upper in zip(levels[:-1], levels[1:]):
        for i in range(4):
            j = (i + 1) % 4
            segments.append((lower[i], upper[i]))      # leg
            segments.append((lower[i], upper[j]))      # X-brace on this face
            segments.append((lower[j], upper[i]))
    for ring in levels:
        for i in range(4):
            segments.append((ring[i], ring[(i + 1) % 4]))  # horizontal bracing

    for c in conductors:
        side = 1.0 if c.attach_y_m >= 0 else -1.0
        if c.insulator_m > 0:
            arm_z = c.attach_z_m + c.insulator_m
            tip = np.array([x_m, c.attach_y_m, arm_z])
            for z in (arm_z, arm_z - cfg.CROSSARM_DEPTH_M):
                hw = half_width(z)
                for sx in (-1, 1):
                    segments.append((np.array([x_m + sx * hw, side * hw, z]), tip))
            segments.append((tip, np.array([x_m, c.attach_y_m, c.attach_z_m])))  # insulator string
        else:
            peak = np.array([x_m, c.attach_y_m, c.attach_z_m])
            for corner in levels[-1]:
                segments.append((corner, peak))
    return segments
