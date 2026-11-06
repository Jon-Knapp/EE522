"""
scene.py - 3D geometry of the simulated span: conductors and lattice towers.

build_conductors() turns each entry of config.CONDUCTORS into a Conductor that
owns its rest catenary, mode shapes, aerodynamic coefficients, and a gallop
dynamics object from physics.py. tower_segments() produces the line segments of a
simple lattice tower with cross-arms and insulator strings. render.py projects and
draws this geometry; simulator.py advances each conductor's dynamics and logs its
displacement as ground truth.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import physics


@dataclass
class Conductor:
    """One conductor (phase or shield wire) on the span, including its current gallop state."""

    name: str
    attach_y_m: float
    attach_z_m: float
    insulator_m: float
    tension_n: float
    mass_per_m: float
    diameter_m: float
    aero_diameter_m: float
    span_m: float
    gravity: float
    mode_numbers: tuple
    x_grid: np.ndarray
    rest_z: np.ndarray
    shapes: np.ndarray
    omegas: np.ndarray
    a1: np.ndarray
    a3: float
    sway_ratio: float
    sway_phase_rad: float
    dynamics: object

    @property
    def frequencies_hz(self):
        """Return the modal natural frequencies in hertz."""
        return self.omegas / (2.0 * math.pi)

    @property
    def sag_m(self):
        """Return the static midspan sag in metres."""
        return float(self.attach_z_m - self.rest_z.min())

    def displacement(self, shapes=None):
        """Return (dz, dy) gallop displacement at the x positions of `shapes` (default: the drawing grid)."""
        shapes = self.shapes if shapes is None else shapes
        return physics.displacement_field(shapes, self.dynamics.q, self.dynamics.v, self.omegas,
                                          self.sway_ratio, self.sway_phase_rad)

    def world_points(self):
        """Return an (N, 3) array of the conductor's current 3D points along the span."""
        dz, dy = self.displacement()
        return np.column_stack([self.x_grid, self.attach_y_m + dy, self.rest_z + dz])

    def sample_points(self, fractions):
        """Return (points, dz, dy) at fractional positions along the span (0 = left tower, 1 = right tower).

        Used for ground truth: `points` is (K, 3) world coordinates; dz and dy are the gallop
        displacements from the rest position at those positions.
        """
        xs = np.asarray(fractions, dtype=float) * self.span_m
        shapes = physics.mode_shapes(xs, self.span_m, self.mode_numbers)
        dz, dy = self.displacement(shapes)
        rest_z = self.attach_z_m - physics.catenary_sag(xs, self.span_m, self.tension_n,
                                                        self.mass_per_m, self.gravity)
        points = np.column_stack([xs, self.attach_y_m + dy, rest_z + dz])
        return points, dz, dy

    def amplitude(self):
        """Return the current vertical gallop amplitude envelope (m), the quantity alerts are based on."""
        return physics.amplitude_envelope(self.dynamics.envelope(), self.shapes)


def build_conductors(cfg, rng):
    """Create a Conductor for every entry in cfg.CONDUCTORS, filling unspecified values from global defaults.

    Computes modal frequencies, aerodynamic coefficients, and the cubic saturation term, then
    attaches Den Hartog or kinematic dynamics with randomized initial conditions from `rng`.
    """
    span = float(cfg.SPAN_LENGTH_M)
    modes = tuple(int(n) for n in cfg.MODE_NUMBERS)
    weights = np.asarray(cfg.MODE_AERO_WEIGHTS, dtype=float)
    ref_mode = int(np.argmax(weights))
    x_grid = np.linspace(0.0, span, int(cfg.CONDUCTOR_POINTS))
    shapes = physics.mode_shapes(x_grid, span, modes)
    conductors = []

    for spec in cfg.CONDUCTORS:
        tension = float(spec.get("tension_n", cfg.CONDUCTOR_TENSION_N))
        mass = float(spec.get("mass_per_m", cfg.CONDUCTOR_MASS_PER_M))
        diameter = float(spec.get("diameter_m", cfg.CONDUCTOR_DIAMETER_M))
        ice = float(spec.get("ice_thickness_m", cfg.ICE_THICKNESS_M))
        aero_d = diameter + 2.0 * ice
        aero_scale = float(spec.get("aero_scale", 1.0))
        attach_z = float(spec["attach_z_m"])

        omegas = 2.0 * math.pi * physics.modal_frequencies_hz(modes, span, tension, mass)
        rest_z = attach_z - physics.catenary_sag(x_grid, span, tension, mass, cfg.GRAVITY_MS2)
        a1_section = cfg.DEN_HARTOG_A1 * weights
        a1 = a1_section * aero_scale

        if cfg.MOTION_MODEL == "den_hartog":
            # a3 describes the iced cross-section, so it is set from the unscaled a1 of the reference mode.
            a3 = physics.cubic_coefficient_for_target(
                cfg.TARGET_AMPLITUDE_M, omegas[ref_mode], mass, cfg.DAMPING_RATIO,
                cfg.AIR_DENSITY_KGM3, aero_d, a1_section[ref_mode], cfg.REFERENCE_WIND_MS)
            q0 = rng.normal(0.0, cfg.INITIAL_PERTURBATION_M, len(modes))
            v0 = rng.normal(0.0, cfg.INITIAL_PERTURBATION_M, len(modes)) * omegas
            dynamics = physics.DenHartogModes(omegas, mass, cfg.DAMPING_RATIO, cfg.AIR_DENSITY_KGM3,
                                              aero_d, a1, a3, q0, v0)
        else:
            a3 = float("nan")
            amplitudes = np.asarray(cfg.KINEMATIC_AMPLITUDES_M, dtype=float) * aero_scale
            phases = rng.uniform(0.0, 2.0 * math.pi, len(modes))
            dynamics = physics.KinematicModes(omegas, amplitudes, cfg.KINEMATIC_ENVELOPE, phases)

        conductors.append(Conductor(
            name=spec["name"], attach_y_m=float(spec["attach_y_m"]), attach_z_m=attach_z,
            insulator_m=float(spec.get("insulator_m", 0.0)), tension_n=tension, mass_per_m=mass,
            diameter_m=diameter, aero_diameter_m=aero_d, span_m=span, gravity=cfg.GRAVITY_MS2,
            mode_numbers=modes, x_grid=x_grid, rest_z=rest_z, shapes=shapes, omegas=omegas,
            a1=a1, a3=a3, sway_ratio=float(spec.get("sway_ratio", cfg.SWAY_RATIO)),
            sway_phase_rad=math.radians(cfg.SWAY_PHASE_DEG), dynamics=dynamics))
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
