"""
config.py - Central configuration for the conductor-gallop synthetic video generator.

This is the single place to adjust the Step 1 simulation: span and conductor
physics, the gallop model, wind, camera, rendering, image degradations, ground
truth, and alert thresholds. gallop_sim.settings.load_config() copies every
UPPERCASE name below into a config object, so main.py can override values from
the command line without editing this file.

[OB x] marks values taken from EPRI's "Transmission Line Reference Book: Wind-Induced
Conductor Motion" (the Orange Book), section/equation/table x. gallop_sim/reference.py
holds the book data and the README lists the citations.

Coordinate system (right-handed, metres):
    X - along the span, from the left tower (X = 0) to the right tower (X = SPAN_LENGTH_M)
    Y - horizontal, perpendicular to the span (transverse / "sway" direction); wind blows toward +Y
    Z - vertical, up from the ground

Units are SI unless a name says otherwise (_PX = pixels, _DEG = degrees,
_BGR = OpenCV colour triplet with 0-255 channels in blue, green, red order).
"""

# =============================================================================
# Run and output
# =============================================================================
RANDOM_SEED = 42                 # Same seed + same config -> identical video and ground truth
OUTPUT_DIR = "output"            # Relative paths are resolved against the project folder
RUN_NAME = None                  # None -> timestamped folder, e.g. output/gallop_20261105_154900
DURATION_S = 120.0               # Length of the generated clip
FPS = 30.0                       # Video frame rate (gallop is 0.08-3 Hz [OB Table 1.3-1], so 30 fps is ample)
VIDEO_CODEC = "mp4v"             # FourCC passed to cv2.VideoWriter; "mp4v" works on Windows out of the box
SAVE_VIDEO = True                # Write the degraded (realistic) video
SAVE_CLEAN_VIDEO = False         # Also write an undegraded copy (no weather, shake, or noise) as a baseline
SAMPLE_FRAME_INTERVAL_S = 10.0   # Save a PNG of the degraded frame every N seconds (0 disables)
PREVIEW = False                  # Show frames in an OpenCV window while rendering (press q to stop)

# =============================================================================
# Span and conductor physics (defaults for every conductor in CONDUCTORS)
# =============================================================================
SPAN_LENGTH_M = 250.0            # Tower-to-tower distance (field galloping data cover 50-450 m [OB Sec. 4.5.4])
CONDUCTOR_PRESET = "drake_acsr"  # Bare-conductor properties from [OB App. 1]; see gallop_sim/reference.py
CONDUCTOR_TENSION_N = 35_000.0   # Horizontal tension: 25 % RTS of Drake; lines are strung at 20-33 % RTS [OB Sec. 4.3.3]
CONDUCTOR_MASS_PER_M = 1.6281    # Bare mass, used only when no preset applies (Drake value)
CONDUCTOR_DIAMETER_M = 0.028143  # Bare diameter, used only when no preset applies (Drake value)
ICE_THICKNESS_M = 0.006          # Equivalent radial glaze ice; thin-to-moderate ice gallops most [OB App. 4.5]
ICE_DENSITY_KGM3 = 900.0         # Glaze ice; the ice mass is added to the bare mass
DAMPING_RATIO = 0.04             # Effective mechanical damping ratio (structural + still-air), fraction of critical
GRAVITY_MS2 = 9.81
AIR_DENSITY_KGM3 = 1.25          # Cold air is denser (1.52 kg/m^3 at -40 C [OB Sec. 2.2.4])
CONDUCTOR_POINTS = 240           # Points used to draw each conductor along the span

# Conductors on the span. Any key from the defaults above can be overridden per conductor:
#   preset, tension_n or tension_pct_rts, mass_per_m (bare), diameter_m (bare), ice_thickness_m, sway_ratio
# attach_y_m / attach_z_m: where the wire hangs at the towers (bottom of the insulator string).
# insulator_m: insulator string length (0 = attached directly to the tower peak, e.g. a shield wire).
# aero_scale: multiplies the gallop driving term (Den Hartog) or amplitudes (kinematic) for this wire.
CONDUCTORS = [
    # 3/8" EHS steel shield wire [OB Table A1-11] at ~18 % RTS, so its sag is close to the phases'.
    {"name": "shield", "attach_y_m": 0.0, "attach_z_m": 37.0, "insulator_m": 0.0,
     "preset": "ehs_3_8_steel", "tension_n": 12_500.0, "aero_scale": 0.6},
    {"name": "phase_A", "attach_y_m": 5.0, "attach_z_m": 30.0, "insulator_m": 2.5, "aero_scale": 1.0},
    {"name": "phase_B", "attach_y_m": 5.5, "attach_z_m": 24.0, "insulator_m": 2.5, "aero_scale": 0.9},
    {"name": "phase_C", "attach_y_m": 5.0, "attach_z_m": 18.0, "insulator_m": 2.5, "aero_scale": 0.8},
]

# =============================================================================
# Gallop model
# =============================================================================
MOTION_MODEL = "den_hartog"      # "den_hartog" (physics-driven onset) or "kinematic" (scripted amplitudes)
MODE_NUMBERS = (1, 2, 3)         # Standing-wave modes simulated (mode n has n loops; spans settle on <= 3 [OB Sec. 4.2])
MODE_COUPLING = True             # Coupled cubic term: the first mode to saturate locks the others out [OB Sec. 4.3.5]

# Orbit shape. Films of 44 galloping events show narrow ellipses, width ~10 % of height, tilted ~5 deg
# from vertical, with the static position at the lower quartile of the motion (rises 3x as far as it
# falls) [OB Sec. 4.5.4, Fig. 4.5-31]. SWAY_RATIO / SWAY_PHASE_DEG = None derive the sway from the
# ORBIT_* values; give numbers to set them directly (ratio 0.3, phase 60 reproduces the old default).
ORBIT_WIDTH_RATIO = 0.10
ORBIT_TILT_DEG = 5.0             # Positive leans the top of the orbit downwind (+Y)
ORBIT_STATIC_QUANTILE = 0.25     # 0.5 = symmetric motion about the static position
SWAY_RATIO = None                # Transverse (Y) amplitude as a fraction of vertical
SWAY_PHASE_DEG = None            # Phase lead of transverse motion relative to vertical

# --- Den Hartog model -------------------------------------------------------
# a1 = -(dCL/dalpha + CD) for the iced cross-section; a1 > 0 means the Den Hartog
# criterion is met and aerodynamic damping turns negative as wind increases [OB Eq. 4.3-1].
DEN_HARTOG_A1 = 2.0
# Per-mode multiplier on a1 (which modes the wind favours). Twisting under eccentric ice makes the
# midspan ice more stable, which favours two loops on single conductors [OB Sec. 4.3.5]. Field reports
# for phase conductors: 1 loop 42, 2 loops 26, 3 loops 34 cases [OB Table 4.2-1] - try (1.0, 0.3, 0.3)
# for one-loop gallop or (0.3, 0.6, 1.0) for three loops.
MODE_AERO_WEIGHTS = (0.3, 1.0, 0.6)
# How the cubic (saturation) term is set:
#   "hunt_richards" - the most-favoured mode reaches f * Ymax / V = HUNT_RICHARDS_RATIO at REFERENCE_WIND_MS,
#                     where Ymax is peak-to-peak; field values spread 0.05-0.30, guide max 0.26 [OB Eq. A4.5-9]
#   "target"        - it reaches TARGET_AMPLITUDE_M (semi-amplitude) at REFERENCE_WIND_MS instead
AMPLITUDE_CALIBRATION = "hunt_richards"
HUNT_RICHARDS_RATIO = 0.10
TARGET_AMPLITUDE_M = 1.5
REFERENCE_WIND_MS = 15.0         # 10 m reference wind (same units as WIND_SCHEDULE) used for the calibration
INITIAL_PERTURBATION_M = 0.05    # Std. dev. of the random initial modal displacement
PHYSICS_SUBSTEPS = 20            # RK4 integration steps per video frame

# --- Kinematic model --------------------------------------------------------
KINEMATIC_AMPLITUDES_M = (0.2, 1.5, 0.3)  # Peak modal amplitude for each entry in MODE_NUMBERS
# Envelope multiplier vs. time as (time_s, scale) pairs, linearly interpolated.
KINEMATIC_ENVELOPE = [(0.0, 0.0), (20.0, 0.0), (50.0, 1.0), (90.0, 1.0), (110.0, 0.0)]

# --- Ice shedding [OB Sec. 7.4] (Den Hartog model only) ----------------------
# (time_s, conductor_name) pairs: at that time the conductor drops its ice, jumps up toward its bare-wire
# sag, rings down, and stops galloping. A realistic event an alerting system should not confuse with gallop.
ICE_SHED_EVENTS = []             # e.g. [(95.0, "phase_C")]

# =============================================================================
# Wind (drives the Den Hartog model, blowout, precipitation drift, and camera shake)
# =============================================================================
# Mean wind speed at the 10 m reference height vs. time as (time_s, speed_ms) pairs, linearly interpolated.
# Galloping is typically seen in steady 7-18 m/s crosswinds [OB Table 1.3-1].
WIND_SCHEDULE = [(0.0, 4.0), (15.0, 4.0), (40.0, 16.0), (85.0, 16.0), (100.0, 5.0), (120.0, 5.0)]
TERRAIN_CATEGORY = 1             # 1 open, 2 farmland, 3 suburban, 4 urban: sets the height profile and turbulence [OB Table 2.2-6]
WIND_ANGLE_DEG = 0.0             # Angle between the wind and the normal to the line; only the normal component acts
WIND_GUST_STD_MS = None          # Gust std. dev.; None = turbulence intensity 1/ln(z/z0) x mean speed [OB Eq. 2.2-7]
WIND_GUST_TIMESCALE_S = 5.0      # Correlation time of gusts (Ornstein-Uhlenbeck process)

# Blowout: the span swings about its chord under mean wind drag [OB Eq. 7.5-4]
BLOWOUT_ENABLED = True
BLOWOUT_DRAG_COEFF = 1.0         # Cd assumed in [OB Eq. 7.5-4]

# =============================================================================
# Camera (pinhole model)
# =============================================================================
IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 720
CAMERA_HFOV_DEG = 65.0           # Horizontal field of view
CAMERA_POSITION_M = (125.0, -210.0, 8.0)   # Broadside view: best for amplitudes [OB Sec. 4.4.3]
CAMERA_TARGET_M = (125.0, 0.0, 24.0)       # Point the optical axis passes through
# Tower-mounted alternative (looking along the span; best for telling modes and phases apart [OB Sec. 4.4.3]):
#   CAMERA_POSITION_M = (-3.0, -2.0, 20.0); CAMERA_TARGET_M = (250.0, 5.0, 22.0); CAMERA_HFOV_DEG = 40.0

# =============================================================================
# Scene rendering
# =============================================================================
SKY_STYLE = "overcast"           # "clear" or "overcast"
CLOUD_COVER = 0.5                # Fraction of sky covered by clouds (0-1)
CLOUD_DRIFT_PX_PER_S = 4.0       # Horizontal cloud motion (a moving background for trackers to ignore)
BACKGROUND_IMAGE = None          # Path to a photo to use as the background instead of the procedural sky
GROUND_COLOR_BGR = (205, 208, 205)     # Snowy field
TREELINE_COLOR_BGR = (60, 72, 58)
TREELINE_HEIGHT_PX = 14
WIRE_COLOR_BGR = (55, 55, 60)          # Dark conductor; try a pale grey to mimic heavy rime ice
MIN_WIRE_ALPHA = 0.35                  # Minimum opacity for wires thinner than a pixel
TOWER_COLOR_BGR = (70, 70, 75)
TOWER_MEMBER_WIDTH_M = 0.15
TOWER_BASE_HALF_WIDTH_M = 4.0
TOWER_TOP_HALF_WIDTH_M = 1.0
TOWER_BRACE_SPACING_M = 4.0
TOWER_PEAK_ABOVE_M = 1.0               # Tower body extends this far above the highest cross-arm
CROSSARM_DEPTH_M = 1.5                 # Vertical depth of each triangular cross-arm

# =============================================================================
# Image degradations (applied in this order: fog, shake, precipitation, blur, illumination, noise)
# =============================================================================
FOG_DENSITY = 0.12               # 0 = none, 1 = everything is fog colour
FOG_COLOR_BGR = (205, 205, 205)
SHAKE_ENABLED = True
SHAKE_STD_PX = 1.5               # Translation jitter at REFERENCE_WIND_MS
SHAKE_ROT_STD_DEG = 0.05         # Rotational jitter at REFERENCE_WIND_MS
SHAKE_TIMESCALE_S = 0.5          # Correlation time of the jitter
SHAKE_SCALES_WITH_WIND = True    # Jitter grows with wind speed (camera pole sway)
WEATHER = "snow"                 # "none", "rain", or "snow"
PRECIP_PARTICLES = 350           # Number of rain streaks / snowflakes on screen
PRECIP_DRIFT_PX_PER_MS = 12.0    # Horizontal particle drift per m/s of wind
BLUR_SIGMA_PX = 0.0              # Gaussian defocus blur (0 disables)
ILLUMINATION_GAIN = 1.0          # Overall brightness multiplier (<1 for dusk)
ILLUMINATION_FLICKER_STD = 0.01  # Frame-to-frame exposure flicker
SENSOR_NOISE_STD = 3.0           # Additive Gaussian noise, 0-255 scale

# =============================================================================
# Ground truth and alerting
# =============================================================================
GT_SAMPLE_FRACTIONS = (0.25, 0.5, 0.75)   # Positions along the span logged every frame (0 = left tower)
ALERT_THRESHOLD_M = 0.75         # Raise an alert when a conductor's amplitude envelope (semi-amplitude) reaches this
ALERT_CLEAR_M = 0.5              # Clear it when the envelope falls below this (hysteresis)
