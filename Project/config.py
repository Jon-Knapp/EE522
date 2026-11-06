"""
config.py - Central configuration for the conductor-gallop synthetic video generator.

This is the single place to adjust the Step 1 simulation: span and conductor
physics, the gallop model, wind, camera, rendering, image degradations, ground
truth, and alert thresholds. gallop_sim.settings.load_config() copies every
UPPERCASE name below into a config object, so main.py can override values from
the command line without editing this file.

Coordinate system (right-handed, metres):
    X - along the span, from the left tower (X = 0) to the right tower (X = SPAN_LENGTH_M)
    Y - horizontal, perpendicular to the span (transverse / "sway" direction)
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
FPS = 30.0                       # Video frame rate (gallop is ~0.1-1 Hz, so 30 fps is ample)
VIDEO_CODEC = "mp4v"             # FourCC passed to cv2.VideoWriter; "mp4v" works on Windows out of the box
SAVE_VIDEO = True                # Write the degraded (realistic) video
SAVE_CLEAN_VIDEO = False         # Also write an undegraded copy (no weather, shake, or noise) as a baseline
SAMPLE_FRAME_INTERVAL_S = 10.0   # Save a PNG of the degraded frame every N seconds (0 disables)
PREVIEW = False                  # Show frames in an OpenCV window while rendering (press q to stop)

# =============================================================================
# Span and conductor physics (defaults for every conductor in CONDUCTORS)
# =============================================================================
SPAN_LENGTH_M = 250.0            # Tower-to-tower distance
CONDUCTOR_TENSION_N = 35_000.0   # Horizontal tension
CONDUCTOR_MASS_PER_M = 1.8       # Linear mass including ice (kg/m)
CONDUCTOR_DIAMETER_M = 0.030     # Bare conductor diameter (sets how thick the wire looks)
ICE_THICKNESS_M = 0.010          # Effective radial ice thickness (enlarges the aerodynamic diameter)
DAMPING_RATIO = 0.04             # Effective mechanical damping ratio (structural + still-air), fraction of critical
GRAVITY_MS2 = 9.81
AIR_DENSITY_KGM3 = 1.25
CONDUCTOR_POINTS = 240           # Points used to draw each conductor along the span

# Conductors on the span. Any key from the defaults above can be overridden per conductor:
#   tension_n, mass_per_m, diameter_m, ice_thickness_m, sway_ratio
# attach_y_m / attach_z_m: where the wire hangs at the towers (bottom of the insulator string).
# insulator_m: insulator string length (0 = attached directly to the tower peak, e.g. a shield wire).
# aero_scale: multiplies the gallop driving term (Den Hartog) or amplitudes (kinematic) for this wire.
CONDUCTORS = [
    {"name": "shield", "attach_y_m": 0.0, "attach_z_m": 37.0, "insulator_m": 0.0,
     "tension_n": 15_000.0, "mass_per_m": 0.6, "diameter_m": 0.012, "aero_scale": 0.6},
    {"name": "phase_A", "attach_y_m": 5.0, "attach_z_m": 30.0, "insulator_m": 2.5, "aero_scale": 1.0},
    {"name": "phase_B", "attach_y_m": 5.5, "attach_z_m": 24.0, "insulator_m": 2.5, "aero_scale": 0.9},
    {"name": "phase_C", "attach_y_m": 5.0, "attach_z_m": 18.0, "insulator_m": 2.5, "aero_scale": 0.8},
]

# =============================================================================
# Gallop model
# =============================================================================
MOTION_MODEL = "den_hartog"      # "den_hartog" (physics-driven onset) or "kinematic" (scripted amplitudes)
MODE_NUMBERS = (1, 2, 3)         # Standing-wave modes simulated (mode n has n loops in the span)
SWAY_RATIO = 0.3                 # Transverse (Y) amplitude as a fraction of vertical -> elliptical orbits
SWAY_PHASE_DEG = 60.0            # Phase lead of transverse motion relative to vertical

# --- Den Hartog model -------------------------------------------------------
# a1 = -(dCL/dalpha + CD) for the iced cross-section; a1 > 0 means the Den Hartog
# criterion is met and aerodynamic damping turns negative as wind increases.
DEN_HARTOG_A1 = 2.0
MODE_AERO_WEIGHTS = (0.3, 1.0, 0.6)   # Per-mode multiplier on a1 (which modes the wind favours)
TARGET_AMPLITUDE_M = 1.5         # Limit-cycle amplitude of the most-favoured mode at REFERENCE_WIND_MS
REFERENCE_WIND_MS = 15.0         # Wind speed at which TARGET_AMPLITUDE_M is reached (sets the cubic term)
INITIAL_PERTURBATION_M = 0.05    # Std. dev. of the random initial modal displacement
PHYSICS_SUBSTEPS = 20            # RK4 integration steps per video frame

# --- Kinematic model --------------------------------------------------------
KINEMATIC_AMPLITUDES_M = (0.2, 1.5, 0.3)  # Peak modal amplitude for each entry in MODE_NUMBERS
# Envelope multiplier vs. time as (time_s, scale) pairs, linearly interpolated.
KINEMATIC_ENVELOPE = [(0.0, 0.0), (20.0, 0.0), (50.0, 1.0), (90.0, 1.0), (110.0, 0.0)]

# =============================================================================
# Wind (uniform along the span; drives the Den Hartog model, precipitation drift, and camera shake)
# =============================================================================
# Mean wind speed vs. time as (time_s, speed_ms) pairs, linearly interpolated.
WIND_SCHEDULE = [(0.0, 4.0), (15.0, 4.0), (40.0, 16.0), (85.0, 16.0), (100.0, 5.0), (120.0, 5.0)]
WIND_GUST_STD_MS = 1.5           # Std. dev. of turbulent gusts around the mean
WIND_GUST_TIMESCALE_S = 5.0      # Correlation time of gusts (Ornstein-Uhlenbeck process)

# =============================================================================
# Camera (pinhole model)
# =============================================================================
IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 720
CAMERA_HFOV_DEG = 65.0           # Horizontal field of view
CAMERA_POSITION_M = (125.0, -210.0, 8.0)   # Side view of the whole span from 210 m away
CAMERA_TARGET_M = (125.0, 0.0, 24.0)       # Point the optical axis passes through
# Tower-mounted alternative (looking down the span, as many line-monitoring cameras do):
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
ALERT_THRESHOLD_M = 0.75         # Raise an alert when a conductor's amplitude envelope reaches this
ALERT_CLEAR_M = 0.5              # Clear it when the envelope falls below this (hysteresis)
