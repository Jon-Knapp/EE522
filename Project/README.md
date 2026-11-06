# Conductor Gallop Simulator (Step 1)

Synthetic video of galloping transmission-line conductors, with exact per-frame
ground truth, for developing and testing computer-vision gallop monitoring.

## Where this fits

The overall project is a camera-based system that continuously watches a
transmission span and notifies engineers when conductor gallop exceeds a
threshold. Real gallop footage is rare and never comes with ground truth, so the
work is staged:

1. **Synthetic 2D video (this code)** – physics-driven gallop rendered through a
   calibrated pinhole camera, with weather and camera nuisances, plus labels.
2. **3D rendering** (e.g. Blender) – more realistic perspective, lighting and backgrounds.
3. **Physical bench rig** – a tensioned wire excited by a fan or shaker, filmed with a real camera.

Every run of this simulator gives you a video *and* the answer key: where each
wire really is, how far it is moving, and when an alert should have fired. That
lets you score detection, tracking, amplitude estimation, and alert latency.

## Physics model

Each conductor is a taut string hanging in a catenary between two lattice towers.
Gallop is a sum of standing-wave modes (mode *n* has *n* loops in the span):

```
z(x, t) = Σ q_n(t) · sin(nπx / L)          f_n = (n / 2L) · √(T / μ)
```

Two models produce the modal coordinates `q_n(t)` (`MOTION_MODEL` in `config.py`):

- **`den_hartog`** (default) – quasi-steady aerodynamics of an iced conductor:

  ```
  m q̈ + c q̇ + m ω² q = ½ ρ D (a1 U q̇ − ¾ a3 q̇³ / U)
  ```

  `a1 = −(dC_L/dα + C_D)`. When `a1 > 0` (the Den Hartog criterion) the
  aerodynamic term is negative damping that grows with wind speed `U`. Above the
  critical wind (`U_crit = 2ζωm / (½ρD·a1)`) the mode grows; the cubic term
  saturates it onto a limit cycle. `a3` is solved automatically so the most
  wind-favoured mode reaches `TARGET_AMPLITUDE_M` at `REFERENCE_WIND_MS`. Gallop
  therefore starts, grows, and dies away as the wind schedule rises and falls.
- **`kinematic`** – prescribed sinusoids with a scripted envelope
  (`KINEMATIC_ENVELOPE`). Use this when you need gallop to hit an exact amplitude
  at an exact time (e.g. testing a threshold).

Transverse sway is the vertical motion scaled by `SWAY_RATIO` and phase-shifted
by `SWAY_PHASE_DEG`, giving elliptical orbits. Wind is uniform along the span
(mean schedule plus Ornstein–Uhlenbeck gusts).

## Installation (Windows / PyCharm)

Python 3.10 or newer. From the `Project` folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In PyCharm you can instead set the project interpreter to a new virtualenv and
run `pip install -r requirements.txt` in the Terminal tab.

## Usage

```powershell
python main.py --summary-only          # check frequencies, sag, critical winds, predicted amplitudes
python main.py                         # full default run (120 s, Den Hartog, snow)
python main.py --duration 30 --name quick_test
python main.py --model kinematic --weather rain --fog 0.3
python main.py --no-shake --noise 0 --weather none --clean-video   # easy case for first experiments
python main.py --preview               # watch while rendering (press q to stop)
python tools/plot_ground_truth.py output/<run_name>                # sanity-check plot
pytest                                 # run the tests
```

Command-line options override `config.py` for that run only; everything else
(camera, conductors, wind schedule, colours, etc.) is edited in `config.py`.
Rendering takes roughly 40–80 ms per 1280×720 frame depending on the CPU and
weather, so the default 2-minute clip takes a few minutes.

## Outputs

Each run creates `output/<run_name>/` (timestamped unless `--name` is given):

| File | Contents |
|---|---|
| `video.mp4` | Degraded, realistic video |
| `video_clean.mp4` | Same scene without degradations (only with `--clean-video`) |
| `frames/frame_*.png` | Snapshot every `SAMPLE_FRAME_INTERVAL_S` seconds |
| `ground_truth.csv` | One row per frame (columns below) |
| `alert_events.csv` | When each conductor's alert was raised and cleared |
| `metadata.json` | Full config, camera K/R/t, per-conductor frequencies, sag, critical winds |
| `ground_truth.png` | Created by `tools/plot_ground_truth.py` |

### `ground_truth.csv` columns

| Column | Meaning |
|---|---|
| `frame`, `time_s` | Frame index and time |
| `wind_ms` | Wind speed (mean + gust) |
| `shake_dx_px`, `shake_dy_px`, `shake_rot_deg` | Camera shake applied to this frame |
| `max_amplitude_m`, `any_alert` | Largest conductor amplitude; 1 if any alert is active |
| `<wire>_amplitude_m` | Vertical amplitude envelope for that conductor (see note) |
| `<wire>_alert` | Ground-truth alert state (hysteresis between `ALERT_CLEAR_M` and `ALERT_THRESHOLD_M`) |
| `<wire>_q<n>_m` | Modal coordinate of mode *n* |
| `<wire>_x<pct>_dz_m`, `_dy_m` | Vertical / transverse displacement from rest at `<pct>`% of the span |
| `<wire>_x<pct>_u_px`, `_v_px` | Pixel position of that point in the saved (shaken) frame |

`<pct>` comes from `GT_SAMPLE_FRACTIONS` (default 25, 50, 75 %). Note that mode 2,
the most common gallop mode, has a node at midspan, so the 25 % and 75 % points
move most. The amplitude envelope is exact for a single active mode and a
conservative upper bound when several modes are active together.

## Configuration guide (`config.py`)

| Section | Key settings |
|---|---|
| Run/output | `DURATION_S`, `FPS`, `RANDOM_SEED`, `SAVE_CLEAN_VIDEO` |
| Span physics | `SPAN_LENGTH_M`, `CONDUCTOR_TENSION_N`, `CONDUCTOR_MASS_PER_M`, `ICE_THICKNESS_M`, `DAMPING_RATIO`, `CONDUCTORS` |
| Gallop | `MOTION_MODEL`, `MODE_NUMBERS`, `MODE_AERO_WEIGHTS`, `TARGET_AMPLITUDE_M`, `REFERENCE_WIND_MS`, `KINEMATIC_*` |
| Wind | `WIND_SCHEDULE`, `WIND_GUST_STD_MS`, `WIND_GUST_TIMESCALE_S` |
| Camera | `IMAGE_*_PX`, `CAMERA_HFOV_DEG`, `CAMERA_POSITION_M`, `CAMERA_TARGET_M` (a tower-mounted preset is in the comments) |
| Scene | `SKY_STYLE`, `CLOUD_*`, `BACKGROUND_IMAGE`, `WIRE_COLOR_BGR`, `MIN_WIRE_ALPHA`, tower geometry |
| Degradations | `FOG_DENSITY`, `SHAKE_*`, `WEATHER`, `PRECIP_*`, `BLUR_SIGMA_PX`, `ILLUMINATION_*`, `SENSOR_NOISE_STD` |
| Alerts | `ALERT_THRESHOLD_M`, `ALERT_CLEAR_M`, `GT_SAMPLE_FRACTIONS` |

Run `python main.py --summary-only` after changing physics settings; it prints
each conductor's modal frequencies, critical wind speeds, predicted amplitudes,
and how many centimetres one pixel covers at midspan (default ≈ 21 cm/px, so a
1.5 m gallop moves the wire about 7 px).

## Suggested experiments

- Start easy (`--weather none --no-shake --noise 0`), then add one nuisance at a time.
- Compare line detectors (Hough, ridge filters, thin-structure segmentation) as `FOG_DENSITY` rises.
- Track `x025`/`x075` points with optical flow; compare to `*_u_px`, `*_v_px`.
- Estimate amplitude from the displacement spectrum (gallop ≈ 0.2–1 Hz) and grade alert latency against `alert_events.csv`.
- Swap in a real sky or site photo with `--background`, or move the camera to the tower-mounted preset.

## Project layout

```
Project/
├── config.py              # all tunable constants
├── main.py                # command-line entry point
├── requirements.txt
├── gallop_sim/
│   ├── settings.py        # loads/validates config, applies overrides
│   ├── physics.py         # catenary, modes, Den Hartog and kinematic dynamics
│   ├── wind.py            # wind schedule + gusts
│   ├── scene.py           # conductors and lattice towers in 3D
│   ├── camera.py          # pinhole projection and calibration
│   ├── render.py          # sky/ground background, anti-aliased wire drawing
│   ├── degrade.py         # fog, shake, rain/snow, blur, flicker, noise
│   ├── ground_truth.py    # CSV/JSON writers, alert hysteresis
│   └── simulator.py       # orchestrates a run
├── tools/plot_ground_truth.py
└── tests/                 # pytest suite
```

## Limitations

- Modes are taut-string modes; sag effects on mode shapes, conductor torsion,
  bundle conductors, and coupling between phases are not modelled.
- Wind is uniform along the span, and aerodynamics are quasi-steady (no vortex shedding or aeolian vibration).
- Drawing ignores depth ordering between towers and wires, and the background is procedural unless you supply a photo.
- Videos are large (tens of MB per minute) because sensor noise defeats compression; `output/` is git-ignored.
