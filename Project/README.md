# Conductor Gallop Simulator (Step 1)

Synthetic video of galloping transmission-line conductors, with exact per-frame
ground truth, for developing and testing computer-vision gallop monitoring.
Physical defaults, motion characteristics, and plausibility checks follow EPRI's
*Transmission Line Reference Book: Wind-Induced Conductor Motion* (the "Orange
Book") [1]; see [Orange Book basis](#orange-book-basis) and [References](#references).

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
z(x, t) = Σ q_n(t) · sin(nπx / L)          f_n = (n / 2L) · √(T / μ)       [1, Eq. 2.5-11]
```

Two models produce the modal coordinates `q_n(t)` (`MOTION_MODEL` in `config.py`):

- **`den_hartog`** (default) – quasi-steady aerodynamics of an iced conductor:

  ```
  m q̈_n + c q̇_n + m ω_n² q_n = ½ ρ D (a1 U q̇_n − a3 P_n[v³] / U)
  ```

  `a1 = −(dC_L/dα + C_D)`. When `a1 > 0` (the Den Hartog criterion [1, Eq. 4.3-1]) the
  aerodynamic term is negative damping that grows with the line-normal wind `U`.
  Above the critical wind (`U_crit = 2ζωm / (½ρD·a1)`) the mode grows; the cubic
  term saturates it onto a limit cycle. `P_n[v³]` projects the cubed physical
  velocity onto mode *n*, which couples the modes: the first mode to reach its limit
  cycle starves the others, as the Orange Book describes for Den Hartog galloping
  [1, Sec. 4.3.5] (`MODE_COUPLING = False` restores independent modes). Gallop
  therefore starts, grows, and dies away as the wind schedule rises and falls.
- **`kinematic`** – prescribed sinusoids with a scripted envelope
  (`KINEMATIC_ENVELOPE`). Use this when you need gallop to hit an exact amplitude
  at an exact time (e.g. testing a threshold).

The cubic coefficient `a3` is calibrated so the most wind-favoured mode reaches a
book-based amplitude at `REFERENCE_WIND_MS` (`AMPLITUDE_CALIBRATION`):
`"hunt_richards"` (default) uses the field relation `f·Ymax / V` with `Ymax` the
peak-to-peak amplitude [1, Eq. A4.5-9]; `"target"` uses a fixed `TARGET_AMPLITUDE_M`.

Around the gallop, three slower effects shape what the camera sees:

- **Orbit shape.** The transverse motion is derived so each point traces the
  narrow ellipse measured from 44 films of real galloping: width 10 % of height,
  tilted 5° from vertical, with the static position at the lower quartile of the
  motion (the wire rises about three times as far as it falls) [1, Sec. 4.5.4, Fig. 4.5-31].
- **Blowout.** The whole span swings about its chord under the mean wind drag,
  `tan θ = ½ρV²C_d d / w` [1, Eq. 7.5-4], using the wind averaged over half the
  swing period [1, Sec. 7.5.7]. Gallop is centred on this blown-out position
  [1, Sec. 4.5.4], so a detector must not read swing as gallop.
- **Wind field.** `WIND_SCHEDULE` is the 10 m reference wind. Each conductor sees
  `V(z) = V_ref (z/10)^α` with `α` from the terrain category [1, Eqs. 2.2-6, 7.5-5,
  Table 2.2-6], only the component normal to the line acts, and gusts default to
  turbulence intensity `I_u = 1/ln(z/z0)` [1, Eq. 2.2-7].

Optionally, **ice shedding** (`ICE_SHED_EVENTS`) drops a conductor's ice at a given
time: the sag shrinks, the span jumps up and rings down at its higher bare-wire
frequency, and it stops galloping [1, Sec. 7.4]. It is a realistic large motion
that is *not* gallop, useful for testing false alarms.

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
python main.py --summary-only          # frequencies, sag, critical winds, amplitudes, Orange Book checks
python main.py                         # full default run (120 s, Den Hartog, snow)
python main.py --duration 30 --name quick_test
python main.py --model kinematic --weather rain --fog 0.3
python main.py --shed 95:phase_C       # phase_C drops its ice at t = 95 s
python main.py --terrain 3 --no-blowout
python main.py --no-shake --noise 0 --weather none --clean-video   # easy case for first experiments
python main.py --preview               # watch while rendering (press q to stop)
python tools/plot_ground_truth.py output/<run_name>                # sanity-check plot
pytest                                 # run the tests
```

Command-line options override `config.py` for that run only; everything else
(camera, conductors, wind schedule, colours, etc.) is edited in `config.py`.
Rendering takes roughly 40–80 ms per 1280×720 frame depending on the CPU and
weather, so the default 2-minute clip takes a few minutes.

`--summary-only` lists every conductor's sag, % of rated strength, modal
frequencies, critical wind, predicted peak-to-peak amplitude, the field-data
amplitude envelope, and the blowout angle at peak wind. Settings outside the
ranges observed on real lines (frequency 0.08–3 Hz, onset wind 7–18 m/s, amplitude
5–300 diameters [1, Table 1.3-1]; the Lilien–Havard envelope [1, Eq. 4.5-16];
peak-to-peak/sag above the filmed 1.7 [1, Sec. 4.5.4]; `f·Ymax/V` outside 0.05–0.30
[1, Fig. A4.5-2]) are listed with a `!` so you can tell an exotic scenario from a typo.

## Outputs

Each run creates `output/<run_name>/` (timestamped unless `--name` is given):

| File | Contents |
|---|---|
| `video.mp4` | Degraded, realistic video |
| `video_clean.mp4` | Same scene without degradations (only with `--clean-video`) |
| `frames/frame_*.png` | Snapshot every `SAMPLE_FRAME_INTERVAL_S` seconds |
| `ground_truth.csv` | One row per frame (columns below) |
| `alert_events.csv` | When each conductor's alert was raised and cleared, plus `ice_shed` events |
| `metadata.json` | Full config, camera K/R/t, per-conductor physics and Orange Book notes |
| `ground_truth.png` | Created by `tools/plot_ground_truth.py` |

### `ground_truth.csv` columns

| Column | Meaning |
|---|---|
| `frame`, `time_s` | Frame index and time |
| `wind_ms` | 10 m reference wind speed (mean + gust) |
| `shake_dx_px`, `shake_dy_px`, `shake_rot_deg` | Camera shake applied to this frame |
| `max_amplitude_m`, `any_alert` | Largest conductor amplitude; 1 if any alert is active |
| `<wire>_amplitude_m` | Vertical amplitude envelope (semi-amplitude) for that conductor (see note) |
| `<wire>_pkpk_m` | Peak-to-peak amplitude, the quantity the Orange Book reports (2 × amplitude) |
| `<wire>_alert` | Ground-truth alert state (hysteresis between `ALERT_CLEAR_M` and `ALERT_THRESHOLD_M`) |
| `<wire>_wind_ms` | Line-normal wind at the conductor's height |
| `<wire>_blowout_deg` | Swing angle of the span about its chord |
| `<wire>_iced` | 1 while the conductor carries ice, 0 after shedding |
| `<wire>_q<n>_m` | Modal coordinate of mode *n* |
| `<wire>_x<pct>_dz_m`, `_dy_m` | Gallop displacement from the blown-out rest position at `<pct>`% of the span |
| `<wire>_x<pct>_u_px`, `_v_px` | Pixel position of that point in the saved (shaken) frame |

`<pct>` comes from `GT_SAMPLE_FRACTIONS` (default 25, 50, 75 %). Two-loop gallop has a
node at midspan, so the 25 % and 75 % points move most. The amplitude envelope is
exact for a single active mode and a conservative upper bound when several modes are
active together. It measures all motion about the rest position, so an ice-shedding
jump also raises it (and can raise an alert); `alert_events.csv` labels those events
`ice_shed` so you can grade either policy.

## Configuration guide (`config.py`)

| Section | Key settings |
|---|---|
| Run/output | `DURATION_S`, `FPS`, `RANDOM_SEED`, `SAVE_CLEAN_VIDEO` |
| Span physics | `SPAN_LENGTH_M`, `CONDUCTOR_PRESET`, `CONDUCTOR_TENSION_N`, `ICE_THICKNESS_M`, `DAMPING_RATIO`, `CONDUCTORS` (per-wire `preset`, `tension_n` or `tension_pct_rts`) |
| Gallop | `MOTION_MODEL`, `MODE_NUMBERS`, `MODE_COUPLING`, `MODE_AERO_WEIGHTS`, `AMPLITUDE_CALIBRATION`, `HUNT_RICHARDS_RATIO`, `REFERENCE_WIND_MS`, `KINEMATIC_*`, `ICE_SHED_EVENTS` |
| Orbit | `ORBIT_WIDTH_RATIO`, `ORBIT_TILT_DEG`, `ORBIT_STATIC_QUANTILE`, or explicit `SWAY_RATIO`/`SWAY_PHASE_DEG` |
| Wind | `WIND_SCHEDULE`, `TERRAIN_CATEGORY`, `WIND_ANGLE_DEG`, `WIND_GUST_STD_MS`, `WIND_GUST_TIMESCALE_S`, `BLOWOUT_*` |
| Camera | `IMAGE_*_PX`, `CAMERA_HFOV_DEG`, `CAMERA_POSITION_M`, `CAMERA_TARGET_M` (a tower-mounted preset is in the comments) |
| Scene | `SKY_STYLE`, `CLOUD_*`, `BACKGROUND_IMAGE`, `WIRE_COLOR_BGR`, `MIN_WIRE_ALPHA`, tower geometry |
| Degradations | `FOG_DENSITY`, `SHAKE_*`, `WEATHER`, `PRECIP_*`, `BLUR_SIGMA_PX`, `ILLUMINATION_*`, `SENSOR_NOISE_STD` |
| Alerts | `ALERT_THRESHOLD_M`, `ALERT_CLEAR_M`, `GT_SAMPLE_FRACTIONS` |

Run `python main.py --summary-only` after changing physics settings. With the
defaults, one pixel covers about 21 cm at midspan, so a 3.4 m peak-to-peak gallop
moves the wire about 16 px.

## Orange Book basis

| Feature | What the book provides | Where in the code |
|---|---|---|
| Conductor properties | Drake ACSR (28.143 mm, 1.6281 kg/m, ~140 kN RTS) [1, Secs. 2.2.5, 2.5.2]; 3/8″ EHS steel strand and Greeley AAAC [1, Tables A1-7, A1-11]; stringing at 20–33 % RTS [1, Sec. 4.3.3] | `reference.CONDUCTOR_PRESETS`, `config.CONDUCTORS` |
| Typical galloping ranges | 0.08–3 Hz, 5–300 diameters pk-pk, 7–18 m/s steady wind [1, Table 1.3-1] | `reference.book_checks` |
| Den Hartog mechanism | Criterion `dC_L/dα + C_D < 0` [1, Eq. 4.3-1]; all modes below a frequency unstable in strong wind; first mode to reach a limit cycle locks the others out [1, Sec. 4.3.5] | `physics.DenHartogModes` |
| Number of loops | Field counts 1/2/3 loops = 42/26/34 phase cases; twisting under ice favours two loops on singles [1, Table 4.2-1, Sec. 4.3.5] | `MODE_AERO_WEIGHTS` comments |
| Amplitude calibration | `f·Ymax/V ≈ 0.26` guide (Hunt & Richards), field spread 0.05–0.30 [1, Eq. A4.5-9, Fig. A4.5-2] | `AMPLITUDE_CALIBRATION` |
| Amplitude envelope | `A/φ = 80 ln(8f/50φ)` single, `170 ln(8f/500φ)` bundle (Lilien & Havard) [1, Eqs. 4.5-16, 4.5-17]; filmed pk-pk/sag up to 1.7 single, 0.93 bundle [1, Sec. 4.5.4] | `reference.lilien_havard_max_pkpk` |
| Orbit shape | Ellipse 10 % wide, 5° tilt, rest at lower quartile; horizontal < 1/5 of vertical (Pon & Havard films) [1, Sec. 4.5.4, Fig. 4.5-31] | `reference.sway_from_orbit`, `physics.displacement_field` |
| Wind profile & turbulence | Power law and terrain categories [1, Eq. 2.2-6, Table 2.2-6, Eq. 7.5-5]; `I_u = 1/ln(z/z0)` [1, Eq. 2.2-7] | `wind.WindModel` |
| Blowout | `tan θ = ½ρV²C_d d / (w)`, `C_d = 1` [1, Eq. 7.5-4]; averaging over half the swing period [1, Sec. 7.5.7] | `Conductor.update_wind` |
| Ice shedding | Jump and ring-down after ice release; sag change drives the jump [1, Secs. 7.4.3, 7.4.4] | `Conductor.shed_ice` |
| Camera placement | Broadside views for amplitude, along-line views for modes and phases [1, Sec. 4.4.3] | `config` camera comments |

## Suggested experiments

- Start easy (`--weather none --no-shake --noise 0 --no-blowout`), then add one nuisance at a time.
- Compare line detectors (Hough, ridge filters, thin-structure segmentation) as `FOG_DENSITY` rises.
- Track `x025`/`x075` points with optical flow; compare to `*_u_px`, `*_v_px`.
- Estimate amplitude from the displacement spectrum (gallop 0.08–3 Hz [1, Table 1.3-1]) and grade alert latency against `alert_events.csv`.
- Separate slow blowout (`*_blowout_deg`, mostly below 0.1 Hz) from gallop with a high-pass filter before thresholding.
- Add `--shed 95:phase_C` and check whether your detector false-alarms on the ice jump.
- Try one-loop or three-loop gallop with `MODE_AERO_WEIGHTS` (see the comments in `config.py`).
- Swap in a real sky or site photo with `--background`, or move the camera to the tower-mounted preset.

## Project layout

```
Project/
├── config.py              # all tunable constants
├── main.py                # command-line entry point
├── requirements.txt
├── gallop_sim/
│   ├── settings.py        # loads/validates config, applies overrides
│   ├── reference.py       # Orange Book data and empirical relations
│   ├── physics.py         # catenary, modes, Den Hartog (coupled) and kinematic dynamics
│   ├── wind.py            # wind schedule, gusts, height profile, line-normal component
│   ├── scene.py           # conductors (blowout, ice shedding) and lattice towers in 3D
│   ├── camera.py          # pinhole projection and calibration
│   ├── render.py          # sky/ground background, anti-aliased wire drawing
│   ├── degrade.py         # fog, shake, rain/snow, blur, flicker, noise
│   ├── ground_truth.py    # CSV/JSON writers, alert hysteresis
│   └── simulator.py       # orchestrates a run
├── tools/plot_ground_truth.py
└── tests/                 # pytest suite (test_reference.py checks the book's worked examples)
```

## Limitations

- Modes are taut-string modes. Sag effects on mode shapes (the "pseudo-one-loop" of
  deadended spans [1, Sec. 4.3.2]), conductor torsion and torsional/vertical flutter
  galloping [1, Sec. 4.3.1], bundle conductors, multi-span coupling through insulator
  swing, and coupling between phases are not modelled.
- Wind is uniform along the span (no spanwise gust structure, which in reality
  selects the number of loops [1, Sec. 4.3.5]); aerodynamics are quasi-steady with no
  vortex shedding or aeolian vibration (3–150 Hz, far above a 30 fps camera's band [1, Table 1.3-1]).
- Blowout treats the span as rotating rigidly about its chord and ignores insulator swing.
- Ice shedding is linear and drops all the ice at once; real shedding is nonlinear and
  usually happens in chunks, and the linear jump is near the top of the measured range [1, Sec. 7.4].
- The Orange Book relations are empirical design aids fitted to sparse field data; the
  book itself cautions against treating them as limits.
- Drawing ignores depth ordering between towers and wires, and the background is procedural unless you supply a photo.
- Videos are large (tens of MB per minute) because sensor noise defeats compression; `output/` is git-ignored.

## References

[1] EPRI, *EPRI Transmission Line Reference Book—Wind-Induced Conductor Motion*,
2nd ed. Palo Alto, CA: Electric Power Research Institute, 2006 (the "Orange Book").
Chapters used: 1 (Introduction, Table 1.3-1); 2 (Aeolian Vibration: Secs. 2.2.4–2.2.5
wind exposure, 2.5.2 conductor modes); 4 (Galloping Conductors: Secs. 4.2–4.5 and
Appendices 4.5–4.7); 7 (Other Motions: Sec. 7.4 ice shedding, Sec. 7.5 gust response);
Appendix 1 (Conductor Tables). Works the book cites and that the code relies on through it:

- J. P. Den Hartog, criterion for galloping instability, as presented in [1, Sec. 4.3.1].
- J. C. R. Hunt and D. J. W. Richards, amplitude relation `f·Ymax/V`, as presented in [1, App. 4.5].
- J.-L. Lilien and D. G. Havard (2000), amplitude envelopes from the field database, as presented in [1, Sec. 4.5.4].
- C. J. Pon and D. G. Havard (1994), orbit analysis of 44 galloping films, as presented in [1, Sec. 4.5.4].
- Houle et al. (1991) and Tsujimoto et al. (1982), conductor swing-out in wind, as presented in [1, Secs. 7.5.5–7.5.7].
- Morgan and Swift (1964), ice-shedding jump tests, as presented in [1, Sec. 7.4].
