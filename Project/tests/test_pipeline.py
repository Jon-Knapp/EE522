"""
test_pipeline.py - Checks the camera, alert labelling, configuration, scene effects, and an end-to-end mini render.

These tests make sure ground-truth pixel positions are geometrically correct,
alert labels follow the configured hysteresis, bad settings are rejected, the
book-derived scene effects (wind profile, blowout, ice shedding) behave as
described, and a short run produces every expected output file.
"""

import csv
import json
import math

import numpy as np
import pytest

from gallop_sim import reference
from gallop_sim.camera import PinholeCamera
from gallop_sim.ground_truth import AlertTracker
from gallop_sim.scene import build_conductors, settle_wind
from gallop_sim.settings import load_config
from gallop_sim.simulator import GallopSimulator
from gallop_sim.wind import WindModel


def test_camera_projects_target_to_image_centre():
    """The look-at target should land exactly on the principal point."""
    cam = PinholeCamera(1280, 720, 65.0, (125.0, -210.0, 8.0), (125.0, 0.0, 24.0))
    uv, valid, _ = cam.project(np.array([[125.0, 0.0, 24.0]]))
    assert valid[0] and np.allclose(uv[0], (640.0, 360.0))


def test_camera_orientation_up_is_up_and_right_is_right():
    """Points higher in the world appear higher (smaller v); points further along X appear further right."""
    cam = PinholeCamera(1280, 720, 65.0, (125.0, -210.0, 8.0), (125.0, 0.0, 24.0))
    uv, _, _ = cam.project(np.array([[125.0, 0.0, 24.0], [125.0, 0.0, 30.0], [150.0, 0.0, 24.0]]))
    assert uv[1, 1] < uv[0, 1]
    assert uv[2, 0] > uv[0, 0]


def test_points_behind_camera_are_invalid():
    """Points behind the camera must be flagged invalid rather than projected."""
    cam = PinholeCamera(640, 480, 60.0, (0.0, 0.0, 10.0), (0.0, 100.0, 10.0))
    _, valid, _ = cam.project(np.array([[0.0, -50.0, 10.0]]))
    assert not valid[0]


def test_alert_hysteresis():
    """Alerts should raise at the threshold, hold between the clear level and threshold, then clear."""
    tracker = AlertTracker(["a"], raise_m=1.0, clear_m=0.5)
    states = [tracker.update(i, i * 0.1, {"a": amp})["a"] for i, amp in enumerate([0.2, 1.0, 0.7, 0.49, 0.8])]
    assert states == [False, True, True, False, False]
    assert [e["event"] for e in tracker.events] == ["raise", "clear"]


def test_invalid_config_is_rejected():
    """Unknown settings and inconsistent values should raise errors before any rendering."""
    with pytest.raises(KeyError):
        load_config({"NOT_A_SETTING": 1})
    with pytest.raises(ValueError):
        load_config({"MOTION_MODEL": "magic"})
    with pytest.raises(ValueError):
        load_config({"ALERT_CLEAR_M": 2.0, "ALERT_THRESHOLD_M": 1.0})
    with pytest.raises(ValueError):
        load_config({"TERRAIN_CATEGORY": 7})
    with pytest.raises(ValueError):
        load_config({"ICE_SHED_EVENTS": [(10.0, "no_such_wire")]})


def test_presets_and_ice_set_conductor_mass():
    """Phase wires should carry Drake properties plus glaze ice; the shield its steel-strand preset."""
    cfg = load_config()
    conductors = {c.name: c for c in build_conductors(cfg, np.random.default_rng(0))}
    drake = reference.CONDUCTOR_PRESETS["drake_acsr"]
    ice = reference.ice_mass_per_m(drake["diameter_m"], cfg.ICE_THICKNESS_M, cfg.ICE_DENSITY_KGM3)
    assert math.isclose(conductors["phase_A"].mass_per_m, drake["mass_per_m"] + ice)
    assert conductors["shield"].diameter_m == reference.CONDUCTOR_PRESETS["ehs_3_8_steel"]["diameter_m"]


def test_wind_increases_with_height():
    """Higher conductors see more wind (power-law profile), and only the line-normal component counts."""
    wind = WindModel([(0, 10.0)], 0.0, 5.0, np.random.default_rng(0), terrain_category=2, wind_angle_deg=60.0)
    assert wind.wind_at(30.0, 10.0) > wind.wind_at(15.0, 10.0)
    assert math.isclose(wind.wind_at(10.0, 10.0), 5.0)   # cos 60 deg = 0.5 at the reference height


def test_blowout_tilts_span_downwind_and_raises_midspan():
    """In steady wind the span swings toward +Y by the Eq. 7.5-4 angle, which lifts the midspan."""
    cfg = load_config({"MOTION_MODEL": "kinematic", "KINEMATIC_AMPLITUDES_M": (0.0, 0.0, 0.0)})
    conductor = settle_wind(build_conductors(cfg, np.random.default_rng(0)), 16.0, cfg)[1]
    expected = reference.blowout_angle_rad(16.0 * conductor.wind_factor, cfg.AIR_DENSITY_KGM3,
                                           cfg.BLOWOUT_DRAG_COEFF, conductor.aero_diameter_m,
                                           conductor.mass_per_m * cfg.GRAVITY_MS2)
    assert math.isclose(conductor.blowout_rad, expected)
    points, _, _ = conductor.sample_points([0.5])
    assert points[0, 1] > conductor.attach_y_m
    assert points[0, 2] > conductor.attach_z_m - conductor.sag_m


def test_ice_shedding_makes_conductor_jump_and_stop_galloping():
    """After shedding, sag shrinks, frequencies rise, the span starts below its new rest (so it jumps up)."""
    cfg = load_config()
    conductor = build_conductors(cfg, np.random.default_rng(0))[1]
    sag_before, f_before = conductor.sag_m, conductor.frequencies_hz[0]
    conductor.dynamics.q[:] = 0.0
    conductor.dynamics.v[:] = 0.0
    conductor.shed_ice()
    assert conductor.sag_m < sag_before and conductor.frequencies_hz[0] > f_before
    assert not conductor.iced and np.all(conductor.a1 == 0)
    dz, _ = conductor.displacement()
    assert dz[len(dz) // 2] < -0.9 * (sag_before - conductor.sag_m)   # starts at the old, lower position


def test_short_run_writes_all_outputs(tmp_path):
    """A one-second, low-resolution run should write video, ground truth, events, and metadata."""
    cfg = load_config({"DURATION_S": 1.0, "FPS": 10.0, "IMAGE_WIDTH_PX": 320, "IMAGE_HEIGHT_PX": 180,
                       "OUTPUT_DIR": str(tmp_path), "RUN_NAME": "mini", "SAMPLE_FRAME_INTERVAL_S": 0.0,
                       "ICE_SHED_EVENTS": [(0.5, "phase_C")]})
    run_dir = GallopSimulator(cfg).run(verbose=False)
    for name in ("video.mp4", "ground_truth.csv", "alert_events.csv", "metadata.json"):
        assert (run_dir / name).exists(), name
    with open(run_dir / "ground_truth.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[0]["phase_C_iced"] == "1" and rows[-1]["phase_C_iced"] == "0"
    assert float(rows[-1]["phase_A_blowout_deg"]) >= 0.0
    with open(run_dir / "alert_events.csv", newline="") as handle:
        assert any(r["event"] == "ice_shed" for r in csv.DictReader(handle))
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert len(meta["camera"]["K"]) == 3 and meta["frames_written"] == 10
    assert "book_notes" in meta["conductors"][0]
