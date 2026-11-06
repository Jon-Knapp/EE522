"""
test_pipeline.py - Checks the camera, alert labelling, configuration, and an end-to-end mini render.

These tests make sure ground-truth pixel positions are geometrically correct,
alert labels follow the configured hysteresis, bad settings are rejected, and a
short run produces every expected output file.
"""

import csv
import json

import numpy as np
import pytest

from gallop_sim.camera import PinholeCamera
from gallop_sim.ground_truth import AlertTracker
from gallop_sim.settings import load_config
from gallop_sim.simulator import GallopSimulator


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


def test_short_run_writes_all_outputs(tmp_path):
    """A one-second, low-resolution run should write video, ground truth, events, and metadata."""
    cfg = load_config({"DURATION_S": 1.0, "FPS": 10.0, "IMAGE_WIDTH_PX": 320, "IMAGE_HEIGHT_PX": 180,
                       "OUTPUT_DIR": str(tmp_path), "RUN_NAME": "mini", "SAMPLE_FRAME_INTERVAL_S": 0.0})
    run_dir = GallopSimulator(cfg).run(verbose=False)
    for name in ("video.mp4", "ground_truth.csv", "alert_events.csv", "metadata.json"):
        assert (run_dir / name).exists(), name
    with open(run_dir / "ground_truth.csv", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 10
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert len(meta["camera"]["K"]) == 3 and meta["frames_written"] == 10
