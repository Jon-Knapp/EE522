"""
simulator.py - Orchestrates a complete synthetic gallop run.

GallopSimulator wires the other modules together: it builds the conductors,
camera, wind, renderer, and degrader from the config, then for every video frame
advances the wind and gallop dynamics, renders and degrades the image, writes it
to video, and logs ground truth and alert labels. It also produces a physics
summary (modal frequencies, sag, critical wind speeds, predicted amplitudes) so
settings can be sanity-checked before committing to a long render. main.py is the
command-line front end for this class.
"""

import datetime
import math
import platform
import time

import cv2
import numpy as np

from . import __version__, physics
from .camera import PinholeCamera
from .degrade import Degrader, transform_points
from .ground_truth import (AlertTracker, GroundTruthWriter, fraction_label, write_events_csv,
                           write_metadata)
from .render import SceneRenderer
from .scene import build_conductors
from .settings import config_to_dict, resolve_output_dir
from .wind import WindModel


class GallopSimulator:
    """Builds the simulated scene from a config namespace and renders it to a run folder."""

    def __init__(self, cfg):
        """Create every component with its own random stream derived from RANDOM_SEED (for reproducibility)."""
        self.cfg = cfg
        seeds = np.random.SeedSequence(cfg.RANDOM_SEED).spawn(5)
        rng_physics, rng_wind, rng_render, rng_degrade, _ = (np.random.default_rng(s) for s in seeds)
        self.camera = PinholeCamera.from_config(cfg)
        self.conductors = build_conductors(cfg, rng_physics)
        self.wind = WindModel(cfg.WIND_SCHEDULE, cfg.WIND_GUST_STD_MS, cfg.WIND_GUST_TIMESCALE_S, rng_wind)
        self.renderer = SceneRenderer(cfg, self.camera, self.conductors, rng_render)
        self.degrader = Degrader(cfg, rng_degrade)

    # ------------------------------------------------------------------ summary
    def physics_summary(self):
        """Return a list of per-conductor dicts with frequencies, sag, critical winds, and predicted amplitudes."""
        cfg = self.cfg
        summary = []
        peak_wind = self.wind.max_mean_speed()
        for c in self.conductors:
            entry = {
                "name": c.name, "sag_m": c.sag_m, "frequencies_hz": c.frequencies_hz,
                "mode_numbers": list(c.mode_numbers), "aero_diameter_m": c.aero_diameter_m,
                "tension_n": c.tension_n, "mass_per_m": c.mass_per_m, "a1": c.a1, "a3": c.a3,
                "midspan_depth_m": float(self.camera.project(
                    np.array([[c.span_m / 2, c.attach_y_m, c.attach_z_m - c.sag_m]]))[2][0]),
            }
            entry["metres_per_pixel_midspan"] = self.camera.metres_per_pixel(entry["midspan_depth_m"])
            if cfg.MOTION_MODEL == "den_hartog":
                args = (c.omegas, c.mass_per_m, cfg.DAMPING_RATIO, cfg.AIR_DENSITY_KGM3, c.aero_diameter_m, c.a1)
                entry["critical_wind_ms"] = physics.critical_wind_speed(*args)
                entry["predicted_amplitude_ref_wind_m"] = physics.predicted_amplitude(*args, c.a3, cfg.REFERENCE_WIND_MS)
                entry["predicted_amplitude_peak_wind_m"] = physics.predicted_amplitude(*args, c.a3, peak_wind)
            else:
                entry["kinematic_amplitudes_m"] = c.dynamics.amplitudes
            summary.append(entry)
        return summary

    def format_summary(self):
        """Return the physics summary as readable text for the console."""
        cfg = self.cfg
        lines = [f"Model: {cfg.MOTION_MODEL} | span {cfg.SPAN_LENGTH_M:.0f} m | modes {list(cfg.MODE_NUMBERS)} | "
                 f"peak mean wind {self.wind.max_mean_speed():.1f} m/s | alert at {cfg.ALERT_THRESHOLD_M} m"]
        for e in self.physics_summary():
            freqs = ", ".join(f"{f:.2f}" for f in e["frequencies_hz"])
            lines.append(f"  {e['name']:<10} sag {e['sag_m']:.2f} m | f_n [{freqs}] Hz | "
                         f"{e['metres_per_pixel_midspan'] * 100:.1f} cm/px at midspan")
            if cfg.MOTION_MODEL == "den_hartog":
                crit = ", ".join("stable" if not math.isfinite(u) else f"{u:.1f}" for u in e["critical_wind_ms"])
                ref = ", ".join(f"{a:.2f}" for a in e["predicted_amplitude_ref_wind_m"])
                peak = ", ".join(f"{a:.2f}" for a in e["predicted_amplitude_peak_wind_m"])
                lines.append(f"  {'':<10} critical wind [{crit}] m/s | amplitude @ref wind [{ref}] m | "
                             f"@peak wind [{peak}] m")
            else:
                amps = ", ".join(f"{a:.2f}" for a in e["kinematic_amplitudes_m"])
                lines.append(f"  {'':<10} scripted amplitudes [{amps}] m")
        return "\n".join(lines)

    # ------------------------------------------------------------------ run
    def _make_run_dir(self):
        """Create and return the output folder for this run (timestamped unless RUN_NAME is set)."""
        name = self.cfg.RUN_NAME or datetime.datetime.now().strftime("gallop_%Y%m%d_%H%M%S")
        run_dir = resolve_output_dir(self.cfg) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _open_video(self, path):
        """Open a cv2.VideoWriter for `path`, raising a clear error if the codec is unavailable."""
        cfg = self.cfg
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cfg.VIDEO_CODEC), cfg.FPS,
                                 (cfg.IMAGE_WIDTH_PX, cfg.IMAGE_HEIGHT_PX))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path} with codec '{cfg.VIDEO_CODEC}'")
        return writer

    def _ground_truth_row(self, frame, t, wind_ms, matrix, alerts, amplitudes):
        """Assemble the ground-truth CSV row for one frame (world displacement and shaken pixel positions)."""
        cfg = self.cfg
        dx, dy, rot = self.degrader.shake.state if self.degrader.shake is not None else (0.0, 0.0, 0.0)
        row = {"frame": frame, "time_s": t, "wind_ms": wind_ms, "shake_dx_px": dx, "shake_dy_px": dy,
               "shake_rot_deg": rot, "max_amplitude_m": max(amplitudes.values()), "any_alert": any(alerts.values())}
        for c in self.conductors:
            row[f"{c.name}_amplitude_m"] = amplitudes[c.name]
            row[f"{c.name}_alert"] = alerts[c.name]
            for n, q in zip(c.mode_numbers, c.dynamics.q):
                row[f"{c.name}_q{n}_m"] = q
            points, dz, dy_disp = c.sample_points(cfg.GT_SAMPLE_FRACTIONS)
            uv, valid, _ = self.camera.project(points)
            uv = transform_points(uv, matrix)
            for i, fraction in enumerate(cfg.GT_SAMPLE_FRACTIONS):
                tag = fraction_label(fraction)
                row[f"{c.name}_{tag}_dz_m"] = dz[i]
                row[f"{c.name}_{tag}_dy_m"] = dy_disp[i]
                row[f"{c.name}_{tag}_u_px"] = uv[i, 0] if valid[i] else float("nan")
                row[f"{c.name}_{tag}_v_px"] = uv[i, 1] if valid[i] else float("nan")
        return row

    def run(self, verbose=True):
        """Render the full clip and write video, ground truth, alert events, and metadata; return the run folder."""
        cfg = self.cfg
        run_dir = self._make_run_dir()
        frames_dir = run_dir / "frames"
        if cfg.SAMPLE_FRAME_INTERVAL_S > 0:
            frames_dir.mkdir(exist_ok=True)
        names = [c.name for c in self.conductors]
        n_frames = int(round(cfg.DURATION_S * cfg.FPS))
        dt_frame = 1.0 / cfg.FPS
        dt_sub = dt_frame / max(int(cfg.PHYSICS_SUBSTEPS), 1)
        sample_every = int(round(cfg.SAMPLE_FRAME_INTERVAL_S * cfg.FPS)) if cfg.SAMPLE_FRAME_INTERVAL_S > 0 else 0

        video = self._open_video(run_dir / "video.mp4") if cfg.SAVE_VIDEO else None
        clean_video = self._open_video(run_dir / "video_clean.mp4") if cfg.SAVE_CLEAN_VIDEO else None
        gt = GroundTruthWriter(run_dir / "ground_truth.csv", names, tuple(cfg.MODE_NUMBERS),
                               cfg.GT_SAMPLE_FRACTIONS)
        alerts = AlertTracker(names, cfg.ALERT_THRESHOLD_M, cfg.ALERT_CLEAR_M)
        if verbose:
            print(self.format_summary())
            print(f"Rendering {n_frames} frames to {run_dir}")

        started = time.time()
        wind_ms = self.wind.current
        last_report = -1
        frames_written = 0
        try:
            for k in range(n_frames):
                t = k * dt_frame
                if k > 0:
                    t_prev = (k - 1) * dt_frame
                    for s in range(max(int(cfg.PHYSICS_SUBSTEPS), 1)):
                        wind_ms = self.wind.step(t_prev + s * dt_sub, dt_sub)
                        for c in self.conductors:
                            c.dynamics.step(dt_sub, wind_ms)

                clean = self.renderer.render(t)
                if clean_video is not None:
                    clean_video.write(np.clip(clean, 0, 255).astype(np.uint8))
                frame, matrix = self.degrader.apply(clean, wind_ms, t)
                if video is not None:
                    video.write(frame)
                if sample_every and k % sample_every == 0:
                    cv2.imwrite(str(frames_dir / f"frame_{k:05d}.png"), frame)

                amplitudes = {c.name: c.amplitude() for c in self.conductors}
                states = alerts.update(k, t, amplitudes)
                gt.write(self._ground_truth_row(k, t, wind_ms, matrix, states, amplitudes))
                frames_written = k + 1

                if cfg.PREVIEW:
                    cv2.imshow("gallop_sim preview (q to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Preview stopped early by user")
                        break
                pct = int(100 * (k + 1) / n_frames) // 5 * 5
                if verbose and pct != last_report:
                    last_report = pct
                    elapsed = time.time() - started
                    print(f"  {pct:3d}%  t={t:6.1f}s  wind={wind_ms:5.1f} m/s  "
                          f"max amp={max(amplitudes.values()):5.2f} m  ({elapsed:.0f}s elapsed)")
        finally:
            for writer in (video, clean_video):
                if writer is not None:
                    writer.release()
            gt.close()
            if cfg.PREVIEW:
                cv2.destroyAllWindows()

        write_events_csv(run_dir / "alert_events.csv", alerts.events)
        write_metadata(run_dir / "metadata.json", self._metadata(frames_written))
        if verbose:
            print(f"Done in {time.time() - started:.0f}s: {frames_written} frames, "
                  f"{len(alerts.events)} alert events -> {run_dir}")
        return run_dir

    def _metadata(self, frames_written):
        """Return the metadata dict saved as metadata.json: config, camera calibration, physics, and versions."""
        cam = self.camera
        return {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "gallop_sim_version": __version__,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
            "frames_written": frames_written,
            "camera": {"K": cam.intrinsic_matrix(), "R": cam.R, "t": cam.t, "position_m": cam.position,
                       "target_m": cam.target, "hfov_deg": cam.hfov_deg, "width_px": cam.width,
                       "height_px": cam.height, "horizon_v_px": cam.horizon_v()},
            "conductors": self.physics_summary(),
            "config": config_to_dict(self.cfg),
        }
