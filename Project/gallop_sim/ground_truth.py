"""
ground_truth.py - Labels and metadata saved alongside every synthetic video.

The point of simulating gallop is having exact answers to grade a CV pipeline
against. This module writes:

* ground_truth.csv  - one row per frame: wind speed, camera shake, and for every
                      conductor its amplitude envelope (semi- and peak-to-peak),
                      alert state, wind at the conductor, blowout angle, ice state,
                      modal coordinates, and the world displacement and pixel
                      position of points at fixed fractions along the span;
* alert_events.csv  - the times each conductor's alert was raised and cleared
                      (plus any ice-shedding events), for measuring detection
                      latency and false alarms;
* metadata.json     - the full config, camera calibration (K, R, t), and the
                      per-conductor physics (frequencies, sag, critical winds,
                      Orange Book plausibility notes).

AlertTracker applies the same raise/clear hysteresis an operational system would,
so the labels define what "correct" alerting looks like.
"""

import csv
import json

import numpy as np


def fraction_label(fraction):
    """Return a compact column tag for a span fraction, e.g. 0.5 -> 'x050' (percent of span)."""
    return f"x{int(round(fraction * 100)):03d}"


class AlertTracker:
    """Per-conductor hysteresis alarm: raise at or above `raise_m`, clear below `clear_m`."""

    def __init__(self, names, raise_m, clear_m):
        """Start with every conductor's alert cleared and an empty event list."""
        self.raise_m = float(raise_m)
        self.clear_m = float(clear_m)
        self.active = {name: False for name in names}
        self.events = []

    def update(self, frame, time_s, amplitudes):
        """Update alert states from {name: amplitude_m}, log any transitions, and return the states dict."""
        for name, amplitude in amplitudes.items():
            if not self.active[name] and amplitude >= self.raise_m:
                self.active[name] = True
                self.log(frame, time_s, name, "raise", amplitude)
            elif self.active[name] and amplitude < self.clear_m:
                self.active[name] = False
                self.log(frame, time_s, name, "clear", amplitude)
        return dict(self.active)

    def log(self, frame, time_s, name, event, amplitude):
        """Append an event row (alert transitions, or other labelled events such as ice shedding)."""
        self.events.append({"frame": frame, "time_s": round(time_s, 4), "conductor": name,
                            "event": event, "amplitude_m": round(amplitude, 4)})


class GroundTruthWriter:
    """Streams per-frame ground truth rows to a CSV file with a fixed, documented column order."""

    def __init__(self, path, conductor_names, mode_numbers, sample_fractions):
        """Open the CSV at `path` and write the header row."""
        self.columns = ["frame", "time_s", "wind_ms", "shake_dx_px", "shake_dy_px", "shake_rot_deg",
                        "max_amplitude_m", "any_alert"]
        for name in conductor_names:
            self.columns += [f"{name}_amplitude_m", f"{name}_pkpk_m", f"{name}_alert", f"{name}_wind_ms",
                             f"{name}_blowout_deg", f"{name}_iced"]
            self.columns += [f"{name}_q{n}_m" for n in mode_numbers]
            for fraction in sample_fractions:
                tag = fraction_label(fraction)
                self.columns += [f"{name}_{tag}_dz_m", f"{name}_{tag}_dy_m",
                                 f"{name}_{tag}_u_px", f"{name}_{tag}_v_px"]
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        self._writer.writeheader()

    def write(self, row):
        """Write one frame's row; floats are rounded to keep the file compact, NaN is written as empty."""
        clean = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                clean[key] = "" if np.isnan(value) else round(float(value), 5)
            elif isinstance(value, (bool, np.bool_)):
                clean[key] = int(value)
            else:
                clean[key] = value
        self._writer.writerow(clean)

    def close(self):
        """Flush and close the CSV file."""
        self._file.close()


def write_events_csv(path, events):
    """Write the alert raise/clear (and ice-shed) events to a CSV (header only if there were none)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "time_s", "conductor", "event", "amplitude_m"])
        writer.writeheader()
        writer.writerows(sorted(events, key=lambda e: (e["frame"], e["conductor"])))


def _json_default(value):
    """Convert numpy types and other non-JSON values so json.dump can serialise run metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_metadata(path, payload):
    """Write the run metadata dict to a pretty-printed JSON file (inf/NaN become null)."""
    def scrub(value):
        """Recursively replace non-finite floats with None so the JSON stays standards-compliant."""
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        if isinstance(value, np.ndarray):
            return scrub(value.tolist())
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scrub(payload), handle, indent=2, default=_json_default)
