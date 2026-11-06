"""
plot_ground_truth.py - Quick-look plots of a simulator run's ground truth.

Reads ground_truth.csv and metadata.json from a run folder and plots, against
time: the wind speed with each conductor's critical speed, every conductor's
amplitude envelope with the alert threshold and alert periods shaded, and the
vertical displacement at the sampled span position that moves the most. Use it to confirm a run contains the gallop
events you intended before testing CV algorithms on the video. Saves
ground_truth.png into the run folder.

Usage:
    python tools/plot_ground_truth.py output/<run_name> [--show]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_columns(csv_path):
    """Load a ground-truth CSV into a dict of column name -> float numpy array (empty cells become NaN)."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        names = reader.fieldnames
    return {name: np.array([float(r[name]) if r[name] != "" else np.nan for r in rows]) for name in names}


def plot_run(run_dir, show=False):
    """Create the three-panel ground-truth figure for a run folder and save it as ground_truth.png."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    data = load_columns(run_dir / "ground_truth.csv")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    names = [c["name"] for c in meta["conductors"]]
    t = data["time_s"]
    # Plot displacement where it is largest: mode 2 (the usual gallop mode) has a node at midspan.
    def spread(fraction):
        """Return the total displacement variance across conductors at one sampled span fraction."""
        tag_f = f"x{int(round(fraction * 100)):03d}"
        return sum(np.nanvar(data[f"{n}_{tag_f}_dz_m"]) for n in names)
    sample = max(cfg["GT_SAMPLE_FRACTIONS"], key=spread)
    tag = f"x{int(round(sample * 100)):03d}"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, data["wind_ms"], color="tab:blue", lw=1)
    for c in meta["conductors"]:
        finite = [u for u in (c.get("critical_wind_ms") or []) if u is not None]
        if finite:
            axes[0].axhline(min(finite), ls=":", lw=0.8, color="grey")
    axes[0].set_ylabel("wind (m/s)")
    axes[0].set_title(f"Run {run_dir.name}  ({cfg['MOTION_MODEL']}; dotted = lowest critical wind per conductor)")

    for name in names:
        axes[1].plot(t, data[f"{name}_amplitude_m"], lw=1.2, label=name)
        axes[2].plot(t, data[f"{name}_{tag}_dz_m"], lw=0.8, label=name)
    axes[1].axhline(cfg["ALERT_THRESHOLD_M"], color="red", ls="--", lw=1, label="alert threshold")
    axes[1].axhline(cfg["ALERT_CLEAR_M"], color="orange", ls=":", lw=1, label="clear level")
    alert = data["any_alert"] > 0
    axes[1].fill_between(t, 0, 1, where=alert, transform=axes[1].get_xaxis_transform(),
                         color="red", alpha=0.08, label="any alert active")
    axes[1].set_ylabel("amplitude envelope (m)")
    axes[1].legend(loc="upper left", fontsize=8, ncol=3)
    axes[2].set_ylabel(f"vertical disp. at {sample:.0%} span (m)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / "ground_truth.png"
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv=None):
    """Parse arguments, plot the requested run, and print where the figure was saved."""
    parser = argparse.ArgumentParser(description="Plot ground truth for a gallop_sim run folder.")
    parser.add_argument("run_dir", help="folder containing ground_truth.csv and metadata.json")
    parser.add_argument("--show", action="store_true", help="open an interactive window as well")
    args = parser.parse_args(argv)
    print(f"Saved {plot_run(args.run_dir, args.show)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
