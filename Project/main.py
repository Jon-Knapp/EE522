"""
main.py - Command-line entry point for the conductor-gallop synthetic video generator.

Loads config.py, applies any command-line overrides, prints a physics summary, and
renders a run (video + ground truth + alert labels + metadata) with
gallop_sim.simulator.GallopSimulator. Use --summary-only to check modal
frequencies, critical wind speeds, and predicted amplitudes without rendering.

Examples:
    python main.py
    python main.py --duration 30 --name quick_test
    python main.py --model kinematic --weather rain --fog 0.3
    python main.py --summary-only
"""

import argparse
import sys

from gallop_sim.settings import load_config
from gallop_sim.simulator import GallopSimulator


def parse_args(argv=None):
    """Parse command-line options; any option left unset keeps its value from config.py."""
    parser = argparse.ArgumentParser(description="Render synthetic conductor-gallop video with ground truth.")
    parser.add_argument("--duration", type=float, help="clip length in seconds (DURATION_S)")
    parser.add_argument("--fps", type=float, help="frame rate (FPS)")
    parser.add_argument("--seed", type=int, help="random seed (RANDOM_SEED)")
    parser.add_argument("--model", choices=["den_hartog", "kinematic"], help="gallop model (MOTION_MODEL)")
    parser.add_argument("--weather", choices=["none", "rain", "snow"], help="precipitation (WEATHER)")
    parser.add_argument("--sky", choices=["clear", "overcast"], help="sky style (SKY_STYLE)")
    parser.add_argument("--fog", type=float, help="fog density 0-1 (FOG_DENSITY)")
    parser.add_argument("--noise", type=float, help="sensor noise std, 0-255 scale (SENSOR_NOISE_STD)")
    parser.add_argument("--no-shake", action="store_true", help="disable camera shake")
    parser.add_argument("--threshold", type=float, help="alert threshold in metres (ALERT_THRESHOLD_M)")
    parser.add_argument("--background", help="background photo to use instead of the procedural sky")
    parser.add_argument("--clean-video", action="store_true", help="also save an undegraded video")
    parser.add_argument("--preview", action="store_true", help="show frames while rendering (q to stop)")
    parser.add_argument("--name", help="run folder name (RUN_NAME)")
    parser.add_argument("--output-dir", help="output root folder (OUTPUT_DIR)")
    parser.add_argument("--summary-only", action="store_true", help="print the physics summary and exit")
    return parser.parse_args(argv)


def overrides_from_args(args):
    """Translate parsed arguments into a {CONFIG_NAME: value} dict containing only the options given."""
    mapping = {"duration": "DURATION_S", "fps": "FPS", "seed": "RANDOM_SEED", "model": "MOTION_MODEL",
               "weather": "WEATHER", "sky": "SKY_STYLE", "fog": "FOG_DENSITY", "noise": "SENSOR_NOISE_STD",
               "threshold": "ALERT_THRESHOLD_M", "background": "BACKGROUND_IMAGE", "name": "RUN_NAME",
               "output_dir": "OUTPUT_DIR"}
    overrides = {key: getattr(args, arg) for arg, key in mapping.items() if getattr(args, arg) is not None}
    if args.no_shake:
        overrides["SHAKE_ENABLED"] = False
    if args.clean_video:
        overrides["SAVE_CLEAN_VIDEO"] = True
    if args.preview:
        overrides["PREVIEW"] = True
    return overrides


def main(argv=None):
    """Run the generator from the command line and return a process exit code."""
    args = parse_args(argv)
    overrides = overrides_from_args(args)
    try:
        if "ALERT_THRESHOLD_M" in overrides:
            # Keep the clear level at the same fraction of the threshold as in config.py.
            base = load_config()
            ratio = base.ALERT_CLEAR_M / base.ALERT_THRESHOLD_M
            overrides.setdefault("ALERT_CLEAR_M", overrides["ALERT_THRESHOLD_M"] * ratio)
        cfg = load_config(overrides)
        simulator = GallopSimulator(cfg)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.summary_only:
        print(simulator.format_summary())
        return 0
    simulator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
