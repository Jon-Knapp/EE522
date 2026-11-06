"""
gallop_sim - Synthetic video generator for conductor gallop (Step 1 of the gallop-monitoring project).

The project goal is a computer-vision system that watches transmission lines and
notifies engineers when conductor gallop exceeds a threshold. Real gallop footage
is rare and has no ground truth, so this package renders physically motivated
synthetic video of a galloping span together with exact per-frame displacement,
pixel positions, and alert labels. Those outputs are the test bench for the
detection, tracking, and alerting algorithms developed in later steps.

Module map:
    settings      - load config.py into a validated config object, apply overrides
    physics       - catenary sag, modal frequencies, Den Hartog and kinematic gallop dynamics
    wind          - mean wind schedule plus turbulent gusts
    scene         - 3D conductors and lattice towers built from the config
    camera        - pinhole camera projecting world points to pixels
    render        - procedural sky/ground background and anti-aliased wire drawing
    degrade       - fog, camera shake, rain/snow, blur, exposure flicker, sensor noise
    ground_truth  - CSV/JSON writers and hysteresis alert labelling
    simulator     - ties everything together and produces a run folder
"""

__version__ = "0.1.0"
