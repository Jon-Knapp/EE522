"""
degrade.py - Real-world image nuisances applied to each clean frame.

A CV detector that only works on a perfect render is not useful in the field, so
this module adds the effects that make line monitoring hard: fog (low contrast),
camera shake from wind on the mounting pole, rain streaks or snowflakes crossing
the wires, defocus blur, exposure flicker, and sensor noise. Each effect is
switchable and tunable from config.py. The camera-shake transform is returned to
the simulator so ground-truth pixel positions stay aligned with what is drawn.
"""

import math

import cv2
import numpy as np

from .render import draw_polylines_alpha


def apply_fog(img, density, color_bgr):
    """Blend the frame toward a uniform fog colour; density 0 leaves it unchanged, 1 replaces it entirely."""
    scaled = cv2.multiply(img, (1.0 - density,) * 3 + (0.0,))
    return cv2.add(scaled, tuple(float(c) * density for c in color_bgr) + (0.0,))


def apply_affine(img, matrix):
    """Warp the frame with a 2x3 affine matrix, reflecting edge pixels so no black borders appear."""
    height, width = img.shape[:2]
    return cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def transform_points(uv, matrix):
    """Apply a 2x3 affine matrix to (N, 2) pixel coordinates (used to keep ground truth aligned with shake)."""
    uv = np.atleast_2d(uv)
    return uv @ matrix[:, :2].T + matrix[:, 2]


def apply_blur(img, sigma_px):
    """Apply Gaussian defocus blur with standard deviation sigma_px (no-op when sigma_px <= 0)."""
    if sigma_px <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma_px)


class SensorNoise:
    """Additive Gaussian sensor noise drawn from a small pre-generated bank for speed.

    Each frame uses a randomly chosen bank entry at a random offset, so the noise looks
    independent from frame to frame while costing only one image addition.
    """

    BANK_SIZE = 3
    MARGIN_PX = 64

    def __init__(self, std, width, height, rng):
        """Pre-generate the noise bank with the given standard deviation (0-255 scale)."""
        self.width, self.height = width, height
        self.rng = rng
        shape = (self.BANK_SIZE, height + self.MARGIN_PX, width + self.MARGIN_PX, 3)
        self.bank = rng.standard_normal(shape, dtype=np.float32) * np.float32(std)

    def apply(self, img):
        """Return img plus a fresh-looking noise field."""
        k = int(self.rng.integers(self.BANK_SIZE))
        dy, dx = (int(v) for v in self.rng.integers(0, self.MARGIN_PX, 2))
        return cv2.add(img, self.bank[k, dy:dy + self.height, dx:dx + self.width])


class CameraShake:
    """Correlated random translation and rotation of the camera, optionally scaled by wind speed."""

    def __init__(self, std_px, rot_std_deg, timescale_s, fps, width, height, rng,
                 scale_with_wind=True, reference_wind_ms=15.0):
        """Store jitter statistics and image geometry; the jitter starts at zero."""
        self.std_px = float(std_px)
        self.rot_std = float(rot_std_deg)
        self.decay = math.exp(-1.0 / (max(timescale_s, 1e-6) * fps))
        self.center = (width / 2.0, height / 2.0)
        self.rng = rng
        self.scale_with_wind = scale_with_wind
        self.reference_wind = max(float(reference_wind_ms), 1e-6)
        self.state = np.zeros(3)   # dx_px, dy_px, rotation_deg

    def step(self, wind_ms):
        """Advance the jitter by one frame and return the 2x3 affine matrix to apply to the image."""
        scale = wind_ms / self.reference_wind if self.scale_with_wind else 1.0
        std = np.array([self.std_px, self.std_px, self.rot_std]) * scale
        innovation = math.sqrt(1.0 - self.decay ** 2)
        self.state = self.state * self.decay + std * innovation * self.rng.standard_normal(3)
        dx, dy, rot = self.state
        matrix = cv2.getRotationMatrix2D(self.center, rot, 1.0)
        matrix[:, 2] += (dx, dy)
        return matrix


class Precipitation:
    """Persistent rain streaks or snowflakes that fall through the frame and drift with the wind."""

    def __init__(self, kind, count, width, height, fps, drift_px_per_ms, rng):
        """Scatter `count` particles with random depth; nearer particles are larger and fall faster."""
        self.kind = kind
        self.width, self.height = width, height
        self.exposure = 0.5 / fps
        self.drift = float(drift_px_per_ms)
        self.rng = rng
        self.x = rng.uniform(0, width, count)
        self.y = rng.uniform(0, height, count)
        self.near = rng.uniform(0.3, 1.0, count)
        self.fall = (1400.0 if kind == "rain" else 70.0) * self.near
        self.phase = rng.uniform(0, 2 * math.pi, count)

    def step(self, dt, wind_ms, t):
        """Move particles by dt seconds and wrap those that leave the frame back in at the top or side."""
        vx = wind_ms * self.drift * self.near
        if self.kind == "snow":
            vx = vx + 15.0 * self.near * np.sin(1.3 * t + self.phase)   # flakes flutter as they fall
        self.vx = vx
        self.x = (self.x + vx * dt) % self.width
        self.y += self.fall * dt
        wrapped = self.y > self.height
        self.y[wrapped] -= self.height
        self.x[wrapped] = self.rng.uniform(0, self.width, wrapped.sum())

    def draw(self, img):
        """Draw the particles into the float32 frame in place (streaks for rain, soft dots for snow)."""
        if self.kind == "rain":
            tails = np.column_stack([self.x - self.vx * self.exposure, self.y - self.fall * self.exposure])
            segments = [np.vstack([head, tail]) for head, tail in zip(np.column_stack([self.x, self.y]), tails)]
            draw_polylines_alpha(img, segments, (215, 215, 215), 0.35, 1)
            return
        mask = np.zeros(img.shape[:2], np.uint8)
        radii = np.maximum(1, np.round(1.0 + 2.0 * self.near)).astype(int)
        for x, y, r in zip(self.x.astype(int), self.y.astype(int), radii):
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1, cv2.LINE_AA)
        rows, cols = np.nonzero(mask)   # blend only the few pixels the flakes cover
        coverage = (mask[rows, cols].astype(np.float32) * (0.85 / 255.0))[:, None]
        img[rows, cols] += (np.float32(245.0) - img[rows, cols]) * coverage


class Degrader:
    """Applies the configured degradations, in a fixed order, to each clean frame."""

    def __init__(self, cfg, rng):
        """Create the stateful effects (shake, precipitation) and remember the static settings."""
        self.cfg = cfg
        self.rng = rng
        width, height = cfg.IMAGE_WIDTH_PX, cfg.IMAGE_HEIGHT_PX
        self.shake = None
        if cfg.SHAKE_ENABLED:
            self.shake = CameraShake(cfg.SHAKE_STD_PX, cfg.SHAKE_ROT_STD_DEG, cfg.SHAKE_TIMESCALE_S, cfg.FPS,
                                     width, height, rng, cfg.SHAKE_SCALES_WITH_WIND, cfg.REFERENCE_WIND_MS)
        self.noise = SensorNoise(cfg.SENSOR_NOISE_STD, width, height, rng) if cfg.SENSOR_NOISE_STD > 0 else None
        self.precip = None
        if cfg.WEATHER != "none" and cfg.PRECIP_PARTICLES > 0:
            self.precip = Precipitation(cfg.WEATHER, int(cfg.PRECIP_PARTICLES), width, height, cfg.FPS,
                                        cfg.PRECIP_DRIFT_PX_PER_MS, rng)

    def apply(self, img, wind_ms, t):
        """Return (uint8 degraded frame, 2x3 shake matrix) for a clean float32 frame at time t.

        Order: fog, camera shake, precipitation, blur, illumination and flicker, sensor noise.
        """
        cfg = self.cfg
        out = apply_fog(img, cfg.FOG_DENSITY, cfg.FOG_COLOR_BGR) if cfg.FOG_DENSITY > 0 else img
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if self.shake is not None:
            matrix = self.shake.step(wind_ms)
            out = apply_affine(out, matrix)
        if self.precip is not None:
            self.precip.step(1.0 / cfg.FPS, wind_ms, t)
            if out is img:
                out = img.copy()   # never draw into the caller's clean frame
            self.precip.draw(out)
        out = apply_blur(out, cfg.BLUR_SIGMA_PX)
        gain = cfg.ILLUMINATION_GAIN * (1.0 + cfg.ILLUMINATION_FLICKER_STD * self.rng.standard_normal())
        out = cv2.multiply(out, (gain,) * 3 + (0.0,))
        if self.noise is not None:
            out = self.noise.apply(out)
        return np.clip(out, 0, 255, out=out).astype(np.uint8), matrix
