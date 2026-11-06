"""
render.py - Draws the clean (undegraded) scene for each video frame.

BackgroundRenderer builds a procedural sky with slowly drifting clouds, a distant
treeline, and snowy ground (or loads a user photo). SceneRenderer adds the lattice
towers and the conductors, projected through camera.py. Wires are drawn with
anti-aliased, sub-pixel polylines whose opacity follows their projected width, so
a 3 cm conductor 200 m away appears as the faint, thin line a real camera sees.
degrade.py then adds weather, shake, and noise on top of this clean frame.
"""

import math

import cv2
import numpy as np

from .scene import tower_segments

FIXED_POINT_SHIFT = 4          # cv2 sub-pixel precision: coordinates are scaled by 2**4
MAX_ABS_COORD_PX = 1e5         # Drop absurd coordinates (points nearly behind the camera)


def _periodic_noise(rng, height, width, scale_px):
    """Return a smooth, zero-mean, unit-variance noise field that wraps around horizontally.

    Gaussian low-pass filtering in the Fourier domain makes the field periodic, so clouds can
    drift across the frame indefinitely without a visible seam.
    """
    noise = rng.standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    response = np.exp(-2.0 * (math.pi * scale_px) ** 2 * (fx ** 2 + fy ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * response))
    return (field - field.mean()) / (field.std() + 1e-12)


def _valid_runs(uv, valid):
    """Split a projected polyline into contiguous runs of drawable points; return a list of (N, 2) arrays."""
    ok = valid & np.all(np.abs(np.nan_to_num(uv, nan=np.inf)) < MAX_ABS_COORD_PX, axis=1)
    runs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append(uv[start:i])
            start = None
    if start is not None:
        runs.append(uv[start:])
    return [r for r in runs if len(r) >= 2]


def wire_style(width_px, min_alpha):
    """Map a projected wire width (pixels) to an integer line thickness and an opacity in [0, 1].

    Wires at least a pixel wide are drawn opaque at their rounded width; thinner wires are drawn
    one pixel wide with opacity equal to their pixel coverage (floored at `min_alpha`).
    """
    if width_px >= 1.0:
        return max(1, int(round(width_px))), 1.0
    return 1, float(np.clip(width_px, min_alpha, 1.0))


def polyline_coverage(polylines, image_shape, alpha, thickness=1, closed=False):
    """Rasterise anti-aliased polylines into a coverage patch; return (y0, x0, coverage) or None.

    `coverage` is a float32 array (0-alpha) covering only the polylines' bounding box, with its
    top-left corner at (y0, x0) in the image. Sub-pixel point positions are honoured.
    """
    polylines = [np.asarray(p, dtype=float) for p in polylines if len(p) >= 2]
    if not polylines or alpha <= 0:
        return None
    height, width = image_shape[:2]
    pts = np.concatenate(polylines)
    pad = thickness + 2
    x0 = max(int(math.floor(pts[:, 0].min())) - pad, 0)
    y0 = max(int(math.floor(pts[:, 1].min())) - pad, 0)
    x1 = min(int(math.ceil(pts[:, 0].max())) + pad, width)
    y1 = min(int(math.ceil(pts[:, 1].max())) + pad, height)
    if x0 >= x1 or y0 >= y1:
        return None
    scale = 1 << FIXED_POINT_SHIFT
    fixed = [np.round((p - (x0, y0)) * scale).astype(np.int32).reshape(-1, 1, 2) for p in polylines]
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.polylines(mask, fixed, closed, 255, int(thickness), cv2.LINE_AA, FIXED_POINT_SHIFT)
    return y0, x0, mask.astype(np.float32) * (alpha / 255.0)


def blend_coverage(img, patch, color_bgr):
    """Blend a solid colour into a float32 image in place, weighted by a patch from polyline_coverage()."""
    if patch is None:
        return
    y0, x0, coverage = patch
    roi = img[y0:y0 + coverage.shape[0], x0:x0 + coverage.shape[1]]
    roi += (np.asarray(color_bgr, np.float32) - roi) * coverage[..., None]


def draw_polylines_alpha(img, polylines, color_bgr, alpha, thickness=1, closed=False):
    """Alpha-blend anti-aliased polylines into a float32 BGR image in place, working only in their bounding box.

    `alpha` scales the pixel coverage so sub-pixel wires render as faint lines, as in real footage.
    """
    blend_coverage(img, polyline_coverage(polylines, img.shape, alpha, thickness, closed), color_bgr)


class BackgroundRenderer:
    """Procedural sky, clouds, treeline, and ground, or a static user-supplied background photo."""

    def __init__(self, cfg, camera, rng):
        """Pre-compute the static layers (sky gradient, cloud texture, ground) so each frame is cheap to make."""
        self.height, self.width = cfg.IMAGE_HEIGHT_PX, cfg.IMAGE_WIDTH_PX
        self.drift = float(cfg.CLOUD_DRIFT_PX_PER_S)
        self.photo = None
        if cfg.BACKGROUND_IMAGE:
            photo = cv2.imread(str(cfg.BACKGROUND_IMAGE), cv2.IMREAD_COLOR)
            if photo is None:
                raise FileNotFoundError(f"Could not read BACKGROUND_IMAGE '{cfg.BACKGROUND_IMAGE}'")
            self.photo = cv2.resize(photo, (self.width, self.height), interpolation=cv2.INTER_AREA).astype(np.float32)
            return

        horizon = camera.horizon_v()
        if horizon is None:
            horizon = float(self.height)
        self.horizon = int(np.clip(round(horizon), 0, self.height))
        rows = np.arange(self.height, dtype=np.float32)[:, None, None]
        if cfg.SKY_STYLE == "clear":
            top, low, cloud = (200, 145, 90), (235, 220, 200), (245, 245, 245)
        else:
            top, low, cloud = (178, 176, 172), (218, 218, 214), (150, 150, 152)
        frac = np.clip(rows / max(self.horizon, 1), 0.0, 1.0)
        self.sky = (np.asarray(top, np.float32) * (1 - frac) + np.asarray(low, np.float32) * frac)
        self.sky = np.broadcast_to(self.sky, (self.height, self.width, 3)).astype(np.float32)
        self.cloud_color = np.asarray(cloud, np.float32)

        field = _periodic_noise(rng, self.height, self.width, 40.0) + 0.5 * _periodic_noise(rng, self.height, self.width, 12.0)
        threshold = np.quantile(field, 1.0 - float(np.clip(cfg.CLOUD_COVER, 0.0, 1.0)))
        alpha = np.clip((field - threshold) / 0.8, 0.0, 1.0) * 0.85
        fade = np.clip((self.horizon - rows[:, :, 0]) / 30.0, 0.0, 1.0)   # clouds thin out near the horizon
        cloud_alpha = (alpha * fade).astype(np.float32)
        # Store the cloud mask twice side by side so drifting is a cheap slice instead of a copy.
        self.cloud_alpha2 = np.repeat(np.concatenate([cloud_alpha, cloud_alpha], axis=1)[..., None], 3, axis=2)
        self.cloud_minus_sky = self.cloud_color - self.sky

        self.ground = None
        if self.horizon < self.height:
            g_rows = self.height - self.horizon
            texture = _periodic_noise(rng, g_rows, self.width, 3.0)[..., None] * 4.0
            shade = np.linspace(0.97, 1.03, g_rows, dtype=np.float32)[:, None, None]
            self.ground = (np.asarray(cfg.GROUND_COLOR_BGR, np.float32) * shade + texture).astype(np.float32)
        tree_profile = _periodic_noise(rng, 1, self.width, 6.0)[0]
        tree_heights = (cfg.TREELINE_HEIGHT_PX * (0.6 + 0.25 * tree_profile)).clip(0).astype(int)

        # The treeline never moves: keep the rows it can occupy and a mask of which pixels are trees.
        self.tree_top = int(max(self.horizon - tree_heights.max(), 0))
        tree_rows = np.arange(self.tree_top, self.horizon)[:, None]
        self.tree_mask = (tree_rows >= (self.horizon - tree_heights)[None, :])[..., None]
        self.tree_color = np.asarray(cfg.TREELINE_COLOR_BGR, np.float32)

    def render(self, t):
        """Return a fresh float32 BGR background for time t (clouds shifted by the drift speed)."""
        if self.photo is not None:
            return self.photo.copy()
        h = self.horizon
        start = self.width - int(round(t * self.drift)) % self.width   # clouds move toward +u
        img = np.empty((self.height, self.width, 3), np.float32)
        np.multiply(self.cloud_minus_sky[:h], self.cloud_alpha2[:h, start:start + self.width], out=img[:h])
        img[:h] += self.sky[:h]
        if self.ground is not None:
            img[h:] = self.ground
        if self.tree_top < h:
            img[self.tree_top:h] = np.where(self.tree_mask, self.tree_color, img[self.tree_top:h])
        return img


class SceneRenderer:
    """Composes background, towers, and conductors into a clean frame as seen by the camera."""

    def __init__(self, cfg, camera, conductors, rng):
        """Prepare the background and pre-project the static tower geometry at both ends of the span."""
        self.cfg = cfg
        self.camera = camera
        self.conductors = conductors
        self.background = BackgroundRenderer(cfg, camera, rng)
        # Towers are static, so rasterise each one once and only blend the cached patch per frame.
        self.tower_patches = []
        for x in (0.0, cfg.SPAN_LENGTH_M):
            lines, depths = [], []
            for p0, p1 in tower_segments(x, conductors, cfg):
                uv, valid, depth = camera.project(np.vstack([p0, p1]))
                if valid.all() and np.all(np.abs(uv) < MAX_ABS_COORD_PX):
                    lines.append(uv)
                    depths.append(depth.mean())
            if lines:
                thickness, alpha = wire_style(cfg.TOWER_MEMBER_WIDTH_M * camera.fx / float(np.median(depths)), 0.5)
                self.tower_patches.append(polyline_coverage(lines, (cfg.IMAGE_HEIGHT_PX, cfg.IMAGE_WIDTH_PX),
                                                            alpha, thickness))

    def render(self, t):
        """Return the clean float32 BGR frame for time t using each conductor's current gallop state."""
        img = self.background.render(t)
        for patch in self.tower_patches:
            blend_coverage(img, patch, self.cfg.TOWER_COLOR_BGR)
        for conductor in self.conductors:
            uv, valid, depth = self.camera.project(conductor.world_points())
            if not valid.any():
                continue
            width_px = conductor.diameter_m * self.camera.fx / float(np.median(depth[valid]))
            thickness, alpha = wire_style(width_px, self.cfg.MIN_WIRE_ALPHA)
            draw_polylines_alpha(img, _valid_runs(uv, valid), self.cfg.WIRE_COLOR_BGR, alpha, thickness)
        return img
