"""
camera.py - Pinhole camera that maps 3D world points (metres) to image pixels.

The simulator renders the scene through this camera, and ground_truth.py logs the
same projection for tracked points, so a CV algorithm's pixel measurements can be
compared directly with the truth. The intrinsic matrix K and extrinsics (R, t) are
saved in each run's metadata.json, which lets later steps convert pixel motion
back into metres exactly as a calibrated field camera would.
"""

import math

import numpy as np

MIN_DEPTH_M = 0.1   # Points closer than this (or behind the camera) are not projected


class PinholeCamera:
    """Ideal pinhole camera with square pixels, defined by a position, a look-at target, and a field of view."""

    def __init__(self, width_px, height_px, hfov_deg, position_m, target_m):
        """Build intrinsics from the field of view and a world-to-camera rotation from the look-at geometry."""
        self.width = int(width_px)
        self.height = int(height_px)
        self.hfov_deg = float(hfov_deg)
        self.position = np.asarray(position_m, dtype=float)
        self.target = np.asarray(target_m, dtype=float)
        self.fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        forward = self.target - self.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-6:
            raise ValueError("Camera cannot look straight up or down; move CAMERA_TARGET_M")
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.forward = forward
        # Rows are the camera axes in world coordinates: x right, y down, z forward (OpenCV convention).
        self.R = np.vstack([right, -up, forward])
        self.t = -self.R @ self.position

    @classmethod
    def from_config(cls, cfg):
        """Create the camera described by the IMAGE_* and CAMERA_* settings."""
        return cls(cfg.IMAGE_WIDTH_PX, cfg.IMAGE_HEIGHT_PX, cfg.CAMERA_HFOV_DEG,
                   cfg.CAMERA_POSITION_M, cfg.CAMERA_TARGET_M)

    def intrinsic_matrix(self):
        """Return the 3x3 intrinsic matrix K."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points_m):
        """Project (N, 3) world points to pixels; return (uv, valid, depth).

        `uv` is (N, 2) float pixel coordinates, `valid` flags points in front of the camera, and
        `depth` is the distance along the optical axis. Invalid points get NaN coordinates.
        """
        cam = (np.atleast_2d(points_m) - self.position) @ self.R.T
        depth = cam[:, 2]
        valid = depth > MIN_DEPTH_M
        uv = np.full((cam.shape[0], 2), np.nan)
        uv[valid, 0] = self.fx * cam[valid, 0] / depth[valid] + self.cx
        uv[valid, 1] = self.fy * cam[valid, 1] / depth[valid] + self.cy
        return uv, valid, depth

    def horizon_v(self):
        """Return the image row of the horizon (where level ground meets the sky), or None if undefined."""
        heading = np.array([self.forward[0], self.forward[1], 0.0])
        if np.linalg.norm(heading) < 1e-6:
            return None
        far_point = self.position + heading / np.linalg.norm(heading) * 1e7
        uv, valid, _ = self.project(far_point)
        return float(uv[0, 1]) if valid[0] else None

    def metres_per_pixel(self, depth_m):
        """Return the size in metres of one pixel at the given depth (useful for choosing thresholds)."""
        return float(depth_m) / self.fx
