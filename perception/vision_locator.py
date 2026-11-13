# vision_locator.py — Multi-view 3D RGB-D point-cloud perception & ray-compensated centroid fusion
import numpy as np


class VisionLocator:
    """
    Pure vision 3D spatial perception (NEVER calls PyBullet ground-truth pose APIs).
    Fuses top and isometric RGB-D views into 3D world point clouds, handling:
      - Illumination-invariant chromatic segmentation for cubes, hollow containers, and obstacles
      - Occlusion fallback when stacked objects hide lower blocks from overhead view
      - Surface-normal / ray-direction volumetric compensation from visible surface to 3D center
    """

    Z_MIN = 0.415
    Z_MAX = 0.78
    MIN_PIXELS_TOP = 14
    MIN_PIXELS_LAT = 10

    def __init__(self, camera, object_dimensions: dict):
        self.camera = camera
        self.object_dimensions = object_dimensions
        self._last_known: dict[str, list[float]] = {}
        self._last_metadata: dict[str, dict] = {}

    def clear_cache(self):
        self._last_known = {}
        self._last_metadata = {}

    def _segment_chromatic(self, rgb: np.ndarray, object_name: str) -> np.ndarray:
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)

        if object_name == "red_cube":
            mask = (r > 120) & (r > 2.1 * g) & (r > 2.1 * b)
        elif object_name == "blue_cube":
            mask = (b > 115) & (b > 1.9 * r) & (b > 1.7 * g)
        elif object_name == "green_cube":
            mask = (g > 105) & (g > 1.8 * r) & (g > 1.8 * b)
        elif object_name == "yellow_cube":
            mask = (r > 130) & (g > 120) & (b < 70) & (np.abs(r - g) < 55)
        elif object_name == "brown_box":
            mask = (
                (r > 75)
                & (r < 205)
                & (g > 30)
                & (g < 120)
                & (b < 45)
                & (r > 1.55 * g)
                & (g > 1.5 * b)
            )
        elif object_name == "obstacle_wall":
            mask = (r > 75) & (b > 85) & (g < 65) & (r > 1.7 * g) & (b > 1.8 * g)
        else:
            return np.zeros(rgb.shape[:2], dtype=bool)
        return mask

    def _extract_view_cloud(
        self, view_name: str, object_name: str, frame: tuple | None = None
    ) -> np.ndarray | None:
        rgb, depth, view_matrix, _ = frame if frame is not None else self.camera.capture_rgbd(view_name)
        mask = self._segment_chromatic(rgb, object_name) & (depth < 0.992)
        ys, xs = np.where(mask)
        min_px = self.MIN_PIXELS_TOP if view_name == "top" else self.MIN_PIXELS_LAT
        if len(ys) < min_px:
            return None

        pts = self.camera.deproject_pixels_to_world(
            xs.astype(np.float64), ys.astype(np.float64), depth[ys, xs].astype(np.float64), view_matrix
        )
        # Filter workspace envelope (table surface and manipulation volume)
        valid = (
            (pts[:, 0] > -0.15)
            & (pts[:, 0] < 0.45)
            & (pts[:, 1] > -0.38)
            & (pts[:, 1] < 0.38)
            & (pts[:, 2] >= self.Z_MIN)
            & (pts[:, 2] <= self.Z_MAX)
        )
        pts = pts[valid]
        if len(pts) < min_px:
            return None

        # Trim 10% outlier boundary pixels via median absolute bounds
        med = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - med, axis=1)
        inliers = pts[dists <= np.percentile(dists, 88) + 1e-4]
        return inliers if len(inliers) >= 5 else pts

    def locate_detailed(
        self, object_name: str, cached_frames: dict[str, tuple] | None = None
    ) -> dict | None:
        if object_name not in self.object_dimensions:
            return None

        spec = self.object_dimensions[object_name]
        half = spec.get("half_extents", [0.025, 0.025, 0.025])
        obj_type = spec.get("type", "cube")

        # 1. Overhead camera first (unbiased XY when unoccluded)
        top_frame = cached_frames.get("top") if cached_frames is not None else None
        top_pts = self._extract_view_cloud("top", object_name, frame=top_frame)
        if top_pts is not None and len(top_pts) >= 24:
            cx = float(np.median(top_pts[:, 0]))
            cy = float(np.median(top_pts[:, 1]))
            z_top = float(np.percentile(top_pts[:, 2], 85))
            if obj_type == "container":
                # Container floor/walls: center is mean of outer envelope, z at base
                xmin, xmax = float(np.percentile(top_pts[:, 0], 5)), float(np.percentile(top_pts[:, 0], 95))
                ymin, ymax = float(np.percentile(top_pts[:, 1], 5)), float(np.percentile(top_pts[:, 1], 95))
                pos = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 0.425]
            elif obj_type == "obstacle":
                xmin, xmax = float(np.percentile(top_pts[:, 0], 5)), float(np.percentile(top_pts[:, 0], 95))
                ymin, ymax = float(np.percentile(top_pts[:, 1], 5)), float(np.percentile(top_pts[:, 1], 95))
                pos = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, z_top - half[2]]
            else:
                pos = [cx, cy, max(0.445, z_top - half[2])]

            info = {
                "position": pos,
                "z_top": z_top,
                "bbox_min": np.percentile(top_pts, 5, axis=0).tolist(),
                "bbox_max": np.percentile(top_pts, 95, axis=0).tolist(),
                "occluded_from_top": False,
                "source_view": "top",
            }
            self._last_known[object_name] = list(pos)
            self._last_metadata[object_name] = info
            return info

        # 2. Multi-view lateral fusion when occluded from overhead (e.g., bottom cube of a tower)
        lateral_views = ["front_isometric", "side_isometric", "opposite_isometric"]
        fused_centers = []
        all_pts = []
        for vname in lateral_views:
            lat_frame = None
            if cached_frames is not None:
                if vname not in cached_frames:
                    cached_frames[vname] = self.camera.capture_rgbd(vname)
                lat_frame = cached_frames[vname]
            pts = self._extract_view_cloud(vname, object_name, frame=lat_frame)
            if pts is None:
                continue
            all_pts.append(pts)
            surf_med = np.median(pts, axis=0)
            eye = np.asarray(self.camera.VIEWS[vname]["eye"], dtype=np.float64)
            # Push surface centroid inward along horizontal viewing ray by half-width
            horiz_ray = surf_med[:2] - eye[:2]
            norm = np.linalg.norm(horiz_ray)
            if norm > 1e-6:
                horiz_unit = horiz_ray / norm
            else:
                horiz_unit = np.zeros(2)
            est_xy = surf_med[:2] + horiz_unit * (half[0] * 0.82)
            # Vertical center from visible side face median
            est_z = float(np.clip(np.median(pts[:, 2]), 0.445, self.Z_MAX))
            fused_centers.append([float(est_xy[0]), float(est_xy[1]), est_z])

        if fused_centers:
            pos = np.median(np.asarray(fused_centers), axis=0).tolist()
            merged = np.concatenate(all_pts, axis=0)
            info = {
                "position": pos,
                "z_top": float(pos[2] + half[2]),
                "bbox_min": np.percentile(merged, 5, axis=0).tolist(),
                "bbox_max": np.percentile(merged, 95, axis=0).tolist(),
                "occluded_from_top": True,
                "source_view": "multi_lateral",
            }
            self._last_known[object_name] = list(pos)
            self._last_metadata[object_name] = info
            return info

        if object_name in self._last_known:
            return self._last_metadata.get(
                object_name,
                {"position": list(self._last_known[object_name]), "occluded_from_top": True, "source_view": "cache"},
            )
        return None

    def locate(self, object_name: str) -> list[float] | None:
        info = self.locate_detailed(object_name)
        return list(info["position"]) if info else None

    def locate_all(self) -> dict[str, list[float]]:
        return {k: list(v["position"]) for k, v in self.locate_all_detailed().items()}

    def locate_all_detailed(self) -> dict[str, dict]:
        frames: dict[str, tuple] = {"top": self.camera.capture_rgbd("top")}
        details = {}
        for name in self.object_dimensions:
            info = self.locate_detailed(name, cached_frames=frames)
            if info is not None:
                details[name] = info
        return details
