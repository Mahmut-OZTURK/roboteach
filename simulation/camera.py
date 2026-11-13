# camera.py — Calibrated multi-view RGB-D sensor with vectorized 3D point-cloud deprojection
import base64
import io
import os
import numpy as np
import pybullet as p
from PIL import Image


class SimCamera:
    WIDTH = 640
    HEIGHT = 480

    VIEWS = {
        "top": {
            "eye": [0.16, 0.0, 1.15],
            "target": [0.16, 0.0, 0.44],
            "up": [0.0, 1.0, 0.0],
        },
        "front_isometric": {
            "eye": [0.68, -0.48, 0.86],
            "target": [0.16, 0.0, 0.45],
            "up": [0.0, 0.0, 1.0],
        },
        "side_isometric": {
            "eye": [0.16, 0.78, 0.86],
            "target": [0.16, 0.0, 0.45],
            "up": [0.0, 0.0, 1.0],
        },
        "opposite_isometric": {
            "eye": [0.68, 0.48, 0.84],
            "target": [0.16, 0.0, 0.45],
            "up": [0.0, 0.0, 1.0],
        },
    }

    def __init__(self, headless: bool | None = None, physics_client_id: int = 0):
        self.fov = 60.0
        self.aspect = self.WIDTH / self.HEIGHT
        self.near = 0.1
        self.far = 3.0
        self.client = physics_client_id
        if headless is None:
            conn_info = p.getConnectionInfo(physicsClientId=self.client)
            headless = conn_info.get("connectionMethod") == p.DIRECT
        self.headless = headless
        self.renderer = (
            p.ER_TINY_RENDERER if self.headless else p.ER_BULLET_HARDWARE_OPENGL
        )
        self._perturbation = {
            "enabled": False,
            "seed": None,
            "depth_noise_mm": 0.0,
            "extrinsics_mm": 0.0,
            "extrinsics_deg": 0.0,
        }
        self._rng = np.random.default_rng()
        self._view_jitter: dict[str, tuple[list[float], list[float]]] = {}

    def set_perturbation(
        self,
        enabled: bool = False,
        seed: int | None = None,
        depth_noise_mm: float = 3.0,
        extrinsics_mm: float = 3.0,
        extrinsics_deg: float = 1.5,
    ):
        """Configures stochastic RGB-D depth noise and camera extrinsics perturbation."""
        self._perturbation = {
            "enabled": bool(enabled),
            "seed": seed,
            "depth_noise_mm": float(depth_noise_mm),
            "extrinsics_mm": float(extrinsics_mm),
            "extrinsics_deg": float(extrinsics_deg),
        }
        self._rng = np.random.default_rng(seed)
        self._view_jitter = {}
        if enabled:
            em = extrinsics_mm / 1000.0
            tan_ang = float(np.tan(np.deg2rad(extrinsics_deg)))
            for vname, v in self.VIEWS.items():
                dist = float(np.linalg.norm(np.array(v["eye"]) - np.array(v["target"])))
                td = dist * tan_ang
                eye_d = self._rng.uniform(-em, em, size=3).tolist()
                tgt_d = self._rng.uniform(-td, td, size=3).tolist()
                self._view_jitter[vname] = (eye_d, tgt_d)

    def capture_rgbd(self, view_name: str = "top"):
        view = self.VIEWS[view_name]
        eye = list(view["eye"])
        target = list(view["target"])
        if self._perturbation["enabled"] and view_name in self._view_jitter:
            ed, td = self._view_jitter[view_name]
            eye = [e + d for e, d in zip(eye, ed)]
            target = [t + d for t, d in zip(target, td)]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=view["up"],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.near, farVal=self.far
        )
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.WIDTH,
            height=self.HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            lightDirection=[0.3, -0.2, 1.5],
            lightAmbientCoeff=0.72,
            lightDiffuseCoeff=0.38,
            shadow=0,
            renderer=self.renderer,
            physicsClientId=self.client,
        )
        rgb_array = np.asarray(rgb, dtype=np.uint8).reshape(self.HEIGHT, self.WIDTH, 4)[:, :, :3]
        depth_array = np.asarray(depth, dtype=np.float32).reshape(self.HEIGHT, self.WIDTH)
        return rgb_array, depth_array, view_matrix, proj_matrix

    def pixel_to_world(self, px: float, py: float, depth_val: float, view_matrix) -> list[float]:
        pts = self.deproject_pixels_to_world(
            np.array([px], dtype=np.float64),
            np.array([py], dtype=np.float64),
            np.array([depth_val], dtype=np.float64),
            view_matrix,
        )
        return pts[0].tolist()

    def deproject_pixels_to_world(
        self, xs: np.ndarray, ys: np.ndarray, depths: np.ndarray, view_matrix
    ) -> np.ndarray:
        """Vectorized back-projection from image pixels + OpenGL depth buffer into 3D world frame."""
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.near, farVal=self.far
        )
        z_ndc = 2.0 * depths - 1.0
        z_eye = (2.0 * self.near * self.far) / (
            self.far + self.near - z_ndc * (self.far - self.near)
        )
        if self._perturbation["enabled"] and self._perturbation["depth_noise_mm"] > 0.0:
            sigma_m = self._perturbation["depth_noise_mm"] / 1000.0
            z_eye = np.clip(z_eye + self._rng.normal(0.0, sigma_m, size=z_eye.shape), self.near, self.far)
        x_ndc = (2.0 * (xs + 0.5) / self.WIDTH) - 1.0
        y_ndc = 1.0 - (2.0 * (ys + 0.5) / self.HEIGHT)

        pm = np.asarray(proj_matrix, dtype=np.float64).reshape(4, 4).T
        x_eye = x_ndc * z_eye / pm[0, 0]
        y_eye = y_ndc * z_eye / pm[1, 1]

        vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4).T
        vm_inv = np.linalg.inv(vm)
        pts_eye = np.stack([x_eye, y_eye, -z_eye, np.ones_like(z_eye)], axis=0)
        pts_world = (vm_inv @ pts_eye).T[:, :3]
        return pts_world

    def capture_b64(self, view_name: str = "top", save_dir: str | None = None) -> dict:
        rgb, _, _, _ = self.capture_rgbd(view_name)
        img = Image.fromarray(rgb)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{view_name}_latest.png"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"b64": base64.b64encode(buf.getvalue()).decode("utf-8"), "view": view_name}

    def capture_multiview_strip_b64(
        self,
        views: tuple[str, ...] = ("front_isometric", "top", "side_isometric"),
        tile_size: tuple[int, int] = (256, 192),
        save_dir: str | None = None,
    ) -> dict:
        """Stitches synchronized multi-view RGB frames into a compact JPEG strip for low-latency VLM judging."""
        tw, th = tile_size
        canvas = Image.new("RGB", (tw * len(views), th))
        for idx, vname in enumerate(views):
            rgb, _, _, _ = self.capture_rgbd(vname)
            tile = Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR)
            canvas.paste(tile, (idx * tw, 0))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            canvas.save(os.path.join(save_dir, "multiview_strip_latest.jpg"), format="JPEG", quality=85)
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85)
        return {"b64": base64.b64encode(buf.getvalue()).decode("utf-8"), "mime": "image/jpeg", "views": list(views)}
