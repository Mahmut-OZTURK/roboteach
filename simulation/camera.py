# camera.py — PyBullet kamera sistemi
# RGB + Depth çekim, pixel→3D dönüşüm
import pybullet as p
import numpy as np
import base64
from PIL import Image
import io
import os


class SimCamera:
    """Simülasyon kamerası — RGB ve Depth buffer ile çalışır."""

    WIDTH = 640
    HEIGHT = 480

    # Kamera pozisyonları — sahne merkezine bakacak şekilde
    VIEWS = {
        "top": {
            "eye": [0.15, 0, 1.2],
            "target": [0.15, 0, 0.45],
            "up": [0, 1, 0]
        },
        "front_isometric": {
            "eye": [0.7, -0.5, 0.9],
            "target": [0.15, 0, 0.45],
            "up": [0, 0, 1]
        },
        "side_isometric": {
            "eye": [0.15, 0.8, 0.9],
            "target": [0.15, 0, 0.45],
            "up": [0, 0, 1]
        },
    }

    def __init__(self):
        self.fov = 60
        self.aspect = self.WIDTH / self.HEIGHT
        self.near = 0.1
        self.far = 3.0
        print("✅ Kamera sistemi hazır (RGB + Depth)")

    def capture_rgbd(self, view_name="top"):
        """RGB + Depth buffer çek — VisionLocator için."""
        view = self.VIEWS[view_name]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=view["eye"],
            cameraTargetPosition=view["target"],
            cameraUpVector=view["up"]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect,
            nearVal=self.near, farVal=self.far
        )
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.WIDTH, height=self.HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(self.HEIGHT, self.WIDTH, 4)[:, :, :3]
        depth_array = np.array(depth, dtype=np.float32).reshape(self.HEIGHT, self.WIDTH)
        return rgb_array, depth_array, view_matrix, proj_matrix

    def pixel_to_world(self, px, py, depth_val, view_matrix):
        """Piksel + depth → 3D dünya koordinatı."""
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect,
            nearVal=self.near, farVal=self.far
        )
        # Depth buffer → gerçek derinlik
        z_ndc = 2.0 * depth_val - 1.0
        z_eye = (2.0 * self.near * self.far) / (self.far + self.near - z_ndc * (self.far - self.near))

        # Piksel → NDC
        x_ndc = (2.0 * px / self.WIDTH) - 1.0
        y_ndc = 1.0 - (2.0 * py / self.HEIGHT)

        # NDC → Eye space
        pm = np.array(proj_matrix).reshape(4, 4).T
        x_eye = x_ndc * z_eye / pm[0, 0]
        y_eye = y_ndc * z_eye / pm[1, 1]

        # Eye → World
        vm = np.array(view_matrix).reshape(4, 4).T
        vm_inv = np.linalg.inv(vm)
        point_eye = np.array([x_eye, y_eye, -z_eye, 1.0])
        point_world = vm_inv @ point_eye
        return list(point_world[:3])

    def capture_b64(self, view_name="top", save_dir=None):
        """Kamera görüntüsünü base64 formatında döndürür (VLM için)."""
        rgb, _, _, _ = self.capture_rgbd(view_name)
        img = Image.fromarray(rgb)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{view_name}_latest.png"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"b64": b64, "view": view_name}
