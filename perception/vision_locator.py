# vision_locator.py — Kamera tabanlı nesne konum tespiti (Pure Vision)
# Simülasyondan doğrudan pozisyon sorgulamaz, sadece kamera RGB+Depth kullanır
import numpy as np


class VisionLocator:
    """
    Renk segmentasyonu + depth buffer ile nesne 3D konumunu tespit eder.
    Pure vision: p.getBasePositionAndOrientation() KULLANMAZ.
    """

    Z_MIN = 0.38
    Z_MAX = 0.68
    MIN_PIXEL_COUNT = 15

    def __init__(self, camera, object_dimensions: dict):
        self.camera = camera

        # Sıkı renk aralıkları — her nesne için benzersiz
        self.color_ranges = {
            "red_cube": {
                "lower": np.array([200, 0, 0]),
                "upper": np.array([255, 60, 60])
            },
            "blue_cube": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([60, 60, 255])
            },
            "green_cube": {
                "lower": np.array([0, 180, 0]),
                "upper": np.array([60, 255, 60])
            },
            "yellow_cube": {
                "lower": np.array([200, 200, 0]),
                "upper": np.array([255, 255, 60])
            },
        }

        self.object_dimensions = object_dimensions
        self._last_known = {}
        print("✅ VisionLocator hazır (pure camera-based, filtered)")

    def clear_cache(self):
        """Cache'i temizle — sahne sıfırlandığında çağrılmalı."""
        self._last_known = {}
        print("  🗑️  Vision cache temizlendi")

    def locate(self, object_name: str) -> list:
        """Kameradan nesne konumunu tespit eder."""
        if object_name not in self.color_ranges:
            print(f"  ❌ '{object_name}' renk tanımı yok!")
            return None

        views_to_try = ["top", "front_isometric", "side_isometric"]

        for view_name in views_to_try:
            rgb, depth, view_matrix, _ = self.camera.capture_rgbd(view_name)
            color_range = self.color_ranges[object_name]

            mask = self._color_segment(rgb, color_range)
            coords = self._find_centroid(mask)
            if coords is None:
                continue

            cy, cx = coords
            
            # Gürültüyü arındırmak için centroid çevresinde 5x5'lik derinlik penceresi al
            h, w = depth.shape
            ymin, ymax = max(0, cy - 2), min(h, cy + 3)
            xmin, xmax = max(0, cx - 2), min(w, cx + 3)
            depth_window = depth[ymin:ymax, xmin:xmax]
            
            # Arka plan piksellerini filtrele (değeri 1.0'a çok yakın olanlar arka plandır)
            valid_depths = depth_window[depth_window < 0.99]
            if len(valid_depths) > 0:
                d = float(np.median(valid_depths))
            else:
                d = float(depth[cy, cx])
                
            world_pos = self.camera.pixel_to_world(cx, cy, d, view_matrix)

            if world_pos[2] < self.Z_MIN or world_pos[2] > self.Z_MAX:
                continue

            # DÜZELTME: Kamera nesnenin üst yüzeyini görür (depth o yüzeye çarpar).
            # Gerçek fiziksel merkeze (Z) inmek için objenin yarı boyunu çıkar.
            half_z = self.object_dimensions[object_name].get("half_extents", [0,0,0])[2]
            if "cube" in object_name:
                world_pos[2] -= half_z

            world_pos = [float(v) for v in world_pos]
            self._last_known[object_name] = list(world_pos)
            print(f"  📍 [Vision/{view_name}] '{object_name}': {[round(v, 3) for v in world_pos]}")
            return world_pos

        cached = self._last_known.get(object_name)
        if cached:
            print(f"  📍 [Cache] '{object_name}': {[round(v, 3) for v in cached]}")
            return list(cached)

        print(f"  ⚠️  '{object_name}' hiçbir kamerada tespit edilemedi!")
        return None

    def locate_all(self) -> dict:
        """Tüm bilinen nesnelerin konumlarını tespit eder."""
        positions = {}
        for name in self.color_ranges:
            pos = self.locate(name)
            if pos:
                positions[name] = pos
        return positions

    def _color_segment(self, rgb: np.ndarray, color_range: dict) -> np.ndarray:
        """RGB görüntüde sıkı renk segmentasyonu uygular."""
        lower = color_range["lower"]
        upper = color_range["upper"]
        mask = np.all((rgb >= lower) & (rgb <= upper), axis=2)
        return mask.astype(np.uint8)

    def _find_centroid(self, mask: np.ndarray):
        """Binary maskede nesne centroid pikselini bulur."""
        ys, xs = np.where(mask > 0)
        if len(ys) < self.MIN_PIXEL_COUNT:
            return None
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        return (cy, cx)
