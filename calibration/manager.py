# manager.py — Kalibrasyon + Bilgi yöneticisi
# Görev tipleri sabit bir listede DEĞİL — LLM dinamik olarak üretir
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class CalibrationManager:
    """
    İki modlu kalibrasyon sistemi:
    - calibrate: Görevi dene, öğren, kalibre et
    - execute: Kalibrasyon var mı kontrol et, varsa çalıştır

    FARK: Görev tipleri sabit bir array'de tutulmaz.
    LLM her yeni görevi dinamik olarak sınıflandırır (snake_case).
    Kalibrasyon dosyaları bu dinamik isimlere göre oluşturulur.
    """

    def __init__(self, calibration_dir="calibrations", knowledge_dir="knowledge"):
        self.calibration_dir = calibration_dir
        self.knowledge_dir = knowledge_dir
        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(knowledge_dir, exist_ok=True)

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print("✅ CalibrationManager hazır")

    # ─── Görev Sınıflandırma (LLM-native, sınırsız) ───

    def classify_task(self, task: str) -> str:
        """LLM ile görev tipini dinamik sınıflandırır — sabit liste YOK."""
        prompt = f"""Categorize this robot manipulation task into a short snake_case action name.

Examples:
- "put the red cube on top of the blue one" -> stack_on_top
- "move the green next to yellow" -> move_beside
- "put all cubes in the box" -> put_inside
- "organize objects by color" -> sort_by_color
- "build a tower with all cubes" -> build_tower
- "swap red and blue positions" -> swap_positions
- "clear the table" -> clear_table
- "push red to the left" -> push_object

Task: "{task}"

Respond with ONLY the snake_case category name, nothing else."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=20
            )
            task_type = resp.choices[0].message.content.strip().lower()
            task_type = "".join(c for c in task_type if c.isalnum() or c == "_")
            print(f"  🏷️  Görev tipi: {task_type}")
            return task_type
        except Exception as e:
            print(f"  ⚠️  Sınıflandırma hatası: {e}")
            return "general_task"

    # ─── Kalibrasyon Kaydet/Yükle ───

    def is_calibrated(self, task_type: str) -> bool:
        """Bu görev tipi kalibre edilmiş mi?"""
        path = os.path.join(self.calibration_dir, f"{task_type}.json")
        if not os.path.exists(path):
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get("calibrated", False)

    def save_calibration(self, task_type: str, task: str, instructions: list):
        """Başarılı kalibrasyon sonrası kaydet."""
        path = os.path.join(self.calibration_dir, f"{task_type}.json")
        data = {
            "task_type": task_type,
            "calibrated": True,
            "last_task": task,
            "instructions": instructions,
            "success_count": 1
        }
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            existing["success_count"] = existing.get("success_count", 0) + 1
            for inst in instructions:
                if inst not in existing["instructions"]:
                    existing["instructions"].append(inst)
            existing["calibrated"] = True
            existing["last_task"] = task
            data = existing

        with open(path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"  ✅ Kalibrasyon kaydedildi: {task_type}")

    def load_calibration(self, task_type: str) -> dict:
        """Kalibrasyon verisini yükle."""
        path = os.path.join(self.calibration_dir, f"{task_type}.json")
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def list_calibrated(self) -> list:
        """Kalibre edilmiş tüm görev tiplerini listele (dosya sisteminden)."""
        calibrated = []
        for f in os.listdir(self.calibration_dir):
            if f.endswith(".json"):
                path = os.path.join(self.calibration_dir, f)
                with open(path, 'r') as fh:
                    data = json.load(fh)
                if data.get("calibrated"):
                    calibrated.append(data.get("task_type", f.replace(".json", "")))
        return calibrated

    # ─── Knowledge (Öğrenilen Notlar) ───

    def save_knowledge(self, task_type: str, instruction: str):
        """Öğrenilen notu kaydet."""
        if not instruction or instruction == "None":
            return
        path = os.path.join(self.knowledge_dir, f"{task_type}.json")
        data = {"task_type": task_type, "instructions": [instruction]}
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            if instruction not in existing["instructions"]:
                existing["instructions"].append(instruction)
            data = existing
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_knowledge(self, task_type: str) -> list:
        """Öğrenilen notları yükle."""
        path = os.path.join(self.knowledge_dir, f"{task_type}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f).get("instructions", [])
        return []
