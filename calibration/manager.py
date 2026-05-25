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
        self.model = "llama-3.3-70b-versatile"
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

    def get_initial_scene_config(self, task: str) -> dict:
        """Görevi analiz ederek başlangıç sahne kurulumunu (stack, inside_box) belirler."""
        prompt = f"""Analyze this robot manipulation task to see if it specifies a special initial scene configuration or starting state for the objects (red_cube, blue_cube, green_cube, yellow_cube).
Specifically, check if any object is described as ALREADY being in a certain state BEFORE the main action starts (e.g. "X is already on top of Y", "X starts inside the box", "X is stacked on Y", "Y'nin üstünde X var").

CRITICAL: Do NOT confuse the TARGET GOAL state (the desired final state that the robot is asked to build, e.g., "build a tower in order X, Y, Z" or "put X on Y") with the INITIAL starting state. If the prompt describes a goal/target stack to build, it is NOT an initial state stack. The initial state should be empty unless the task EXPLICITLY states that they START or ALREADY are in a stack before the task begins.

For example:
- "put green on yellow but red is already on blue" -> {{"stacks": [["red_cube", "blue_cube"]], "inside_box": []}}
- "build a tower: blue at bottom, red on top, then green" -> {{"stacks": [], "inside_box": []}} (This is a target goal state, NOT an initial state!)
- "kutuları kule gibi üst üste diz: en altta mavi üstünde kırmızı" -> {{"stacks": [], "inside_box": []}} (Target goal state, NOT initial state!)

Format the response ONLY as a JSON with two keys:
1. "stacks": a list of lists of two strings: [["top_cube_name", "bottom_cube_name"]]. Use exact names: "red_cube", "blue_cube", "green_cube", "yellow_cube".
2. "inside_box": a list of strings of cubes that start inside the box.

Task: "{task}"

Respond with ONLY the JSON object, nothing else."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=150
            )
            content = resp.choices[0].message.content.strip()
            import json
            import re
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                print(f"  📐 Başlangıç kurulumu: {data}")
                return data
        except Exception as e:
            print(f"  ⚠️  Başlangıç kurulumu analiz hatası: {e}")
        return {"stacks": [], "inside_box": []}

    def break_down_task(self, task: str) -> list:
        """Görevi ardışık adımlara / kontrol noktalarına böler."""
        prompt = f"""Analyze this robot manipulation task and break it down into sequential checkpoints (steps) that need to be accomplished.
Each checkpoint must be a simple, short task description focusing on a single step/action (e.g. "grasp A and place on B", "move A inside the box", "move A to an empty space to unstack").

Examples:
- "put the blue cube on red, but red is already on blue" -> [
    "move red_cube to an empty space to unstack",
    "put blue_cube on top of red_cube"
  ]
- "put all cubes in the box" -> [
    "put red_cube inside brown_box",
    "put blue_cube inside brown_box",
    "put green_cube inside brown_box",
    "put yellow_cube inside brown_box"
  ]
- "put green on yellow next to red" -> [
    "put green_cube on top of yellow_cube"
  ]
- "kutuları üst üste diz 4 tanesini kule gibi" -> [
    "put red_cube on the table",
    "put blue_cube on top of red_cube",
    "put green_cube on top of blue_cube",
    "put yellow_cube on top of green_cube"
  ]

Task: "{task}"

Respond with ONLY a JSON list of strings, nothing else."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=250
            )
            content = resp.choices[0].message.content.strip()
            import json
            import re
            match = re.search(r"(\[.*\])", content, re.DOTALL)
            if match:
                steps = json.loads(match.group(1))
                print(f"  📌 Kontrol Noktaları (Checkpoints): {steps}")
                return steps
        except Exception as e:
            print(f"  ⚠️  Görev bölme hatası: {e}")
        return [task]
