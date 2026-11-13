# manager.py — Canonical Task Calibration & Structured Experience Manager
import json
import os
from groq import Groq
from dotenv import load_dotenv

from calibration.ontology import CanonicalTaskOntology, TaskSpec
from calibration.memory import CalibrationMemory, FailureDiagnostic

load_dotenv()


class CalibrationManager:
    """
    Coordinates canonical task classification, scene precondition extraction,
    and structured causal/geometric experience persistence.
    """

    def __init__(
        self,
        calibration_dir: str = "calibrations",
        knowledge_dir: str = "knowledge",
        llm_client=None,
    ):
        self.calibration_dir = calibration_dir
        self.knowledge_dir = knowledge_dir
        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(knowledge_dir, exist_ok=True)

        api_key = os.getenv("GROQ_API_KEY")
        self.client = (
            llm_client
            if llm_client is not None
            else (Groq(api_key=api_key, max_retries=1, timeout=20.0) if api_key else None)
        )
        self.model = "openai/gpt-oss-20b"
        self.ontology = CanonicalTaskOntology(llm_client=self.client, model=self.model)
        self.memory = CalibrationMemory(storage_dir=calibration_dir)

    def parse_task(self, task: str) -> TaskSpec:
        return self.ontology.parse(task)

    def classify_task(self, task: str) -> str:
        return self.ontology.parse(task).canonical_type.lower()

    def is_calibrated(self, task_type: str) -> bool:
        if self.memory.is_calibrated(task_type.upper()):
            return True
        path = os.path.join(self.calibration_dir, f"{task_type.lower()}.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("calibrated", False))

    def record_structured_failure(self, task_type: str, diag: FailureDiagnostic):
        self.memory.record_failure(task_type.upper(), diag)
        if diag.symbolic_rule:
            self.save_knowledge(task_type.lower(), diag.symbolic_rule)

    def get_priors(self, task_type: str) -> dict:
        return self.memory.get_kinematic_priors(task_type.upper())

    def get_structured_rules(self, task_type: str) -> list[str]:
        rules = self.memory.get_symbolic_rules(task_type.upper())
        if rules:
            return rules
        return self.load_knowledge(task_type.lower())

    def save_calibration(self, task_type: str, task: str, instructions: list):
        self.memory.mark_calibrated(task_type.upper(), task)
        path = os.path.join(self.calibration_dir, f"{task_type.lower()}.json")
        data = {
            "task_type": task_type.lower(),
            "calibrated": True,
            "last_task": task,
            "instructions": list(dict.fromkeys(instructions)),
            "success_count": 1,
        }
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            existing["success_count"] = existing.get("success_count", 0) + 1
            for inst in instructions:
                if inst and inst not in existing.setdefault("instructions", []):
                    existing["instructions"].append(inst)
            existing["calibrated"] = True
            existing["last_task"] = task
            data = existing
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load_calibration(self, task_type: str) -> dict:
        path = os.path.join(self.calibration_dir, f"{task_type.lower()}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def list_calibrated(self) -> list[str]:
        found = set()
        for fname in os.listdir(self.calibration_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.calibration_dir, fname), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("calibrated"):
                label = data.get("canonical_type") or data.get("task_type") or fname[:-5]
                found.add(str(label).lower())
        return sorted(found)

    def save_knowledge(self, task_type: str, instruction: str):
        if not instruction or instruction == "None":
            return
        path = os.path.join(self.knowledge_dir, f"{task_type.lower()}.json")
        data = {"task_type": task_type.lower(), "instructions": [instruction]}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if instruction not in existing.setdefault("instructions", []):
                existing["instructions"].append(instruction)
            data = existing
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load_knowledge(self, task_type: str) -> list[str]:
        path = os.path.join(self.knowledge_dir, f"{task_type.lower()}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("instructions", [])
        return []

    def get_initial_scene_config(self, task: str) -> dict:
        return self.ontology.parse(task).to_scene_config()

    def break_down_task(self, task: str) -> list[str]:
        spec = self.ontology.parse(task)
        return self.ontology.decompose_checkpoints(spec)
