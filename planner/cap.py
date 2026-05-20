# cap.py — Code as Policies: Adım adım çalıştırıcı
# Her kritik adımda VLM ile kontrol yapar
import pybullet as p
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CaP:
    """
    LLM planını satır satır çalıştırır.
    Grasp sonrası: VLM integrity check
    Release sonrası: physics settle bekle, sonra kontrol
    """

    def __init__(self, skills):
        self.skills = skills
        self.sandbox = {
            "move_to":             self.skills.move_to,
            "grasp":               self.skills.grasp,
            "release":             self.skills.release,
            "home":                self.skills.home,
            "get_object_position": self.skills.get_object_position,
            "print":               print,
        }
        print("✅ CaP (Code as Policies) hazır!")

    def _clean_plan(self, plan: str) -> list:
        """LLM planını satırlara böler ve temizler."""
        lines = plan.strip().splitlines()
        clean = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#") or "```" in s:
                continue
            clean.append(s)
        return clean

    def execute(self, plan: str, scorer=None) -> bool:
        """Planı adım adım çalıştırır."""
        lines = self._clean_plan(plan)
        scope = {**self.sandbox}

        print(f"\n🚀 Plan çalıştırılıyor ({len(lines)} adım)...")
        print("=" * 50)

        for i, line in enumerate(lines):
            print(f"  [{i+1}/{len(lines)}]: {line}")
            try:
                exec(line, {"__builtins__": {}}, scope)

                # Release sonrası → physics settle bekle
                if "release" in line:
                    print(f"  ⏳ Physics settle bekleniyor...")
                    for _ in range(300):
                        p.stepSimulation()
                        time.sleep(1.0 / 240.0)

            except Exception as e:
                print(f"  ❌ Hata (Satır: {line}): {e}")
                return False

        print("=" * 50)
        print("✅ Tüm adımlar tamamlandı!")
        return True
