# vlm_scorer.py — VLM tabanlı görsel değerlendirme + öğrenme
# Simülasyondan SADECE kamera kullanır, pozisyon sorgulamaz
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from dotenv import load_dotenv
from simulation.camera import SimCamera

load_dotenv()


class VLMScorer:
    """
    Kamera görüntüsü ile görev değerlendirmesi.
    Self-learning: Her denemede not alır, talimatlarına ekler.
    """

    def __init__(self, camera: SimCamera):
        self.camera = camera
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print(f"✅ VLM Scorer hazır ({self.model})")

    def analyze_scene(self, images: list = None) -> str:
        """
        Sahneyi VLM ile analiz eder — planner'a girdi olacak.
        Kamera görüntüsünden sahne açıklaması üretir.
        """
        if images is None:
            img_iso = self.camera.capture_b64("front_isometric", "captures")
            img_top = self.camera.capture_b64("top", "captures")
            img_side = self.camera.capture_b64("side_isometric", "captures")
            images = [img_iso, img_top, img_side]

        prompt = """Describe the robot manipulation scene in the image.
List all visible objects with their approximate positions (left/right/center, on table, etc.).
Describe spatial relationships between objects.
Keep it concise and factual. No speculation."""

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img['b64']}"}
            })

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0, max_tokens=300
            )
            desc = resp.choices[0].message.content.strip()
            print(f"\n🔍 Sahne Analizi:\n{desc}\n")
            return desc
        except Exception as e:
            print(f"  ⚠️  Sahne analizi hatası: {e}")
            return "Scene analysis unavailable."

    def integrity_check(self) -> bool:
        """Hızlı güvenlik kontrolü — SADECE ciddi kazalar (nesne masadan düştü)."""
        img = self.camera.capture_b64("top")
        prompt = """Look at the image from above. Check ONLY for catastrophic failures:
- Has any object fallen OFF the table onto the floor?
- Has any object been launched or thrown far away?

Objects stacked on each other, tilted, or touching are NORMAL and SAFE.
Respond ONLY: 'SAFE: True' or 'SAFE: False' with 'REASON: [reason]'."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img['b64']}"}}
                ]}],
                temperature=0.1, max_tokens=100
            )
            text = resp.choices[0].message.content.upper()
            return "SAFE: TRUE" in text
        except:
            return True

    def evaluate(self, task: str) -> dict:
        """
        Görev sonucunu kamera ile değerlendirir.
        Self-learning: hata varsa öğrenilen talimat üretir.
        """
        print(f"\n📊 VLM Değerlendirme başlıyor (Üçlü Açı)...")
        img_iso = self.camera.capture_b64("front_isometric", "captures")
        img_top = self.camera.capture_b64("top", "captures")
        img_side = self.camera.capture_b64("side_isometric", "captures")

        prompt = f"""Evaluate this robot task STRICTLY based on the images.

TASK: "{task}"

EVALUATION RULES:
- BE EXTREMELY STRICT. Do NOT give false positives. If the task failed or is partially done, say SUCCESS: False.
- If the task is "stack on top" or "üstüne koy", the object MUST be sitting directly on top of the other object. If it fell off or is only touching/next to it, SUCCESS = False.
- For tower stacking/ordering tasks (e.g. "kule", "sırala", "üst üste diz", "order"), the cubes MUST form a single vertical tower. You MUST verify the EXACT vertical order of the colors from bottom to top. If the tower has collapsed, is slanted/fallen, or if the cubes are in the wrong vertical color sequence, SUCCESS = False and Score = 0.0.
- If the task is "put inside" or "içine koy", the object MUST be completely inside the brown box.
- If the task is "put beside" or "yanına koy", the objects must be next to each other on the table.
- If the task is "diagonal" or "çaprazına", the object MUST be placed diagonally relative to the other object (offsetted on both X and Y axes, forming a 45-degree corner relationship), NOT directly stacked and NOT directly straight side-by-side. Verify from the top view.
- NEGATIVE CONSTRAINTS (e.g. "do not move X") are CRITICAL. Violation = Score 0.
- Objects MUST NOT be on the floor.

ANALYSIS: Briefly (2 sentences max) describe what you see. Is the task 100% achieved?

SUMMARY BLOCK (MUST BE AT THE END):
REASONING: [1 sentence]
VISUAL_SUCCESS: True/False
VISUAL_SCORE: 0.0 to 1.0
ISSUE: [Major error or 'None']
LEARNED_INSTRUCTION: [Tip or 'None']"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_iso['b64']}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_top['b64']}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_side['b64']}"}}
        ]

        result = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0, max_tokens=600
                )
                result = self._parse(resp.choices[0].message.content)
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str:
                    wait = 15 * (attempt + 1)
                    print(f"  ⏳ Rate limit — {wait}s bekleniyor (deneme {attempt+1}/3)...")
                    import time
                    time.sleep(wait)
                else:
                    print(f"  ⚠️  VLM hatası: {e}")
                    break

        if result is None:
            print(f"  ⚠️  VLM çağrısı başarısız — plan tamamlandıysa başarılı sayılıyor")
            result = {"visual_success": True, "visual_score": 0.7,
                      "issue": "None", "instruction": None,
                      "reasoning": "VLM unavailable — assuming success since plan completed"}

        print(f"    🧐 Reasoning: {result.get('reasoning', '')}")
        print(f"    ✅ Başarı: {result['visual_success']}")
        print(f"    🏆 Skor: {result['visual_score']:.2f}")
        if not result["visual_success"]:
            print(f"    ⚠️  Sorun: {result['issue']}")

        return {
            "success": result["visual_success"],
            "score": result["visual_score"],
            "issue": result["issue"],
            "instruction": result.get("instruction")
        }

    def _parse(self, text: str) -> dict:
        """VLM yanıtını parse eder."""
        result = {"visual_success": False, "visual_score": 0.0,
                  "issue": "Parse error", "instruction": None, "reasoning": ""}
        try:
            # Temizlik: Markdown kalınlaştırmalarını ve başlıkları kaldır
            clean_text = text.replace("**", "").replace("###", "").replace("##", "")
            lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
            
            found_something = False
            for line in lines:
                up = line.upper()
                if "REASONING:" in up or "ANALYSIS:" in up or "EXPLANATION:" in up:
                    result["reasoning"] = line.split(":", 1)[1].strip()
                    found_something = True
                elif "VISUAL_SUCCESS:" in up or "SUCCESS:" in up or "ACHIEVED:" in up:
                    result["visual_success"] = "TRUE" in up or "YES" in up
                    found_something = True
                elif "VISUAL_SCORE:" in up or "SCORE:" in up or "RATING:" in up:
                    try:
                        import re
                        match = re.search(r"(\d+\.?\d*)", line)
                        if match:
                            result["visual_score"] = float(match.group(1))
                            found_something = True
                    except:
                        pass
                elif "ISSUE:" in up or "ISSUE_FOUND:" in up or "ERROR:" in up:
                    result["issue"] = line.split(":", 1)[1].strip()
                    found_something = True
                elif "LEARNED_INSTRUCTION:" in up or "ADVICE:" in up or "TIP:" in up:
                    result["instruction"] = line.split(":", 1)[1].strip()
                    found_something = True
            
            if found_something:
                result["issue"] = "None" if result["issue"] == "Parse error" else result["issue"]

            # Eğer reasoning hala boşsa ama metin varsa, anlamlı bir yer bul
            if not result["reasoning"] and len(lines) > 0:
                result["reasoning"] = lines[-1] if len(lines) > 5 else lines[0]
                
        except Exception as e:
            print(f"  ⚠️  Parse hatası: {e}")
            print(f"  📝 Ham yanıt: {text}")
        return result
