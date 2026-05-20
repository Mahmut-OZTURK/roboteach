# progprompt.py — LLM tabanlı dinamik görev planlayıcı
# Herhangi bir dilde, herhangi bir görev tipini anlayıp çalıştırılabilir Python planı üretir
import os
from groq import Groq


class ProgPrompt:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        print(f"✅ Planner hazır ({self.model})")

    def plan(self, task: str, scene_description: str,
             objects: list, robot_position: list,
             extra_instructions: list = None) -> str:

        objects_str = "\n".join([
            f"  - {obj['name']}: position {[round(x, 3) for x in obj['position']]}"
            for obj in objects
        ])

        learning_str = ""
        if extra_instructions:
            learning_str = "\nLEARNED TIPS:\n"
            learning_str += "\n".join([f"- {inst}" for inst in extra_instructions[-3:]])

        obj_names = [obj['name'] for obj in objects]

        prompt = f"""Generate Python commands to control a robot arm for ANY manipulation task.

SKILLS:
- grasp("name")              → Picks up object (auto-approaches + grabs)
- release()                  → Drops held object at current position
- move_to([x, y, z])         → Move end-effector to [x,y,z]
- get_object_position("name") → Returns [x,y,z] from camera
- home()                     → Go to rest position

AVAILABLE OBJECTS (use ONLY these exact strings):
{obj_names}

OBJECTS WITH POSITIONS:
{objects_str}

TASK: {task}{learning_str}

OBJECT NAME MAPPING (Turkish → ID):
- "kırmızı" / "red" → "red_cube"
- "mavi" / "blue" → "blue_cube"
- "yeşil" / "green" → "green_cube"
- "sarı" / "yellow" → "yellow_cube"
- "kutu" / "box" / "karton" → "brown_box"

PHYSICS RULES:
1. "on top" / "üstüne" / "üzerine": Use target_z + 0.05 (exactly places a 5cm cube on top of another)
2. "inside" / "içine": Use target_z + 0.02 (drops inside the box base)
3. "beside" / "yanına": Use target_z (same height)
4. "touching" / "temas": Use 0.06m offset from target center in XY
5. Each object must be grasped individually. Release before grasping the next.
6. ALWAYS end with home()

MULTI-OBJECT EXAMPLE — "tüm küpleri kutunun içine koy":
box = get_object_position("brown_box")
grasp("red_cube")
move_to([box[0], box[1], box[2] + 0.20])
move_to([box[0], box[1], box[2] + 0.05])
release()
home()
grasp("blue_cube")
move_to([box[0], box[1], box[2] + 0.20])
move_to([box[0], box[1], box[2] + 0.05])
release()
home()

STACKING EXAMPLE — "yeşili kırmızının üstüne koy":
target = get_object_position("red_cube")
grasp("green_cube")
move_to([target[0], target[1], target[2] + 0.20])
move_to([target[0], target[1], target[2] + 0.02])
release()
home()

NEGATIVE CONSTRAINT EXAMPLE — "sarıyı kırmızının yanına koy ama maviye dokunma":
target = get_object_position("red_cube")
grasp("yellow_cube")
move_to([target[0] + 0.06, target[1], target[2] + 0.20])
move_to([target[0] + 0.06, target[1], target[2]])
release()
home()

OUTPUT: ONLY flat Python lines. No comments, no markdown, no def/class/if/for.
IF THE TASK IS PHYSICALLY IMPOSSIBLE (e.g. "fly", "paint", "destroy", "shrink", objects don't exist, or contradicts physics), output ONLY: IMPOSSIBLE: <reason>
"""
        print(f"\n🧠 Plan üretiliyor ({self.model}): '{task}'")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are a robot motion planner. Output ONLY executable Python lines. No comments, no markdown. Follow the example format exactly. Always end with home(). If the task is physically impossible for a pick-and-place robot arm, output ONLY: IMPOSSIBLE: <reason>"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        plan = response.choices[0].message.content.strip()
        plan = plan.replace("```python", "").replace("```", "").strip()

        # İmkansız görev kontrolü
        if plan.upper().startswith("IMPOSSIBLE"):
            reason = plan.split(":", 1)[1].strip() if ":" in plan else "Bilinmeyen sebep"
            print(f"\n🚫 İMKANSIZ GÖREV: {reason}")
            return None

        print(f"\n📋 Üretilen Plan:\n{plan}\n")
        return plan

