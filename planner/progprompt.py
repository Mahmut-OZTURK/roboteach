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
PHYSICS RULES:
1. "on top" / "üstüne" / "üzerine": Move above at target_z + 0.15 first, then LOWER to target_z + 0.052. This gently places a 5cm cube on top.
2. For multiple stacks (towers), ALWAYS query the position of the immediately underlying cube right before placing (using get_object_position), and place it at exactly that underlying cube's Z + 0.052. NEVER guess or accumulate manual height offsets like +0.1, +0.2 etc.
3. "beside" / "yanına": Use target_z (same height)
4. "touching" / "temas": Use 0.06m offset from target center in XY
5. Each object must be grasped individually. Release before grasping the next.
6. ALWAYS end with home()
7. CRITICAL: After grasping an object, ALWAYS lift it straight up to a safe high Z (z = 0.65) at its current XY coordinate BEFORE moving horizontally. Never move horizontally from low Z as it will collide with and knock over other cubes.
8. CRITICAL: When building towers, ALWAYS call home() right after release() and BEFORE querying the position of the placed cube (using get_object_position). This ensures the robot arm moves out of the camera's view, allowing 100% accurate top-view vision without occlusion.

STACKING EXAMPLE — "yeşili kırmızının üstüne koy":
target = get_object_position("red_cube")
green_pos = get_object_position("green_cube")
grasp("green_cube")
move_to([green_pos[0], green_pos[1], 0.65]) # Lift straight up to safe Z!
move_to([target[0], target[1], 0.65])        # Move horizontally at safe Z!
move_to([target[0], target[1], target[2] + 0.052]) # Lower straight down!
release()
home()

TOWER STACKING EXAMPLE (3 levels) — "red_cube, blue_cube ve green_cube'u kule gibi üst üste diz":
red_pos = get_object_position("red_cube")
blue_pos = get_object_position("blue_cube")
grasp("blue_cube")
move_to([blue_pos[0], blue_pos[1], 0.65]) # Lift straight up!
move_to([red_pos[0], red_pos[1], 0.65])   # Move horizontally!
move_to([red_pos[0], red_pos[1], red_pos[2] + 0.052]) # Lower!
release()
home() # ALWAYS go home to clear camera occlusion!

blue_placed_pos = get_object_position("blue_cube") # Query after going home!
green_pos = get_object_position("green_cube")
grasp("green_cube")
move_to([green_pos[0], green_pos[1], 0.65]) # Lift straight up!
move_to([blue_placed_pos[0], blue_placed_pos[1], 0.65]) # Move horizontally!
move_to([blue_placed_pos[0], blue_placed_pos[1], blue_placed_pos[2] + 0.052]) # Lower!
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

