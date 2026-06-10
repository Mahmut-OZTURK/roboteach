# progprompt.py — LLM tabanlı dinamik görev planlayıcı
# Herhangi bir dilde, herhangi bir görev tipini anlayıp çalıştırılabilir Python planı üretir
import os
from groq import Groq


class ProgPrompt:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
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
7. CRITICAL: After grasping an object, ALWAYS lift it straight up to a safe high Z (z = 0.65 or higher if there are taller stacked towers in the scene) at its current XY coordinate BEFORE moving horizontally. Never move horizontally from low Z as it will collide with and knock over other cubes.
8. CRITICAL: When building towers, ALWAYS call home() right after release() and BEFORE querying the position of the placed cube (using get_object_position). This ensures the robot arm moves out of the camera's view, allowing 100% accurate top-view vision without occlusion.
9. LOGICAL STACKING ORDER: When stacking multiple objects into a tower (e.g., "A, B, C, D'yi üst üste diz" or "küpleri üst üste diz"), choose one object as the BASE (bottom-most object, e.g., A). LEAVE the base object A at its original position (do NOT grasp it). Then, grasp B and place it on top of A. Then grasp C and place it on B. Then grasp D and place it on C. CRITICAL: NEVER grasp the base object (A) after other objects have been placed on top of it. Grasping an object that is at the bottom of a stack is physically impossible, will collapse the tower, and is strictly prohibited.
10. "diagonal" / "çaprazına": Use an offset in BOTH X and Y directions relative to the target center. Since cubes are 0.05m wide, a diagonal placement should have +/- 0.06m offset in X AND +/- 0.06m offset in Y. For example, to place blue_cube diagonal to red_cube, use: [red_pos[0] + 0.06, red_pos[1] + 0.06, red_pos[2]]. NEVER place them with only 1 axis offset, as that would be side-by-side (beside), not diagonal.
11. OBSTACLE AVOIDANCE: When there are existing stacks or kuleler in the scene, make sure to plan horizontal movements (`move_to`) with a safe high Z (z = 0.65 or 0.70) to avoid any collision with existing towers on the path.
12. "beside" / "yanına" / "yan yana": Place objects side-by-side by applying a 0.06m offset to ONLY ONE axis (either X or Y, but not both). For example, to place blue_cube beside red_cube along the Y axis, use: [red_pos[0], red_pos[1] + 0.06, red_pos[2]]. Never apply offsets to both X and Y, as that is diagonal placement. Never use 0 offset as they will collide.
13. MULTIPLE BESIDE / PLURAL LINE-UP: When asked to place all cubes side-by-side or in a row (e.g., "küpleri yan yana koy" or "küpleri yan yana diz"), you MUST arrange ALL available cubes in a single row (line) using one cube as the base (e.g. red_cube) and placing the others at progressive offsets on the Y-axis: B at base_y + 0.06, C at base_y + 0.12, D at base_y + 0.18. Do NOT leave any cubes out. Make sure all cubes are moved into the row.





BESIDE EXAMPLE (Plural/Multiple cubes beside each other) — "küpleri yan yana koy":
# Choose red_cube as the base of the row, and place blue, green, and yellow cubes sequentially beside it on the Y-axis.
red_pos = get_object_position("red_cube")
blue_pos = get_object_position("blue_cube")
grasp("blue_cube")
move_to([blue_pos[0], blue_pos[1], 0.65])
move_to([red_pos[0], red_pos[1] + 0.06, 0.65])
move_to([red_pos[0], red_pos[1] + 0.06, red_pos[2]])
release()
home()

# Place green_cube next to the placed blue_cube
green_pos = get_object_position("green_cube")
grasp("green_cube")
move_to([green_pos[0], green_pos[1], 0.65])
move_to([red_pos[0], red_pos[1] + 0.12, 0.65])
move_to([red_pos[0], red_pos[1] + 0.12, red_pos[2]])
release()
home()

# Place yellow_cube next to the placed green_cube
yellow_pos = get_object_position("yellow_cube")
grasp("yellow_cube")
move_to([yellow_pos[0], yellow_pos[1], 0.65])
move_to([red_pos[0], red_pos[1] + 0.18, 0.65])
move_to([red_pos[0], red_pos[1] + 0.18, red_pos[2]])
release()
home()

TOWER STACKING EXAMPLE (4 levels) — "red_cube, blue_cube, green_cube ve yellow_cube'u kule gibi üst üste diz":
# Step 1: red_cube is the base. Leave it at its position! Grasp blue_cube and stack it on red_cube.
red_pos = get_object_position("red_cube")
blue_pos = get_object_position("blue_cube")
grasp("blue_cube")
move_to([blue_pos[0], blue_pos[1], 0.65])
move_to([red_pos[0], red_pos[1], 0.65])
move_to([red_pos[0], red_pos[1], red_pos[2] + 0.052])
release()
home()

# Step 2: Grasp green_cube and stack it on the placed blue_cube.
blue_placed_pos = get_object_position("blue_cube")
green_pos = get_object_position("green_cube")
grasp("green_cube")
move_to([green_pos[0], green_pos[1], 0.65])
move_to([blue_placed_pos[0], blue_placed_pos[1], 0.65])
move_to([blue_placed_pos[0], blue_placed_pos[1], blue_placed_pos[2] + 0.052])
release()
home()

# Step 3: Grasp yellow_cube and stack it on the placed green_cube.
green_placed_pos = get_object_position("green_cube")
yellow_pos = get_object_position("yellow_cube")
grasp("yellow_cube")
move_to([yellow_pos[0], yellow_pos[1], 0.65])
move_to([green_placed_pos[0], green_placed_pos[1], 0.65])
move_to([green_placed_pos[0], green_placed_pos[1], green_placed_pos[2] + 0.052])
release()
home()
# TASK COMPLETE. Red_cube was NEVER grasped because it was the base! Only blue, green, and yellow were moved.


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
        if "</think>" in plan:
            plan = plan.split("</think>")[-1].strip()
        elif "<think>" in plan:
            # If thinking got cut off, try to remove everything up to the last character or handle gracefully
            parts = plan.split("<think>")
            if len(parts) > 1:
                # If there's content after <think> but it didn't finish, it's probably all thoughts
                plan = parts[-1].strip()
        plan = plan.replace("```python", "").replace("```", "").strip()

        # İmkansız görev kontrolü
        if plan.upper().startswith("IMPOSSIBLE"):
            reason = plan.split(":", 1)[1].strip() if ":" in plan else "Bilinmeyen sebep"
            print(f"\n🚫 İMKANSIZ GÖREV: {reason}")
            return None

        print(f"\n📋 Üretilen Plan:\n{plan}\n")
        return plan

