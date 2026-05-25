import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from calibration.manager import CalibrationManager
from simulation.env import RoboTeachEnv

def test_initial_state_parsing():
    print("🧪 1. LLM ile Başlangıç Sahne Kurulumu Analiz Testi:")
    mgr = CalibrationManager()
    
    tasks = [
        "put the blue cube on the red cube, but red is already on blue",
        "take the red cube out of the box and put it next to green",
        "put green cube on top of yellow cube"
    ]
    
    for t in tasks:
        print(f"\nTask: '{t}'")
        config = mgr.get_initial_scene_config(t)
        print(f"Result: {config}")

def test_checkpoint_breakdown():
    print("\n🧪 2. LLM ile Görev Checkpoint Bölme Testi:")
    mgr = CalibrationManager()
    
    tasks = [
        "put the blue cube on red, but red is already on blue",
        "kutuları üst üste diz 4 tanesini kule gibi",
        "put all cubes inside the box"
    ]
    
    for t in tasks:
        print(f"\nTask: '{t}'")
        checkpoints = mgr.break_down_task(t)
        print(f"Checkpoints: {checkpoints}")

def test_sim_setup():
    print("\n🧪 3. Simülasyon Dinamik Kurulum Testi (GUI Olmadan):")
    import pybullet as p
    # Headless connect to verify no display server issues on test runner
    client = p.connect(p.DIRECT)
    
    # Custom load minimal env setup or check imports
    try:
        from simulation.env import RoboTeachEnv
        print("RoboTeachEnv import ve initialize kontrolü...")
    except Exception as e:
        print(f"Error importing Env: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    test_initial_state_parsing()
    test_checkpoint_breakdown()
    test_sim_setup()
    print("\n✅ Dryrun başarıyla tamamlandı!")

