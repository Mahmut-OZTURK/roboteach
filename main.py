# main.py — RoboTeach: İki Modlu Robot Kontrol Sistemi
# calibrate: Görev öğren + kalibre et
# execute: Kalibre edilmiş görevi çalıştır (zero-shot da desteklenir)
# Görev tipleri sabit DEĞİL — LLM dinamik üretir
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from calibration.manager import CalibrationManager
from skills.primitives import RobotSkills
from planner.progprompt import ProgPrompt
from planner.cap import CaP
from evaluation.vlm_scorer import VLMScorer


MAX_CALIBRATION_RETRIES = 999


def calibrate(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator):
    """
    Kalibrasyon modu:
    1. LLM görevi dinamik sınıflandırır
    2. Plan üret → çalıştır → VLM değerlendir
    3. Başarısızsa öğren, tekrar dene
    4. Başarılıysa kalibrasyon kaydet
    """
    print(f"\n{'='*60}")
    print(f"  🔧 KALİBRASYON MODU: '{task}'")
    print(f"{'='*60}")

    task_type = calib_mgr.classify_task(task)
    knowledge = calib_mgr.load_knowledge(task_type)

    for attempt in range(1, MAX_CALIBRATION_RETRIES + 1):
        print(f"\n{'─'*30} Deneme {attempt} {'─'*30}")

        env.reset()
        env.randomize()
        vision_locator.clear_cache()

        # Sahne analizi
        print("\n🔍 Sahne Analizi:")
        scene_desc = scorer.analyze_scene()

        # Nesne konumları (kameradan)
        positions = vision_locator.locate_all()
        scene_objects = [{"name": n, "position": p} for n, p in positions.items()]

        if not scene_objects:
            print("  ❌ Hiçbir nesne tespit edilemedi!")
            skills.home()
            continue

        # Plan üret
        plan = planner.plan(
            task=task,
            scene_description=scene_desc,
            objects=scene_objects,
            robot_position=[0, -0.5, 0.42],
            extra_instructions=knowledge if knowledge else None
        )

        # Çalıştır
        success_exec = cap.execute(plan, scorer=scorer)

        if not success_exec:
            print("\n⚠️  Çalıştırma hatası!")
            calib_mgr.save_knowledge(task_type, "Execution failed - simplify the plan")
            knowledge.append("Simplify the plan")
            skills.home()
            continue

        # VLM değerlendirme
        print("\n📊 VLM Değerlendirme başlıyor...")
        result = scorer.evaluate(task)

        if result["success"] and result["score"] >= 0.8:
            print(f"\n✅ KALİBRASYON BAŞARILI! (Skor: {result['score']:.2f})")

            if attempt == 1 and not knowledge:
                print("  ✨ İlk denemede kusursuz!")
            
            calib_mgr.save_calibration(task_type, task, knowledge)
            skills.home()
            return True

        # Başarısız — öğren
        if result.get("instruction") and result["instruction"] != "None":
            calib_mgr.save_knowledge(task_type, result["instruction"])
            knowledge.append(result["instruction"])
            print(f"  📝 Öğrenilen: {result['instruction']}")

        print(f"\n⚠️  Deneme {attempt} başarısız: {result.get('issue', 'düşük skor')}")
        skills.home()

    return False


def execute(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator):
    """
    Execute modu:
    1. Kalibrasyon kontrolü (yoksa zero-shot dener)
    2. Kalibre bilgiyle çalıştır
    3. 3 kere dene
    """
    print(f"\n{'='*60}")
    print(f"  ▶️  EXECUTE MODU: '{task}'")
    print(f"{'='*60}")

    task_type = calib_mgr.classify_task(task)
    is_calib = calib_mgr.is_calibrated(task_type)

    if is_calib:
        print(f"  ✅ Kalibrasyon mevcut: {task_type}")
    else:
        print(f"  ⚠️  '{task_type}' kalibre edilmemiş — zero-shot denenecek")

    for attempt in range(1, 4):
        print(f"\n{'─'*30} Execute Deneme {attempt}/3 {'─'*30}")

        env.reset()
        env.randomize()
        vision_locator.clear_cache()

        # Bilgi birikimini yükle
        calib = calib_mgr.load_calibration(task_type)
        instructions = calib.get("instructions", [])
        task_knowledge = calib_mgr.load_knowledge(task_type)
        combined = list(set(instructions + task_knowledge))

        if combined:
            print(f"  📚 {len(combined)} deneyim yüklendi")

        # Sahne analizi
        print("\n🔍 Sahne Analizi:")
        scene_desc = scorer.analyze_scene()

        positions = vision_locator.locate_all()
        scene_objects = [{"name": n, "position": p} for n, p in positions.items()]

        if not scene_objects:
            print("  ❌ Hiçbir nesne tespit edilemedi!")
            continue

        # Plan üret + çalıştır
        plan = planner.plan(
            task=task,
            scene_description=scene_desc,
            objects=scene_objects,
            robot_position=[0, -0.4, 0.42],
            extra_instructions=combined if combined else None
        )

        success_exec = cap.execute(plan, scorer=scorer)
        if not success_exec:
            print("\n⚠️  Çalıştırma hatası!")
            skills.home()
            continue

        # VLM değerlendir
        print("\n📊 VLM Değerlendirme başlıyor...")
        result = scorer.evaluate(task)
        skills.home()

        if result["success"] and result["score"] >= 0.7:
            print(f"\n✅ Görev başarıyla tamamlandı! (Skor: {result['score']:.2f})")
            return True
        else:
            print(f"\n⚠️  Deneme {attempt} başarısız: {result.get('issue', 'düşük skor')}")

    print(f"\n❌ 3 deneme başarısız. 'calibrate' modunu deneyin.")
    return False


def main():
    print("\n" + "=" * 60)
    print("  🤖 RoboTeach — İki Modlu Robot Kontrol Sistemi")
    print("  Planner: llama-3.3-70b-versatile")
    print("  Vision : llama-4-scout-17b-16e-instruct")
    print("  Görev Tipleri: LLM-native (sınırsız)")
    print("=" * 60)

    print("\n🔧 Sistem başlatılıyor...\n")

    env = RoboTeachEnv()
    camera = SimCamera()
    vision_locator = VisionLocator(camera, env.object_dimensions)
    calib_mgr = CalibrationManager()
    skills = RobotSkills(env.robot_id, env.objects, vision_locator)
    planner = ProgPrompt()
    cap = CaP(skills)
    scorer = VLMScorer(camera)

    skills.home()

    print("\n" + "=" * 60)
    print("  Komutlar:")
    print("    calibrate / c → Kalibrasyon modu (öğren + kalibre et)")
    print("    execute   / e → Çalıştırma modu")
    print("    status    / s → Kalibre edilmiş görevleri göster")
    print("    quit      / q → Çıkış")
    print("  💡 Görev tipleri sınırsız — LLM dinamik üretir!")
    print("=" * 60)

    while True:
        try:
            mode = input("\n🎮 Mod seç (c/e/s/q): ").strip().lower()

            if mode in ("quit", "q"):
                break

            elif mode in ("status", "s"):
                calibrated = calib_mgr.list_calibrated()
                print("\n📊 Kalibre Edilmiş Görev Tipleri:")
                if calibrated:
                    for t in calibrated:
                        print(f"  ✅ {t}")
                else:
                    print("  (henüz kalibrasyon yok)")
                continue

            elif mode in ("calibrate", "c"):
                task = input("🎯 Görev: ").strip()
                if not task:
                    continue
                calibrate(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator)

            elif mode in ("execute", "e"):
                task = input("🎯 Görev: ").strip()
                if not task:
                    continue
                execute(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator)

            else:
                print("  ⚠️  Geçersiz! c/e/s/q")

        except KeyboardInterrupt:
            print("\n\n⚡ Ctrl+C")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()

    skills.home()
    env.close()
    print("\n👋 RoboTeach kapatıldı.")


if __name__ == "__main__":
    main()
