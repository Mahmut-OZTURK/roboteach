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
    2. Görevi ardışık kontrol noktalarına (checkpoints) böler
    3. Her adımda Plan üret → çalıştır → VLM değerlendir (kapalı döngü)
    4. Başarısız olan adımda öğren, tekrar dene
    5. Başarılıysa kalibrasyonu kaydet
    """
    print(f"\n{'='*60}")
    print(f"  🔧 KALİBRASYON MODU: '{task}'")
    print(f"{'='*60}")

    task_type = calib_mgr.classify_task(task)
    knowledge = calib_mgr.load_knowledge(task_type)
    initial_state = calib_mgr.get_initial_scene_config(task)
    # Kullanıcı talebi üzerine aşamalı planlama (break_down_task) kaldırıldı, tek adımda planlama yapılacak
    checkpoints = [task]

    for attempt in range(1, MAX_CALIBRATION_RETRIES + 1):
        print(f"\n{'─'*30} Deneme {attempt} {'─'*30}")

        env.reset()
        env.randomize(initial_state)
        
        success_all_checkpoints = True
        for cp_idx, checkpoint in enumerate(checkpoints):
            print(f"\n🚩 [Kontrol Noktası {cp_idx+1}/{len(checkpoints)}]: '{checkpoint}'")
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
                success_all_checkpoints = False
                break

            # Plan üret
            plan = planner.plan(
                task=checkpoint,
                scene_description=scene_desc,
                objects=scene_objects,
                robot_position=[0, -0.5, 0.42],
                extra_instructions=knowledge if knowledge else None
            )

            if not plan:
                print("\n⚠️  Plan üretilemedi veya görev imkansız!")
                success_all_checkpoints = False
                break

            # Çalıştır
            success_exec = cap.execute(plan, scorer=scorer)

            if not success_exec:
                print("\n⚠️  Çalıştırma hatası!")
                calib_mgr.save_knowledge(task_type, f"Execution failed for '{checkpoint}' - simplify the plan")
                knowledge.append(f"Simplify plan for '{checkpoint}'")
                skills.home()
                success_all_checkpoints = False
                break

            # VLM değerlendirme (Token tasarrufu için sadece son adımda kontrol et)
            is_last_checkpoint = (cp_idx == len(checkpoints) - 1)
            if is_last_checkpoint:
                print(f"\n📊 VLM Değerlendirme başlıyor (Final Adım: '{checkpoint}')...")
                result = scorer.evaluate(checkpoint)
            else:
                print(f"\n📊 Ara adım '{checkpoint}' için VLM kontrolü atlanıyor (Token tasarrufu)...")
                result = {"success": True, "score": 1.0, "issue": None, "instruction": "None"}

            if not result["success"] or result["score"] < 0.8:
                print(f"\n⚠️  Kontrol noktası '{checkpoint}' başarısız: {result.get('issue', 'düşük skor')}")

                # Başarısız — öğren
                if result.get("instruction") and result["instruction"] != "None":
                    calib_mgr.save_knowledge(task_type, result["instruction"])
                    knowledge.append(result["instruction"])
                    print(f"  📝 Öğrenilen: {result['instruction']}")

                skills.home()
                success_all_checkpoints = False
                break

            if is_last_checkpoint:
                print(f"  ✅ Görev başarıyla geçildi! (Final Skor: {result['score']:.2f})")
            else:
                print(f"  ✅ Ara adım '{checkpoint}' başarıyla geçildi! (Atlandı)")
            skills.home()

        if success_all_checkpoints:
            print(f"\n✅ KALİBRASYON BAŞARILI! Tüm kontrol noktaları geçildi.")
            calib_mgr.save_calibration(task_type, task, knowledge)
            return True

    return False


def execute(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator):
    """
    Execute modu:
    1. Kalibrasyon kontrolü (yoksa zero-shot dener)
    2. Görevi ardışık adımlara böler
    3. Her adımda plan üretip kapalı döngüde VLM ile doğrulayarak çalıştırır
    4. 3 kere dene
    """
    print(f"\n{'='*60}")
    print(f"  ▶️  EXECUTE MODU: '{task}'")
    print(f"{'='*60}")

    task_type = calib_mgr.classify_task(task)
    is_calib = calib_mgr.is_calibrated(task_type)
    initial_state = calib_mgr.get_initial_scene_config(task)
    # Kullanıcı talebi üzerine aşamalı planlama (break_down_task) kaldırıldı, tek adımda planlama yapılacak
    checkpoints = [task]

    if is_calib:
        print(f"  ✅ Kalibrasyon mevcut: {task_type}")
    else:
        print(f"  ⚠️  '{task_type}' kalibre edilmemiş — zero-shot denenecek")

    for attempt in range(1, 4):
        print(f"\n{'─'*30} Execute Deneme {attempt}/3 {'─'*30}")

        env.reset()
        env.randomize(initial_state)

        # Bilgi birikimini yükle
        calib = calib_mgr.load_calibration(task_type)
        instructions = calib.get("instructions", [])
        task_knowledge = calib_mgr.load_knowledge(task_type)
        combined = list(set(instructions + task_knowledge))

        if combined:
            print(f"  📚 {len(combined)} deneyim yüklendi")

        success_all_checkpoints = True
        for cp_idx, checkpoint in enumerate(checkpoints):
            print(f"\n🚩 [Kontrol Noktası {cp_idx+1}/{len(checkpoints)}]: '{checkpoint}'")
            vision_locator.clear_cache()

            # Sahne analizi
            print("\n🔍 Sahne Analizi:")
            scene_desc = scorer.analyze_scene()

            positions = vision_locator.locate_all()
            scene_objects = [{"name": n, "position": p} for n, p in positions.items()]

            if not scene_objects:
                print("  ❌ Hiçbir nesne tespit edilemedi!")
                success_all_checkpoints = False
                break

            # Plan üret + çalıştır
            plan = planner.plan(
                task=checkpoint,
                scene_description=scene_desc,
                objects=scene_objects,
                robot_position=[0, -0.4, 0.42],
                extra_instructions=combined if combined else None
            )

            if not plan:
                print("\n⚠️  Plan üretilemedi!")
                success_all_checkpoints = False
                break

            success_exec = cap.execute(plan, scorer=scorer)
            if not success_exec:
                print("\n⚠️  Çalıştırma hatası!")
                skills.home()
                success_all_checkpoints = False
                break

            # VLM değerlendir (Token tasarrufu için sadece son adımda kontrol et)
            is_last_checkpoint = (cp_idx == len(checkpoints) - 1)
            if is_last_checkpoint:
                print(f"\n📊 VLM Değerlendirme başlıyor (Final Adım: '{checkpoint}')...")
                result = scorer.evaluate(checkpoint)
            else:
                print(f"\n📊 Ara adım '{checkpoint}' için VLM kontrolü atlanıyor (Token tasarrufu)...")
                result = {"success": True, "score": 1.0, "issue": None}
            skills.home()

            if not result["success"] or result["score"] < 0.7:
                print(f"\n⚠️  Kontrol noktası '{checkpoint}' başarısız: {result.get('issue', 'düşük skor')}")
                success_all_checkpoints = False
                break

            if is_last_checkpoint:
                print(f"  ✅ Görev başarıyla geçildi! (Final Skor: {result['score']:.2f})")
            else:
                print(f"  ✅ Ara adım '{checkpoint}' başarıyla geçildi! (Atlandı)")

        if success_all_checkpoints:
            print(f"\n✅ Görev başarıyla tamamlandı!")
            return True

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
