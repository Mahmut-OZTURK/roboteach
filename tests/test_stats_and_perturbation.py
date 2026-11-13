import numpy as np
import pybullet as p

from evaluation.stats import mcnemar_exact, percentile_summary, wilson_ci
from planner.openai_client import OpenAIChatClient
from simulation.camera import SimCamera
from simulation.env import RoboTeachEnv


def test_wilson_ci_and_mcnemar_and_percentiles():
    lo, hi = wilson_ci(24, 30)
    assert 60.0 < lo < 80.0 < hi < 95.0
    assert wilson_ci(0, 0) == (0.0, 0.0)

    mc = mcnemar_exact([True] * 12 + [False] * 4, [False] * 12 + [False] * 4)
    assert mc["b_only_a"] == 12
    assert mc["c_only_b"] == 0
    assert mc["p_value"] < 0.001

    mc_tie = mcnemar_exact([True, False, True], [True, False, True])
    assert mc_tie["p_value"] == 1.0

    pct = percentile_summary([100.0, 200.0, 300.0, 400.0, 1000.0], n_boot=200, seed=7)
    assert pct["p50"] == 300.0
    assert pct["p99"] >= pct["p95"] >= pct["p50"]
    assert pct["p50_ci95"][0] <= pct["p50"] <= pct["p50_ci95"][1]


def test_openai_client_payload_for_gpt6_luna():
    client = OpenAIChatClient(api_key="sk-test-dummy")
    p_low = client._build_payload(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=900,
    )
    assert p_low["max_completion_tokens"] == 900
    assert "max_tokens" not in p_low
    assert p_low["reasoning_effort"] == "low"
    assert "temperature" not in p_low

    p_none = client._build_payload(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}]}],
        temperature=0.0,
        max_tokens=200,
        reasoning_effort="none",
    )
    assert p_none["reasoning_effort"] == "none"
    assert p_none["temperature"] == 0.0
    assert p_none["max_completion_tokens"] == 200


def test_domain_randomization_and_camera_perturbation():
    env = RoboTeachEnv(headless=True, seed=11, include_container=True, include_obstacle=True)
    cam = SimCamera(headless=True, physics_client_id=env.client)
    try:
        env.randomize({"stacks": [], "has_obstacle": True}, seed=42, domain_randomization=True)
        dp = env.last_domain_params
        assert dp["domain_randomization"] is True
        for m in dp["cube_masses_kg"].values():
            assert 0.084 <= m <= 0.156
        for f in dp["cube_frictions"].values():
            assert 1.2 <= f <= 2.8

        dyn = p.getDynamicsInfo(env.objects["red_cube"], -1, physicsClientId=env.client)
        assert abs(dyn[0] - dp["cube_masses_kg"]["red_cube"]) < 1e-4

        cam.set_perturbation(enabled=False)
        _, depth_nom, vm_nom, _ = cam.capture_rgbd("top")
        pt_nom = cam.pixel_to_world(320, 240, float(depth_nom[240, 320]), vm_nom)

        cam.set_perturbation(enabled=True, seed=7, depth_noise_mm=4.0, extrinsics_mm=3.0, extrinsics_deg=1.5)
        _, depth_p, vm_p, _ = cam.capture_rgbd("top")
        pt_p = cam.pixel_to_world(320, 240, float(depth_p[240, 320]), vm_p)
        assert np.linalg.norm(np.array(pt_nom) - np.array(pt_p)) > 1e-4

        env.randomize({"stacks": [], "has_obstacle": True}, seed=42, domain_randomization=False)
        assert env.last_domain_params["cube_masses_kg"]["red_cube"] == 0.12
        assert env.last_domain_params["cube_frictions"]["red_cube"] == 1.6
    finally:
        env.close()

