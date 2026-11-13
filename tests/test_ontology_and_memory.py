from calibration.ontology import CanonicalTaskOntology
from calibration.memory import CalibrationMemory, FailureDiagnostic


def test_canonical_ontology_eliminates_synonym_split_brain():
    ont = CanonicalTaskOntology(llm_client=None)

    s1 = ont.parse("kırmızının çaprazına mavi kutuyu sarı kutunun çaprazına yeşil kutuyu koy")
    s2 = ont.parse("place blue cube diagonally to red and green diagonally to yellow")
    s3 = ont.parse("put blue_cube diagonal relative to red_cube, yellow_cube with green_cube")
    assert s1.canonical_type == "DIAGONAL_PAIR"
    assert s2.canonical_type == "DIAGONAL_PAIR"
    assert s3.canonical_type == "DIAGONAL_PAIR"

    t1 = ont.parse("kutuları kule gibi üst üste diz: en altta mavi üstünde kırmızı üstünde yeşil en üstte sarı")
    t2 = ont.parse("stack in order from bottom to top: blue, red, green, yellow")
    assert t1.canonical_type == "STACK_ORDERED"
    assert t2.canonical_type == "STACK_ORDERED"
    assert t1.ordered_objects == ["blue_cube", "red_cube", "green_cube", "yellow_cube"]
    assert t2.ordered_objects == ["blue_cube", "red_cube", "green_cube", "yellow_cube"]


def test_unstack_precondition_and_container_and_obstacle_parsing():
    ont = CanonicalTaskOntology(llm_client=None)

    swap = ont.parse("mavi küpü kırmızının üstüne koy ama kırmızı zaten mavinin üstünde")
    assert swap.canonical_type == "UNSTACK_SWAP"
    assert ["red_cube", "blue_cube"] in swap.initial_stacks

    inside = ont.parse("kırmızı ve mavi küpleri kahverengi kutunun içine koy")
    assert inside.canonical_type == "PLACE_INSIDE"
    assert inside.target_container == "brown_box"

    obs = ont.parse("engelin üzerinden aşırarak mavi küpü kırmızı küpün üstüne koy")
    assert obs.canonical_type == "OBSTACLE_CLEARANCE"
    assert obs.has_obstacle is True


def test_structured_calibration_memory_deduplication_and_priors(tmp_path):
    mem = CalibrationMemory(storage_dir=str(tmp_path))
    d1 = FailureDiagnostic(
        failure_code="F_COLLISION",
        summary="Arm collided with obstacle wall at z=0.50m during transit",
        param_updates={"transit_z": 0.66},
        symbolic_rule="Lift end-effector to z >= 0.66m before crossing obstacle_wall.",
    )
    d2 = FailureDiagnostic(
        failure_code="F_COLLISION",
        summary="Higher obstacle clearance needed",
        param_updates={"transit_z": 0.68},
        symbolic_rule="Lift end-effector to z >= 0.68m before crossing obstacle_wall.",
    )
    mem.record_failure("OBSTACLE_CLEARANCE", d1)
    mem.record_failure("OBSTACLE_CLEARANCE", d2)

    priors = mem.get_kinematic_priors("OBSTACLE_CLEARANCE")
    assert priors["transit_z"] == 0.68
    rules = mem.get_symbolic_rules("OBSTACLE_CLEARANCE")
    # Deduplicated by failure_code so prompt context never bloats
    assert len(rules) == 1
    assert "0.68" in rules[0]

