"""180-Paraphrase Bilingual Ontology Benchmark & 60-Pair Freeform Key Fragmentation Test."""
import argparse
import json
import os
import re
import sys
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"), override=True)

from benchmark.ablation_suite import freeform_llm_task_key
from calibration.ontology import ALL_CUBES, COLOR_ALIASES, CanonicalTaskOntology, TaskSpec
from evaluation.stats import mcnemar_exact, wilson_ci
from planner.openai_client import OpenAIChatClient

# 15 TR + 15 EN diverse paraphrases per task family T1-T6 (180 total)
PARAPHRASES = {
    "T1_STACK_ORDERED": {
        "tr": [
            "kutuları kule gibi üst üste diz: en altta mavi, üstünde kırmızı, üstünde yeşil, en üstte sarı",
            "mavi, kırmızı, yeşil ve sarı küpleri sırasıyla üst üste istifle",
            "tabanda mavi, üzerinde kırmızı, onun üstünde yeşil ve en tepede sarı olacak şekilde blokları yığ",
            "alttan üste doğru mavi, kırmızı, yeşil, sarı küplerle dört katlı bir kule kur",
            "sırasıyla mavi, kırmızı, yeşil ve sarı kutuları birbiri üstüne yerleştir",
            "en altta mavi küp dursun; üstüne sırasıyla kırmızı, yeşil ve sarı küpleri kule yap",
            "dört küpü tabandan tepeye mavi, kırmızı, yeşil, sarı düzeninde üst üste diz",
            "mavi taban olmak üzere kırmızı, yeşil ve sarı blokları üst üste istifle",
            "tabana maviyi alıp sırasıyla mavi, kırmızı, yeşil, sarı kulesini oluştur",
            "blokları dikey olarak en altta mavi, sonra kırmızı, yeşil ve sarı şeklinde üst üste yığ",
            "mavi, kırmızı, yeşil, sarı sırasını koruyarak dört küpten tek bir kule inşa et",
            "tabanda mavi küp kalacak biçimde kırmızı, yeşil ve sarı küpleri tepeye doğru sırala",
            "alttan başlayarak mavi, kırmızı, yeşil ve sarı blokları kule halinde üst üste koy",
            "dört renkli küpü sırasıyla mavi, kırmızı, yeşil ve sarı düzeninde üst üste topla",
            "en altta mavi blok, üstünde kırmızı, onun tepesinde yeşil ve sarı ile kule diz",
        ],
        "en": [
            "stack in order from bottom to top: blue, red, green, yellow",
            "build a four-cube tower with blue at the bottom, then red, green, and yellow on top",
            "pile up the blue, red, green, and yellow blocks vertically in that exact bottom-to-top order",
            "assemble a single tower starting from blue at the base followed by red, green, and yellow",
            "stack the cubes in order: blue base, red second, green third, yellow at the top",
            "make a vertical tower using blue, red, green, and yellow from lowest to highest",
            "place blue at the bottom and stack red, green, and yellow in order above it",
            "construct a 4-layer stack ordered bottom-to-top as blue, red, green, yellow",
            "stack blue, red, green, and yellow cubes one on top of another in sequence",
            "form a tower in order from bottom to top using blue, red, green, and yellow blocks",
            "pile blue first at the base, then red, green, and finally yellow on top",
            "build a column of cubes ordered from base to peak: blue, red, green, yellow",
            "arrange blue, red, green, and yellow into a vertical stack in bottom-up sequence",
            "tier the four blocks in order: blue at bottom, red, green, and yellow uppermost",
            "create an ordered vertical tower of blue, red, green, and yellow cubes",
        ],
    },
    "T2_ALIGN_ROW": {
        "tr": [
            "kırmızı, mavi, yeşil ve sarı küpleri yan yana tek sıra halinde diz", "tüm küpleri yan yana hizala: kırmızı, mavi, yeşil, sarı",
            "kırmızı, mavi, yeşil ve sarı blokları doğrusal bir sıraya diz", "dört küpü kırmızı, mavi, yeşil, sarı düzeninde tek sıra yap",
            "kırmızı, mavi, yeşil ve sarı kutuları masa üstünde yan yana yerleştir", "kırmızı, mavi, yeşil, sarı küpleri aynı hizada sırala",
            "blokları kırmızı, mavi, yeşil ve sarı düzeninde tek sıra hizala", "kırmızı, mavi, yeşil ve sarı renkli küpleri yan yana koy",
            "dört bloğu kırmızı, mavi, yeşil ve sarı sırasıyla yan yana getir", "kırmızı, mavi, yeşil, sarı küpleri tek bir doğru boyunca hizala",
            "kırmızı, mavi, yeşil ve sarı küpleri aralıksız yan yana diz", "masada kırmızı, mavi, yeşil ve sarı bloklardan tek sıra oluştur",
            "kırmızı, mavi, yeşil ve sarı kutuları yan yana aynı çizgiye hizala", "dört küpü kırmızı, mavi, yeşil ve sarı sırasında tek sıra diz",
            "kırmızı, mavi, yeşil ve sarı blokları düzgün bir sıraya yerleştir",
        ],
        "en": [
            "line up the red, blue, green, and yellow cubes in a row", "arrange the red, blue, green, and yellow blocks side by side",
            "place red, blue, green, and yellow cubes next to each other in a single line", "align red, blue, green, and yellow in a collinear row on the table",
            "put the red, blue, green, and yellow blocks in a straight row", "form a single row of red, blue, green, and yellow cubes",
            "place red, blue, green, and yellow side-by-side along one line", "line up red, blue, green, and yellow beside one another",
            "set the red, blue, green, and yellow cubes into a neat row", "arrange red, blue, green, and yellow cubes in a horizontal line",
            "organize the red, blue, green, and yellow blocks side by side in a row", "position red, blue, green, and yellow next to each other in a line",
            "create a collinear line of red, blue, green, and yellow cubes", "place all four cubes beside each other in order: red, blue, green, yellow",
            "lay out red, blue, green, and yellow blocks in a single row",
        ],
    },
    "T3_DIAGONAL_PAIR": {
        "tr": [
            "kırmızının çaprazına mavi kutuyu, sarının çaprazına yeşil kutuyu koy", "kırmızı küpün çaprazına mavi küpü ve sarı küpün çaprazına yeşil küpü yerleştir",
            "mavi küpü kırmızının çaprazına, yeşil küpü sarının çaprazına koy", "kırmızının köşegenine maviyi, sarının köşegenine yeşili yerleştir",
            "maviyi kırmızının çapraz konumuna, yeşili sarının çapraz konumuna taşı", "kırmızı bloğun çaprazına mavi bloğu ve sarı bloğun çaprazına yeşil bloğu bırak",
            "mavi kutuyu kırmızı kutunun köşegenine, yeşil kutuyu sarı kutunun köşegenine koy", "kırmızının çapraz hizasına maviyi, sarının çapraz hizasına yeşili getir",
            "maviyi kırmızının çapraz tarafına ve yeşili sarının çapraz tarafına yerleştir", "kırmızı küpün köşegen yönüne mavi küpü, sarı küpün köşegen yönüne yeşil küpü koy",
            "kırmızının çapraz köşesine mavi bloğu, sarının çapraz köşesine yeşil bloğu yerleştir", "mavi bloğu kırmızı bloğun çaprazına ve yeşil bloğu sarı bloğun çaprazına bırak",
            "kırmızının çaprazına mavi küpü; sarının çaprazına yeşil küpü koy", "maviyi kırmızıya göre çapraz konuma, yeşili sarıya göre çapraz konuma al",
            "kırmızı kutunun çaprazına mavi kutuyu, sarı kutunun çaprazına yeşil kutuyu yerleştir",
        ],
        "en": [
            "place blue diagonally to red and green diagonally to yellow", "put the blue cube diagonal to the red cube, and the green cube diagonal to the yellow cube",
            "move blue diagonally relative to red and green diagonally relative to yellow", "position the blue block diagonally from red, and the green block diagonally from yellow",
            "set blue diagonal to red and set green diagonal to yellow", "place blue crosswise to red and place green crosswise to yellow",
            "shift the blue cube diagonally beside red, and shift green diagonally beside yellow", "put blue at a diagonal offset from red and green at a diagonal offset from yellow",
            "locate blue diagonally opposite red, and green diagonally opposite yellow", "arrange blue diagonally relative to red; arrange green diagonally relative to yellow",
            "transfer the blue block diagonal to red and the green block diagonal to yellow", "place blue cube diagonally to red cube and green cube diagonally to yellow cube",
            "offset blue diagonally from red and offset green diagonally from yellow", "put blue on a diagonal to red, and put green on a diagonal to yellow",
            "align blue diagonally with red and green diagonally with yellow",
        ],
    },
    "T4_PLACE_INSIDE": {
        "tr": [
            "kırmızı, mavi ve yeşil küpleri kahverengi kutunun içine koy", "kırmızı, mavi ve yeşil blokları açık kutunun içine yerleştir",
            "kırmızı, mavi ve yeşil küpleri kabın içerisine bırak", "kırmızı, mavi, yeşil kutuları kahverengi kaba taşı",
            "kırmızı, mavi ve yeşil blokları kutuya koy", "kırmızı, mavi ve yeşil küpleri kahverengi kutunun içerisine yerleştir",
            "kırmızı, mavi ve yeşil blokları açık kabın içine topla", "kırmızı, mavi, yeşil küpleri kahverengi kutu içine bırak",
            "kırmızı, mavi ve yeşil küpleri kabın içine aktar", "kırmızı, mavi ve yeşil blokları kahverengi hazneye yerleştir",
            "kırmızı, mavi, yeşil küpleri açık kutuya koy", "kırmızı, mavi ve yeşil küpleri kutunun içine doldur",
            "kırmızı, mavi ve yeşil blokları kabın içine bırak", "kırmızı, mavi, yeşil küpleri kahverengi kaba yerleştir",
            "kırmızı, mavi ve yeşil kutuları açık kabın içerisine koy",
        ],
        "en": [
            "put the red, blue, and green cubes inside the brown box", "place the red, blue, and green blocks into the container",
            "drop red, blue, and green cubes into the box", "move the red, blue, and green blocks inside the hollow box",
            "set red, blue, and green cubes in the box", "put red, blue, and green inside the brown container",
            "place the red, blue, and green cubes within the box", "pack the red, blue, and green blocks into the brown box",
            "transfer red, blue, and green cubes inside the open box", "store the red, blue, and green blocks in the container",
            "deposit red, blue, and green cubes into the brown box", "insert the red, blue, and green blocks inside the box",
            "collect red, blue, and green cubes into the open container", "place red, blue, and green inside the brown_box",
            "put the red, blue, and green blocks in the brown box",
        ],
    },
    "T5_UNSTACK_SWAP": {
        "tr": [
            "mavi küpü kırmızının üstüne koy ama kırmızı zaten mavinin üstünde", "maviyi kırmızı küpün üzerine istifle fakat kırmızı zaten mavinin üzerinde",
            "mavi bloğu kırmızının üstüne yerleştir ancak kırmızı başlangıçta mavinin üstünde", "maviyi kırmızının tepesine koy ama kırmızı şu anda mavinin üstünde duruyor",
            "mavi küpü kırmızının üstüne al fakat kırmızı halihazırda mavinin üzerinde", "maviyi kırmızının üzerine koy oysa kırmızı zaten mavinin üstünde",
            "başlangıçta kırmızı mavinin üstünde; maviyi kırmızının üstüne koy", "şu anda kırmızı mavinin üzerinde, mavi küpü kırmızının üstüne yerleştir",
            "mavi küpü kırmızının üstüne diz ama kırmızı zaten mavinin tepesinde", "maviyi kırmızının üstüne oturt ancak kırmızı başlangıçta mavinin üzerinde",
            "mavi kutuyu kırmızının üstüne taşı fakat kırmızı halihazırda mavinin üstünde", "maviyi kırmızı bloğun üstüne yerleştir ama kırmızı zaten mavinin üstünde",
            "mavi küpü kırmızının üstüne kule yap ancak kırmızı zaten mavinin üzerinde", "maviyi kırmızının üzerine al fakat kırmızı başlangıçta mavinin üstünde duruyor",
            "mavi küpü kırmızının üstüne koy oysa kırmızı halihazırda mavinin üstünde",
        ],
        "en": [
            "put the blue cube on the red cube, but red is already on blue", "stack blue on top of red, although red is already on blue",
            "place the blue block onto red while red starts on blue", "put blue on red, whereas red is currently on top of blue",
            "although red is already on blue, put the blue cube on red", "given that red starts on blue, stack blue on top of red",
            "currently red sits on blue; place blue on top of red", "move blue onto the red cube, but red is already atop blue",
            "stack the blue cube on red even though red is initially on blue", "place blue on top of red; however, red is already on blue",
            "initially red is on blue, put blue on top of red", "swap the stack so blue is on top of red, while red is already on blue",
            "put blue on red, though red starts atop blue", "since red is currently on blue, stack blue onto red",
            "stack blue on the red block, but red is initially on blue",
        ],
    },
    "T6_OBSTACLE_CLEARANCE": {
        "tr": [
            "engelin üzerinden aşırarak mavi küpü kırmızı küpün üstüne koy", "duvar engelini aşarak mavi bloğu kırmızının üzerine yerleştir",
            "bariyerin üzerinden geçirerek maviyi kırmızı küpün üstüne koy", "duvarı aşarak mavi küpü kırmızının üstüne istifle",
            "engeli atlatarak mavi kutuyu kırmızı kutunun üzerine yerleştir", "bariyeri aşırarak mavi küpü kırmızı bloğun tepesine koy",
            "duvarın üzerinden atlatarak maviyi kırmızının üstüne koy", "engelin ötesine geçip mavi küpü kırmızının üstüne yerleştir",
            "bariyer engelini aşarak mavi bloğu kırmızının üstüne diz", "duvar engelinin üstünden geçirerek maviyi kırmızının üzerine koy",
            "engeli aşarak mavi küpü kırmızı küpün üstüne çıkar", "mor duvarın üzerinden aşırarak mavi bloğu kırmızı bloğun üstüne koy",
            "bariyeri aşarak maviyi kırmızının üstüne yerleştir", "duvarı atlatarak mavi küpü kırmızının tepesine bırak",
            "engelin üzerinden geçirerek mavi küpü kırmızının üstüne taşı",
        ],
        "en": [
            "lift over the obstacle wall and place the blue cube on top of the red cube", "stack the blue cube onto red by clearing the barrier wall",
            "move blue over the wall and put it on top of red", "clear the obstacle and stack the blue block on the red block",
            "carry the blue cube over the divider to place it on red", "pass above the barrier and set blue on top of red",
            "reach over the purple wall and stack blue onto red", "bring the blue block across the hurdle onto the red cube",
            "avoid hitting the wall while stacking blue on top of red", "lift blue over the partition and put it onto red",
            "clear the wall barrier and place blue on the red cube", "hop over the obstacle to stack the blue cube on red",
            "transport blue above the obstacle wall and set it on red", "cross over the barrier to put blue on top of red",
            "go over the wall and stack the blue block onto red",
        ],
    },
}


def keyword_baseline_parse(task: str) -> tuple[TaskSpec, int]:
    """Naive keyword-only baseline without clause/precondition/role/checkpoint resolution."""
    low = task.lower()
    hits = []
    for canonical, aliases in COLOR_ALIASES.items():
        idxs = [m.start() for a in aliases for m in [re.search(rf"\b{re.escape(a)}", low)] if m]
        if idxs:
            hits.append((min(idxs), canonical))
    hits.sort(key=lambda x: x[0])
    mentions = [c for _, c in hits]
    if any(w in low for w in ("wall", "obstacle", "engel", "duvar")):
        fam = "OBSTACLE_CLEARANCE"
    elif any(w in low for w in ("already", "zaten")):
        fam = "UNSTACK_SWAP"
    elif any(w in low for w in ("inside", "into the box", "kutunun içine", "içine", "kutuya")):
        fam = "PLACE_INSIDE"
    elif any(w in low for w in ("diagonal", "diagonally", "çapraz")):
        fam = "DIAGONAL_PAIR"
    elif any(w in low for w in ("in a row", "yan yana", "line up", "beside")):
        fam = "ALIGN_ROW"
    else:
        fam = "STACK_ORDERED"
    pairs = [(mentions[i], mentions[i + 1]) for i in range(0, len(mentions) - 1, 2)] if fam == "DIAGONAL_PAIR" and len(mentions) >= 2 else []
    spec = TaskSpec(task, fam, mentions or list(ALL_CUBES), pairs, "brown_box" if fam == "PLACE_INSIDE" else None, [], fam == "OBSTACLE_CLEARANCE")
    return spec, 1


def _expected_spec(fam_id: str, text: str, ont: CanonicalTaskOntology) -> dict:
    if fam_id == "T1_STACK_ORDERED":
        return {"family": "STACK_ORDERED", "ordered": ["blue_cube", "red_cube", "green_cube", "yellow_cube"], "pairs": [], "container": None, "stacks": [], "checkpoints": 1}
    if fam_id == "T2_ALIGN_ROW":
        return {"family": "ALIGN_ROW", "ordered": ["red_cube", "blue_cube", "green_cube", "yellow_cube"], "pairs": [], "container": None, "stacks": [], "checkpoints": 1}
    if fam_id == "T3_DIAGONAL_PAIR":
        return {"family": "DIAGONAL_PAIR", "ordered": ont._extract_ordered_mentions(text), "pairs": [("blue_cube", "red_cube"), ("green_cube", "yellow_cube")], "container": None, "stacks": [], "checkpoints": 1}
    if fam_id == "T4_PLACE_INSIDE":
        return {"family": "PLACE_INSIDE", "ordered": ["red_cube", "blue_cube", "green_cube"], "pairs": [], "container": "brown_box", "stacks": [], "checkpoints": 1}
    if fam_id == "T5_UNSTACK_SWAP":
        return {"family": "UNSTACK_SWAP", "ordered": ["red_cube", "blue_cube"], "pairs": [], "container": None, "stacks": [["red_cube", "blue_cube"]], "checkpoints": 2}
    return {"family": "OBSTACLE_CLEARANCE", "ordered": ["red_cube", "blue_cube"], "pairs": [], "container": None, "stacks": [], "checkpoints": 1}


def _is_exact(sp: TaskSpec, cp_cnt: int, exp: dict) -> bool:
    return (
        sp.canonical_type == exp["family"] and sp.ordered_objects == exp["ordered"]
        and sp.relational_pairs == exp["pairs"] and sp.target_container == exp["container"]
        and sp.initial_stacks == exp["stacks"] and cp_cnt == exp["checkpoints"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    ont = CanonicalTaskOntology(llm_client=None)
    ont_hits, kw_hits, misses = [], [], []
    by_family, by_lang = {}, {"tr": {"n": 0, "ont": 0, "kw": 0}, "en": {"n": 0, "ont": 0, "kw": 0}}
    for fam_id, langs in PARAPHRASES.items():
        f_ont = f_kw = f_n = 0
        for lang in ("tr", "en"):
            for text in langs[lang]:
                exp = _expected_spec(fam_id, text, ont)
                sp = ont.parse(text)
                cp_cnt = len(ont.decompose_checkpoints(sp))
                ok_o = _is_exact(sp, cp_cnt, exp)
                kw_sp, kw_cp = keyword_baseline_parse(text)
                ok_k = _is_exact(kw_sp, kw_cp, exp)
                ont_hits.append(ok_o); kw_hits.append(ok_k)
                f_n += 1; f_ont += int(ok_o); f_kw += int(ok_k)
                by_lang[lang]["n"] += 1; by_lang[lang]["ont"] += int(ok_o); by_lang[lang]["kw"] += int(ok_k)
                if not ok_o:
                    misses.append({"family": fam_id, "lang": lang, "text": text, "got": sp.__dict__, "cp": cp_cnt, "exp": exp})
        by_family[fam_id] = {"n": f_n, "ontology_exact": f_ont, "ontology_pct": round(100.0 * f_ont / f_n, 2), "keyword_exact": f_kw, "keyword_pct": round(100.0 * f_kw / f_n, 2)}

    total_n, ont_s, kw_s = len(ont_hits), sum(ont_hits), sum(kw_hits)
    mc = mcnemar_exact(ont_hits, kw_hits)
    freeform_res = None
    if not args.offline_only:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
        client = OpenAIChatClient(api_key=api_key)
        pair_logs, ff_hits, onto_pair_hits = [], 0, 0
        for fam_id, langs in PARAPHRASES.items():
            for idx in range(10):
                a = langs["tr"][idx]
                b = langs["en"][idx] if idx < 5 else langs["tr"][idx + 5]
                ka, kb = freeform_llm_task_key(client, a, model=model), freeform_llm_task_key(client, b, model=model)
                ff_hit = (ka == kb)
                ont_hit = (ont.parse(a).canonical_type == ont.parse(b).canonical_type)
                ff_hits += int(ff_hit); onto_pair_hits += int(ont_hit)
                pair_logs.append({"family": fam_id, "pair_idx": idx, "key_a": ka, "key_b": kb, "freeform_hit": ff_hit, "ontology_hit": ont_hit})
        st = client.get_stats()
        cost_usd = st["prompt_tokens"] * 0.10e-6 + st["completion_tokens"] * 0.50e-6
        freeform_res = {
            "model": model, "n_pairs": len(pair_logs), "freeform_key_hits": ff_hits,
            "freeform_hit_pct": round(100.0 * ff_hits / len(pair_logs), 2), "freeform_ci95_pct": list(wilson_ci(ff_hits, len(pair_logs))),
            "ontology_pair_hits": onto_pair_hits, "ontology_pair_hit_pct": round(100.0 * onto_pair_hits / len(pair_logs), 2),
            "mcnemar_ontology_vs_freeform": mcnemar_exact([p["ontology_hit"] for p in pair_logs], [p["freeform_hit"] for p in pair_logs]),
            "api_usage": {**st, "cost_usd": round(cost_usd, 6)}, "pairs": pair_logs,
        }

    payload = {
        "n_paraphrases": total_n,
        "exact_match_criteria": ["canonical_type", "ordered_objects", "relational_pairs", "target_container", "initial_stacks", "checkpoint_count"],
        "overall": {"ontology_exact": ont_s, "ontology_pct": round(100.0 * ont_s / total_n, 2), "ontology_ci95_pct": list(wilson_ci(ont_s, total_n)),
                    "keyword_exact": kw_s, "keyword_pct": round(100.0 * kw_s / total_n, 2), "keyword_ci95_pct": list(wilson_ci(kw_s, total_n)), "mcnemar_ontology_vs_keyword": mc},
        "by_language": {l: {"n": d["n"], "ontology_exact": d["ont"], "ontology_pct": round(100.0 * d["ont"] / d["n"], 2), "keyword_exact": d["kw"], "keyword_pct": round(100.0 * d["kw"] / d["n"], 2)} for l, d in by_lang.items()},
        "by_family": by_family, "ontology_misses": misses, "freeform_key_pair_test_60": freeform_res,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "paraphrase_benchmark_180.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"PARAPHRASE_180: Ontology={ont_s}/{total_n} ({payload['overall']['ontology_pct']}%) vs Keyword={kw_s}/{total_n} ({payload['overall']['keyword_pct']}%) p={mc['p_value']:.4g}")
    if misses:
        print("MISSES:", json.dumps(misses[:5], ensure_ascii=False))
    if freeform_res:
        print(f"FREEFORM_60: Freeform={freeform_res['freeform_key_hits']}/60 ({freeform_res['freeform_hit_pct']}%) vs Ontology={freeform_res['ontology_pair_hits']}/60 cost=${freeform_res['api_usage']['cost_usd']:.6f}")


if __name__ == "__main__":
    main()
