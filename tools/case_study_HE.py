import json
import os

HIGH_PATH = "results/high_hexphi/hexphi_results_with_category.jsonl"
LOW_PATH  = "results/low_hexphi/hexphi_results_with_category.jsonl"
OUT_PATH  = "results/case_study_hexphi_136_with_low_high.json"

TARGET_INDEX = 136


def load_jsonl_line(path, index):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range for {path}")


def get_prompt(obj):
    return obj.get("que", obj.get("prompt", obj.get("query", "")))


def get_response(obj):
    return obj.get("response", obj.get("answer", obj.get("output", "")))


high_obj = load_jsonl_line(HIGH_PATH, TARGET_INDEX)
low_obj  = load_jsonl_line(LOW_PATH, TARGET_INDEX)

case_study = {
    "case_id": TARGET_INDEX,
    "dataset": "HEx-PHI",
    "category": high_obj.get("category", low_obj.get("category", "")),
    "prompt": get_prompt(high_obj),
    "comparison": {
        "power_low_response": get_response(low_obj),
        "power_high_response": get_response(high_obj),
    }
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(case_study, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUT_PATH}")
print(json.dumps(case_study, indent=2, ensure_ascii=False))