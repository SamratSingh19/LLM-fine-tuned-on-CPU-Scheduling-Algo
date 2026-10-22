import json
import random

NUMERICALS_PATH = "data/processed/finetune_data.jsonl"
CONCEPTS_PATH = "data/processed/concepts_template.jsonl"  
OUT_PATH = "data/processed/combined_training_data.jsonl"

DUPLICATE_COUNT = 15

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def merge_and_balance():
    numericals = load_jsonl(NUMERICALS_PATH)
    concepts = load_jsonl(CONCEPTS_PATH)
    duplicated_concepts = concepts * DUPLICATE_COUNT
    combined = numericals + duplicated_concepts
    random.shuffle(combined)   # mix them up, don't leave theory all at the end

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    theory_share = len(duplicated_concepts) / len(combined)
    print(f"Numericals: {len(numericals)}")
    print(f"Theory examples: {len(concepts)} (duplicated {DUPLICATE_COUNT}x -> {len(duplicated_concepts)})")
    print(f"Combined total: {len(combined)}  |  theory share: {theory_share:.1%}")
    print(f"Written to: {OUT_PATH}")
    print("\nPoint train_lora.py at this file instead of finetune_data.jsonl.")


if __name__ == "__main__":
    merge_and_balance()