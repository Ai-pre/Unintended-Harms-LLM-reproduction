import json

input_path = "results/low_hexphi/HEx-PHI_results.jsonl"
output_path = "results/low_hexphi/HEx-PHI_results1.jsonl"

with open(input_path) as f, open(output_path, "w") as out:
    for line in f:
        row = json.loads(line)

        new_row = {
            "prompt": row["query"],      # rename
            "response": row["answer"],   # rename
        }

        out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

print("DONE")