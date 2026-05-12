import json
import pandas as pd
from pathlib import Path

# 1) BeaverTails 원본(정답 category_id 들어있음)
SRC_DATA = "./data/BeaverTails-Evaluation.csv"   # repo에 있는 파일 기준
# 2) inference 결과 (네가 만든 beavertails_results.json)
INFER_JSON = "./results/mistral-7b-VIM-beavertails_results_low/beavertails_results.json"
# 3) 평가용 jsonl 출력
OUT_JSONL = "./results/mistral-7b-VIM-beavertails_results_low/beavertails_results.eval.jsonl"

def main():
    bt = pd.read_csv(SRC_DATA)
    # 보통 BeaverTails-Evaluation.csv는 prompt / category_id 컬럼이 있음 (없으면 컬럼명 확인해서 수정)
    # print(bt.columns)

    with open(INFER_JSON, "r") as f:
        preds = json.load(f)   # list of {query, answer, prompt}

    # inference 결과의 query가 원본 prompt와 동일하다는 가정(네 코드상 query=dataset['prompt'][i])
    # 혹시 공백/개행 차이 있으면 strip()로 normalize
    bt_map = {}
    for i, r in bt.iterrows():
        bt_map[str(r["prompt"]).strip()] = int(r["category_id"])

    out = []
    miss = 0
    for x in preds:
        q = str(x.get("query", "")).strip()
        ans = x.get("answer", "")
        if q not in bt_map:
            miss += 1
            cid = -1
        else:
            cid = bt_map[q]
        out.append({
            "prompt": q,
            "response": ans,
            "category_id": cid,
        })

    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", OUT_JSONL)
    print("missing prompt match:", miss, "/", len(out))

if __name__ == "__main__":
    main()
