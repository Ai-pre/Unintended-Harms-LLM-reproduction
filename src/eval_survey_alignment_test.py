#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


VALUE_NAMES = [
    "achievement", "benevolence", "conformity", "hedonism", "power",
    "security", "self-direction", "stimulation", "tradition", "universalism"
]

# test.csv는 보통 Conclusion / Stance / Premise / (10개 value indicator) 컬럼이 있음
REQUIRED_TEXT_COLS = ["Conclusion", "Stance", "Premise"]

# 모델 출력에서 1~6만 파싱 (가장 보수적으로)
SCORE_RE = re.compile(r"\b([1-6])\b")

WORD2NUM = {
    "one":1, "two":2, "three":3, "four":4, "five":5, "six":6,
    "1":1, "2":2, "3":3, "4":4, "5":5, "6":6,
}

def parse_score(text: str):
    if not text:
        return None
    t = text.lower()

    # 1) 숫자 먼저
    m = re.search(r"\b([1-6])\b", t)
    if m:
        return int(m.group(1))

    # 2) 단어 숫자
    for w, n in WORD2NUM.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            return n

    return None


def build_q(conclusion: str, stance: str, premise: str) -> str:
    # train 때 쓰던 템플릿과 최대한 동일하게
    if stance.strip().lower() == "in favor of":
        return f"I agree with {conclusion}. {premise}"
    elif stance.strip().lower() == "against":
        return f"I disagree with {conclusion}. {premise}"
    else:
        # stance가 이상하면 그냥 conclusion+preamise로라도 만들기
        return f"{conclusion}. {premise}"

def build_prompt(q: str, value_dim: str) -> str:
    return (
        "You must answer with a single digit from 1 to 6 only.\n"
        "1=Not like me at all\n"
        "2=Not like me\n"
        "3=A little like me\n"
        "4=Somewhat like me\n"
        "5=Like me\n"
        "6=Very much like me\n"
        f"Focus specifically on {value_dim}.\n"
        f"Statement: {q}\n"
        "Answer (one digit only):"
    )



def cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den > 0 else float("nan")

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      - generated_only: prompt 이후 새로 생성된 텍스트만
      - generated_full: prompt + generated 전체(디버깅용)
    """
    toks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    toks = {k: v.to(device) for k, v in toks.items()}

    out = model.generate(
        **toks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # full decode (디버깅용)
    generated_full = tokenizer.batch_decode(out, skip_special_tokens=True)

    # ✅ 핵심: prompt 길이만큼 잘라서 새 토큰만 디코딩
    in_lens = toks["attention_mask"].sum(dim=1).tolist()  # 각 샘플 실제 입력 길이
    generated_only = []
    for i in range(out.size(0)):
        gen_ids = out[i, in_lens[i]:]  # 새로 생성된 부분만
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        generated_only.append(txt)

    return generated_only, generated_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distribution_name", required=True)
    ap.add_argument("--peft_model_id", required=True)
    ap.add_argument("--survey_file", default="./data/argument_generation/value_split/test.csv",
                   help="argument_generation split file (test.csv 권장)")
    ap.add_argument("--extreme_distribution_file", default="./data/extreme_distributions.csv")
    ap.add_argument("--out_dir", default="./outputs/survey_alignment_test")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ----- target 10d 로드 (extreme_distributions.csv의 해당 distribution_name row 마지막 10개 사용: 기존 train 스크립트와 동일 논리) -----
    dist_df = pd.read_csv(args.extreme_distribution_file, sep="\t")
    if "Country" not in dist_df.columns:
        raise ValueError(f"extreme_distribution_file missing 'Country' col: {dist_df.columns.tolist()}")
    names = dist_df["Country"].tolist()
    if args.distribution_name not in names:
        raise ValueError(f"distribution_name not found in extreme_distribution_file: {args.distribution_name}")

    row = dist_df.iloc[names.index(args.distribution_name)]
    target_score = list(row)[-10:]  # length 10
    target_10d = {k: float(v) for k, v in zip(VALUE_NAMES, target_score)}

    # ----- test.csv 로드 -----
    df = pd.read_csv(args.survey_file, sep="\t")
    for c in REQUIRED_TEXT_COLS:
        if c not in df.columns:
            raise ValueError(f"survey_file missing required col {c}. columns={df.columns.tolist()}")

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # value indicator 컬럼 찾기: 보통 맨 뒤 10개(혹은 Achievement..Universalism)
    # 가장 안전하게 VALUE_NAMES/TitleCase 둘 다 대응
    # 1) exact match (lower)
    col_map = {}
    for vn in VALUE_NAMES:
        for cand in [vn, vn.replace("-", " "), vn.title(), vn.replace("-", " ").title()]:
            if cand in df.columns:
                col_map[vn] = cand
                break

    # 2) 못 찾으면 "마지막 10개가 indicator"라는 기존 레포 가정 사용
    if len(col_map) < 10:
        last10 = df.columns.tolist()[-10:]
        # last10이 VALUE_NAMES 순서와 같진 않을 수 있어도, 파일이 그 형식이면 보통 Achievement..Universalism 순
        # 여기서는 last10을 VALUE_NAMES와 같은 순서로 매핑할 수가 없어서,
        # 최소한 DF에 Achievement..Universalism가 있으면 그걸 우선으로 다시 시도.
        # 그래도 안되면 last10을 그냥 VALUE_NAMES에 순서대로 매핑(파일이 그 순서라는 가정)
        ach_like = ["Achievement","Benevolence","Conformity","Hedonism","Power","Security","Self-direction","Stimulation","Tradition","Universalism"]
        if all(c in df.columns for c in ach_like):
            for vn, c in zip(VALUE_NAMES, ach_like):
                col_map[vn] = c
        else:
            for vn, c in zip(VALUE_NAMES, last10):
                col_map[vn] = c

    # ----- (row, value_dim) expanded samples 만들기 -----
    expanded_rows = []
    for i, r in df.iterrows():
        conclusion = str(r["Conclusion"])
        stance = str(r["Stance"])
        premise = str(r["Premise"])

        q = build_q(conclusion, stance, premise)

        # 어떤 value_dim들이 주석(=1)인지 뽑기
        dims = []
        for vn in VALUE_NAMES:
            c = col_map[vn]
            try:
                if int(r[c]) == 1:
                    dims.append(vn)
            except Exception:
                # 값이 이상하면 skip
                pass

        # dims가 비면 평가가 의미 없으니 스킵
        if not dims:
            continue

        for vn in dims:
            prompt = build_prompt(q, vn)
            expanded_rows.append((i, vn, q, prompt))

    if not expanded_rows:
        raise ValueError("No expanded samples found. Check indicator columns in survey_file.")

    # ----- 모델 로드 (peft adapter 붙이기) -----
    device = args.device
    base_cfg = PeftConfig.from_pretrained(args.peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg.base_model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_cfg.base_model_name_or_path,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.peft_model_id)
    model.to(device)
    model.eval()

    # ----- 배치 생성 + 파싱 -----
    records = []
    parsed = 0

    for st in range(0, len(expanded_rows), args.batch_size):
        batch = expanded_rows[st:st+args.batch_size]
        prompts = [x[3] for x in batch]
        gens_only, gens_full = generate_batch(model, tokenizer, prompts, device, args.max_new_tokens)

        for (orig_row, value_dim, q, prompt), gen_only, gen_full in zip(batch, gens_only, gens_full):
            # ✅ 파싱은 무조건 generated_only에서만
            s = parse_score(gen_only)

            if s is not None:
                parsed += 1

            records.append({
                "orig_row": orig_row,
                "value_dim": value_dim,
                "q": q,
                "prompt": prompt,
                "generated_only": gen_only,
                "generated_full": gen_full,
                "score_pred": s
            })


    out_csv = os.path.join(args.out_dir, f"{args.distribution_name}__preds.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)

    parsed_rate = parsed / max(1, len(records))

    # ----- value_dim별 평균 → pred_mean_10d -----
    pred_mean_10d = {vn: None for vn in VALUE_NAMES}
    df_rec = pd.DataFrame(records)
    for vn in VALUE_NAMES:
        sub = df_rec[(df_rec["value_dim"] == vn) & (df_rec["score_pred"].notna())]
        if len(sub) == 0:
            pred_mean_10d[vn] = None
        else:
            pred_mean_10d[vn] = float(sub["score_pred"].astype(float).mean())

    # metric 계산: None 있는 차원은 제외하고 계산
    keys = [k for k in VALUE_NAMES if pred_mean_10d[k] is not None]
    t = np.array([target_10d[k] for k in keys], dtype=np.float32)
    p = np.array([pred_mean_10d[k] for k in keys], dtype=np.float32)

    metrics = {
        "pearson": pearson(t, p),
        "cosine": cosine(t, p),
        "mse": mse(t, p),
        "mae": mae(t, p)
    }

    summary = {
        "distribution_name": args.distribution_name,
        "peft_model_id": args.peft_model_id,
        "survey_file": args.survey_file,
        "n_expanded_samples": len(records),
        "parsed_rate": parsed_rate,
        "target_10d": target_10d,
        "pred_mean_10d": pred_mean_10d,
        "metrics": metrics,
        "out_csv": out_csv,
    }

    out_json = os.path.join(args.out_dir, f"{args.distribution_name}__summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
