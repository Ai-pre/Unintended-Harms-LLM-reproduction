import os
import sys
import json
import random
import dotenv
dotenv.load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import fire
import vllm
import pandas as pd

from peft import PeftConfig
from vllm.lora.request import LoRARequest
from utils.utils import _find_save_path


def resolve_peft_model_id(peft_path: str) -> str:
    epoch_num = _find_save_path(peft_path)
    if epoch_num == -1:
        return peft_path
    return f"{peft_path}/{epoch_num}"


def load_base_model_id(peft_model_id: str) -> str:
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    return peft_config.base_model_name_or_path


def build_llm(base_model_id: str) -> vllm.LLM:
    return vllm.LLM(
        model=base_model_id,
        task="generate",
        enable_lora=True,
        enforce_eager=True,
        dtype="half",
        max_model_len=1024,
        tokenizer=base_model_id,
        tokenizer_mode="auto",
        hf_overrides={"head_dim": 128},
        gpu_memory_utilization=0.8,
    )


def pick_prompt(df: pd.DataFrame, keyword: str, seed: int = 42):
    """
    category 컬럼에 keyword가 포함된 row 중 하나를 고른다.
    """
    sub = df[df["category"].str.contains(keyword, case=False, na=False)].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for keyword='{keyword}'")

    sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)
    row = sub.iloc[0]
    return {
        "category": row["category"],
        "query": row["prompt"],
    }


def main(
    csv_path: str = "data/BeaverTails-Evaluation.csv",
    power_low_path: str = "./ckpt/argument_survey/mistral/min_TH_3/power_low_631",
    power_high_path: str = "./ckpt/argument_survey/mistral/min_TH_3/power_high_631",
    output_path: str = "./results/beavertails_case_study_outputs.json",
    seed: int = 42,
    max_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    # 1) CSV 로드
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns or "category" not in df.columns:
        raise ValueError(f"CSV must contain 'prompt' and 'category' columns. got: {list(df.columns)}")

    print(f"Loaded CSV: {csv_path}")
    print(f"Shape: {df.shape}")
    print("Unique categories:")
    print(df["category"].value_counts())

    # 2) 실제 프롬프트 선택
    prompts = [
        pick_prompt(df, "hate_speech", seed=seed),
        pick_prompt(df, "financial_crime", seed=seed + 1),
    ]

    print("\nSelected prompts:")
    for p in prompts:
        print(f"- [{p['category']}] {p['query']}")

    # 3) adapter 경로 resolve
    power_low_model_id = resolve_peft_model_id(power_low_path)
    power_high_model_id = resolve_peft_model_id(power_high_path)

    print(f"\n[Power-Low]  PEFT model: {power_low_model_id}")
    print(f"[Power-High] PEFT model: {power_high_model_id}")

    # 4) base model 확인
    base_low = load_base_model_id(power_low_model_id)
    base_high = load_base_model_id(power_high_model_id)

    if base_low != base_high:
        raise ValueError(
            f"Base model mismatch:\n"
            f"  power_low  -> {base_low}\n"
            f"  power_high -> {base_high}"
        )

    base_model_id = base_low
    print(f"Base model: {base_model_id}")

    # 5) vllm 로드
    llm = build_llm(base_model_id)

    sampling_params = vllm.SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    formatted_prompts = [f"QUESTION: {item['query']}\nANSWER: " for item in prompts]

    # 6) Power-Low 생성
    low_outputs = llm.generate(
        formatted_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("power_low", 1, power_low_model_id),
        use_tqdm=False,
    )

    # 7) Power-High 생성
    high_outputs = llm.generate(
        formatted_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("power_high", 2, power_high_model_id),
        use_tqdm=False,
    )

    # 8) 결과 정리
    results = []
    for i, item in enumerate(prompts):
        low_answer = low_outputs[i].outputs[0].text if len(low_outputs[i].outputs) > 0 else ""
        high_answer = high_outputs[i].outputs[0].text if len(high_outputs[i].outputs) > 0 else ""

        rec = {
            "category": item["category"],
            "query": item["query"],
            "prompt": formatted_prompts[i],
            "power_low_answer": low_answer,
            "power_high_answer": high_answer,
        }
        results.append(rec)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}\n")

    for rec in results:
        print("=" * 120)
        print(f"[CATEGORY] {rec['category']}")
        print(f"[QUERY] {rec['query']}\n")
        print("[Power-Low]")
        print(rec["power_low_answer"])
        print("\n[Power-High]")
        print(rec["power_high_answer"])
        print("=" * 120)


if __name__ == "__main__":
    fire.Fire(main)