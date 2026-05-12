import os
import dotenv
dotenv.load_dotenv()

import fire
import vllm
import json
from tqdm import tqdm

from peft import PeftConfig
from vllm.lora.request import LoRARequest
from utils.utils import (
    _find_save_path, load_datasets
)

def main(
    dataset_name: str = 'holisticbiasr',  # 'rtp', 'holisticbiasr', 'HEx-PHI', 'beavertails'
    home_directory='.',
    sanity_check=-1,
    batch_size=8,
    max_tokens=64,
    peft_path: str = './ckpt/argument_survey/mistral/min_TH_3/power_high_631',
    output_path: str = '',  # './results/...'
    save_jsonl: bool = True,   # ✅ 기본 jsonl 저장(안 깨짐)
    debug_blank: bool = True,  # ✅ 빈 답변 디버그 로그
):
    # Load Dataset
    dataset = load_datasets(dataset_name, home_directory=home_directory)
    if sanity_check != -1:
        dataset = dataset[:sanity_check]
        print(f"Sanity check: {len(dataset)} samples")
    else:
        print(f"Dataset: {len(dataset)} samples")

    # Determine Output Path
    if output_path == '':
        output_file = os.path.join(peft_path, 'results', f'{dataset_name}_results.json')
    else:
        output_file = os.path.join(output_path, f'{dataset_name}_results.json')

    # ✅ jsonl로 저장하면 중간 overwrite로 망가질 일이 거의 없음
    if save_jsonl and output_file.endswith(".json"):
        output_file = output_file[:-5] + ".jsonl"

    epoch_num = _find_save_path(peft_path)
    if epoch_num == -1:
        peft_model_id = peft_path
    else:
        peft_model_id = f"{peft_path}/{epoch_num}"
        print(f"Loading peft model from {peft_model_id}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    keys_and_max_tokens = {
        'rtp': 64,
        'holisticbiasr': 128,
        'HEx-PHI': 64,
        'beavertails': 64,
    }
    n = 10 if dataset_name == 'rtp' else 1
    max_tokens = keys_and_max_tokens.get(dataset_name, max_tokens)

    sampling_params = vllm.SamplingParams(
        n=n,
        temperature=0.1,
        top_p=0.75,
        max_tokens=max_tokens,
    )

    # Load base + peft
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_id}")
    print(f"Loading peft model from {peft_model_id}")

    llm = vllm.LLM(
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

    # Build prompt list
    if dataset_name in ['holisticbiasr']:
        dataset = dataset['prompt_text'].to_list()

    prompt_list = []
    data_query_list = []

    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}", total=len(dataset)):
        if dataset_name in ['rtp']:
            query = dataset[i][0]
            prompt = dataset[i][0]
        elif dataset_name in ['holisticbiasr']:
            query = dataset[i]
            prompt = dataset[i]
        else:
            # HEx-PHI / beavertails는 보통 dataframe
            query = dataset['prompt'][i]
            prompt = f"QUESTION: {query} \nANSWER: "

        prompt_list.append(prompt)
        data_query_list.append(query)

    # ✅ jsonl이면 파일 비우고 append 시작
    if save_jsonl:
        with open(output_file, "w", encoding="utf-8") as f:
            pass

    result_dict = []  # json 저장 모드에서만 사용

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(dataset), batch_size),
                  desc=f"Generating {dataset_name} responses",
                  total=total_batches):

        batch_prompt = prompt_list[i:i + batch_size]
        output = llm.generate(
            batch_prompt,
            sampling_params=sampling_params,
            lora_request=LoRARequest("peft", 1, peft_model_id),
            use_tqdm=False,
        )

        # ✅ 여기가 핵심: j 루프는 반드시 배치 루프 안에 있어야 함
        batch_records = []
        for j in range(i, min(i + batch_size, len(dataset))):
            index = j - i
            outs = output[index].outputs  # ✅ output[0] 말고 index 기준

            if len(outs) == 0:
                ans = ""
                fr = None
                if debug_blank:
                    print("[WARN] empty outputs:", j, repr(str(data_query_list[j])[:120]))
            elif len(outs) == 1:
                ans = outs[0].text
                fr = getattr(outs[0], "finish_reason", None)
            else:
                ans = [o.text for o in outs]
                fr = getattr(outs[0], "finish_reason", None)

            if debug_blank and (ans == "" or (isinstance(ans, str) and ans.strip() == "")):
                print("[DEBUG] blank answer:", j, "finish_reason=", fr,
                      "prompt_head=", repr(prompt_list[j][:120]))

            rec = {
                "query": data_query_list[j],
                "answer": ans,
                "prompt": prompt_list[j],
            }
            batch_records.append(rec)

        # ✅ 저장: jsonl이면 append, json이면 리스트에 쌓아서 마지막에 dump
        if save_jsonl:
            with open(output_file, "a", encoding="utf-8") as f:
                for rec in batch_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            result_dict.extend(batch_records)

    # json 모드면 마지막에 한 번만 dump
    if not save_jsonl:
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")

    # ✅ sanity: jsonl이면 줄 수로 확인
    if save_jsonl:
        try:
            import subprocess
            out = subprocess.check_output(["wc", "-l", output_file]).decode().strip()
            print("[CHECK] wc -l:", out)
        except Exception:
            pass

if __name__ == '__main__':
    fire.Fire(main)
