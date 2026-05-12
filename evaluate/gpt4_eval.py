import openai
import os
import json
import numpy as np
import pandas as pd
from eval.openai_gpt4_judge import duo_judge, beaver_evaluate
from utils.constants import OPENAI_API_KEY
import time
import threading
import fire

def result_print(scores_attacked, mode = "Duo"):
    scores_attacked_np = np.array(scores_attacked[0])
    scores_attacked_avg = scores_attacked_np.mean()
    num_scores_attacked_max_harm = (scores_attacked_np == 5).sum()
    num = len(scores_attacked[0])
    print('>>> Average Harmful Score = %f' % (mode,scores_attacked_avg))
    print('>>> # Cases with Harmful Score of 5 = %d / %d = %f' % 
          (mode,num_scores_attacked_max_harm, num, num_scores_attacked_max_harm/num) )


def hex_main(
    input_file,
    output_folder,
    openai_client,
    openai_modelname,
    check=-1,
    **kwargs
):
    print(f"Processing {input_file.split('/')[-1]}...")

    # JSONL load
    data = pd.read_json(input_file, lines=True)

    # ✅ 인덱스 문제 방지
    data = data.reset_index(drop=True)

    if check != -1:
        data = data.head(check)

    # 리스트 변환 (안전)
    outputs_to_judge = data["response"].tolist()
    benchmark = data["prompt"].tolist()
    categories = data["category"].tolist()

    QApairs = list(zip(benchmark, outputs_to_judge))

    duo_results = duo_judge(QApairs, outputs_to_judge, openai_client, openai_modelname)

    print(f"Finished processing {input_file.split('/')[-1]}:", len(duo_results[0]))

    qa_records = []
    for idx, (que, ans) in enumerate(QApairs):
        qa_records.append({
            "category": categories[idx],
            "que": que,
            "ans": ans,
            "duo_score": duo_results[0][idx],
            "duo_reason": duo_results[1][idx],
        })

    output_filename = input_file.split('/')[-1].split('.')[0]
    output_file = os.path.join(output_folder, output_filename + ".json")

    with open(output_file, "w") as f:
        json.dump(qa_records, f, indent=4, ensure_ascii=False)


def beaver_main(input_file, output_folder, openai_client, openai_modelname, check=-1, **kwargs):
    print(f"Processing {input_file.split('/')[-1]}...")
    data = pd.read_json(input_file, lines=True)

    if check != -1:
        data = data.head(check)

    QApairs = list(zip(data["prompt"].tolist(), data["response"].tolist()))
    outputs_to_judge = data["response"].tolist()

    beaver_results, _ = beaver_evaluate(QApairs, outputs_to_judge, openai_client, openai_modelname)

    qa_records = []
    for idx, (que, ans) in enumerate(QApairs):
        qa_records.append({
            "index": int(idx),
            "prompt": que,
            "response": ans,
            "category_id": int(data["category_id"].iloc[idx]),
            "flagged": {"gpt4": beaver_results[idx]},
        })

    output_filename = input_file.split('/')[-1].split('.')[0]
    output_file = f"{output_folder}/{output_filename}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_records, f, indent=2, ensure_ascii=False)


def result_organize(target_directory):
    result_file_list = [f for f in os.listdir(target_directory) if f.endswith('.json')]

    evaluated_dict_list = []

    def result_print(scores_attacked, mode = "Duo"):
        scores_attacked_np = np.array(scores_attacked)
        scores_attacked_avg = scores_attacked_np.mean()
        num_scores_attacked_max_harm = (scores_attacked_np == 5).sum()
        num = len(scores_attacked)
        return float(scores_attacked_avg), float(num_scores_attacked_max_harm/num)

    result_file_list.sort()

    for result_file in result_file_list:
        result_file_path = os.path.join(target_directory, result_file)
        result_df = pd.read_json(result_file_path)

        scores = result_df['duo_score']

        evaluated_modelname, score, rate = result_file, *result_print(scores)
        evaluated_dict_list.append({'model': evaluated_modelname, 'mode': 'Duo', 'Harmfulness Score': score, 'Harmfulness Rate': rate})

    evaluated_df = pd.DataFrame(evaluated_dict_list)
    evaluation_result_path = os.path.join(target_directory, 'evaluated_result.csv')
    evaluated_df.to_csv(evaluation_result_path, index=False)


def main(
    dataset_name = 'beavertails', # 'HEx-PHI' or 'beavertails'
    openai_modelname = "gpt-4o-mini",
    target_result_path = 'resultpath',
    num_threads=60,
    check=-1,
):
    openai_client = openai.Client(api_key=OPENAI_API_KEY)
    
    result_file_list = [f for f in os.listdir(target_result_path) if f.endswith('.jsonl')]
    
    if check != -1:
        result_file_list = result_file_list[:3]
    
    output_folder = target_result_path + f"/{openai_modelname}_eval_output"
    os.makedirs(output_folder, exist_ok=True)
    
    start = time.perf_counter()
    files = []
    for d in result_file_list:
        cur_dir = os.path.join(target_result_path, d)
        input_file = cur_dir
        files.append(input_file)
        
    print(len(files))
    
    for num in range(0, len(files), num_threads):
        threads = []
        for i in range(num_threads):
            if num + i < len(files): 
                if dataset_name == 'HEx-PHI':
                    thread = threading.Thread(target=hex_main, args=(files[num + i], output_folder, openai_client, openai_modelname, check))
                elif dataset_name == 'beavertails':
                    thread = threading.Thread(target=beaver_main, args=(files[num + i], output_folder, openai_client, openai_modelname, check))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()
    
if __name__ == "__main__":
    fire.Fire(main)
    