### Script for evaluating the RealToxicityPrompts inference results ###
export API_KEY="YOUR_PERSPECTIVE_API_KEY"

CUDA_VISIBLE_DEVICES=$GPU_NUM python src/eval_RTP.py \
    --data_file ./results/rtp_results/rtp.json \
    --result_file ./results/rtp_eval_results/rtp.json \
    --num_threads 50

### Script for evaluating the HolisticBiasR inference results ###
dir=./results/llama-2-7b-VIM-holisticbiasr_results
target_output_directory=$dir
regard_result_directory=$dir/holisticbiasr_result

CUDA_VISIBLE_DEVICES=$GPU_NUM python src/eval_HBR.py \
    --target_fpath $dir/holisticbiasr.json \
    --result_fname $regard_result_directory/holisticbiasr.json \
    --batch_size 1024

python ./evaluate/bias_score_0.py \
    --result_directory $target_output_directory \
    --regard_result_directory $regard_result_directory \
    --original_data ./data/HolisticBiasR.jsonl

python ./evaluate/bias_score_1.py \
    --linked_directory $target_output_directory/linked

python ./evaluate/bias_score_2.py \
    --directory $target_output_directory/linked \
    --output_fname ${target_output_directory}/holisticbiasr_results_final.json


### Script for evaluating the HEx-PHI results or Beavertails results ###
result_dirs_HExPHI=(
    ./results/mistral-7b-VIM-beavertails_results-processed
)
result_dirs_beavertails=(
    ./results/mistral-7b-VIM-beavertails_results-processed
)

for dir in "${result_dirs_HExPHI[@]}"; do
    echo "dir: $dir"
    python ./evaluate/gpt4_eval.py \
        --target_result_path $dir \
        --dataset_name HEx-PHI
done
for dir in "${result_dirs_beavertails[@]}"; do
    python ./evaluate/gpt4_eval.py \
        --target_result_path $dir \
        --dataset_name beavertails
done