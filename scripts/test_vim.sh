# Script for testing the VIM models trained on 154 value distributions
mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names_full.txt

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

ckpt_home_path=./ckpt/argument_survey/mistral/min_TH_3

dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')

GPU_NUM=2
start=0
end=${length}

for ((i=${start}; i<${end}; i++)); do
    for dataset_name in "${dataset_names[@]}"; do
        number=$(($i+1))
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
        ckpt_path=${ckpt_home_path}/${TARGET_DISTRIBUTIONS[i]}

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference.py \
            --home_directory . \
            --batch_size 500 \
            --peft_path $ckpt_path \
            --dataset_name $dataset_name
    done
done