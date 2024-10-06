tasks=("number_string" "kv_retrieval" "longbook_sum_eng" "longbook_choice_eng" "longbook_qa_eng" "longbook_qa_chn" "longdialogue_qa_eng" "math_find" "math_calc" "code_run" "code_debug")
weight_percents=(60 80 90 95 99 100)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for task in "${tasks[@]}"; do
    for weight_percent in "${weight_percents[@]}"; do
        python3 eval_llama3.1.py --task $task --model_path /data/Meta-Llama-3.1-8B-Instruct --model_name llama3.1 --weight_percent $weight_percent
    done
done