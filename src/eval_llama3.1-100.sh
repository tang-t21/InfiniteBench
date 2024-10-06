# tasks=("number_string" "kv_retrieval" "longbook_sum_eng" "longbook_choice_eng" "longbook_qa_eng" "longbook_qa_chn" "longdialogue_qa_eng" "math_find")
tasks=("kv_retrieval")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for task in "${tasks[@]}"; do
        python3 eval_llama3.1.py --task $task --model_path /data/Meta-Llama-3.1-8B-Instruct --model_name llama3.1 --stop_idx 100 --start_idx 80
done