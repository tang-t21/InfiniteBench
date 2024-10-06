# tasks=("number_string" "kv_retrieval" "longbook_sum_eng" "longbook_choice_eng" "longbook_qa_eng" "longbook_qa_chn" "longdialogue_qa_eng" "math_find")
tasks=("kv_retrieval" "math_find" "code_debug" "longbook_choice_eng")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for task in "${tasks[@]}"; do
        python3 profile_chunk_num.py --task $task --model_path /data/Meta-Llama-3.1-8B-Instruct --model_name llama3.1
done