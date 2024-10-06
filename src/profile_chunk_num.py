import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from llama.modeling_llama import LlamaForCausalLM
from eval_utils import (
    create_prompt,
    load_data,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from args import parse_args

from eval_yarn_mistral import TRUNCATE_LEN


def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    model = LlamaForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, _attn_implementation="eager"
    )
    print("Time taken:", round(time.time() - start_time))
    return model, tok  # type: ignore



if __name__ == "__main__":
    model_name = "llama3.1"
    args = parse_args()
    torch.set_num_threads(32)
    print(json.dumps(vars(args), indent=4))
    data_name = args.task
    
    # Model
    model, tok = load_model(args.model_path)
    model.set_weight_percent(args.weight_percent)
    # Data
    examples = load_data(data_name, data_dir=args.data_dir)
    chunk_len = 1024
    max_chunk_num = (TRUNCATE_LEN + chunk_len -1) // chunk_len
    num_chunks=[]
    for example in examples:
        input_text = create_prompt(example,args.task,model_name,args.data_dir)
        tokens = tok.encode(input_text)
        num_chunks.append((len(tokens) + chunk_len -1) // chunk_len if len(tokens) < TRUNCATE_LEN else max_chunk_num) 
    print(f"=============Results of {args.task}==============") 
    print(f"Average number of chunks: {sum(num_chunks)/len(num_chunks)}")
    print(f"Max number of chunks: {max(num_chunks)}")
    print(f"Min number of chunks: {min(num_chunks)}")