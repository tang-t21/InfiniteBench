from llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
from typing import Tuple
import argparse
import torch
from eval_utils import load_data, create_prompt
from eval_yarn_mistral import truncate_by_tokens
def load_model(model_path, device="cuda")-> Tuple[LlamaForCausalLM, AutoTokenizer]:
    model = LlamaForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, _attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate(model, tokenizer, input_text, max_tokens=64):
    # print("Prompt:", input_text)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=model.device).unsqueeze(0)
    kv_cache = None
    past_kv_length = 0
    output_token_ids = []
    for i in range(max_tokens):
        past_kv_length += input_ids.size(1)
        outputs = model.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        logits = model.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        output_token_ids.append(next_token_id.item())
        input_ids = next_token_id.unsqueeze(0)
        position_ids = torch.tensor([past_kv_length], dtype=torch.long, device=model.device).unsqueeze(0)
        kv_cache = past_key_values
        if next_token_id == tokenizer.eos_token_id:
            break
    output_text = [tokenizer.decode(id) for id in output_token_ids]
    # print("".join(output_text))
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--task", type=str, default="kv_retrieval")
    parser.add_argument("--data_dir", type=str, default="/home/infiniBench/data")
    parser.add_argument("--model_name", type=str, default="llama3.1")
    args = parser.parse_args()
    model, tok = load_model(args.model_path)
    model.set_weight_percent(100)
    examples = load_data(args.task, data_dir=args.data_dir)
    prompt = create_prompt(examples[0], args.task, args.model_name, args.data_dir)
    prompt = truncate_by_tokens(prompt, tok, 1024)
    generate(model, tok, prompt, args.max_tokens)