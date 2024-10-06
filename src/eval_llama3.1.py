import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from llama.modeling_llama import LlamaForCausalLM
from args import parse_args


MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1024


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = 128 * 1024,
    chunk_size: int = 2500,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)
        print(f"input_ids.shape:{input_ids.shape}")
        attention_mask: Tensor = inputs.attention_mask  # (b, n)
        position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, value=1)
        seq_len = input_ids.shape[-1]
        print("seq_len:", seq_len)
        kv_cache: Any = None
        # Split into chunks for pre-filling
        chunk_idxs = []
        n = seq_len - 1
        while n > 0:
            chunk_idxs.append(n)
            n -= chunk_size
        chunk_idxs.append(0)
        chunk_idxs = chunk_idxs[::-1]
        chunk_lo = chunk_idxs[:-1]
        chunk_hi = chunk_idxs[1:]
        print(f"Number of chunks: {len(chunk_lo)}, generating...")
        start_time = time.time()
        for chunk_i, (chunk_lo, chunk_hi) in enumerate(
            zip(chunk_lo, chunk_hi)
        ):
            if verbose:
                print(
                    f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                    round(time.time() - start_time),
                )
            chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
            if kv_cache is not None:
                mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
            else:
                mask_start_idx = chunk_lo
            chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
            chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
            outputs: BaseModelOutputWithPast = model.model.forward(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                position_ids=chunk_position_ids,
                past_key_values=kv_cache,
                return_dict=True,
                use_cache=True,
            )
            kv_cache = outputs.past_key_values
            # Discard KV states on the left beyond the window
            new_cache = ()
            n_layers = len(kv_cache)
            for layer_i in range(n_layers):
                keys = kv_cache[layer_i][0][:, :, -sliding_window:]
                values = kv_cache[layer_i][1][:, :, -sliding_window:]
                new_cache += ((keys, values),)
            kv_cache = new_cache
        kv_cache_len = kv_cache[0][0].shape[2]
        print("kv_cache_len:", kv_cache_len)
        print("KV cache prepared!")
        outputs = model.generate(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask[:, -kv_cache_len - 1 :],
            max_new_tokens=max_tokens,
            past_key_values=kv_cache,
            eos_token_id=tok.pad_token_id,
            use_cache=True,
        )
        # output_token_ids = []
        # position_ids = torch.tensor([kv_cache_len], dtype=torch.long, device=model.device).unsqueeze(0)
        # for i in range(max_tokens):
        #     outputs = model.model.forward(
        #         input_ids=input_ids[:, -1:],
        #         position_ids=position_ids,
        #         past_key_values=kv_cache,
        #         use_cache=True,
        #     )
        #     hidden_states = outputs.last_hidden_state
        #     kv_cache = outputs.past_key_values
        #     logits = model.lm_head(hidden_states)
        #     next_token_logits = logits[:, -1, :]
        #     next_token_id = torch.argmax(next_token_logits, dim=-1)
        #     output_token_ids.append(next_token_id.item())
        #     input_ids = next_token_id.unsqueeze(0)
        #     position_ids = torch.tensor([kv_cache[0][0].shape[2]], dtype=torch.long, device=model.device).unsqueeze(0)
        #     if next_token_id == tok.eos_token_id:
        #         break
        responses = [
            tok.decode(t[1:], skip_special_tokens=True) for t in outputs
        ]
    return responses


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    output = chunk_generate(
        model,
        tok,
        [input_text],
        max_tokens=max_tokens,
        chunk_size=1024,
        verbose=verbose,
    )[0]
    print("Chunked generation:", output)
    return output


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
    torch.set_num_threads(64)
    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)
    model.set_weight_percent(args.weight_percent)
    # Data
    result_dir = Path(args.output_dir, model_name, str(args.weight_percent))
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)
    args.start_idx = 0 if args.start_idx is None else args.start_idx
    args.stop_idx = args.start_idx + min(20,len(examples)//4) if args.stop_idx is None else args.stop_idx
    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose,
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
