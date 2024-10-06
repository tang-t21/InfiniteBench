from eval_utils import iter_jsonl
from compute_scores import get_preds
from args import parse_args
from pathlib import Path

def assert_pred(pred_path1, pred_path2, length=20):
    data1=list(iter_jsonl(pred_path1))[:length]
    data2=list(iter_jsonl(pred_path2))[:length]
    pred1=get_preds(data1)
    pred2=get_preds(data2)
    scores=[]
    for pred1, pred2 in zip(pred1, pred2):
        scores.append(pred1==pred2)
    print(f"Accuracy-: {sum(scores)/len(scores)}")    

if __name__ == "__main__":
    args = parse_args()
    result_dir = Path(args.output_dir, args.model_name, str(args.weight_percent))
    pred1_path = result_dir / f"preds_{args.task}_{args.start_idx}-{args.stop_idx}-new.jsonl"
    pred2_path = result_dir / f"preds_{args.task}_{args.start_idx}-{args.stop_idx}.jsonl"
    assert_pred(pred1_path, pred2_path)