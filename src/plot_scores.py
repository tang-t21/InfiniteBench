from pathlib import Path
import json
from args import parse_args
import matplotlib.pyplot as plt
from compute_scores import ALL_TASKS

def process_data(args, weight_percents, tasks):
    result_dir = Path(args.output_dir, args.model_name)
    scores_by_task = {task:[] for task in tasks}
    for weight_percent in weight_percents:
        with open(f"{result_dir}/{weight_percent}/scores.json", "r") as f:
            scores = json.load(f)
        for task in tasks:
            scores_by_task[task].append(scores[task])
    return scores_by_task

def plot_scores(scores_by_task, weight_percents, tasks, output_dir):
    plt.figure(figsize=(10,8))
    plt.xlabel("Weight_percent")
    plt.ylabel("Score")
    for task in tasks:
        print(task, scores_by_task[task])
        plt.plot(weight_percents, scores_by_task[task], label=f"{task}")
    plt.xticks(weight_percents)
    plt.title(f"Scores of tasks VS weight percents")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{task}_scores.png")


if __name__ == "__main__":
    args = parse_args()
    if args.task == "all":
        tasks = ALL_TASKS
    else:
        tasks = [args.task]
    weight_percents = [50,60,80,90,95,99]
    scores_by_task = process_data(args, weight_percents, tasks)
    plot_scores(scores_by_task, weight_percents, tasks, args.output_dir)