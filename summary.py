import json
import argparse
import pandas as pd
from pathlib import Path

from pytorch_lightning.core.memory import get_human_readable_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--save_path", type=str, default="./output/summary.md")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    infos = {}
    for d in Path(args.model_dir).iterdir():
        metric_path = d / "metric.json"
        if not metric_path.exists():
            continue

        with open(metric_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["test_acc"] *= 100
            origin_model_params = data["params"]
            origin_model_test_acc = data["test_acc"]
            data["params"] = get_human_readable_count(data["params"])
            data["size reduce"] = ""
            infos[d.name] = data

        for pruned_d in d.iterdir():
            if str(pruned_d.name).startswith("pruned"):
                pruned_metric_path = pruned_d / "metric.json"
                if not pruned_metric_path.exists():
                    continue

                with open(pruned_metric_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["test_acc"] *= 100
                    infos[f"{d.name} {pruned_metric_path.parent.name}"] = data

                pruned_model_params = data["params"]
                pruned_model_test_acc = data["test_acc"]
                data["params"] = get_human_readable_count(data["params"])
                data[
                    "size reduce"
                ] = f"{(origin_model_params-pruned_model_params)/origin_model_params*100:.2f}%"
                data[
                    "test_acc_diff"
                ] = f"{(pruned_model_test_acc - origin_model_test_acc):.2f}"

    columns = {
        "net": "Net",
        "s": "Sparsity",
        "prune_ratio": "Prune Ratio",
        "test_acc": "Test Acc",
        "test_acc_diff": "Test Acc Diff",
        "params": "Params",
        "size reduce": "Size Reduce",
    }

    markdown_data = []
    for it in infos.values():
        _data = {}
        for n, _it in it.items():
            if n in columns:
                _data[columns[n]] = _it
        markdown_data.append(_data)

    df = pd.DataFrame(sorted(markdown_data, key=lambda x: x["Net"]))
    df = df.reindex(columns=list(columns.values()))
    markdown = df.to_markdown()

    print(markdown)
    with open(args.save_path, "w", encoding="utf-8") as f:
        f.write(markdown)
