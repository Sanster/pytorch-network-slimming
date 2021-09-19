import argparse
import json

import torch

from backbone.build import build_model
from pns import SlimPruner
from pns.tracker import gen_pruning_schema
from pns.functional import summary_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="resnet18")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = build_model(args.net)
    summary_model(model)
    x = torch.Tensor(1, 3, 32, 32)
    y = model.features(x)
    print(f"input shape: {x.shape}, backbone output shape: {y.shape}")
    # import ipdb;ipdb.set_trace()

    config = gen_pruning_schema(model, x)
    config["prefix"] = args.prefix

    if "RepVGG" in args.net and "woid" not in args.net:
        shortcuts = []
        last_group = None
        for i, group in enumerate(config["shortcuts"]):
            if last_group is not None:
                if any(["identity" in _ for _ in group["names"]]):
                    last_group["names"].extend(group["names"])
                else:
                    last_group = group
                    shortcuts.append(last_group)
            else:
                shortcuts.append(group)
                last_group = group
        config["shortcuts"] = shortcuts

    if "mobilenet_v3_large" in args.net and "nose" not in args.net:
        # BN in block with se module should have same channels
        se_shortcuts = []
        modules = []
        for m in config["modules"]:
            if m["name"].endswith("fc2"):
                # e.g: features.5.block
                feature_name = ".".join(m["name"].split(".")[:-2])
                m["next_bn"] = f"{feature_name}.1.1"
            elif m["name"].endswith("fc1"):
                m["next_bn"] = ""

            modules.append(m)
        # config["shortcuts"].extend(se_shortcuts)
        config["modules"] = modules

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    try:
        pruner = SlimPruner(model, args.save_path)
        pruner.run(0.6)
        pruner.pruned_model.eval()

        print("Summary of pruned_model")
        summary_model(pruner.pruned_model)

        print("Run forward on pruned model")
        x = torch.Tensor(1, 3, 224, 224)
        pruner.pruned_model(x)
    except Exception as e:
        import traceback

        traceback.print_exc()
        exit(-1)

    print("Schema is ok")
