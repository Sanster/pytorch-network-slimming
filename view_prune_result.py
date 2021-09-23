import torch
import typer
from pns import SlimPruner
from pns.functional import summary_model

from backbone.build import build_model


def main(net: str, ckpt: str):
    model = build_model(net)
    pruner = SlimPruner(model)
    checkpoint = torch.load(ckpt)
    for it in checkpoint[SlimPruner.PRUNING_RESULT_KEY]:
        if it["name"].startswith("model."):
            it["name"] = it["name"].replace("model.", "")

    pruner.apply_pruning_result(checkpoint[SlimPruner.PRUNING_RESULT_KEY])

    summary_model(pruner.pruned_model)


if __name__ == "__main__":
    typer.run(main)
