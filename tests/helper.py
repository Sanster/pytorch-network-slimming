import json
import tempfile

import torch
from pns import SlimPruner
from pns.schema_post_process import mbv3_large_schema_post_process
from pns.tracker import gen_pruning_schema
from torch import Tensor


def check_gen_schema(model, net: str = "", in_channel: int = 3):
    x = torch.Tensor(1, in_channel, 224, 224)
    y = model(x)
    if isinstance(y, Tensor):
        print(f"model ouput shape: {y.shape}")
    elif isinstance(y, dict):
        for k, v in y.items():
            print(f"{k} {v.shape}")

    config = gen_pruning_schema(model, x)

    if "mobilenet_v3_large" in net and "nose" not in net:
        mbv3_large_schema_post_process(config)

    with tempfile.NamedTemporaryFile("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.flush()
        pruner = SlimPruner(model, f.name)
        pruner.run(0.6)
        pruner.pruned_model.eval()
        x = torch.Tensor(1, in_channel, 224, 224)
        y = pruner.pruned_model(x)
        print(y)
