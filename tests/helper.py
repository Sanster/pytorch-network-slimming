import json
import tempfile

import torch
from pns import SlimPruner
from pns.tracker import gen_pruning_schema


def check_gen_schema(model):
    x = torch.Tensor(1, 3, 224, 224)
    config = gen_pruning_schema(model, x)

    with open("test.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.flush()
        pruner = SlimPruner(model, f.name)
        pruner.run(0.6)
        pruner.pruned_model.eval()
        x = torch.Tensor(1, 3, 224, 224)
        pruner.pruned_model(x)
