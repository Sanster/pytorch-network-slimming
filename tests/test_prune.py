import json
import os
import sys
from pathlib import Path

import torch
from pns.tracker import gen_pruning_schema

from helper import check_gen_schema
from pns import SlimPruner


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
SCHEMA_DIR = CURRENT_DIR / "schema"


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_fixed_bn_ratio():
    model = Model1()

    pruner = SlimPruner(model, str(SCHEMA_DIR / "model1.json"))
    pruner.run(0.6)
    pruner.pruned_model.eval()
    x = torch.Tensor(1, 3, 224, 224)
    pruner.pruned_model(x)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))
