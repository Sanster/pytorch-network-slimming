import copy
import json
from functools import reduce
from itertools import chain
from typing import Dict, List

import pandas as pd
import torch
from torch.nn import Conv2d, BatchNorm2d, Linear

from .functional import (
    cal_threshold_by_bn2d_weights,
    mask2idxes,
    top_k_idxes,
    prune_conv2d,
    prune_fc,
    prune_bn2d,
)

SHORTCUTS_MERGE_OR = "or"
SHORTCUTS_MERGE_AND = "and"


class Conv2dWrapper:
    def __init__(self, module, name, prev_bn_name=None, next_bn_name=None):
        self.module: Conv2d = copy.deepcopy(module)
        self.pruned_module: Conv2d = module
        self.name = name
        self.prev_bn_name = prev_bn_name
        self.next_bn_name = next_bn_name

        self.is_pruned = False
        self.in_channels_keep_idxes = None
        self.out_channels_keep_idxes = None

    def prune(self, prev_bn=None, next_bn=None):
        if prev_bn is not None:
            assert prev_bn.is_pruned
        if next_bn is not None:
            assert next_bn.is_pruned

        self.in_channels_keep_idxes, self.out_channels_keep_idxes = prune_conv2d(
            self.pruned_module,
            prev_bn.keep_idxes if prev_bn else None,
            next_bn.keep_idxes if next_bn else None,
        )

        self.is_pruned = True

    def prune_info(self):
        in_channels = self.module.in_channels
        out_channels = self.module.out_channels
        pruned_in_channels = self.pruned_module.in_channels
        pruned_out_channels = self.pruned_module.out_channels

        return {
            "name": self.name,
            "weight shape": f"[{in_channels},{out_channels}] g={self.module.groups}",
            "pruned weight shape": f"[{pruned_in_channels},{pruned_out_channels}] g={self.pruned_module.groups}",
            "prune percent": f"{100 - (pruned_in_channels * pruned_out_channels) / (in_channels * out_channels) * 100:.2f}%",
        }

    def prune_result(self):
        return {
            "name": self.name,
            "in_channels_keep_idxes": self.in_channels_keep_idxes,
            "out_channels_keep_idxes": self.out_channels_keep_idxes,
        }


class LinearWrapper:
    def __init__(self, module, name, prev_bn_name=None):
        self.module = copy.deepcopy(module)
        self.pruned_module: Linear = module
        self.name = name
        self.prev_bn_name = prev_bn_name
        self.is_pruned = False
        self.in_features_keep_idxes = None

    def prune(self, prev_bn):
        """
        Args:
            prev_bn:

        Returns:

        """
        self.in_features_keep_idxes = prune_fc(
            self.pruned_module, prev_bn.keep_idxes, prev_bn.module.num_features
        )
        self.is_pruned = True

    def prune_info(self):
        in_features = self.module.in_features
        pruned_in_features = self.pruned_module.in_features
        return {
            "name": self.name,
            "in_features": f"{pruned_in_features}/{in_features}",
            "prune percent": f"{100 - pruned_in_features/in_features*100:.2f}",
        }

    def prune_result(self):
        return {
            "name": self.name,
            "in_features_keep_idxes": self.in_features_keep_idxes,
        }


class BN2dWrapper:
    def __init__(self, module: BatchNorm2d, name):
        self.module = copy.deepcopy(module)
        self.pruned_module = module
        self.name = name
        self.keep_idxes = None
        self.is_pruned = False

    @property
    def is_idxes_calculated(self):
        return self.keep_idxes is not None

    def in_channels(self) -> int:
        return self.module.num_features

    def cal_keep_idxes(self, threshold: float, min_keep_ratio: float = 0):
        """
        根据所有 BatchNorm2d 层的 gamma 系计算每层 bn 的 keep_idxes
        """
        mask = self.module.weight.data.abs().gt(threshold).cpu().numpy()
        idxes = mask2idxes(mask)

        if len(idxes) == 0:
            if min_keep_ratio == 0:
                raise RuntimeError("")
            else:
                idxes = top_k_idxes(self.module, min_keep_ratio)

        self.keep_idxes = idxes

    def set_fixed_ratio(self, keep_ratio: float = 1):
        idxes = top_k_idxes(self.module, keep_ratio)
        self.keep_idxes = idxes

    def prune(self):
        assert self.keep_idxes is not None
        prune_bn2d(self.pruned_module, self.keep_idxes)
        self.is_pruned = True

    def prune_info(self):
        str1 = f"{len(self.keep_idxes)}/{self.in_channels()}"
        str2 = (
            f"{(self.in_channels()-len(self.keep_idxes))/self.in_channels()*100:.2f}%"
        )
        return {"name": self.name, "channels": str1, "prune percent": str2}

    def prune_result(self):
        return {"name": self.name, "keep_idxes": self.keep_idxes}


class SlimPruner:
    PRUNING_RESULT_KEY = "_slim_pruning_result"

    def __init__(self, model, schema: str = None):
        """

        Args:
            model: Module will be modified inplace
            schema: pruning schema
        """
        self.pruned_model = model

        if schema is not None:
            if isinstance(schema, str):
                with open(schema, "r") as f:
                    schema = json.loads(f.read())

            self._add_prefix_to_config_name(schema)

            modules = {}
            for it in schema["modules"]:
                modules[it["name"]] = it

            self.conv2d_modules = {}
            self.bn2d_modules = {}
            self.fc_modules = {}
            self.shortcuts = schema.get("shortcuts", [])
            self.depthwise_conv_adjacent_bn = schema.get(
                "depthwise_conv_adjacent_bn", []
            )
            self.fixed_bn_ratio = schema.get("fixed_bn_ratio", [])

            for name, module in self.pruned_model.named_modules():
                if isinstance(module, Conv2d):
                    self.conv2d_modules[name] = Conv2dWrapper(
                        module,
                        modules[name]["name"],
                        prev_bn_name=modules[name].get("prev_bn", ""),
                        next_bn_name=modules[name].get("next_bn", ""),
                    )
                elif isinstance(module, Linear):
                    if name in modules:
                        self.fc_modules[name] = LinearWrapper(
                            module, name, prev_bn_name=modules[name].get("prev_bn", "")
                        )
                elif isinstance(module, BatchNorm2d):
                    self.bn2d_modules[name] = BN2dWrapper(module, name)

    def _add_prefix_to_config_name(self, config):
        prefix = config.get("prefix", "")
        for it in config["modules"]:
            it["name"] = prefix + it["name"]
            if "prev_bn" in it and it["prev_bn"]:
                it["prev_bn"] = prefix + it["prev_bn"]
            if "next_bn" in it and it["next_bn"]:
                it["next_bn"] = prefix + it["next_bn"]
        """
        "shortcuts": [
            {
                "names": [
                    "bn1",
                    "layer1.0.bn2",
                    "layer1.1.bn2"
                ],
                "method": "or"
            }
        ]
        """
        for it in config.get("shortcuts", []):
            it["names"] = [prefix + _ for _ in it["names"]]
        for it in config.get("depthwise_conv_adjacent_bn", []):
            it["names"] = [prefix + _ for _ in it["names"]]

    def run(self, ratio: float):
        threshold = cal_threshold_by_bn2d_weights(
            [it.module for it in self.bn2d_modules.values()], ratio
        )

        bn2d_prune_info = []
        for name, bn2d in self.bn2d_modules.items():
            bn2d.cal_keep_idxes(threshold, min_keep_ratio=0.02)

        self._apply_fix_bn_ratio()
        self._merge_shortcuts()
        self._merge_depthwise_conv2d_adjacent_bn()

        for bn2d in self.bn2d_modules.values():
            bn2d.prune()
            bn2d_prune_info.append(bn2d.prune_info())

        df = pd.DataFrame(bn2d_prune_info)
        print("\nBatchNorm2d prune info")
        print(df.to_markdown() + "\n")

        conv2d_prune_info = []
        for conv2d in self.conv2d_modules.values():
            conv2d.prune(
                self.bn2d_modules[conv2d.prev_bn_name] if conv2d.prev_bn_name else None,
                self.bn2d_modules[conv2d.next_bn_name] if conv2d.next_bn_name else None,
            )
            conv2d_prune_info.append(conv2d.prune_info())
        df = pd.DataFrame(conv2d_prune_info)
        print("\nConv2d prune info")
        print(df.to_markdown() + "\n")

        fc_prune_info = []
        for linear in self.fc_modules.values():
            if not linear.prev_bn_name:
                continue
            linear.prune(self.bn2d_modules[linear.prev_bn_name])
            fc_prune_info.append(linear.prune_info())
        if len(fc_prune_info):
            df = pd.DataFrame(fc_prune_info)
            print("\nLinear prune info")
            print(df.to_markdown() + "\n")

        return self._export_pruning_result()

    def apply_pruning_result(self, pruning_result: List[Dict]):
        info = {it["name"]: it for it in pruning_result}

        for name, module in self.pruned_model.named_modules():
            if name not in info:
                continue

            if isinstance(module, Conv2d):
                prune_conv2d(
                    module,
                    info[name]["in_channels_keep_idxes"],
                    info[name]["out_channels_keep_idxes"],
                )

            elif isinstance(module, Linear):
                prune_fc(module, info[name]["in_features_keep_idxes"])
            elif isinstance(module, BatchNorm2d):
                prune_bn2d(module, info[name]["keep_idxes"])

    def _export_pruning_result(self):
        prune_result = []
        for it in chain(
            self.bn2d_modules.values(),
            self.conv2d_modules.values(),
            self.fc_modules.values(),
        ):
            if it.is_pruned:
                prune_result.append(it.prune_result())

        # return {self.PRUNING_RESULT_KEY: prune_result}
        return prune_result

    def _merge_depthwise_conv2d_adjacent_bn(self):
        self._align_bns(
            self.depthwise_conv_adjacent_bn,
            min_keep_ratio=0.05,
            log_name="depthwise conv bn",
        )

    def _merge_shortcuts(self):
        self._align_bns(self.shortcuts, min_keep_ratio=0.05, log_name="shortcuts")

    def _align_bns(self, bn_groups, min_keep_ratio: float, log_name: str):
        """
        bn layer is changed inplace
        [
            {
                "names": ["bn1", "bn2"]
                "method": "or" / "and"
            }
        ]

        Returns:

        """
        merged = []
        for i, shortcuts in enumerate(bn_groups):
            assert "method" in shortcuts
            assert shortcuts["method"] in [SHORTCUTS_MERGE_OR, SHORTCUTS_MERGE_AND]

            merge_func = {
                SHORTCUTS_MERGE_AND: lambda x, y: set(x) & set(y),
                SHORTCUTS_MERGE_OR: lambda x, y: set(x) | set(y),
            }[shortcuts["method"]]

            bn2d_layers = []
            bn2d_names = shortcuts["names"]
            print(f"============{log_name} [{i}]===========")
            for bn2d_name in bn2d_names:
                assert (
                    bn2d_name in self.bn2d_modules
                ), f"{bn2d_name} is not a BatchNorm2d layer"
                # 防止 bn 出现在多个 shortcuts 中
                assert bn2d_name not in merged

                bn2d = self.bn2d_modules[bn2d_name]
                assert bn2d.is_idxes_calculated
                bn2d_layers.append(bn2d)
                merged.append(bn2d_name)
                print(f"{bn2d_name}: {len(bn2d.keep_idxes)}")

            merged_idxes = list(
                reduce(
                    merge_func,
                    [it.keep_idxes for it in bn2d_layers],
                )
            )
            print(f"merged indexes length: {len(merged_idxes)}")

            for bn2d in bn2d_layers:
                _merged_idxes = merged_idxes
                if len(merged_idxes) == 0:
                    _merged_idxes = top_k_idxes(bn2d.module, min_keep_ratio)
                bn2d.keep_idxes = _merged_idxes

    def _apply_fix_bn_ratio(self):
        for it in self.fixed_bn_ratio:
            name = it["name"]
            ratio = it["ratio"]
            assert 0 < ratio < 1
            if isinstance(name, str):
                name = [name]
            for _name in name:
                if _name in self.bn2d_modules:
                    self.bn2d_modules[_name].set_fixed_ratio(1 - ratio)
