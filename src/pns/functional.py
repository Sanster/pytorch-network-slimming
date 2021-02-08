import copy
from typing import List

import torch
from torch.nn import BatchNorm2d, Conv2d, Linear
import numpy as np
import pandas as pd


def update_bn_grad(model, s=0.0001):
    """
    根据 BN 的 gamma 系数稀疏化训练 BN 层
    在 loss.backward() 之后执行

    Args:
        model: nn.Module
        s: 系数训练的权重

    Returns:

    """
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))


def summary_model(
    model: torch.nn.Module, prune_related_layer_types=(Conv2d, BatchNorm2d, Linear)
):
    """
    打印 model 中和剪枝有关的层
    """
    info = []
    for name, module in model.named_modules():
        if type(module) in prune_related_layer_types:
            info.append({"name": name, "module": module})

    df = pd.DataFrame(info)
    df = df.reindex(columns=["name", "module"])
    print(df.to_markdown())


def prune_bn2d(module: BatchNorm2d, keep_idxes):
    module.num_features = len(keep_idxes)
    module.weight.data = module.weight.data[keep_idxes]
    module.weight.grad = None
    module.bias.data = module.bias.data[keep_idxes]
    module.bias.grad = None
    module.running_mean = module.running_mean[keep_idxes]
    module.running_var = module.running_var[keep_idxes]


def prune_conv2d(module: Conv2d, in_keep_idxes=None, out_keep_idxes=None):
    if in_keep_idxes is None:
        in_keep_idxes = list(range(module.weight.shape[1]))

    if out_keep_idxes is None:
        out_keep_idxes = list(range(module.weight.shape[0]))

    assert len(in_keep_idxes) <= module.weight.shape[1]
    assert len(out_keep_idxes) <= module.weight.shape[0]

    module.out_channels = len(out_keep_idxes)
    module.in_channels = len(in_keep_idxes)

    module.weight.data = module.weight.data[out_keep_idxes, :, :, :]
    module.weight.data = module.weight.data[:, in_keep_idxes, :, :]
    module.weight.grad = None

    if module.bias is not None:
        module.bias.data = module.bias.data[out_keep_idxes]
        module.bias.grad = None

    return in_keep_idxes, out_keep_idxes


def prune_fc(module: Linear, keep_idxes: List[int], bn_num_channels: int = None):
    """

    Args:
        module:
        keep_idxes:
        bn_num_channels: prev bn num_channels

    Returns:

    """
    if bn_num_channels is not None:
        assert module.in_features % bn_num_channels == 0

        channel_step = module.in_features // bn_num_channels

        _keep_idxes = []
        for idx in keep_idxes:
            _keep_idxes.extend(
                np.asarray(list(range(channel_step))) + idx * channel_step
            )

        keep_idxes = _keep_idxes

    module.in_features = len(keep_idxes)
    module.weight.data = module.weight.data[:, keep_idxes]
    module.weight.grad = None
    return keep_idxes


def cal_threshold_by_bn2d_weights(bn2d_list: List[BatchNorm2d], sparsity: float):
    """
    sparsity: 要剪枝的比例
    """
    assert 0 < sparsity < 1

    bn_weight_list = []
    for module in bn2d_list:
        bn_weight_list.append(module.weight.data.cpu().abs().clone())

    bn_weights = torch.cat(bn_weight_list)
    k = int(bn_weights.shape[0] * sparsity)

    sorted_bn = torch.sort(bn_weights)[0]
    thresh = sorted_bn[k]
    return thresh


def mask2idxes(mask):
    idxes = np.squeeze(np.argwhere(mask))
    if idxes.size == 1:
        idxes = np.resize(idxes, (1,))
    return idxes


def top_k_idxes(module, ratio):
    weights = module.weight.data.abs().clone()
    k = max(int(weights.shape[0] * ratio), 2)
    idxes = torch.topk(weights.view(-1), k, largest=True)[1]
    return idxes.cpu().numpy()
