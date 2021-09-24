## Pytorch Network Slimming

This repository contains tools to make implement
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) on different backbones easier.

## Features

- [x] Auto generate pruning schema. [Tracker](https://github.com/Sanster/pytorch-network-slimming/blob/master/src/pns/tracker.py) code is inspired by [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [x] Channel pruning
- [x] Save and load pruned network graph and params (without pruning schema)
- [x] Export pruned model to onnx
- [x] Channels round up to 8x (for train/infer speed) (TODO: handle shortcuts channels round up)
- [ ] Layer pruning

Supported model arches:
- ResNet
- RepVGG
- vgg_bn
- mobilenet_v2
- mobilenet_v3([doc](./docs/mobilenet_v3.md))


Supported layers:
- Conv2d
- depthwise/pointwise Conv2d
- SqueezeExcitation(SE block)
- Linear
- BatchNorm2d

## Quick start on cifar10

1. Install pns as a python package: `python3 setup.py develop`
2. Install `requirements.txt`, this is the dependency needed to train the cifar10 demo,
   pns itself only depends on pytorch

3. Generate pruning schema of `resnet18`:

   ```bash
   python3 gen_schema.py --net resnet18 --save_path ./schema/resnet18.json
   ```

4. Train on cifar10，and do fine tuning after apply Network Slimming.

   ```bash
   python3 main.py \
   --save_dir output \
   --dataset cifar10 \
   --net resnet18 \
   --epochs 120 \
   --batch_size 64 \
   --learning_rate 0.01 \
   --sparsity_train \
   --s 0.0001 \
   --fine_tune \
   --prune_schema ./schema/resnet18.json \
   --fine_tune_epochs 120 \
   --fine_tune_learning_rate 0.001 \
   --prune_ratio 0.75
   ```

5. After apply Network Slimming, pruning result will be saved in checkpoint with `_slim_pruning_result` key and pruned params will be saved in another checkpoint.

Eval model without pruning result:

```bash
python3 main.py \
--dataset cifar10 \
--net resnet18 \
--ckpt ./output/last.ckpt
```

Eval model with pruning result:

```bash
python3 main.py \
--dataset cifar10 \
--net resnet18 \
--ckpt ./output/pruned_0.75/model_with_pruning_result.ckpt \
--ckpt_pruned ./output/pruned_0.75/last.ckpt
```

Export pruned model to ONNX

```bash
python3 main.py \
--net resnet18 \
--ckpt ./output/pruned_0.75/model_with_pruning_result.ckpt \
--ckpt_pruned ./output/pruned_0.75/last.ckpt \
--export_onnx_path ./output/pruned_0.75/last.onnx
```

Eval ONNX model(demo script only support CPU)
```bash
python3 main.py \
--dataset cifar10 \
--net resnet18 \
--ckpt ./output/pruned_0.75/last.onnx \
--device cpu
```

Benchmark onnx model performance
```bash
python3 benchmark.py \
/path/to/onnx_model1.onnx \
/path/to/onnx_model2.onnx
```

## Experiments Result on CIFAR10

- batch_size: 64
- epochs: 120
- sparsity: 1e-4
- learning rate: 0.01
- fine tune epochs: 120
- fine tune learning rate: 0.01
- onnx latency: input shape=1x3x224x224, OMP_NUM_THREADS=4, cpu=Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz

|     | Net            | Prune Ratio | Test Acc | Test Acc Diff | Params | Params Reduce | ONNX File size(MB) | ONNX Latency | ONNX Memory |
| --: | :------------- | ----------: | -------: | ------------: | :----- | :---------- | :----------------- | :-------------------| :----|
|  0 | RepVGG-A0-woid   |          0    |      93.6  |          0    | 7.8 M    |               |
|  1 | RepVGG-A0-woid   |          0.75 |      93.88 |            +0.28 | 2.2 M    | 71.78%        |
|  2 | RepVGG-A0-woid   |          0.5  |      93.86 |            +0.26 | 3.8 M    | 52.14%        |
|   3 | resnet18       |           0 |    94.48 |             0 | 11.2 M |             |
|   4 | resnet18       |        0.75 |    94.14 |         -0.34 | 4.5 M  | 59.29%      |
|   5 | resnet18       |         0.5 |     94.8 |         +0.32 | 3.5 M  | 68.83%      |
|   6 | resnet50       |           0 |    94.65 |             0 | 23.5 M |             |
|   7 | resnet50       |        0.75 |    95.29 |         +0.64 | 5.3 M  | 77.59%      |
|   8 | resnet50       |         0.5 |    95.42 |         +0.77 | 14.8 M | 37.04%      |
|   9 | vgg11_bn       |           0 |     91.7 |             0 | 128 M  |             |
|  10 | vgg11_bn       |        0.75 |    89.85 |         -1.85 | 28.9 M | 77.53%      |
|  11 | vgg11_bn       |         0.5 |    91.46 |         -0.24 | 58.5 M | 54.58%      |
|  12 | mobilenet_v2   |        0    |    94.52 |         0     | 2.2M   |             | [8.5](https://github.com/Sanster/models/raw/master/pytorch-network-slimming/mobilenet_v2_s_0.0001.onnx) |
|  13 | mobilenet_v2   |       0.75  |    91.17 |         -3.35 | 661K   | 70.41%      | [2.6](https://github.com/Sanster/models/raw/master/pytorch-network-slimming/mobilenet_v2_s_0.0001_0.75.onnx) |
|  14 | mobilenet_v2(s=1e-5)  |       0    |    94.42 |         0     | 2.2M   |             | [8.5](https://github.com/Sanster/models/raw/master/pytorch-network-slimming/mobilenet_v2_s_0.00001.onnx) |
|  15 | mobilenet_v2(s=1e-5)  |      0.75  |    93.12 |         -1.3  | 597k   | 73.30%      | [2.3](https://github.com/Sanster/models/raw/master/pytorch-network-slimming/mobilenet_v2_s_0.00001_0.75.onnx) |
|  16 | mobilenet_v3_large_nose |          0    |      92.61 |          0      | 2.7 M    |               |
|  18 | mobilenet_v3_large_nose |          0.5  |      93.33 |           +0.72 | 1.9 M    | 30.09%        |
|  17 | mobilenet_v3_large_nose |          0.75 |      91.42 |           -1.19 | 1.4 M    | 48.33%        |
|  19 | mobilenet_v3_small_nose |          0    |      90.69 |          0      | 1.1 M    |               |
|  20 | mobilenet_v3_small_nose |          0.5  |      91.08 |           +0.39 | 777 K    | 27.11%        |
|  21 | mobilenet_v3_small_nose |          0.75 |      87.25 |           -3.44 | 564 K    | 47.11%        |
|  22 | mobilenet_v3_large      |          0    |      92.96 |          0      | 4.2 M    |               | [16](https://github.com/Sanster/models/blob/master/pytorch-network-slimming/mobilenet_v3_large_s_0.0001.onnx)  | 100ms | 165mb |
|  23 | mobilenet_v3_large      |          0.75 |      92.18 |           -0.78 | 1.6 M    | 63.12%        | [5.9](https://github.com/Sanster/models/blob/master/pytorch-network-slimming/mobilenet_v3_large_s_0.0001_0.75.onnx) | 65ms | 151mb |
|  24 | mobilenet_v3_large      |          0.5  |      92.87 |           -0.09 | 2.3 M    | 45.57%        | [8.7](https://github.com/Sanster/models/blob/master/pytorch-network-slimming/mobilenet_v3_large_s_0.0001_0.5.onnx) | 81ms | 123mb |



- for RepVGG-A0-woid(prune ratio 0.7), fine tune learning rate = 0.001
- woid：RepVGGBlock without identity layer
- nose: MobileNetV3 without SE block

Experiments result without sparsity train + prune:

|     | Net      | Sparsity | Prune Ratio | Test Acc | Test Acc Diff | Params | Size Reduce |
| --: | :------- | -------: | ----------: | -------: | ------------: | :----- | :---------- |
|   0 | resnet18 |        0 |           0 |    93.65 |             0 | 11.2 M |             |
|   1 | resnet18 |        0 |        0.75 |    91.07 |         -2.58 | 389 K  | 96.52%      |

TODO: Understand why the size of the model is reduced so much?

## How to use pns in your project

1. Understand the content of this paper [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)，install pns by run `python3 setup.py install`
2. Refer to the `gen_schema.py` script to generate the pruning schema. You may need to implement your own `build_model` section
3. Training: call `update_bn_grad` after `loss.backward()`

```python
...
loss.backward()
update_bn_grad(model, s=0.0001)
...
```

4. Fine tune:

   - Restore model weights according to the normal flow of your project
   - Pruning according to the `prune_schema` of your network
   - Save pruning result anywhere you like
   - Fine tune pruned model

   ```python
   pruner = SlimPruner(restored_model, prune_schema)
   pruning_result: List[Dict] = pruner.run(prune_ratio=0.75)
   # save pruning_result
   ...
   # fine tune pruned_model
   pruner.pruned_model
   ...
   # save pruned_model state_dict()
   torch.save(pruner.pruned_model.state_dict())
   ```

5. Loading pruning result/params when do forward or pruning again:

```python
#  build model
pruner = SlimPruner(model)
# load pruning_result from some where to get a slim network
pruning_result: List[Dict]
pruner.apply_pruning_result(pruning_result)
# load pruned state_dict from some where
pruned_state_dict: Dict
pruner.pruned_model.load_state_dict(pruned_state_dict)
# do forward or train with update_bn_grad again
pruner.pruned_model
```

## Pruning Schema

```json
{
  "prefix": "model.",
  "channel_rounding": "eight",
  "modules": [
    {
      "name": "conv1",
      "prev_bn": "",
      "next_bn": "bn1"
    }
  ],
  "shortcuts": [
    {
      "names": ["bn1", "layer1.0.bn2", "layer1.1.bn2"],
      "method": "or"
    }
  ],
  "depthwise_conv_adjacent_bn": [
    {
      "names": ["bn1", "layer1.0.bn2", "layer1.1.bn2"],
      "method": "or"
    }
  ],
  "fixed_bn_ratio": [
    {
      "name": "name1",
      "ratio": 0.8
    },
    {
      "name": ["name2"],
      "ratio": 0.8
    }
  ]
}
```

- prefix: common prefix added to all module name
- channel_rounding: none/eight/two_pow
- modules: Conv2d or Linear layers
- shortcuts/depthwise_conv_adjacent_bn: BatchNorm2d Layers
  - or: All bn layer reserved channels take the merged set
  - and: All bn layer reserved channels take the intersection set
- fixed_bn_ratio: BatchNorm2d Layers fix prune percent, will applied before merge shortcuts
  - name: string or List[string]
  - ratio: prune ratio

## Development
Run test
```bash
pytest -v tests
```
