import torch
from torchvision import models
from .repvgg import func_dict as repvgg
from .mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large


def build_model(net, num_classes=10):
    if net in ["resnet18", "resnet34", "resnet50"]:
        model = getattr(models, net)(num_classes=num_classes)
        # to get better result on cifar10
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
    elif net in ["vgg11_bn"]:
        model = models.vgg11_bn(num_classes=num_classes)
    elif net in repvgg:
        model = repvgg[net](num_classes=num_classes)
    elif net in ["shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
        model = getattr(models, net)(num_classes=num_classes)
        model.maxpool = torch.nn.Identity()
    elif net in ["mobilenet_v2"]:
        # to get better result on cifar10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        model = getattr(models, net)(
            num_classes=num_classes, inverted_residual_setting=inverted_residual_setting
        )
        model.features[0][0].stride = (1, 1)
    elif net in ["mobilenet_v3_large_nose", "mobilenet_v3_small_nose"]:
        mbv3 = {
            "mobilenet_v3_large": mobilenet_v3_large,
            "mobilenet_v3_small": mobilenet_v3_small,
        }
        no_se = True if "nose" in net else False
        model = mbv3[net.replace("_nose", "")](num_classes=num_classes, no_se=no_se)
    else:
        raise NotImplementedError(f"{net}")

    return model
