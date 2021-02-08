import torch
from torchvision import models
from .repvgg import func_dict as repvgg


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
    else:
        raise NotImplementedError(f"{net}")

    return model
