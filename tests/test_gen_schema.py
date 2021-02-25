import torch

from helper import check_gen_schema


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 1)
        self.bn2 = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.conv1(x)
        x_bn1_out = self.bn1(x)
        x = torch.nn.functional.relu(x_bn1_out)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_bn1_out
        return x


def test_nn_functional_relu():
    check_gen_schema(Model1())
