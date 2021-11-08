import torch

from helper import check_gen_schema


class AddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
        self.bn2 = torch.nn.BatchNorm2d(3)

        self.conv3 = torch.nn.Conv2d(3, 3, 1)
        self.bn3 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))

        x = x1 + x2
        x = self.bn3(self.conv3(x))
        return x


def test_add():
    check_gen_schema(AddModel())
