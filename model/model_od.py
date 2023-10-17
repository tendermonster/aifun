from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import typing

if typing.TYPE_CHECKING:
    from torch import Tensor
    from typing import Tuple
# resnet usually consists of 4 residual blocks

# resnet34 consists of [6,8,12,6] meaning(6*3+8*3+12*3+6*3)
# residual blocks aswell as input cnn layer aswell as last classification layer


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = ClassificationHead(
            in_channels, num_anchors, num_classes
        )
        self.objectiveness_head = ObjectivenessHead(in_channels, num_anchors)
        self.bbox_head = BboxHead(in_channels, num_anchors)

    def forward(self, x) -> tuple[Tensor, Tensor, Tensor]:
        # for now we need to hardcode the input params
        return (
            self.classification_head(x),
            self.objectiveness_head(x),
            self.bbox_head(x),
        )


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.reshape = Rearrange("b (c p) h w -> b c h w p", c=num_anchors)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.reshape(x)
        return x


class ObjectivenessHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.reshape = Rearrange("b (c p) h w -> b c h w p", c=num_anchors)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.reshape(x)
        return x


class BboxHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        # number of real numbers = 4
        # p is x_min, y_min, x_max, y_max, confidence
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.reshape = Rearrange("b (c p) h w -> b c h w p", c=num_anchors)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.reshape(x)
        return x


# this block consist of 2 conv layers
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        # change dimensionality of in_channels != out_channels or stride != 1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # BasicBlock forward
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    in_channels = 64
    name = "ResNet"

    def __init__(self, block, num_blocks) -> None:
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        # first block sometimes have different stride
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # some dirty workaround to collect intermediate outputs
        intermediate = []
        out = self.layer1(out)  # [-1, 64, 32, 32 ]
        intermediate.append(out)
        out = self.layer2(out)  # [-1, 128, 16, 16]
        intermediate.append(out)
        out = self.layer3(out)  # [-1, 256, 8, 8]
        intermediate.append(out)
        out = self.layer4(out)  # [-1, 512, 4, 4]
        intermediate.append(out)
        # out = F.avg_pool2d(out, 4)  # [-1, 512, 1, 1]
        # out = out.view(
        #     out.shape[0], -1
        # )  # the size -1 is inferred from other dimensions
        return intermediate


class ModelWithHead(nn.Module):
    def __init__(self, model: ResNet, head: DetectionHead):
        super().__init__()
        self.name = ""
        self.model = model
        self.head = head

    def forward(self, x):
        x = self.model(x)
        x = self.head(x[0])
        return x


def ResNet18() -> ModelWithHead:
    r = ResNet(BasicBlock, [2, 2, 2, 2])
    print(get_num_parameters(r))
    model = ModelWithHead(r, DetectionHead(64, 3, 10))
    model.name = "ResNet18"
    return model


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


def get_num_parameters(model, include_grad=False) -> int:
    if include_grad:
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


if __name__ == "__main__":
    model: ModelWithHead = ResNet18()
    print(get_num_parameters(model, include_grad=False))
    x: Tensor = torch.randn(4, 3, 224, 224)

    out: Tuple[Tensor, Tensor, Tensor] = model(x)
    # out = torch.max(out, 1)
    print(out)
    print(type(out))
    # print class prediction
    cls_pred = out[0]
    print(cls_pred.shape)
    # print objectiveness prediction
    obj_pred = out[1]
    print(obj_pred.shape)
    # print bbox prediction
    bbox_pred = out[2]
    print(bbox_pred.shape)
