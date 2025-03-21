import torch
import torch.nn as nn
import torch.nn.functional as F

# resnet usually consists of 4 residual blocks

# resnet34 consists of [6,8,12,6] meaning(6*3+8*3+12*3+6*3)
# residual blocks aswell as input cnn layer aswell as last classification layer


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
    # lets use the 224x224 imagenet input size
    in_channels = 64
    name = "ResNet"

    def __init__(self, block, num_blocks, num_classes=10) -> None:
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512, num_classes)
        # self.softmax = nn.Softmax(dim=1)

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
        # print(out.shape)
        out = self.layer1(out)  # [-1, 64, 224, 224]
        # print(out.shape)
        out = self.layer2(out)  # [-1, 128, 112, 112]
        # print(out.shape)
        out = self.layer3(out)  # [-1, 256, 56, 56]
        # print(out.shape)
        out = self.layer4(out)  # [-1, 512, 28, 28]
        # print(out.shape)
        out = F.avg_pool2d(out, out.shape[-1])  # [-1, 512, 1, 1]
        out = out.view(
            out.shape[0], -1
        )  # the size -1 is inferred from other dimensions
        out = self.linear(out)
        # out = self.softmax(out)
        return out


def ResNet18():
    r = ResNet(BasicBlock, [2, 2, 2, 2])
    r.name = "ResNet18"
    return r


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


if __name__ == "__main__":
    model = ResNet18()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    out = torch.max(out, 1)
    print(out)
    print(type(out))
