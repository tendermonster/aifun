import torch
import random
import torch.nn as nn


def make_layer(in_planes, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append([in_planes, planes, stride])
        in_planes = planes
    # return nn.Sequential(*layers)
    print(layers)


if __name__ == "__main__":
    a = [(1, 2), (3, 4), (5, 6)]
    print(a)
    random.shuffle(a)
    print(a)
