import torch
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
    # Create some example data
    input = torch.randn(3, 5, requires_grad=True)
    print(input)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(target)
    num_classes = 10
    batch_size = 1

    # Simulated predicted logits and target labels
    logits = torch.randn(batch_size, num_classes)
    print(logits)
    target_labels = torch.randint(0, num_classes, (batch_size,))
    print(target_labels)
    print(target_labels.dtype)

    # Instantiate the CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Calculate the loss
    loss = criterion(logits, target_labels)

    print("Loss:", loss.item())
