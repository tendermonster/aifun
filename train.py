import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import random

from model.model import ResNet18
from tools.load_cifar import CIFAR10

# todo swap loss function
# todo swap optimizer
# todo

if __name__ == "__main__":
    cifar = CIFAR10()
    traindata = list(zip(cifar.data, cifar.labels))
    # randomize data
    random.shuffle(traindata)

    train = traindata[:30000]
    test = traindata[30000:40000]
    val = traindata[40000:]

    net = ResNet18()
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    save = True
    # todo make batch size
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train):
            # todo make the thing with
            input, label = data
            if input.dim() == 3:
                input = input.unsqueeze(0)
            if label.dim() == 0:
                label = label.unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    if save:
        torch.save(net.state_dict(), "model.pth")
