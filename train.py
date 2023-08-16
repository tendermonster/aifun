import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
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
    test = traindata[30000:45000]
    val = traindata[45000:]

    # Initialize a DataLoader
    batch_size = 32
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    net = ResNet18()
    if torch.cuda.is_available():
        print("NVIDIA GPU is available!")
        net = net.to("cuda")
    else:
        print("NVIDIA GPU is not available.")
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()  # setting this changes behaviour of some modules like e.g dropout and batchnorm
    save = True
    # todo make batch size
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            # todo make the thing with
            input, targets = data
            if input.dim() == 3:
                input = input.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            if torch.cuda.is_available():
                input = input.to("cuda")
                targets = targets.to("cuda")
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if i >= int(2000 / batch_size):  # print every 20 mini-batches
                print(f"minibatch {int(2000 / batch_size)}")
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / int(2000 / batch_size):.8f}"
                )
                print(f"Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0

    print("Finished Training")

    if save:
        torch.save(net.state_dict(), "model.pth")
