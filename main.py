import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from model.model import ResNet18
from tools.load_cifar import CIFAR10
from tools.load_model import ModelLoader
from typing import Tuple
from tools.logger import Logger


import os
import time

# todo swap loss function
# todo swap optimizer
# todo


class Trainer:
    criterion = CrossEntropyLoss()
    root = "checkpoints"

    def __init__(self, net, train_set, test_set, val_set, device, logger=None):
        if logger is None:
            self.log = Logger()
        else:
            self.log = logger
        self.save_dir = os.path.join(
            self.root, net.name + "_" + self.log.get_timestamp()
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.net = net
        self.net.train()
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.device = device
        lr = 0.0001
        # weight_decay is L2 regularization
        # self.optimizer = SGD(net.parameters(), lr=0.07, momentum=0.9, weight_decay=5e-3)
        self.optimizer = Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
        self.log.log(f"Using optimizer: {self.optimizer} with start lr: {lr}")
        self.best_accuracy = 0.0
        self.best_loss = torch.inf

    def get_dir_name(self):
        return os.path.dirname(self.save_dir)

    def train(self, epochs, save):
        self.log.log(f"Training for {epochs} epochs")
        no_improvement = 0
        self.log.log("learning rate: {}".format(self.scheduler.get_last_lr()))
        for epoch in range(epochs):  # loop over the dataset multiple times
            for _, data in enumerate(self.train_set):
                # todo make the thing with
                input, targets = data
                input = input.to(device)
                targets = targets.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(input)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()  # adjust lr for next epoch
            # print("learning rate: ", self.optimizer.get_last_lr())
            self.log.log(
                "learning rate: {}".format(self.optimizer.param_groups[0]["lr"])
            )
            self.log.log("epoch: {}".format(epoch)!dataset/.gitkeep)
            loss, acc = self.test(self.val_set)
            if loss < self.best_loss:
                self.best_loss = loss
                self.log.log("Saving model")
                if save:
                    torch.save(
                        net.state_dict(),
                        os.path.join(self.save_dir, f"model_{epoch}.pth"),
                    )
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == 10:
                    self.log.log("Early stopping")
                    break

    def test(self, dataset) -> Tuple[float, float]:
        self.log.log("Testing/Evaluating")
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                # todo make the thing with
                input, targets = data
                input = input.to(device)
                targets = targets.to(device)
                # forward + backward + optimize
                outputs = net(input)
                loss = self.criterion(outputs, targets)

                # statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            # print statistics
            total_loss = running_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            self.log.log(f"[Batches:{len(train_data)}] loss: {total_loss:.8f}")
            self.log.log(f"Accuracy: {accuracy:.2f}%")
            return total_loss, accuracy


if __name__ == "__main__":
    log = Logger()
    log.log("Starting session")
    # Initialize a Dataset
    log.log("Loading dataset")
    cifar = CIFAR10()
    train = cifar.train
    test = cifar.test
    val = cifar.val

    # Initialize a DataLoader
    batch_size = 128
    log.log(f"Training dataset size: {len(train)}")
    log.log(f"Batch size: {batch_size}")

    train_data = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=False)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=False)

    net = ResNet18()
    log.log(f"init nnet {net.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # todo scheduler

    net.train()  # setting this changes behaviour of some modules like e.g dropout and batchnorm

    # net = ModelLoader(net)
    # net.load_model("checkpoints/model_7.pth")
    # net = net.model
    # net.train()

    trainer = Trainer(net, train_data, val_data, test_data, device, logger=log)
    save = True
    trainer.train(10, save)
    trainer.test(trainer.test_set)
    log.log("Finished session")
