from __future__ import annotations
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from typing import Tuple

import tqdm
from utils.logger import Logger
import os
import typing

# from tools.load_cifar import CIFAR10
from torch.utils.data import DataLoader

if typing.TYPE_CHECKING:
    from utils.dataset.dataset import Dataset


class Trainer:
    criterion = CrossEntropyLoss()
    root = "checkpoints"

    def __init__(
        self,
        net,
        dataset: Dataset,  # : CIFAR10 or MNISTM10
        logger: Logger,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.dataset = dataset
        if logger is None:
            self.log = Logger()
        else:
            self.log = logger
        self.save_dir = os.path.join(
            self.root, net.name + "_" + self.log.get_timestamp()
        )
        self.net = net
        self.train_set = DataLoader(
            dataset.get_train(), batch_size=batch_size, shuffle=True
        )
        self.test_set = DataLoader(
            dataset.get_test(), batch_size=batch_size, shuffle=False
        )
        self.val_set = DataLoader(
            dataset.get_val(), batch_size=batch_size, shuffle=False
        )
        self.device = device
        lr = 0.001
        # weight_decay is L2 regularization
        # self.optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
        # self.optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-2)
        self.optimizer = Adam(net.parameters(), lr=lr, weight_decay=6e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )
        self.log.log(f"Using optimizer: {self.optimizer} with start lr: {lr}")
        self.best_accuracy = 0.0
        self.best_loss = torch.inf

    def get_dir_name(self):
        return os.path.dirname(self.save_dir)

    def train(self, epochs, save=True, augment=True):
        if save and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        training_loss = 0.0
        total = 0
        correct = 0
        self.log.log(f"Training for {epochs} epochs")
        no_improvement = 0
        self.log.log("learning rate: {}".format(self.scheduler.get_last_lr()))
        for epoch in range(epochs):  # loop over the dataset multiple times
            self.net.train()
            # for batch_idx, data in enumerate(self.train_set):
            for batch_idx in tqdm.tqdm(range(len(self.train_set)), "Training: "):
                data = next(iter(self.train_set))
                # todo make the thing with
                input, targets = data
                if augment:
                    input = self.dataset.augment_train(input)
                input = input.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(input)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # statistics
                training_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            self.scheduler.step()  # adjust lr for next epoch
            # print("learning rate: ", self.optimizer.get_last_lr())
            self.log.log(
                "learning rate: {}".format(self.optimizer.param_groups[0]["lr"])
            )
            self.log.log("epoch: {}".format(epoch))

            # print statisticsnet.eval()
            total_loss = training_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            self.log.log("Training/Evaluating")
            self.log.log(f"[Batches:{len(self.train_set)}] loss: {total_loss:.8f}")
            self.log.log(f"Accuracy: {accuracy:.2f}%")
            training_loss = 0.0
            total = 0
            correct = 0

            loss, acc = self.test(self.val_set)
            if loss < self.best_loss:
                self.best_loss = loss
                self.log.log("Saving model")
                if save:
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(self.save_dir, f"model_{epoch}.pth"),
                    )
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == 15:
                    self.log.log("Early stopping")
                    break

    def test(self, dataset: DataLoader) -> Tuple[float, float]:
        self.log.log("Testing/Evaluating")
        self.net.eval()
        testing_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(len(self.train_set)), "Testing: "):
                data = next(iter(self.train_set))
                # todo make the thing with
                input, targets = data
                input = self.dataset.augment_test(input)
                input = input.to(self.device)
                targets = targets.to(self.device)
                # forward + backward + optimize
                outputs = self.net(input)
                loss = self.criterion(outputs, targets)

                # statistics
                testing_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            # print statistics
            total_loss = testing_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            self.log.log(f"[Batches:{len(dataset)}] loss: {total_loss:.8f}")
            self.log.log(f"Accuracy: {accuracy:.2f}%")
            return total_loss, accuracy
