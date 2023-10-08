import torch
import torch.backends.cudnn as cudnn

from model.model import ResNet18
from tools.load_mnist_m import MNISTM10
from tools.load_model import ModelLoader
from tools.logger import Logger
from tools.trainer import Trainer

import matplotlib.pyplot as plt

# todo undersatand best practices for pytorch and ways to optimize stuff
# implement warmup
# see that training process runs for long time without overfitting
# great strategy is to augment randomly every epoch on different data

# todo implement tensorboard or similar monitoring methods like plt
# implement warmup
# see what is the bound for early stopping
# programm quality resume methods so that stuff resumes after model been trained
# -> use json to load last parameters


if __name__ == "__main__":
    log = Logger()
    log.log("Starting session")
    # Initialize a Dataset
    log.log("Loading MNISTM10 dataset")
    mnistm = MNISTM10()
    train = mnistm.train
    test = mnistm.test
    val = mnistm.val

    # Initialize a DataLoader
    batch_size = 128
    log.log(f"Training dataset size: {len(train)}")
    log.log(f"Batch size: {batch_size}")

    net = ResNet18()
    log.log(f"init nnet {net.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # todo scheduler

    # eval model
    if False:
        net = ModelLoader(net)
        net.load_model("checkpoints/ResNet18_mnistm/model_35.pth")  # mnistm best model
        net = net.model
        trainer = Trainer(net, mnistm, batch_size, device, logger=log)
        trainer.test(trainer.val_set)
        exit()

    if True:
        net = ModelLoader(net)
        net.load_model(
            "checkpoints/ResNet18_1696794933/model_26.pth"
        )  # mnist model trained with mnistm stats
        net = net.model
        trainer = Trainer(net, mnistm, batch_size, device, logger=log)
        trainer.test(trainer.val_set)
        exit()

    trainer = Trainer(
        net,
        mnistm,
        batch_size,
        device,
        logger=log,
    )
    save = True
    trainer.train(100, save=True)
    trainer.test(trainer.test_set)
    log.log("Finished session")