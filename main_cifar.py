import torch
import torch.backends.cudnn as cudnn

from model.model import ResNet18
from utils.dataset.cifar import CIFAR10
from utils.load_model import ModelLoader
from utils.logger import Logger
from utils.trainer import Trainer
from utils.dataset.dataset import Dataset

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
    log.log("Loading dataset")
    cifar: Dataset = CIFAR10()
    train = cifar.get_train()
    test = cifar.get_test()
    val = cifar.get_val()

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
        net.load_model("checkpoints/ResNet18_cifar/model_23.pth")
        net = net.model
        trainer = Trainer(
            net=net,
            dataset=cifar,
            logger=log,
            batch_size=batch_size,
            device=str(device),
        )
        trainer.test(trainer.val_set)
        exit()

    trainer = Trainer(
        net=net,
        dataset=cifar,
        logger=log,
        batch_size=batch_size,
        device=str(device),
    )
    save = True
    trainer.train(100, save=True)
    trainer.test(trainer.test_set)
    log.log("Finished session")
