import torch
import torch.backends.cudnn as cudnn

from model.model import ResNet18
from utils.dataset.mnist import MNIST10
from utils.dataset.mnistm import MNISTM10
from utils.load_model import ModelLoader
from utils.logger import Logger
from utils.trainer import Trainer

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
    log.log("Loading MNIST10 dataset")
    mnist = MNIST10()
    mnistm = MNISTM10()
    train = mnist.get_train()
    test = mnist.get_test()
    val = mnist.get_val()

    # Initialize a DataLoader
    batch_size = 32
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
    if True:
        net = ModelLoader(net)
        net.load_model(
            "checkpoints/ResNet18_mnist_224_224/model_6.pth"
        )  # -> this is mnistm best model
        net = net.model
        trainer = Trainer(
            net=net,
            dataset=mnist,
            logger=log,
            batch_size=batch_size,
            device=str(device),
        )
        trainer.test(trainer.val_set)
        exit()

    if False:
        net = ModelLoader(net)
        net.load_model("checkpoints/ResNet18_mnist_with_mnistm_stats/model_16.pth")
        net = net.model
        trainer = Trainer(
            net=net,
            dataset=mnist,
            logger=log,
            batch_size=batch_size,
            device=str(device),
        )
        trainer.test(trainer.val_set)
        exit()

    if False:
        mnistm.std = mnist.std
        mnistm.mean = mnist.mean
        net = ModelLoader(net)
        net.load_model(
            "checkpoints/ResNet18_1696803877/model_27.pth"
        )  # plain mnist with mnist stats used on mnistm data
        net = net.model
        trainer = Trainer(
            net=net,
            dataset=mnistm,
            logger=log,
            batch_size=batch_size,
            device=str(device),
        )
        trainer.test(trainer.val_set)
        exit()

    trainer = Trainer(
        net=net,
        dataset=mnist,
        logger=log,
        batch_size=batch_size,
        device=str(device),
    )
    save = True
    trainer.train(100, save=True)
    trainer.test(trainer.test_set)
    log.log("Finished session")
