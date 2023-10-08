# TODO this class should ca a root class that other loaders extend to save coding
import os


class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.root = "datasets"
        self.dataset_path = os.path.join(self.root, self.dataset_name)
        self.train = None
        self.test = None
        self.val = None
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def get_dataset(self):
        return self.train, self.test, self.val

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_dataset_root(self):
        return self.root
