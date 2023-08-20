import torch
import torch.backends.cudnn as cudnn


class ModelLoader:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str):
        """
        Loads the model from the provided model path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            None
        """
        self.__try_loading_model(model_path)

    def __try_loading_model(self, model_path: str):
        """
        Tries to load the model from the provided model path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            None
        """
        self.model = self.model.to(self.device)
        try:
            # todo do some checks if file exists or so
            if self.device.type == "cuda":
                # self.model = torch.nn.DataParallel(self.model)
                cudnn.benchmark = True
                model_load = torch.load(model_path)
            else:
                # see this https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
                self.model = torch.nn.DataParallel(self.model)
                model_load = torch.load(
                    model_path, map_location=torch.device(self.device)
                )
            self.model.load_state_dict(model_load)
            self.model.eval()
        except Exception as e:
            print(e)
