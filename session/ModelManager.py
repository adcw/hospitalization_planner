import os
from pickle import load, dump

from models.utils import get_sequences
from session.helpers import train_model_helper, test_model_helper, ModelPayload
from session.prompts import prompt_mode, prompt_model_file, prompt_model_name
from src.config.config_classes import parse_config


class ModelManager:
    def __init__(self,
                 models_dir: str,
                 config_path: str,
                 test_perc: float = 0.1
                 ):
        """

        :param models_dir: Path to save and load models
        """
        self.models_dir = models_dir

        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        self.model_params, self.train_params = parse_config(config_path)

        sequences, self.preprocessor = get_sequences()
        split_point = round(test_perc * len(sequences))
        self.sequences_train = sequences[:split_point]
        self.sequences_test = sequences[:split_point]

        pass

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            trained_model = train_model_helper(model_params=self.model_params, train_params=self.train_params,
                                               sequences=self.sequences_train)
            model_name = prompt_model_name()
            if model_name:
                payload = ModelPayload(model=trained_model, model_params=self.model_params,
                                       train_params=self.train_params)

                with open(f"{self.models_dir}/{model_name}", "wb+") as file:
                    dump(payload, file)

        elif mode == "test":
            model_filename = prompt_model_file(self.models_dir)

            with open(f"{self.models_dir}/{model_filename}", "rb") as file:
                model_payload: ModelPayload = load(file)

                test_model_helper(model_payload, sequences=self.sequences_test, preprocessor=self.preprocessor)

        elif mode == "eval":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    manager = ModelManager(models_dir='./manager_test_dir', config_path="../models/config.yaml")
    manager.start()

pass
