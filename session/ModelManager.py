import os
import time
from dataclasses import dataclass

from src.config.config_classes import ModelParams, TrainParams, parse_config
import numpy.random

from session.helpers import train_model_helper, test_model_helper, ModelPayload
from session.prompts import prompt_mode, prompt_model_file, prompt_model_name

from pickle import load, dump

from src.nn import StatePredictionModule


class ModelManager:
    def __init__(self,
                 models_dir: str,
                 config_path: str
                 ):
        """

        :param models_dir: Path to save and load models
        """
        self.models_dir = models_dir

        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        self.model_params, self.train_params = parse_config(config_path)

        pass

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            trained_model = train_model_helper(model_params=self.model_params, train_params=self.train_params)
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

                test_model_helper(model_payload)
            pass
        elif mode == "eval":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pass


pass

if __name__ == '__main__':
    manager = ModelManager(models_dir='./manager_test_dir', config_path="../models/config.yaml")
    manager.start()

pass
