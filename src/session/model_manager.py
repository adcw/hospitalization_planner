import os
import time
from pickle import load, dump

import numpy as np
import pandas as pd

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.config.parsing import parse_config
from src.preprocessing.preprocessor import Preprocessor
from src.session.helpers.eval import eval_model
from src.session.helpers.model_payload import ModelPayload
from src.session.helpers.test import test_model
from src.session.helpers.train import train_model
from src.session.utils.prompts import prompt_mode, prompt_model_file, prompt_model_name

CSV_PATH = './data/input.csv'


def _get_sequences() -> tuple[list[pd.DataFrame], Preprocessor]:
    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object, usecols=COLS)

    # replace literals with values
    whole_df.replace("YES", 1., inplace=True)
    whole_df.replace("NO", 0., inplace=True)
    whole_df.replace("MISSING", np.NAN, inplace=True)

    # impute_dict = {
    #     c.CREATININE: [c.LEVONOR, c.TOTAL_BILIRUBIN, c.HEMOSTATYCZNY],
    #     c.TOTAL_BILIRUBIN: [c.RTG_PDA, c.ANTYBIOTYK, c.PENICELINA1, c.STERYD],
    #     c.PTL: [c.TOTAL_BILIRUBIN, c.ANTYBIOTYK, c.KARBAPENEM, c.GENERAL_PDA_CLOSED]
    # }

    rankings = {
        c.RESPIRATION: ["WLASNY", "CPAP", "MAP1", "MAP2", "MAP3"]
    }

    preprocessor = Preprocessor(group_cols=[c.PATIENTID],
                                group_sort_col=c.DATEID,
                                rank_dict=rankings,
                                impute_dict=None,
                                drop_na=True)

    sequences = preprocessor.fit_transform(whole_df)

    return sequences, preprocessor


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
        self.config_path = config_path

        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        self.model_params, self.train_params, self.eval_params = parse_config(config_path)

        sequences, self.preprocessor = _get_sequences()
        split_point = round(test_perc * len(sequences))

        self.sequences_train = sequences[split_point:]
        self.sequences_test = sequences

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            print("Read model params:")
            print(self.model_params)
            print("Read train params:")
            print(self.train_params)

            time.sleep(1)

            trained_model = train_model(model_params=self.model_params, train_params=self.train_params,
                                        sequences=self.sequences_train)

            payload = ModelPayload(model=trained_model, model_params=self.model_params,
                                   train_params=self.train_params, eval_params=self.eval_params)

            test_model(payload, sequences=self.sequences_test, limit=30)

            model_name = prompt_model_name()
            if model_name:
                payload = ModelPayload(model=trained_model, model_params=self.model_params,
                                       train_params=self.train_params, eval_params=self.eval_params)

                with open(f"{self.models_dir}/{model_name}", "wb+") as file:
                    dump(payload, file)

        elif mode == "test":
            model_filename = prompt_model_file(self.models_dir)

            if model_filename is None:
                raise UserWarning("Where are no models inside directory.")

            with open(f"{self.models_dir}/{model_filename}", "rb") as file:
                model_payload: ModelPayload = load(file)

                print("Read model params:")
                print(model_payload.model_params)
                print("Read train params:")
                print(model_payload.train_params)

                time.sleep(1)

                test_model(model_payload, sequences=self.sequences_test, limit=30)

        elif mode == "eval":
            model_params, train_params, eval_params = parse_config(self.config_path)
            model_payload = ModelPayload(model_params=model_params, train_params=train_params, eval_params=eval_params,
                                         model=None)

            print("Read eval params:")
            print(model_payload.eval_params)

            time.sleep(1)

            eval_model(model_payload, self.sequences_train + self.sequences_test)
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")
