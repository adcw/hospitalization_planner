import os
import time
from pickle import load, dump

import numpy as np
import pandas as pd

import data.raw.colnames_original as c

from src.preprocessing import Preprocessor
from src.session.helpers import train_model_helper, test_model_helper, ModelPayload
from src.session.prompts import prompt_mode, prompt_model_file, prompt_model_name
from src.config.parsing import parse_config

CSV_PATH = './data/clean/input.csv'


def _get_sequences() -> tuple[list[pd.DataFrame], Preprocessor]:
    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object)

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

        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        self.model_params, self.train_params = parse_config(config_path)

        sequences, self.preprocessor = _get_sequences()
        split_point = round(test_perc * len(sequences))

        self.sequences_train = sequences[split_point:]
        self.sequences_test = sequences[:split_point]

    def start(self):

        mode = prompt_mode()

        if mode == "train":
            print("Read model params:")
            print(self.model_params)
            print("Read train params:")
            print(self.train_params)

            time.sleep(1)

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

            if model_filename is None:
                raise UserWarning("Where are no models inside direcotry.")

            with open(f"{self.models_dir}/{model_filename}", "rb") as file:
                model_payload: ModelPayload = load(file)

                print("Read model params:")
                print(model_payload.model_params)
                print("Read train params:")
                print(model_payload.train_params)

                time.sleep(1)

                test_model_helper(model_payload, sequences=self.sequences_test)

        elif mode == "eval":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")
