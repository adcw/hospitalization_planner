import os
import time
from copy import deepcopy
from pickle import load, dump
from typing import Optional

import numpy as np
import pandas as pd

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.config.parsing import parse_config
from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.preprocessing.preprocessor import Preprocessor
from src.session.helpers.eval import eval_model
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model
from src.session.helpers.train import train_model
from src.session.utils.prompts import prompt_mode, prompt_model_file, prompt_model_name

CSV_PATH = './data/input.csv'


def _get_sequences(path: str = CSV_PATH, limit: int = None) -> tuple[list[pd.DataFrame], Preprocessor]:
    # read data
    whole_df = pd.read_csv(path, dtype=object, usecols=COLS)

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
        c.RESPIRATION: ["MAP3", "MAP2", "MAP1", "CPAP", "WLASNY"]
    }

    preprocessor = Preprocessor(group_cols=[c.PATIENTID],
                                group_sort_col=c.DATEID,
                                rank_dict=rankings,
                                impute_dict=None,
                                drop_na=True)

    sequences = preprocessor.fit_transform(whole_df)
    np.random.shuffle(sequences)

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
        self.sequences_test = None
        self.sequences_train = None
        self.models_dir = models_dir
        self.config_path = config_path

        self.test_perc = test_perc

        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        model_params, train_params, eval_params = parse_config(config_path)
        self.session_payload = SessionPayload(model_params=model_params, train_params=train_params,
                                              eval_params=eval_params,
                                              model=None)

        self.sequences, self.preprocessor = _get_sequences()

    # TODO: This is hardcoded
    def _split_sequences(self, limit: int = None):
        splitter = RegressionTrainTestSplitter()
        self.sequences_train, self.sequences_test = splitter.fit_split(
            self.sequences[:limit], test_size=self.test_perc,
            n_clusters=5)
        splitter.plot_split(title="Train and test split", axe_titles=['a', 'b', 'std'])

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            print("Read model session_payload:")
            print(self.session_payload.model_params)
            print("Read train session_payload:")
            print(self.session_payload.train_params)

            self._split_sequences(self.session_payload.train_params.sequence_limit)

            time.sleep(1)

            trained_model = train_model(payload=self.session_payload,
                                        sequences=self.sequences_train)

            payload = SessionPayload(model=trained_model, model_params=self.session_payload.model_params,
                                     train_params=self.session_payload.train_params,
                                     eval_params=self.session_payload.eval_params)

            test_model(payload, sequences=self.sequences_test, limit=None)

            model_name = prompt_model_name()
            if model_name:
                payload = deepcopy(self.session_payload)
                payload.model = trained_model
                with open(f"{self.models_dir}/{model_name}", "wb+") as file:
                    dump(payload, file)

        elif mode == "test":
            model_filename = prompt_model_file(self.models_dir)
            self._split_sequences(self.session_payload.train_params.sequence_limit)

            if model_filename is None:
                raise UserWarning("Where are no models inside directory.")

            with open(f"{self.models_dir}/{model_filename}", "rb") as file:
                model_payload: SessionPayload = load(file)

                print("Read model session_payload:")
                print(model_payload.model_params)
                print("Read train session_payload:")
                print(model_payload.train_params)

                time.sleep(1)

                test_model(model_payload, sequences=self.sequences_test, limit=30)

        elif mode == "eval":
            self._split_sequences(self.session_payload.eval_params.sequence_limit)

            print("Read eval session_payload:")
            print(self.session_payload.eval_params)

            time.sleep(1)

            eval_model(self.session_payload, self.sequences_train + self.sequences_test)
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")
