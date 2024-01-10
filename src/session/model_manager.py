import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pickle import load, dump
from typing import Optional

import numpy as np
import pandas as pd
import shutil

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.config.parsing import parse_config
from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.preprocessing.preprocessor import Preprocessor
from src.session.helpers.eval import eval_model
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model
from src.session.helpers.train import train_model
from src.session.utils.prompts import prompt_mode, choose_model_name, prompt_model_name
from src.session.utils.save_plots import base_dir, save_plot

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


def _generate_session_id():
    now = datetime.now()
    now_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now_formatted


class ModelManager:
    def __init__(self,
                 base_path: str,
                 config_path: str,
                 test_perc: float = 0.1
                 ):
        """
        :param base_path: Path to store session data, such as plots, configs and models.
        """
        self.sequences_test = None
        self.sequences_train = None
        self.session_path = base_path
        self.config_path = config_path

        self.session_id = _generate_session_id()

        self.test_perc = test_perc

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
        # splitter.plot_split(title="Train and test split", axe_titles=['a', 'b', 'std'])

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            model_name = prompt_model_name()
            train_dir = base_dir(f"{self.session_path}/train_{model_name}_{self.session_id}")

            print("Read model session_payload:")
            print(self.session_payload.model_params)
            print("Read train session_payload:")
            print(self.session_payload.train_params)

            self._split_sequences(self.session_payload.train_params.sequence_limit)

            trained_model = train_model(payload=self.session_payload,
                                        sequences=self.sequences_train)

            payload = SessionPayload(model=trained_model, model_params=self.session_payload.model_params,
                                     train_params=self.session_payload.train_params,
                                     eval_params=self.session_payload.eval_params)

            test_model(payload, sequences=self.sequences_test)

            if model_name:
                payload = deepcopy(self.session_payload)
                payload.model = trained_model

                if not os.path.exists(train_dir):
                    os.makedirs(train_dir)

                with open(f"{train_dir}/model.pkl", "wb+") as file:
                    dump(payload, file)

                shutil.copy(self.config_path, f"{train_dir}/config.yaml")

        elif mode == "test":
            model_name = choose_model_name(f"{self.session_path}")

            if model_name is None:
                sys.exit(0)

            only_name = "_".join(model_name.split("_")[1:-2])
            base_dir(f"{self.session_path}/test_{only_name}_{self.session_id}")

            self._split_sequences(self.session_payload.train_params.sequence_limit)

            with open(f"{self.session_path}/{model_name}/model.pkl", "rb") as file:
                model_payload: SessionPayload = load(file)

                print("Read model session_payload:")
                print(model_payload.model_params)
                print("Read train session_payload:")
                print(model_payload.train_params)

                time.sleep(1)

                test_model(model_payload, sequences=self.sequences_test)
                save_plot(f"preds.png")

        elif mode == "eval":
            eval_dir = base_dir(f"{self.session_path}/eval_{self.session_id}")
            self._split_sequences(self.session_payload.eval_params.sequence_limit)

            print("Read eval session_payload:")
            print(self.session_payload.eval_params)

            eval_model(self.session_payload, self.sequences_train + self.sequences_test)
            shutil.copy(self.config_path, f"{eval_dir}/config.yaml")
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")
