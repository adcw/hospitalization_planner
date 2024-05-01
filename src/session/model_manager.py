import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pickle import load, dump
from typing import Optional, List

import numpy as np
import pandas as pd
import shutil

from matplotlib import pyplot as plt

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.config.parsing import parse_config
from src.error_analysis.analyse import perform_error_analysis
from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.model_selection.stratified_sampling import stratified_sampling
from src.preprocessing.preprocessor import Preprocessor
from src.session.helpers.eval import eval_model
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model, test_model_state_optimal
from src.session.helpers.train import train_model
from src.session.utils.prompts import prompt_mode, choose_model_name, prompt_model_name
from src.session.utils.save_plots import base_dir, save_plot

CSV_PATH = './data/input.csv'
import logging


def _get_sequences(path: str = CSV_PATH, limit: int = None, usecols: Optional[List[str]] = None) -> tuple[
    list[pd.DataFrame], Preprocessor]:
    if usecols is None:
        usecols = COLS

    if limit is not None:
        logging.warning("The limit argument can trim sequences. Use only for dev.")
    # read data
    whole_df = pd.read_csv(path, dtype=object, usecols=usecols, nrows=limit)

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
                 ):
        """
        :param base_path: Path to store session data, such as plots, configs and models.
        """
        self.sequences_test = None
        self.sequences_train = None
        self.session_path = base_path
        self.config_path = config_path

        self.session_id = _generate_session_id()

        main_params, train_params, eval_params, test_params = parse_config(config_path)
        self.session_payload = SessionPayload(main_params=main_params,
                                              train_params=train_params,
                                              eval_params=eval_params,
                                              test_params=test_params,
                                              model=None)

        self.sequences, self.preprocessor = _get_sequences()
        self.splitter: Optional[RegressionTrainTestSplitter] = None

    # TODO: This is hardcoded
    def _split_sequences(self, limit: int = None):
        splitter = RegressionTrainTestSplitter()
        self.sequences_train, self.sequences_test = splitter.fit_split(
            self.sequences[:limit], test_size=self.session_payload.main_params.test_size,
            n_clusters=5)
        self.splitter = splitter
        # splitter.plot_split(title="Train and test split", axe_titles=['a', 'b', 'std'])

    def _test_and_save_results(self, max_plots: int = 16):
        testing_mode = self.session_payload.test_params.mode.lower()
        model_type = self.session_payload.main_params.model_type

        plot_indexes = stratified_sampling(self.splitter._clusters[self.splitter._test_indices], max_plots)

        if model_type == "window" and testing_mode in ["full", "both"]:
            test_loss, _ = test_model(self.session_payload, sequences=self.sequences_test, mode="full",
                                      plot_indexes=plot_indexes, max_plots=max_plots)
            plt.subplots_adjust(top=0.95)
            plt.suptitle(f"MAE Test loss: {test_loss}", fontsize=20)
            save_plot(f"test_full.png")

        # For step model and full testing mode, choose more optimal solution
        if model_type == "step" and testing_mode in ["full", "both"]:
            test_loss, _ = test_model_state_optimal(self.session_payload, sequences=self.sequences_test,
                                                    plot_indexes=plot_indexes, max_plots=max_plots)
            plt.subplots_adjust(top=0.95)
            plt.suptitle(f"MAE Test loss: {test_loss}", fontsize=20)
            save_plot(f"test_full.png")

        if testing_mode in ["pessimistic", "both"]:
            test_loss, _ = test_model(self.session_payload, sequences=self.sequences_test, mode="pessimistic",
                                      plot_indexes=plot_indexes, max_plots=max_plots)
            plt.subplots_adjust(top=0.95)
            plt.suptitle(f"MAE Test loss: {test_loss}", fontsize=20)
            save_plot(f"test_pessimistic.png")

        if model_type == "window":
            _, seq_losses = test_model(self.session_payload, sequences=[*self.sequences_train, *self.sequences_test],
                                       mode="full", plot=False)
        elif model_type == "step":
            _, seq_losses = test_model_state_optimal(self.session_payload,
                                                     sequences=[*self.sequences_train, *self.sequences_test],
                                                     plot=False)
        else:
            raise ValueError(f"Unsuported model type: {model_type}")

        perform_error_analysis(sequences=[*self.sequences_train, *self.sequences_test], losses=seq_losses)

    @staticmethod
    def _draw_and_save_losses(train_mae_losses, val_mae_losses, train_final_loss, val_final_loss, filename):
        plt.plot(train_mae_losses, label=f"Train MAE loss = {train_final_loss:.4f}")
        plt.plot(val_mae_losses, label=f"Val MAE loss = {val_final_loss:.4f}")
        plt.legend()
        plt.title("MAE Losses")
        save_plot(filename)

    def start(self):
        mode = prompt_mode()

        if mode == "train":
            model_name = prompt_model_name()
            train_dir = base_dir(f"{self.session_path}/train_{model_name}_{self.session_id}")

            print("Main params:")
            print(self.session_payload.main_params)
            print("Train params:")
            print(self.session_payload.train_params)

            self._split_sequences(self.session_payload.train_params.sequence_limit)

            trained_model, (train_mae_losses, val_mae_losses, last_train_loss, last_val_loss) = train_model(
                payload=self.session_payload,
                sequences=self.sequences_train)

            self._draw_and_save_losses(train_mae_losses, val_mae_losses, last_train_loss, last_val_loss,
                                       f"mae_losses.png")

            payload = SessionPayload(model=trained_model,
                                     main_params=self.session_payload.main_params,
                                     train_params=self.session_payload.train_params,
                                     eval_params=self.session_payload.eval_params,
                                     test_params=self.session_payload.test_params)

            self.session_payload = payload
            self._test_and_save_results()

            if model_name:
                payload = deepcopy(self.session_payload)
                payload.model = trained_model

                os.makedirs(train_dir, exist_ok=True)

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
                session_payload: SessionPayload = load(file)
                session_payload.test_params = self.session_payload.test_params
                self.session_payload = session_payload

                print("Main Params:")
                print(session_payload.main_params)
                print("Test Params")
                print(session_payload.test_params)

                self._test_and_save_results()

                pass

        elif mode == "eval":
            eval_dir = base_dir(f"{self.session_path}/eval_{self.session_id}")
            # self._split_sequences(self.session_payload.eval_params.sequence_limit)

            print("Main params:")
            print(self.session_payload.main_params)
            print("Eval params:")
            print(self.session_payload.eval_params)

            eval_model(self.session_payload, self.sequences)
            shutil.copy(self.config_path, f"{eval_dir}/config.yaml")
            pass

        else:
            raise ValueError(f"Unknown mode: {mode}")
