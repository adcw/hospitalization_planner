from datetime import datetime
import os
from typing import Literal

import inquirer


def prompt_mode() -> Literal["train", "test", "validate"]:
    prompt_name = 'mode'
    questions = [
        inquirer.List(prompt_name,
                      message="Choose Mode",
                      choices=['train', 'test', 'eval'],
                      ),
    ]

    answers = inquirer.prompt(questions)
    return answers[prompt_name]


def choose_model_name(models_dir: str):
    prompt_name = 'file_choice'
    filenames = os.listdir(models_dir)

    filenames = [f for f in filenames if f.split("_")[0] == "train"]

    if len(filenames) == 0:
        print(f"There are no models in {models_dir}. Please train a model first.")
        return None

    questions = [
        inquirer.List(prompt_name,
                      message="Choose model to test",
                      choices=filenames,
                      ),
    ]

    answers = inquirer.prompt(questions)
    return answers[prompt_name]


def prompt_model_name():
    text_question = inquirer.Text('model_name', "Enter model name")
    answers = inquirer.prompt([text_question])
    return f"{answers['model_name']}"
