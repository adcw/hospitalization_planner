from datetime import datetime
import os
from typing import Literal

import inquirer


def prompt_mode() -> Literal["train", "test", "validate"]:
    prompt_name = 'mode'
    questions = [
        inquirer.List(prompt_name,
                      message="Choose Mode",
                      choices=['train', 'test', 'validate'],
                      ),
    ]

    answers = inquirer.prompt(questions)
    return answers[prompt_name]


def prompt_model_file(models_dir: str):
    prompt_name = 'file_choice'
    filenames = os.listdir(models_dir)

    if len(filenames) == 0:
        print(f"There are no models in {models_dir}. Please place them or train a nwe one.")
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
    questions = [
        inquirer.Confirm('save_model', message="Save current model?"),
    ]

    answers = inquirer.prompt(questions)

    if answers['save_model']:
        text_question = inquirer.Text('model_name', "Enter model name")
        answers.update(inquirer.prompt([text_question]))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M")
        return f"{answers['model_name']}_{timestamp}.pkl"
    else:
        return None
