from dataclasses import dataclass
from typing import Optional

from src.config.parsing import ModelParams, TrainParams, EvalParams
from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel


@dataclass
class SessionPayload:
    model_params: ModelParams
    train_params: TrainParams
    eval_params: EvalParams
    model: Optional[StepModel | WindowModel]
