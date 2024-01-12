from dataclasses import dataclass
from typing import Optional

from src.config.dataclassess import WindowModelParams
from src.config.parsing import StepModelParams, TrainParams, EvalParams
from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel


@dataclass
class SessionPayload:
    model_params: StepModelParams | WindowModelParams
    train_params: TrainParams
    eval_params: EvalParams
    model: Optional[StepModel | WindowModel]