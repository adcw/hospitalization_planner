from dataclasses import dataclass
from typing import Optional

from src.config.parsing import ModelParams, TrainParams, EvalParams
from src.models.stateful_prediction_module import StatePredictionModule
from src.models.windowed_module import WindowedModule


@dataclass
class SessionPayload:
    model_params: ModelParams
    train_params: TrainParams
    eval_params: EvalParams
    model: Optional[StatePredictionModule | WindowedModule]
