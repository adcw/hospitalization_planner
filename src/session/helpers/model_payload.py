from dataclasses import dataclass
from typing import Optional

from src.config.parsing import ModelParams, TrainParams, EvalParams
from src.models.state_prediction_module import StatePredictionModule


@dataclass
class ModelPayload:
    model_params: ModelParams
    train_params: TrainParams
    eval_params: EvalParams
    model: Optional[StatePredictionModule]
