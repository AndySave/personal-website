
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class LayerType(str, Enum):
    dense = "dense"
    dropout = "dropout"
    relu = "relu"
    sigmoid = "sigmoid"
    softmax = "softmax"

class Layer(BaseModel):
    type: LayerType
    in_features: Optional[int] = None
    out_features: Optional[int] = None

class TrainConfig(BaseModel):
    dataset_name: str
    layers: list[Layer]

class CustomInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    workclass: str
    education: str
    marital_status: str
    race: str
    sex: Literal["Male", "Female"]
    work_hours: int
