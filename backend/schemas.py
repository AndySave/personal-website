
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Layer(BaseModel):
    type: Literal["dense", "dropout", "relu", "sigmoid", "softmax"]
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
