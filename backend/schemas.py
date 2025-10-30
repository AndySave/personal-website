
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

class AdultIncomeInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    workclass: Literal["Federal-gov", "State-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "Without-pay"]
    education: Literal["<HS", "HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"]
    marital_status: Literal["Never-married", "Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent", "Separated", "Divorced", "Widowed"]
    race: Literal["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    sex: Literal["Male", "Female"]
    work_hours: int = Field(..., ge=0, le=200)
