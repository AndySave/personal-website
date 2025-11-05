
from typing import Literal
from .base_dataset import BaseDataset
from loaders import BaseCsvLoader
from schemas import MedicalCostInput, FeatureOption, FeatureMetadata, DatasetMetadata, TaskType

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


CAT_COLS = ["sex", "smoker", "region"]
NUM_COLS = ["age", "bmi", "children"]


class MedicalCostDataset(BaseDataset):
    def __init__(self, csv_loader: BaseCsvLoader) -> None:
        self.csv_loader = csv_loader
        self.X, self.y, self.column_transformer, self.feature_columns = self._get_dataset()

    def _get_dataset(self, path="datasets/insurance.csv"):
        column_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore",
                 drop="if_binary", sparse_output=False), CAT_COLS),
                ("num", StandardScaler(), NUM_COLS)
            ]
        )

        df = self.csv_loader.load(path)

        y = df["charges"].to_numpy()

        X = column_transformer.fit_transform(df.drop(columns=["charges"]))

        feature_columns = column_transformer.get_feature_names_out().tolist()

        return np.array(X), np.array(y), column_transformer, feature_columns

    def transform_one(self, input: MedicalCostInput):
        df = pd.DataFrame([input.model_dump()])
        X = self.column_transformer.transform(df)
        return np.array(X)

    def num_features(self) -> int:
        return len(self.feature_columns)

    def num_outputs(self) -> int:
        return 1
    
    def task_type(self) -> TaskType:
        return TaskType.regression

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="medical_cost",
            display_name="Medical Cost",
            task_type=TaskType.regression,
            description="Predicts the estimated medical insurance charges billed to an individual based on demographic and lifestyle factors.",
            features=[
                FeatureMetadata(
                    name="sex",
                    display_name="Sex",
                    type="categorical",
                    options=[
                        FeatureOption(value="male", display_name="Male"),
                        FeatureOption(value="female", display_name="Female"),
                    ],
                ),
                FeatureMetadata(
                    name="smoker",
                    display_name="Smoker",
                    type="categorical",
                    options=[
                        FeatureOption(value="yes", display_name="Yes"),
                        FeatureOption(value="no", display_name="No"),
                    ],
                ),
                FeatureMetadata(
                    name="region",
                    display_name="Region",
                    type="categorical",
                    options=[
                        FeatureOption(value="northeast", display_name="Northeast"),
                        FeatureOption(value="northwest", display_name="Northwest"),
                        FeatureOption(value="southeast", display_name="Southeast"),
                        FeatureOption(value="southwest", display_name="Southwest"),
                    ],
                ),
                FeatureMetadata(
                    name="age",
                    display_name="Age",
                    type="numeric",
                    min=18,
                    max=100,
                ),
                FeatureMetadata(
                    name="bmi",
                    display_name="Body Mass Index",
                    type="numeric",
                    min=10,
                    max=50,
                ),
                FeatureMetadata(
                    name="children",
                    display_name="Number of Children",
                    type="numeric",
                    min=0,
                    max=10,
                ),
            ],
        )
