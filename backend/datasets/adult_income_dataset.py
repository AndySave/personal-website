
from typing import Literal
from .base_dataset import BaseDataset
from loaders import BaseCsvLoader
from schemas import AdultIncomeInput, FeatureOption, FeatureMetadata, DatasetMetadata
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


CAT_COLS = ["workclass", "education", "marital_status", "race", "sex"]
NUM_COLS = ["age", "work_hours"]

EDUCATION_MAP = {
    'Preschool': '<HS',
    '1st-4th': '<HS',
    '5th-6th': '<HS',
    '7th-8th': '<HS',
    '9th': '<HS',
    '10th': '<HS',
    '11th': '<HS',
    '12th': '<HS',
    'HS-grad': 'HS-grad',
    'Assoc-acdm': 'Some-college',
    'Assoc-voc': 'Some-college',
    'Some-college': 'Some-college',
    'Bachelors': 'Bachelors',
    'Masters': 'Masters',
    'Prof-school': 'Doctorate',
    'Doctorate': 'Doctorate',
}

DROP_COLUMNS = [
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "native-country",
    "occupation",
    "relationship",
]


class AdultIncomeDataset(BaseDataset):
    def __init__(self, csv_loader: BaseCsvLoader) -> None:
        self.csv_loader = csv_loader
        self.X, self.y, self.column_transformer, self.feature_columns = self._get_dataset()

    def _get_dataset(self, path="datasets/adult.csv"):
        column_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore",
                 drop="if_binary", sparse_output=False), CAT_COLS),
                ("num", StandardScaler(), NUM_COLS)
            ]
        )

        df = self.csv_loader.load(path, skipinitialspace=True)

        df.drop(
            columns=DROP_COLUMNS,
            inplace=True
        )

        df['education'] = df['education'].replace(EDUCATION_MAP)

        y = (df["income"] == ">50K").to_numpy()

        X = column_transformer.fit_transform(df.drop(columns=["income"]))

        feature_columns = column_transformer.get_feature_names_out().tolist()

        return np.array(X), np.array(y), column_transformer, feature_columns

    def transform_one(self, input: AdultIncomeInput):
        df = pd.DataFrame([input.model_dump()])
        X = self.column_transformer.transform(df)
        return np.array(X)

    def num_features(self) -> int:
        return len(self.feature_columns)

    def num_outputs(self) -> int:
        return 1
    
    def task_type(self) -> Literal["regression", "binary_classification", "multi_classification"]:
        return "binary_classification"

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="adult_income",
            display_name="Adult Income",
            task_type="binary_classification",
            description="Predicts whether income exceeds $50K/yr based on census data.",
            features=[
                FeatureMetadata(
                    name="workclass",
                    display_name="Work Class",
                    type="categorical",
                    options=[
                        FeatureOption(value="Federal-gov", display_name="Federal government"),
                        FeatureOption(value="State-gov", display_name="State government"),
                        FeatureOption(value="Local-gov", display_name="Local government"),
                        FeatureOption(value="Never-worked", display_name="Never worked"),
                        FeatureOption(value="Private", display_name="Private"),
                        FeatureOption(value="Self-emp-inc", display_name="Self-employed (incorporated)"),
                        FeatureOption(value="Self-emp-not-inc", display_name="Self-employed (not incorporated)"),
                        FeatureOption(value="Without-pay", display_name="Without pay"),
                    ],
                ),
                FeatureMetadata(
                    name="education",
                    display_name="Education Level",
                    type="categorical",
                    options=[
                        FeatureOption(value="<HS", display_name="Below high school"),
                        FeatureOption(value="HS-grad", display_name="High school"),
                        FeatureOption(value="Some-college", display_name="Some college"),
                        FeatureOption(value="Bachelors", display_name="Bachelors"),
                        FeatureOption(value="Masters", display_name="Masters"),
                        FeatureOption(value="Doctorate", display_name="Doctorate (PhD/EdD)"),
                    ],
                ),
                FeatureMetadata(
                    name="marital_status",
                    display_name="Marital Status",
                    type="categorical",
                    options=[
                        FeatureOption(value="Never-married", display_name="Never married"),
                        FeatureOption(value="Married-civ-spouse", display_name="Married (civilian spouse)"),
                        FeatureOption(value="Married-AF-spouse", display_name="Married (Armed Forces spouse)"),
                        FeatureOption(value="Married-spouse-absent", display_name="Married (spouse living elsewhere)"),
                        FeatureOption(value="Separated", display_name="Separated"),
                        FeatureOption(value="Divorced", display_name="Divorced"),
                        FeatureOption(value="Widowed", display_name="Widowed"),
                    ],
                ),
                FeatureMetadata(
                    name="race",
                    display_name="Race",
                    type="categorical",
                    options=[
                        FeatureOption(value="White", display_name="White"),
                        FeatureOption(value="Black", display_name="Black"),
                        FeatureOption(value="Asian-Pac-Islander", display_name="Asian"),
                        FeatureOption(value="Amer-Indian-Eskimo", display_name="Native American"),
                        FeatureOption(value="Other", display_name="Other"),
                    ],
                ),
                FeatureMetadata(
                    name="sex",
                    display_name="Sex",
                    type="categorical",
                    options=[
                        FeatureOption(value="Male", display_name="Male"),
                        FeatureOption(value="Female", display_name="Female"),
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
                    name="work_hours",
                    display_name="Weekly Work Hours",
                    type="numeric",
                    min=0,
                    max=200,
                ),
            ],
        )
