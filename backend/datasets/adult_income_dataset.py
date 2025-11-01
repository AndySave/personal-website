
from .base_dataset import BaseDataset
from loaders import BaseCsvLoader
from schemas import AdultIncomeInput
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

    def num_classes(self) -> int:
        return 1

    def metadata(self) -> dict:
        return {
            "name": "adult_income",
            "display_name": "Adult Income",
            "description": "Predicts whether income exceeds $50K/yr based on census data.",
            "features": [
                {
                    "name": "workclass",
                    "display_name": "Work Class",
                    "type": "categorical",
                    "options": [
                        {"value": "Federal-gov", "display_name": "Federal government"},
                        {"value": "State-gov", "display_name": "State government"},
                        {"value": "Local-gov", "display_name": "Local government"},
                        {"value": "Never-worked", "display_name": "Never worked"},
                        {"value": "Private", "display_name": "Private"},
                        {"value": "Self-emp-inc",
                            "display_name": "Self-employed (incorporated)"},
                        {"value": "Self-emp-not-inc",
                            "display_name": "Self-employed (not incorporated)"},
                        {"value": "Without-pay", "display_name": "Without pay"},
                    ],
                },
                {
                    "name": "education",
                    "display_name": "Education Level",
                    "type": "categorical",
                    "options": [
                        {"value": "<HS", "display_name": "Below high school"},
                        {"value": "HS-grad", "display_name": "High school"},
                        {"value": "Some-college", "display_name": "Some college"},
                        {"value": "Bachelors", "display_name": "Bachelors"},
                        {"value": "Masters", "display_name": "Masters"},
                        {"value": "Doctorate", "display_name": "Doctorate (PhD/EdD)"},
                    ],
                },
                {
                    "name": "marital_status",
                    "display_name": "Marital Status",
                    "type": "categorical",
                    "options": [
                        {"value": "Never-married", "display_name": "Never married"},
                        {"value": "Married-civ-spouse",
                            "display_name": "Married (civilian spouse)"},
                        {"value": "Married-AF-spouse",
                            "display_name": "Married (Armed Forces spouse)"},
                        {"value": "Married-spouse-absent",
                            "display_name": "Married (spouse living elsewhere)"},
                        {"value": "Separated", "display_name": "Separated"},
                        {"value": "Divorced", "display_name": "Divorced"},
                        {"value": "Widowed", "display_name": "Widowed"},
                    ],
                },
                {
                    "name": "race",
                    "display_name": "Race",
                    "type": "categorical",
                    "options": [
                        {"value": "White", "display_name": "White"},
                        {"value": "Black", "display_name": "Black"},
                        {"value": "Asian-Pac-Islander", "display_name": "Asian"},
                        {"value": "Amer-Indian-Eskimo",
                            "display_name": "Native American"},
                        {"value": "Other", "display_name": "Other"},
                    ],
                },
                {
                    "name": "sex",
                    "display_name": "Sex",
                    "type": "categorical",
                    "options": [
                        {"value": "Male", "display_name": "Male"},
                        {"value": "Female", "display_name": "Female"},
                    ],
                },
                {
                    "name": "age",
                    "display_name": "Age",
                    "type": "numeric",
                    "min": 18,
                    "max": 100,
                },
                {
                    "name": "work_hours",
                    "display_name": "Weekly Work Hours",
                    "type": "numeric",
                    "min": 0,
                    "max": 200,
                },
            ],
        }
