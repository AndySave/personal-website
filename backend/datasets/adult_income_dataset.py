
from .base_dataset import BaseDataset
from schemas import CustomInput
import pandas as pd
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
            'Prof-school': 'Prof-school',
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
    def __init__(self) -> None:
        self.X, self.y, self.column_transformer, self.feature_columns = self._get_dataset()

    def _get_dataset(self, csv_file="adult.csv"):
        column_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CAT_COLS),
                ("num", StandardScaler(), NUM_COLS)
            ]
        )

        df = pd.read_csv(csv_file, skipinitialspace=True)

        df.drop(
            columns=DROP_COLUMNS,
            inplace=True
        )

        df['education'] = df['education'].replace(EDUCATION_MAP)

        y = (df["income"] == ">50K").to_numpy()

        X = column_transformer.fit_transform(df.drop(columns=["income"]))

        feature_columns = column_transformer.get_feature_names_out().tolist()

        return X, y, column_transformer, feature_columns
    

    def transform_one(self, custom_input: CustomInput):
        df = pd.DataFrame([custom_input.model_dump()])
        df["education"] = df["education"].replace(EDUCATION_MAP)

        X = self.column_transformer.transform(df)

        return X
    
    def num_features(self) -> int:
        return len(self.feature_columns)

    def num_classes(self) -> int:  # TODO: Maybe change?
        return 1
    
    def metadata(self) -> dict:
        raise NotImplementedError
