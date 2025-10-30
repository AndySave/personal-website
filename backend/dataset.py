
from schemas import CustomInput
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset:
    def _get_dataset(self):
        raise NotImplementedError

    def transform_one(self, custom_input: CustomInput):
        raise NotImplementedError
    
    def num_features(self) -> int:
        raise NotImplementedError

    def num_classes(self) -> int:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class AdultIncomeDataset(Dataset):
    def __init__(self) -> None:
        self.X, self.y, self.scaler, self.feature_columns = self._get_dataset()

    def _get_dataset(self):
        education_mapping = {
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

        df = pd.read_csv("adult.csv", skipinitialspace=True)

        df.drop(
            columns=[
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "native-country",
                "occupation",
                "relationship",
            ],
            inplace=True
        )

        df['education'] = df['education'].replace(education_mapping)

        for col in df.select_dtypes(include='object'):
            if df[col].nunique() == 2:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            else:
                df = pd.get_dummies(df, columns=[col], drop_first=False)

        print(df.head())
        print(df.columns)

        target_col = "income_>50K"

        n_per_class = df[target_col].value_counts().min()
        df_bal = (
            df.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(n=n_per_class, random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        df = df_bal


        X = df.drop("income_>50K", axis=1).values
        y = df["income_>50K"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
        feature_columns = df.drop("income_>50K", axis=1).columns.tolist()
        return X, np.asarray(y), scaler, feature_columns

    def transform_one(self, custom_input: CustomInput):
        df = pd.DataFrame([custom_input.model_dump()])
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.feature_columns, fill_value=False)

        X = np.asarray(df.values)
        X = self.scaler.transform(X)
        return X
    
    def num_features(self) -> int:
        return len(self.feature_columns)

    def num_classes(self) -> int:  # TODO: Maybe change?
        return 1

    def __len__(self):
        return len(self.X)
