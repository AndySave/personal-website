
from unittest.mock import Mock
import pandas as pd
import numpy as np
from datasets.adult_income_dataset import AdultIncomeDataset
from schemas import AdultIncodeInput


def _tiny_df():
    return pd.DataFrame([
        {
            "workclass": "Self-emp-not-inc",
            "education": "10th",
            "marital_status": "Never-married",
            "race": "White",
            "sex": "Male",
            "age": 37,
            "work_hours": 40,
            "income": "<=50K",
            "fnlwgt": 1, "education-num": 10, "capital-gain": 0, "capital-loss": 0,
            "native-country": "United-States", "occupation": "Tech-support", "relationship": "Not-in-family",
        },
        {
            "workclass": "Federal-gov",
            "education": "7th-8th",
            "marital_status": "Never-married",
            "race": "White",
            "sex": "Female",
            "age": 45,
            "work_hours": 70,
            "income": ">50K",
            "fnlwgt": 2, "education-num": 13, "capital-gain": 0, "capital-loss": 0,
            "native-country": "United-States", "occupation": "Exec-managerial", "relationship": "Husband",
        },
        {
            "workclass": "Private",
            "education": "Masters",
            "marital_status": "Married-AF-spouse",
            "race": "White",
            "sex": "Female",
            "age": 18,
            "work_hours": 10,
            "income": "<=50K",
            "fnlwgt": 2, "education-num": 13, "capital-gain": 0, "capital-loss": 0,
            "native-country": "United-States", "occupation": "Exec-managerial", "relationship": "Husband",
        },
        {
            "workclass": "State-gov",
            "education": "Doctorate",
            "marital_status": "Married-civ-spouse",
            "race": "Asian",
            "sex": "Male",
            "age": 69,
            "work_hours": 20,
            "income": ">50K",
            "fnlwgt": 2, "education-num": 13, "capital-gain": 0, "capital-loss": 0,
            "native-country": "United-States", "occupation": "Exec-managerial", "relationship": "Husband",
        },
    ])



def test_shapes_ok():
    mock_loader = Mock()
    mock_loader.load.return_value = _tiny_df()

    dataset = AdultIncomeDataset(csv_loader=mock_loader)

    assert len(dataset.X) == 4
    assert dataset.num_features() == len(dataset.feature_columns)
    assert len(dataset.X[0]) == dataset.num_features()


def test_feature_columns_ok():
    mock_loader = Mock()
    mock_loader.load.return_value = _tiny_df()

    dataset = AdultIncomeDataset(csv_loader=mock_loader)

    assert "cat__race_White" in dataset.feature_columns
    assert "cat__education_<HS" in dataset.feature_columns
    assert "cat__education_10th" not in dataset.feature_columns
    assert "cat__education_7th-8th" not in dataset.feature_columns
    assert "num__age" in dataset.feature_columns
    assert "num__work_hours" in dataset.feature_columns


def test_transform_one_returns_correct_shape():
    mock_loader = Mock()
    mock_loader.load.return_value = _tiny_df()

    dataset = AdultIncomeDataset(csv_loader=mock_loader)

    inp = AdultIncodeInput(
        age=33,
        workclass="Private",
        education="Masters",
        marital_status="Married-civ-spouse",
        race="Black",
        sex="Female",
        work_hours=45,
    )

    X = dataset.transform_one(inp)

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, dataset.num_features())
