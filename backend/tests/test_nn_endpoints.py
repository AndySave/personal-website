
from fastapi.testclient import TestClient
import pytest
from pydantic import BaseModel
import numpy as np
import nano_nn as nn

from backend.main import app, get_repo, get_model_factory
from backend.nn.datasets.base_dataset import BaseDataset
from backend.nn.utils import normalize_activation
from backend.repositories.in_memory_repo import InMemoryRepository
from backend.schemas import TaskType, DatasetMetadata, NetworkMetadata, ModelSize


class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def train(self):
        pass

    def forward(self, X, y=None):
        preds = np.full((len(X), 1), 0.9)
        if y is None:
            return preds
        loss = 0.8
        return preds, loss

    def backward(self):
        pass
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Small Network", model_size=ModelSize.small)


class DummyDataset(BaseDataset):
    def __init__(self):
        self.X = np.array([[0.5],
                           [0.8],
                           [0.6],
                           [0.9],
                           [0.2]])
        self.y = np.array([0, 0, 1, 1, 0])

    def _get_dataset(self): ...

    def transform_one(self, input: BaseModel) -> np.ndarray:
        return np.array([[0.2]])

    def num_features(self) -> int:
        return 3

    def num_outputs(self) -> int:
        return 1

    def task_type(self) -> TaskType:
        return TaskType.binary_classification

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="dummy",
            display_name="dummy",
            task_type=TaskType.binary_classification,
            description="dummy dataset",
            features=[]
        )

class DummyNetworkFactory:
    def create(self, model_size: ModelSize, dataset: BaseDataset, task_type: TaskType):
        return DummyNetwork()

@pytest.fixture
def test_network_factory():
    return DummyNetworkFactory()

@pytest.fixture
def test_repo():
    repo = InMemoryRepository()
    repo.add_dataset("adult_income", DummyDataset())
    repo.add_model("real_model_id", DummyNetwork())
    return repo

@pytest.fixture
def client(test_repo, test_network_factory):
    app.dependency_overrides[get_repo] = lambda: test_repo
    app.dependency_overrides[get_model_factory] = lambda: test_network_factory
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_get_network_metadata(client):
    response = client.get("/api/nn-framework/networks-metadata")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 3

    for item in data:
        assert "display_name" in item
        assert "model_size" in item


def test_get_dataset_metadata(client):
    response = client.get("/api/nn-framework/datasets-metadata")

    assert response.status_code == 200

    data = response.json()


    assert len(data) == 1

    for item in data:
        assert "name" in item
        assert "display_name" in item
        assert "task_type" in item
        assert "description" in item
        assert "features" in item


def test_train_model_valid_response(client):
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": 1
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "model_id" in data
    assert "training_loss" in data
    assert "training_accuracy" in data


def test_train_model_invalid_dataset(client):
    payload = {
        "dataset_name": "non_existant_dataset",
        "model_size": "small",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 404


def test_train_model_invalid_model_size(client):
    payload = {
        "dataset_name": "adult_income",
        "model_size": "non_existant_model_size",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 422


def test_train_model_invalid_epoch(client):
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": 501,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 422


def test_model_is_stored_after_training(client, test_repo):
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 200

    model_id = response.json()["model_id"]

    assert model_id in test_repo.models
    assert test_repo.models[model_id] is not None


def test_training_loss_matches_epochs(client):
    NUM_EPOCHS = 3
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": NUM_EPOCHS,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    data = response.json()

    assert len(data["training_loss"]) == NUM_EPOCHS


def test_predict_adult_model_not_found(client):
    params = {"model_id": "non_existance_model_id"}
    payload = {
        "age": 49,
        "workclass": "Federal-gov",
        "education": "Some-college",
        "marital_status": "Married-civ-spouse",
        "race": "White",
        "sex": "Female",
        "work_hours": 50
    }
    
    response = client.post("/api/nn-framework/predict/adult_income", params=params, json=payload)

    assert response.status_code == 404


def test_predict_valid_response(client):
    params = {"model_id": "real_model_id"}
    payload = {
        "age": 49,
        "workclass": "Federal-gov",
        "education": "Some-college",
        "marital_status": "Married-civ-spouse",
        "race": "White",
        "sex": "Female",
        "work_hours": 50
    }

    response = client.post("/api/nn-framework/predict/adult_income", params=params, json=payload)
    data = response.json()
    
    assert response.status_code == 200
    assert "output" in data
    assert "activations" in data


def test_predict_invalid_education_input(client):
    params = {"model_id": "real_model_id"}
    payload = {
        "age": 49,
        "workclass": "Federal-gov",
        "education": "Non Existant Education",
        "marital_status": "Married-civ-spouse",
        "race": "White",
        "sex": "Female",
        "work_hours": 50
    }

    response = client.post("/api/nn-framework/predict/adult_income", params=params, json=payload)

    assert response.status_code == 422


def test_normalize_activation_gives_zeroes():
    out = normalize_activation(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))

    assert all([value == 0 for value in out])


def test_normalize_activation_expected():
    out = normalize_activation(np.array([0, 0.25, 0.5, 0.75, 1]))
    expected = np.array([0, 0.25, 0.5, 0.75, 1])

    assert all([out_v == exp_v for out_v, exp_v in zip(out, expected)])
