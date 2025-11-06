
from fastapi.testclient import TestClient

from .main import app, models

client = TestClient(app)


def test_get_network_metadata():
    response = client.get("/api/nn-framework/networks-metadata")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 3

    for item in data:
        assert "display_name" in item
        assert "model_size" in item


def test_get_dataset_metadata():
    response = client.get("/api/nn-framework/datasets-metadata")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    for item in data:
        assert "name" in item
        assert "display_name" in item
        assert "task_type" in item
        assert "description" in item
        assert "features" in item


def test_train_model_valid_response():
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


def test_train_model_invalid_dataset():
    payload = {
        "dataset_name": "non_existant_dataset",
        "model_size": "small",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 404


def test_train_model_invalid_model_size():
    payload = {
        "dataset_name": "adult_income",
        "model_size": "non_existant_model_size",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 422


def test_train_model_invalid_epoch():
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": 501,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    assert response.status_code == 422


def test_model_is_stored_after_training():
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": 1,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    model_id = response.json()["model_id"]

    assert model_id in models


def test_training_loss_matches_epochs():
    NUM_EPOCHS = 3
    payload = {
        "dataset_name": "adult_income",
        "model_size": "small",
        "epochs": NUM_EPOCHS,
    }

    response = client.post("/api/nn-framework/train", json=payload)
    data = response.json()

    assert len(data["training_loss"]) == NUM_EPOCHS


def test_predict_adult_model_not_found():
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
