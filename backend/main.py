
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

import uuid
import numpy as np
import nano_nn as nn
from sklearn.metrics import accuracy_score
from pydantic import BaseModel
from .nn.datasets.base_dataset import BaseDataset
from .nn.datasets.adult_income_dataset import AdultIncomeDataset
from .nn.datasets.medical_cost_dataset import MedicalCostDataset
from .nn.network_factory import ModelFactory
from .loaders import PandasCsvLoader
from .schemas import TrainConfig, AdultIncomeInput, MedicalCostInput, DatasetMetadata, NetworkMetadata
from .nn.network import SmallModel, MediumModel, LargeModel
from .nn.utils import get_activations
from .repositories.in_memory_repo import InMemoryRepository


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

repo = InMemoryRepository()
repo.add_dataset("adult_income", AdultIncomeDataset(csv_loader=PandasCsvLoader()))
repo.add_dataset("medical_cost", MedicalCostDataset(csv_loader=PandasCsvLoader()))

def get_repo():
    return repo

model_factory = ModelFactory()

def get_model_factory():
    return model_factory


@app.post("/api/nn-framework/train")
async def train_model(train_config: TrainConfig, repo: InMemoryRepository = Depends(get_repo), model_factory: ModelFactory = Depends(get_model_factory)):
    if train_config.dataset_name not in repo.datasets:
        raise HTTPException(status_code=404, detail=f"Couldn't find dataset with name: {train_config.dataset_name}")

    model_id = str(uuid.uuid4())
    dataset = repo.get_dataset(train_config.dataset_name)
    task_type = dataset.task_type()

    model = model_factory.create(train_config.model_size, dataset, task_type)

    dataset = repo.datasets[train_config.dataset_name]

    training_loss = []
    training_accuracy = []
    for _ in range(train_config.epochs):
        model.train()
        output, loss = model.forward(dataset.X, dataset.y)
        training_loss.append(loss)

        if dataset.task_type() != "regression":
            training_accuracy.append(accuracy_score(
                dataset.y, (output.squeeze() > 0.5)))
        model.backward()

    repo.add_model(model_id, model)

    return {"model_id": model_id, "training_loss": training_loss, "training_accuracy": training_accuracy if len(training_accuracy) > 0 else None}


def predict(dataset: BaseDataset, model: nn.Module, input: BaseModel):
    X = dataset.transform_one(input)

    model.eval()
    output = float(np.asarray(model.forward(X)).squeeze())
    layer_activations = get_activations(model, X)

    return {
        "output": output,
        "activations": layer_activations,
    }


@app.post("/api/nn-framework/predict/adult_income")
async def predict_adult(model_id: str, input: AdultIncomeInput, repo: InMemoryRepository = Depends(get_repo)):
    if model_id not in repo.models:
        raise HTTPException(status_code=404, detail=f"Could not find model with model id: {model_id}")

    model = repo.get_model(model_id)
    dataset = repo.get_dataset("adult_income")

    return predict(dataset, model, input)


@app.post("/api/nn-framework/predict/medical_cost")
async def predict_medical(model_id: str, input: MedicalCostInput, repo: InMemoryRepository = Depends(get_repo)):
    if model_id not in repo.models:
        raise HTTPException(status_code=404, detail=f"Could not find model with model id: {model_id}")

    model = repo.get_model(model_id)
    dataset = repo.get_dataset("medical_cost")

    return predict(dataset, model, input)


@app.get("/api/nn-framework/datasets-metadata", response_model=list[DatasetMetadata])
async def datasets_metadata(repo: InMemoryRepository = Depends(get_repo)):
    metadatas = []
    for dataset_name in repo.datasets:
        dataset = repo.get_dataset(dataset_name)
        metadatas.append(dataset.metadata())
    return metadatas


@app.get("/api/nn-framework/networks-metadata", response_model=list[NetworkMetadata])
async def networks_metadata():
    metadatas = [SmallModel.metadata(), MediumModel.metadata(),
                 LargeModel.metadata()]
    return metadatas
