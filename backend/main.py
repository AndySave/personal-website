
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uuid
import numpy as np
import nano_nn as nn
from sklearn.metrics import accuracy_score
from pydantic import BaseModel
from .nn.datasets.base_dataset import BaseDataset
from .nn.datasets.adult_income_dataset import AdultIncomeDataset
from .nn.datasets.medical_cost_dataset import MedicalCostDataset
from .loaders import PandasCsvLoader
from .schemas import TrainConfig, AdultIncomeInput, MedicalCostInput, DatasetMetadata, NetworkMetadata, ModelSize
from .nn.network import SmallModel, MediumModel, LargeModel
from .nn.utils import get_activations



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models: dict[str, nn.Module] = {}
datasets: dict[str, BaseDataset] = {
    "adult_income": AdultIncomeDataset(csv_loader=PandasCsvLoader()),
    "medical_cost": MedicalCostDataset(csv_loader=PandasCsvLoader()),
}


@app.post("/api/nn-framework/train")
async def train_model(train_config: TrainConfig):
    if train_config.dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Couldn't find dataset with name: {train_config.dataset_name}")

    model_id = str(uuid.uuid4())
    dataset = datasets[train_config.dataset_name]
    task_type = dataset.task_type()

    if train_config.model_size == "small":
        model = SmallModel(dataset.num_features(),
                           dataset.num_outputs(), task_type)
    elif train_config.model_size == "medium":
        model = MediumModel(dataset.num_features(),
                            dataset.num_outputs(), task_type)
    else:
        model = LargeModel(dataset.num_features(),
                           dataset.num_outputs(), task_type)

    dataset = datasets[train_config.dataset_name]

    training_loss = []
    training_accuracy = []
    for i in range(train_config.epochs):
        model.train()
        output, loss = model.forward(dataset.X, dataset.y)
        training_loss.append(loss)

        if dataset.task_type() != "regression":
            training_accuracy.append(accuracy_score(
                dataset.y, (output.squeeze() > 0.5)))
        model.backward()

        print(f'epoch: {i+1}/{train_config.epochs}, loss: {loss:.6f}')

    models[model_id] = model

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
async def predict_adult(model_id: str, input: AdultIncomeInput):
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Could not find model with model id: {model_id}")

    model = models[model_id]
    dataset = datasets["adult_income"]

    return predict(dataset, model, input)


@app.post("/api/nn-framework/predict/medical_cost")
async def predict_medical(model_id: str, input: MedicalCostInput):
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Could not find model with model id: {model_id}")

    model = models[model_id]
    dataset = datasets["medical_cost"]

    return predict(dataset, model, input)


@app.get("/api/nn-framework/datasets-metadata", response_model=list[DatasetMetadata])
async def datasets_metadata():
    metadatas = []
    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        metadatas.append(dataset.metadata())
    return metadatas


@app.get("/api/nn-framework/networks-metadata", response_model=list[NetworkMetadata])
async def networks_metadata():
    metadatas = [SmallModel.metadata(), MediumModel.metadata(),
                 LargeModel.metadata()]
    return metadatas
