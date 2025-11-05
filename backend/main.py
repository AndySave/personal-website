
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uuid
import numpy as np
import nano_nn as nn
from sklearn.metrics import accuracy_score
from datasets.adult_income_dataset import AdultIncomeDataset
from datasets.medical_cost_dataset import MedicalCostDataset
from loaders import PandasCsvLoader
from schemas import TrainConfig, AdultIncomeInput, MedicalCostInput, DatasetMetadata
from nn.network import SmallModel, MediumModel, LargeModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models: dict[str, nn.Module] = {}
datasets = {
    "adult_income": AdultIncomeDataset(csv_loader=PandasCsvLoader()),
    "medical_cost": MedicalCostDataset(csv_loader=PandasCsvLoader()),
}


@app.post("/api/nn-framework/train")
async def train_model(train_config: TrainConfig):
    model_id = str(uuid.uuid4())
    dataset = datasets[train_config.dataset_name]
    task_type = dataset.task_type()

    if train_config.model_size == "small":
        model = SmallModel(dataset.num_features(),
                           dataset.num_outputs(), task_type)
    elif train_config.model_size == "medium":
        model = MediumModel(dataset.num_features(),
                            dataset.num_outputs(), task_type)
    elif train_config.model_size == "large":
        model = LargeModel(dataset.num_features(),
                           dataset.num_outputs(), task_type)
    else:
        return {"status": "error", "message": f"There is no model with size {train_config.model_size}"}

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


async def predict(dataset_name: str, input, model_id):
    if model_id not in models:
        return {"status": "error", "message": f"Could not find model with model id: {model_id}"}

    model = models[model_id]
    dataset = datasets[dataset_name]

    X = dataset.transform_one(input)

    model.eval()

    output = float(np.asarray(model.forward(X)).squeeze())

    layer_activations = []
    temp = X[:]
    for layer in model.layers:
        temp = layer.forward(temp)

        # We want the raw output from the last layer
        if layer is model.layers[-1]:
            layer_activations.append(temp[0])
        # TODO: Improve the nn framework so I can use isinstance(layer, nn.Activation) instead
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Softmax)):
            min_val = np.min(temp)
            max_val = np.max(temp)

            if max_val > min_val:
                activation_normalized = (temp - min_val) / (max_val - min_val)
            else:
                activation_normalized = np.zeros_like(temp)

            layer_activations.append(activation_normalized[0])

    layer_activations = [a.tolist() for a in layer_activations]

    return {
        "output": output,
        "activations": layer_activations,
    }


@app.post("/api/nn-framework/predict/adult_income")
async def predict_adult(model_id: str, input: AdultIncomeInput):
    return await predict("adult_income", input, model_id)


@app.post("/api/nn-framework/predict/medical_cost")
async def predict_medical(model_id: str, input: MedicalCostInput):
    return await predict("medical_cost", input, model_id)


@app.get("/api/nn-framework/datasets-metadata", response_model=list[DatasetMetadata])
async def datasets_metadata():
    metadatas = []
    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        metadatas.append(dataset.metadata())
    return metadatas


@app.get("/api/nn-framework/networks-metadata")
async def networks_metadata():
    metadatas = [SmallModel.metadata(), MediumModel.metadata(),
                 LargeModel.metadata()]
    return metadatas


@app.get("/api/nn-framework/test")
async def test():
    return {"features": datasets["adult_income"].num_features()}
