
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import uuid
import numpy as np
from sklearn.metrics import accuracy_score
from datasets.adult_income_dataset import AdultIncomeDataset
from datasets.medical_cost_dataset import MedicalCostDataset
from loaders import PandasCsvLoader
from schemas import TrainConfig, AdultIncomeInput, MedicalCostInput, DatasetMetadata
from nn.builder import SmallModel, MediumModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models = {}
datasets = {
    "adult_income": AdultIncomeDataset(csv_loader=PandasCsvLoader()),
    "medical_cost": MedicalCostDataset(csv_loader=PandasCsvLoader()),
}


@app.post("/api/nn-framework/train/")
async def train_model(train_config: TrainConfig):
    model_id = str(uuid.uuid4())
    dataset = datasets[train_config.dataset_name]
    task_type = dataset.task_type()

    if train_config.model_size == "small":
        model = SmallModel(dataset.num_features(),
                           dataset.num_outputs(), task_type)
    else:
        # TODO: Add more models
        model = MediumModel(dataset.num_features(),
                            dataset.num_outputs(), task_type)

    dataset = datasets[train_config.dataset_name]

    training_loss = []
    training_accuracy = []
    for i in range(train_config.epochs):
        model.train()
        output, loss = model.forward(dataset.X, dataset.y)
        training_loss.append(loss)

        if dataset.task_type() != "regression":
            training_accuracy.append(accuracy_score(dataset.y, (output.squeeze() > 0.5)))
        model.backward()

        print(f'epoch: {i+1}/{train_config.epochs}, loss: {loss:.6f}')

    models[model_id] = model

    return {"model_id": model_id, "training_loss": training_loss, "training_accuracy": training_accuracy}


async def predict(dataset_name: str, input, model_id):
    if model_id not in models:
        return {"status": "error", "message": f"Could not find model with model id: {model_id}"}

    model = models[model_id]
    dataset = datasets[dataset_name]

    X = dataset.transform_one(input)

    model.eval()

    output = float(np.asarray(model.forward(X)).squeeze())

    return {
        "output": output,
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
    metadatas = [SmallModel.metadata(), MediumModel.metadata()]
    return metadatas


@app.get("/api/nn-framework/test")
async def test():
    return {"features": datasets["adult_income"].num_features()}
