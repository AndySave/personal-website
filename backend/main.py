
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import uuid
import numpy as np
from sklearn.metrics import accuracy_score
from datasets.adult_income_dataset import AdultIncomeDataset
from loaders import PandasCsvLoader
from schemas import TrainConfig, AdultIncomeInput, DatasetMetadata
from nn.builder import build_model


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
    "adult_income": AdultIncomeDataset(csv_loader=PandasCsvLoader())
}


@app.post("/api/nn-framework/train/")
async def train_model(train_config: TrainConfig, response: Response):
    model_id = str(uuid.uuid4())

    model = build_model(train_config.layers)
    dataset = datasets[train_config.dataset_name]

    training_loss = []
    training_accuracy = []
    for i in range(train_config.epochs):
        model.train()
        output, loss = model.forward(dataset.X, dataset.y)
        training_loss.append(loss)
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

    output = np.asarray(model.forward(X))
    prob = float(output.squeeze())
    label = ">50K" if prob > 0.5 else "<=50K"

    return {
        "prediction": label,
        "probability": prob
    }


@app.post("/api/nn-framework/predict/adult_income")
async def predict_adult(model_id: str, input: AdultIncomeInput):
    return await predict("adult_income", input, model_id)


@app.get("/api/nn-framework/datasets-metadata", response_model=list[DatasetMetadata])
async def datasets_metadata():
    metadatas = []
    for dataset in datasets:
        metadatas.append(datasets[dataset].metadata())
    return metadatas
