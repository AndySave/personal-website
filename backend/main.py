
from fastapi import FastAPI, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware

from typing import Annotated
import uuid
import numpy as np
from sklearn.metrics import accuracy_score
from datasets.adult_income_dataset import AdultIncomeDataset
from loaders import PandasCsvLoader
from schemas import TrainConfig, AdultIncomeInput
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
    response.set_cookie(
        key="model_id",
        value=model_id,
        httponly=True,
        secure=False
    )

    model = build_model(train_config.layers)
    dataset = datasets[train_config.dataset_name]

    epochs = 60
    for i in range(epochs):
        model.train()
        output, loss = model.forward(dataset.X, dataset.y)
        model.backward()
        
        print(f'epoch: {i+1}/{epochs}, loss: {loss:.6f}')
    
    models[model_id] = model
    
    model.eval()
    output = np.asarray(model.forward(dataset.X))

    accuracy = accuracy_score(dataset.y, (output.squeeze() > 0.5))

    return {"accuracy": accuracy}


async def predict(dataset_name: str, input, model_id):
    if not model_id or model_id not in models:
        return {"status": "error", "message": "No model trained"}

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
async def predict_adult(input: AdultIncomeInput, model_id: Annotated[str | None, Cookie()] = None):
    return await predict("adult_income", input, model_id)
