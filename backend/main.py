
from fastapi import FastAPI, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware

from typing import Annotated
import uuid
import nano_nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
# from dataset import AdultIncomeDataset
from datasets.adult_income_dataset import AdultIncomeDataset
from loaders import PandasCsvLoader
from schemas import Layer, LayerType, TrainConfig, AdultIncodeInput


def add_layer(module: nn.Module, layer: Layer):
    match layer.type:
        case LayerType.dense:
            module.add(nn.Dense(layer.in_features, layer.out_features))
        case LayerType.dropout:
            module.add(nn.Dropout(0.5))  # TODO: Change later to accept custom rate
        case LayerType.relu:
            module.add(nn.ReLU())
        case LayerType.sigmoid:
            module.add(nn.Sigmoid())
        case LayerType.softmax:
            module.add(nn.Softmax())


def build_model(layers: list[Layer]):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            for layer in layers:
                add_layer(self, layer)

    model = Model()
    model.set_learning_rate(0.01)
    model.set_loss_fn(nn.BinaryCrossEntropy())
    model.set_optimizer(nn.Adam())

    return model



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models = {}
dataset = AdultIncomeDataset(csv_loader=PandasCsvLoader())


@app.post("/api/nn-framework/train")
async def train_model(train_config: TrainConfig, response: Response):
    model_id = str(uuid.uuid4())
    response.set_cookie(
        key="model_id",
        value=model_id,
        httponly=True,
        secure=False
    )

    model = build_model(train_config.layers)

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


@app.post("/api/nn-framework/predict")
async def predict(input: AdultIncodeInput, model_id: Annotated[str | None, Cookie()] = None):
    if not model_id or model_id not in models:
        return {"status": "error", "message": "No model trained"}

    model = models[model_id]

    X = dataset.transform_one(input)

    model.eval()

    output = np.asarray(model.forward(X))
    prob = float(output.squeeze())
    label = ">50K" if prob > 0.5 else "<=50K"

    return {
        "prediction": label,
        "probability": prob
    }

