
from fastapi import FastAPI, Response, Cookie 
from typing import Literal, Optional, Annotated
import uuid
import nano_nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import AdultIncomeDataset
from schemas import Layer, TrainConfig, CustomInput


def build_model(layers: list[Layer]):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            for layer in layers:
                if layer.type == "dense":
                    self.add(nn.Dense(layer.in_features, layer.out_features))
                elif layer.type == "dropout":
                    self.add(nn.Dropout(0.5))  # TODO: Change later to accept custom rate
                elif layer.type == "relu":
                    self.add(nn.ReLU())
                elif layer.type == "sigmoid":
                    self.add(nn.Sigmoid())
                elif layer.type == "softmax":
                    self.add(nn.Softmax())

    model = Model()
    model.set_learning_rate(0.01)
    model.set_loss_fn(nn.BinaryCrossEntropy())
    model.set_optimizer(nn.Adam())

    return model



app = FastAPI()

models = {}
dataset = AdultIncomeDataset()


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

    epochs = 50
    for i in range(epochs):
        # Training phase: enable dropout, etc.
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
async def predict(custom_input: CustomInput, model_id: Annotated[str | None, Cookie()] = None):
    if not model_id or model_id not in models:
        return {"status": "error", "message": "No model trained"}

    model = models[model_id]

    X = dataset.transform_one(custom_input)

    model.eval()

    output = np.asarray(model.forward(X))
    prob = float(output.squeeze())
    label = ">50K" if prob > 0.5 else "<=50K"

    return {
        "prediction": label,
        "probability": prob
    }

