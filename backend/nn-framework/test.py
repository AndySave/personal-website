
import nano_nn as nn
import numpy as np

def normalize_inputs(x_train, x_test):
    """
    Normalize features (height, weight) to have mean 0 and std 1.
    Returns normalized x_train, x_test, and normalization parameters (mean, std).
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm, mean, std


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.add(nn.Dense(2, 10))    
        self.add(nn.ReLU())

        self.add(nn.Dense(10, 10))
        self.add(nn.ReLU())

        self.add(nn.Dense(10, 1))
        self.add(nn.Sigmoid())




model = Model()
model.set_loss_fn(nn.BinaryCrossEntropy())
model.set_learning_rate(0.01)
model.set_optimizer(nn.Adam())


# Heights and weights (x)
x = np.array([
    [172.07, 83.50],
    [167.67, 61.93],
    [174.87, 79.05],
    [183.99, 64.66],
    [179.50, 71.99],
    [163.83, 61.12],
    [174.02, 83.46],
    [162.91, 55.84],
    [183.88, 80.15],
    [177.21, 66.57]
])

# Gender labels (y) — 0 = male, 1 = female
y = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0])

x_test = np.array([
    [169.4, 59.8],
    [182.7, 82.3],
    [160.5, 55.2],
    [176.1, 77.9],
    [165.3, 63.4]
])

# Test labels: 0 = male, 1 = female
y_test = np.array([1, 0, 1, 0, 1])

x, x_test, mean, std = normalize_inputs(x, x_test)


epochs = 100
for epoch in range(epochs):
    model.train()
    output, loss = model.forward(x, y)
    model.backward()

    model.eval()
    output_test, loss_test = model.forward(x_test, y_test)

    print(f'epoch: {epoch+1}/{epochs}, loss: {loss:.6f}, test loss: {loss_test:.6f}')



# -----------------------------
# 🔮 Prediction
# -----------------------------
# Example new sample
x_new = np.array([[185, 80]])

# Apply same normalization using training mean/std
x_new = (x_new - mean) / std

model.eval()
output = model.forward(x_new)
predicted_class = "male" if output[0][0] < 0.5 else "female"
print(f"Predicted output: {predicted_class}")
print(output)
