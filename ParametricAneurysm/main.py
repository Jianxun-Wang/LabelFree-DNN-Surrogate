import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import geo_train
import mesh_gen

# Constants and parameters
PARAMS = {
    "device": torch.device("cpu"),
    "epochs": 3,
    "L": 1,
    "xStart": 0,
    "rInlet": 0.05,
    "n_x": 100,
    "N_y": 20,
    "nu": 1e-3,
    "sigma": 0.1,
    "scaleStart": -0.02,
    "scaleEnd": 0.0,
    "Ng": 50,
    "dP": 0.1,
    "g": 9.8,
    "rho": 1,
    "batchsize": 50,
    "learning_rate": 1e-3,
    "path": "Cases/"
}

def generate_2d_mesh(unique_x, N_y):
    x_2d = np.tile(unique_x, N_y)
    return np.reshape(x_2d, (len(x_2d), 1))

def generate_3d_mesh(unique_x, x_2d, scale_1d):
    return mesh_gen.create_3d_mesh(unique_x, x_2d, scale_1d)

def calculate_y_values(x, unique_x, yUp):
    y = np.zeros([len(x), 1])
    for x0 in unique_x:
        index = np.where(x[:, 0] == x0)[0]
        Rsec = max(yUp[index])
        tmpy = np.linspace(-Rsec, Rsec, len(index)).reshape(len(index), -1)
        y[index] = tmpy
    return y

# Mesh generation
xEnd = PARAMS["xStart"] + PARAMS["L"]
PARAMS["xEnd"] = xEnd
mu = 0.5 * (PARAMS["xEnd"] - PARAMS["xStart"])
PARAMS["mu"] = mu
unique_x = np.linspace(PARAMS["xStart"], xEnd, PARAMS["n_x"])
x_2d = generate_2d_mesh(unique_x, PARAMS["N_y"])
scale_1d = np.linspace(PARAMS["scaleStart"], PARAMS["scaleEnd"], PARAMS["Ng"], endpoint=True)
x, scale = generate_3d_mesh(unique_x, x_2d, scale_1d)

# Calculate stenosis and y values
R = scale * 1 / np.sqrt(2 * np.pi * PARAMS["sigma"]**2) * np.exp(-(x - mu)**2 / (2 * PARAMS["sigma"]**2))
yUp = (PARAMS["rInlet"] - R) * np.ones_like(x)
yDown = (-PARAMS["rInlet"] + R) * np.ones_like(x)
y = calculate_y_values(x, unique_x, yUp)

# Plot the stenosis for the first scale value
idx = np.where(scale == PARAMS["scaleStart"])
plt.figure()
plt.scatter(x, yUp)
plt.scatter(x, yDown)
plt.axis('equal')
plt.show()

# Ensure the path exists
if not os.path.exists(PARAMS["path"]):
    os.makedirs(PARAMS["path"])

# Train the model using geo_train
tic = time.time()
geo_train.train_network(device=PARAMS["device"], sigma= PARAMS["sigma"], scale= scale, xStart=PARAMS["xStart"], xEnd=PARAMS["xEnd"], L=PARAMS["L"],
                    rInlet=PARAMS["rInlet"], x=x, y=y, dP=PARAMS["dP"], nu=PARAMS["nu"], rho=PARAMS["rho"], batchsize=PARAMS["batchsize"], learning_rate=PARAMS["learning_rate"], epochs=PARAMS["epochs"], path=PARAMS["path"], mu=PARAMS["mu"])

toc = time.time()
elapsedTime = toc - tic
print("Elapsed time in serial =", elapsedTime)