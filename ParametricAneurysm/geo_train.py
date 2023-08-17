import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Constants
H_ND = 30
H_N = 20
INPUT_N = 3


# Activation Function
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(torch.sigmoid(x)) if self.inplace else x * torch.sigmoid(x)


# Neural Network Definitions
class BaseNet(nn.Module):
    def __init__(self, input_dim, layer_structure, final_out=1, activation_fn=Swish):
        super(BaseNet, self).__init__()
        layers = []
        for i in range(len(layer_structure)):
            layers.append(nn.Linear(input_dim, layer_structure[i]))
            layers.append(activation_fn())
            input_dim = layer_structure[i]
        layers.append(nn.Linear(layer_structure[-1], final_out))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Training Utilities
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)


import torch
import numpy as np

def criterion(x, y, scale, net2, net3, net4, sigma, rInlet, xStart, xEnd, dP, L, nu, rho, mu, device):
    """
    Compute the loss based on given parameters and neural network outputs.
    
    Args:
    - x, y: Input tensors representing spatial locations.
    - scale: Input scale tensor.
    - net2, net3, net4: Neural network models.
    - sigma, rInlet, xStart, xEnd, dP, L, nu, rho, mu: Model-specific parameters.
    - device: Device to which tensors are to be moved for computation.
    
    Returns:
    - loss: Computed loss value.
    """

    # Convert inputs to the specified device
    x = torch.FloatTensor(x).to(device)
    y = torch.FloatTensor(y).to(device)
    scale = torch.FloatTensor(scale).to(device)

    # Enable gradients for these variables
    x.requires_grad = True
    y.requires_grad = True
    scale.requires_grad = True

    # Concatenate inputs for neural network
    net_in = torch.cat((x, y, scale), 1)

    # Get the output from the neural networks
    u = net2(net_in)
    v = net3(net_in)
    P = net4(net_in)
    u = u.view(len(u),-1)
    v = v.view(len(v),-1)
    P = P.view(len(P),-1)

    # Analytical symmetric boundary conditions
    R = scale * 1/np.sqrt(2 * np.pi * sigma**2) * torch.exp(-(x - mu)**2 / (2 * sigma**2)).to(device)
    h = rInlet - R.to(device)
    
    # Hard constraints for velocity and pressure
    u_hard = u * (h**2 - y**2)
    v_hard = (h**2 - y**2) * v
    P_hard = (xStart - x) * 0 + dP * (xEnd - x) / L + 0 * y + (xStart - x) * (xEnd - x) * P

    # Compute derivatives w.r.t x and y for u and P using autograd
    u_x = torch.autograd.grad(u_hard, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    u_y = torch.autograd.grad(u_hard, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    P_x = torch.autograd.grad(P_hard, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    # Loss component for u (momentum equation in x-direction)
    loss_1 = (u_hard * u_x + v_hard * u_y - nu * (u_xx + u_yy) + 1/rho * P_x)

    # Compute derivatives w.r.t x and y for v using autograd
    v_x = torch.autograd.grad(v_hard, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    v_y = torch.autograd.grad(v_hard, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    P_y = torch.autograd.grad(P_hard, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    # Loss component for v (momentum equation in y-direction)
    loss_2 = (u_hard * v_x + v_hard * v_y - nu * (v_xx + v_yy) + 1/rho * P_y)

    # Loss component for continuity equation
    loss_3 = (u_x + v_y)

    # Calculate final MSE loss across all loss components
    loss_f = torch.nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1)) + loss_f(loss_2, torch.zeros_like(loss_2)) + loss_f(loss_3, torch.zeros_like(loss_3))

    return loss


def train_network(device, sigma, x, y, scale, batchsize, learning_rate, epochs, path, dP, nu, rho, rInlet, xStart, xEnd, L, mu):
    # Preparing Data
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y), torch.Tensor(scale))
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)

    # Initializing Networks
    net1_structure = [H_ND, H_ND, H_ND]
    net2_structure = [H_N, H_N, H_N]

    net2 = BaseNet(3, net2_structure).to(device)
    net3 = BaseNet(3, net2_structure).to(device)
    net4 = BaseNet(3, net2_structure).to(device)

    # Weights Initialization
    for net in [net2, net3, net4]:
        net.apply(init_normal)

    # Optimizers
    optimizers = [
        optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
        for net in [net2, net3, net4]
    ]

    # Main loop
    LOSS = []
    tic = time.time()

    for epoch in range(epochs):
        for batch_idx, (x_in, y_in, scale_in) in enumerate(dataloader):
            for optimizer, net in zip(optimizers, [net2, net3, net4]):
                optimizer.zero_grad()

            loss = criterion(x_in, y_in, scale_in, net2, net3, net4, sigma, rInlet, xStart, xEnd, dP, L, nu, rho, mu, device)
            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(x), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
                LOSS.append(loss.item())

            if epoch % 100 == 0:
                for net, name in zip([net2, net3, net4], ["hard_u", "hard_v", "hard_P"]):
                    torch.save(net.state_dict(), path + f"geo_para_axisy_sigma{sigma}_epoch{epoch}{name}.pt")

    toc = time.time()
    print("Elapsed time = ", toc - tic)

    # Save loss
    LOSS = np.array(LOSS)
    np.savetxt('Loss_track_pipe_para.csv', LOSS)

    # Save networks
    for net, name in zip([net2, net3, net4], ["hard_u", "hard_v", "hard_P"]):
        torch.save(net.state_dict(), path + f"geo_para_axisy_sigma{sigma}_epoch{epochs}{name}.pt")


# The main function call can be:
# train_network(device, sigma, x, y, scale, batchsize, learning_rate, epochs, path)
