import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
from pathlib import Path
import sys
import os

import geo_train
import mesh_gen

#import seaborn as sns
device = torch.device("cpu")
epochs  = 500

L = 1
xStart = 0
xEnd = xStart+L
rInlet = 0.05

nPt = 100
unique_x = np.linspace(xStart, xEnd, nPt)
mu = 0.5*(xEnd-xStart)

######################################################

N_y = 20
x_2d = np.tile(unique_x,N_y)
x_2d = np.reshape(x_2d,(len(x_2d),1))

###################################################################

nu = 1e-3

##########
sigma = 0.1
## negative means aneurysm
scaleStart = -0.02
scaleEnd = 0
Ng = 50
scale_1d = np.linspace(scaleStart,scaleEnd,Ng,endpoint= True)
x,scale = mesh_gen.ThreeD_mesh(unique_x,x_2d,scale_1d,sigma,mu)


# axisymetric boundary
R = scale * 1/sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
#R = 0

# Generate stenosis
yUp = (rInlet - R)*np.ones_like(x)
yDown = (-rInlet + R)*np.ones_like(x)

idx = np.where(scale == scaleStart)
plt.figure()
plt.scatter(x[idx], yUp[idx])
plt.scatter(x[idx], yDown[idx])
plt.axis('equal')
plt.show()

########################################


y = np.zeros([len(x),1])
for x0 in unique_x:
	index = np.where(x[:,0]==x0)[0]
	Rsec = max(yUp[index])
	#print('shape of index',index.shape)
	tmpy = np.linspace(-Rsec,Rsec,len(index)).reshape(len(index),-1)
	#print('shape of tmpy',tmpy.shape)
	y[index] = tmpy

print('shape of x',x.shape)
print('shape of y',y.shape)
print('shape of sacle',scale.shape)




#######################################

dP = 0.1
g = 9.8

rho = 1


############################################
#batchsize = 1480
batchsize = 50
learning_rate = 1e-3


path = "Cases/"

#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train.geo_train(device,sigma,scale,mu,xStart,xEnd,L,rInlet,x,y,R,yUp,dP,nu,rho,g,batchsize,learning_rate,epochs,path)
tic = time.time()

elapseTime = toc - tic
print ("elapse time in serial = ", elapseTime)




