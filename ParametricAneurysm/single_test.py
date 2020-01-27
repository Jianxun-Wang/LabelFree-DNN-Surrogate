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
#def det_test(xStart,xEnd,nPt,rInlet,L,dP,mu,sigma,scale,epochs,path,device):
def det_test(x,y,nu,dP,mu,sigma,scale,epochs,path1,device,caseIdx):
	h_nD = 30
	h_n = 20
	h_np = 20
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class Net1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(3,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),

				nn.Linear(h_nD,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(3,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net3(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net3, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(3,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net4(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net4, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(3,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				################## below are added layers

				nn.Linear(h_np,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	################################################################
	net1 = Net1()
	net2 = Net2()
	net3 = Net3()
	net4 = Net4()

	pre = ''
	#pre = "../ensemble500"
	#load network
	#net1.load_state_dict(torch.load(pre+"stenosis_para_axisy_sigma"+str(sigma)+"scale"+str(scale)+"_epoch"+str(epochs)+"boundary.pt",map_location = 'cpu'))
	#net1.eval()
	## Geometry cases
	net2.load_state_dict(torch.load(pre+path1+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_u.pt",map_location = 'cpu'))
	net3.load_state_dict(torch.load(pre+path1+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_v.pt",map_location = 'cpu'))
	net4.load_state_dict(torch.load(pre+path1+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_P.pt",map_location = 'cpu'))
	### viscosity cases
	#net2.load_state_dict(torch.load(pre+path1+"stenosis_para_axisy_sigma"+str(sigma)+'scale'+str(scale)+"_epoch"+str(epochs)+"hard_u.pt",map_location = 'cpu'))
	#net3.load_state_dict(torch.load(pre+path1+"stenosis_para_axisy_sigma"+str(sigma)+'scale'+str(scale)+"_epoch"+str(epochs)+"hard_v.pt",map_location = 'cpu'))
	#net4.load_state_dict(torch.load(pre+path1+"stenosis_para_axisy_sigma"+str(sigma)+'scale'+str(scale)+"_epoch"+str(epochs)+"hard_P.pt",map_location = 'cpu'))
	##
	net2.eval()
	net3.eval()
	net4.eval()




	R = scale * 1/sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
	rInlet = 0.05
	#R = np.zeros_like(x)
	yUp = rInlet - R	
	########################################
	xt = torch.FloatTensor(x).to(device)
	yt = torch.FloatTensor(y).to(device)
	xt = xt.view(len(xt),-1)
	yt = yt.view(len(yt),-1)
	scalet = scale*torch.ones_like(xt)
	#nut = nu*torch.ones_like(xt)
	Rt = torch.FloatTensor(yUp).to(device)
	Rt = Rt.view(len(Rt),-1)
	#print('shape of Rt is',Rt.shape)
	###########################
	###################################
	xt.requires_grad = True
	yt.requires_grad = True
	scalet.requires_grad = True
	#nut.requires_grad = True

	net_in = torch.cat((xt,yt,scalet),1)

	u_t = net2(net_in)
	v_t = net3(net_in)
	P_t = net4(net_in)

	u_hard = u_t*(Rt**2 - yt**2)
	v_hard = (Rt**2 -yt**2)*v_t
	L = 1
	xStart = 0
	xEnd = L
	P_hard = (xStart-xt)*0 + dP*(xEnd-xt)/L + 0*yt + (xStart - xt)*(xEnd - xt)*P_t
	#P_hard = (-4*xt**2+3*xt+1)*dP +(xStart - xt)*(xEnd - xt)*P_t

	u_hard = u_hard.cpu().data.numpy()
	v_hard = v_hard.cpu().data.numpy()
	P_hard = P_hard.cpu().data.numpy()
	#path = "/home/luning/OpenFOAM/luning-v1806/run/aneurysmsigma01scale0005_100pt-tmp_"+str(ii)
	np.savez(path1+str(int(caseIdx))+'ML_WallStress_uvp',x_center = x,y_center = y,u_center = u_hard,v_center = v_hard,p_center = P_hard)

	return u_hard,v_hard,P_hard

