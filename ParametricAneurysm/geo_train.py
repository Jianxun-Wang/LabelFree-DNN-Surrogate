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
def geo_train(device,sigma,scale,mu,xStart,xEnd,L,rInlet,x,y,R,yUp,dP,nu,rho,g,batchsize,learning_rate,epochs,path):
	dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y),torch.Tensor(scale))
	dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
	h_nD = 30
	h_n = 20
	input_n = 3
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
				nn.Linear(2,h_nD),
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
				nn.Linear(input_n,h_n),
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
				nn.Linear(input_n,h_n),
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
				nn.Linear(input_n,h_n),
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
				################## below are added layers

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	################################################################
	net1 = Net1().to(device)
	net2 = Net2().to(device)
	net3 = Net3().to(device)
	net4 = Net4().to(device)
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

		# use the modules apply function to recursively apply the initialization
	net1.apply(init_normal)
	net2.apply(init_normal)
	net3.apply(init_normal)
	net4.apply(init_normal)
	############################################################################
	#continue traning network
	#net2.load_state_dict(torch.load("stenosis_para"+"_epoch"+str(epochs)+"hard.pt",map_location = 'cpu'))
	#net2.eval()
	#net2.load_state_dict(torch.load("stenosis_para"+"_epoch"+str(epochs)+"hard_u.pt",map_location = 'cpu'))
	#net3.load_state_dict(torch.load("stenosis_para"+"_epoch"+str(epochs)+"hard_v.pt",map_location = 'cpu'))
	#net4.load_state_dict(torch.load("stenosis_para"+"_epoch"+str(epochs)+"hard_P.pt",map_location = 'cpu'))
	#net2.eval()
	#net3.eval()
	#net4.eval()

	############################################################################

	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer3	= optim.Adam(net3.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer4	= optim.Adam(net4.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

	def criterion(x,y,scale):

		x = torch.FloatTensor(x).to(device)
		y = torch.FloatTensor(y).to(device)
		scale = torch.FloatTensor(scale).to(device)

		x.requires_grad = True
		y.requires_grad = True
		scale.requires_grad = True
		
		net_in = torch.cat((x,y,scale),1)
		u = net2(net_in)
		v = net3(net_in)
		P = net4(net_in)
		u = u.view(len(u),-1)
		v = v.view(len(v),-1)
		P = P.view(len(P),-1)

		###############
		#analytical symmetric boundary
		R = scale * 1/sqrt(2*np.pi*sigma**2)*torch.exp(-(x-mu)**2/(2*sigma**2)).to(device)
		h = rInlet - R
		h = h.to(device)

		u_hard = u*(h**2 - y**2)
		v_hard = (h**2 -y**2)*v
		P_hard = (xStart-x)*0 + dP*(xEnd-x)/L + 0*y + (xStart - x)*(xEnd - x)*P


		
		u_x = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#P_xx = torch.autograd.grad(P_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		loss_1 = (u_hard*u_x+v_hard*u_y-nu*(u_xx+u_yy)+1/rho*P_x)

		v_x = torch.autograd.grad(v_hard,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		v_y = torch.autograd.grad(v_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		#P_yy = torch.autograd.grad(P_y,y,grad_outputs=torch.ones_like(x),create_graph = True,allow_unused = True)[0]


		loss_2 = (u_hard*v_x+v_hard*v_y - nu*(v_xx+v_yy)+1/rho*P_y)
		#Main_deriv = torch.cat((u_x,u_xx,u_y,u_yy,P_x,v_x,v_xx,v_y,v_yy,P_y),1)
		loss_3 = (u_x + v_y)
		#loss_3 = u_x**2 + 2*u_y*v_x + v_y**2+1/rho*(P_xx + P_yy)
		#loss_3 = loss_3*100




		# MSE LOSS
		loss_f = nn.MSELoss()


		loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))

		return loss

	###################################################################

	# Main loop
	LOSS = []
	tic = time.time()

	for epoch in range(epochs):
		for batch_idx, (x_in,y_in,scale_in) in enumerate(dataloader):


			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			net3.zero_grad()
			net4.zero_grad()
			loss = criterion(x_in,y_in,scale_in)
			loss.backward()
			#return loss
			#loss = closure()
			#optimizer2.step(closure)
			#optimizer3.step(closure)
			#optimizer4.step(closure)
			optimizer2.step() 
			optimizer3.step()
			optimizer4.step()
			if batch_idx % 100 ==0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
					epoch, batch_idx * len(x), len(dataloader.dataset),
					100. * batch_idx / len(dataloader), loss.item()))
				LOSS.append(loss.item())
			if epoch %100 == 0:
				torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
				torch.save(net3.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_v.pt")
				torch.save(net4.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_P.pt")
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	############################################################
	#save loss
	#myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	#with myFile:
		#writer = csv.writer(myFile)
		#writer.writerows(LOSS)
	LOSS = np.array(LOSS)
	np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################

	#save network
	#torch.save(net1.state_dict(),"stenosis_para_axisy_sigma"+str(sigma)+"scale"+str(scale)+"_epoch"+str(epochs)+"boundary.pt")
	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_u.pt")
	torch.save(net3.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_v.pt")
	torch.save(net4.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_P.pt")
	#####################################################################