import numpy as np 
def ThreeD_mesh(x_1d,x_2d,tmp_1d,sigma,mu):
	tmp_3d = np.expand_dims(np.tile(tmp_1d,len(x_2d)),1).astype('float')
	x = []

	for x0 in x_2d:
		  tmpx = np.tile(x0,len(tmp_1d))
		  x.append(tmpx)

	x = np.reshape(x,(len(tmp_3d),1))

	return x,tmp_3d