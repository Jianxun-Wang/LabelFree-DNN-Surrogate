import numpy as np
import tensorflow as tf
import pdb
from matplotlib import pyplot as plt


re = 200.0    # Reynolds number re = U(2R)/nu
nuMean = 0.001
nuStd = 0.9
L = 1.0      # length of pipe
R = 0.05       #
rho = 1       # density
periodicBC = True # or false
dP = 0.1

eps = 1e-4
coef_reg = 1e-5

learning_rate = 5e-3
npoch = 3000# 5000
batch_size = 128

N_x = 10
N_y = 50
N_p = 50

n_h = 50
display_step = 100

xStart = 0
xEnd = xStart + L
yStart = -R
yEnd = yStart + 2*R

## prepare data with (?, 2)
data_1d_x = np.linspace(xStart, xEnd, N_x, endpoint=True)
data_1d_y = np.linspace(yStart, yEnd, N_y, endpoint=True)
nuStart = nuMean-nuMean*nuStd
nuEnd = nuMean+nuMean*nuStd
# nuStart = 0.0001
# nuEnd = 0.1
data_1d_nu = np.linspace(nuStart, nuEnd, N_p, endpoint=True)
print('train_nu is',data_1d_nu)
np.savez('train_nu',nu_1d = data_1d_nu)


data_2d_xy_before = np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu))
data_2d_xy_before_reshape = data_2d_xy_before.reshape(3, -1)
data_2d_xy = data_2d_xy_before_reshape.T



num_steps = npoch*(N_x*N_y*N_p)/batch_size


def myswish_beta(x):
    """
    Swish activation - with beta not-traininable!

    """
    beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
    return x * tf.nn.sigmoid(x*beta)

class classDataGenerator(object):

    def __init__(self, data_2d_xy):

        self.data = np.copy(data_2d_xy)
        np.random.shuffle(self.data)

        self.batch_index = 0
        self.total_data_num = self.data.shape[0]

    def next_batch(self, batch_size):

        if self.batch_index + batch_size < self.total_data_num:

            batch_x = self.data[self.batch_index: self.batch_index + batch_size, 0]
            batch_y = self.data[self.batch_index: self.batch_index + batch_size, 1]
            batch_nu = self.data[self.batch_index: self.batch_index + batch_size, 2]
            self.batch_index = self.batch_index + batch_size

        else:

            batch_x = self.data[self.batch_index: self.total_data_num, 0]
            batch_y = self.data[self.batch_index: self.total_data_num, 1]
            batch_nu = self.data[self.batch_index: self.batch_index + batch_size, 2]
            self.batch_index = 0

        batch_x = batch_x.reshape(-1,1)
        batch_y = batch_y.reshape(-1,1)
        batch_nu = batch_nu.reshape(-1,1)

        return batch_x, batch_y, batch_nu


act = myswish_beta


x = tf.placeholder('float',[None, 1])
y = tf.placeholder('float',[None, 1])
nu = tf.placeholder('float', [None, 1])
res_true = tf.placeholder('float', [None, 1])

if periodicBC:
    b = 2*np.pi/(xEnd-xStart)
    c = np.pi*(xStart+xEnd)/(xStart-xEnd)
    sin_x = xStart*tf.sin(b*x+c)
    cos_x = xStart*tf.cos(b*x+c)
    n1Layer = 4
    input = tf.concat([sin_x, cos_x, y, nu], axis=1)
else:
    n1Layer = 3
    input = tf.concat([x, y, nu], axis=1)

init = tf.contrib.layers.xavier_initializer()
# U
W_1_u = tf.Variable(init([n1Layer, n_h]))
W_2_u = tf.Variable(init([n_h, n_h]))
W_3_u = tf.Variable(init([n_h, n_h]))
W_4_u = tf.Variable(init([n_h, 1]))

b_1_u = tf.Variable(init([1, n_h]))
b_2_u = tf.Variable(init([1, n_h]))
b_3_u = tf.Variable(init([1, n_h]))
b_4_u = tf.Variable(init([1, 1]))

# v
W_1_v = tf.Variable(init([n1Layer, n_h]))
W_2_v = tf.Variable(init([n_h, n_h]))
W_3_v = tf.Variable(init([n_h, n_h]))
W_4_v = tf.Variable(init([n_h, 1]))

b_1_v = tf.Variable(init([1, n_h]))
b_2_v = tf.Variable(init([1, n_h]))
b_3_v = tf.Variable(init([1, n_h]))
b_4_v = tf.Variable(init([1, 1]))


# p
W_1_p = tf.Variable(init([n1Layer, n_h]))
W_2_p = tf.Variable(init([n_h, n_h]))
W_3_p = tf.Variable(init([n_h, n_h]))
W_4_p = tf.Variable(init([n_h, 1]))

b_1_p = tf.Variable(init([1, n_h]))
b_2_p = tf.Variable(init([1, n_h]))
b_3_p = tf.Variable(init([1, n_h]))
b_4_p = tf.Variable(init([1, 1]))


# u_nn(x,y)
u_nn = tf.matmul(act(tf.matmul(act(tf.matmul(act(tf.matmul(input, W_1_u) + b_1_u), W_2_u) + b_2_u), W_3_u) + b_3_u), W_4_u) + b_4_u

# v_nn(x,y)
v_nn = tf.matmul(act(tf.matmul(act(tf.matmul(act(tf.matmul(input, W_1_v) + b_1_v), W_2_v) + b_2_v), W_3_v) + b_3_v), W_4_v) + b_4_v

# p_nn(x,y)
p_nn = tf.matmul(act(tf.matmul(act(tf.matmul(act(tf.matmul(input, W_1_p) + b_1_p), W_2_p) + b_2_p), W_3_p) + b_3_p), W_4_p) + b_4_p


# data generator
dataGenerator = classDataGenerator(data_2d_xy=data_2d_xy)


#################################
# enforcing boudnary condition 
#################################

# u = tf.nn.tanh(eps/(1.0 + 0.2*eps - y)) + (1.0 - x**2)*(1.0 - y**2)*u_nn

# u = tf.nn.tanh(eps/(1.0 + 0.2*eps - y)) + (1.0 - x**2)*(1.0 - y**2)*u_nn


# Impose pressure gradient as a constant
# u = u_nn*(R**2 - y**2)
# v = (R - y**2)*v_nn
# p = dP - dP*(x-xStart)/L + 0*y

# Impose pressure drop
u = u_nn*(R**2 - y**2)
v = (R**2 - y**2)*v_nn
p = (xStart-x)*0 + dP*(xEnd-x)/L + 0*y + (xStart - x)*(xEnd - x)*p_nn
#p = (1-x)*200 + (1+x)*0 + (1 - x**2)*p_nn # iniitial loss is super large

# Impose velocity
#u = tf.nn.tanh(eps/(1.0 + 0.2*eps + x)) + u_nn*(1.0 - y**2)*(1.0 + x)
#v = (1.0 - x**2)*(1.0 - y**2)*v_nn
#p = p_nn


#################################
# enforcing PDE loss 
#################################

dudx = tf.gradients(u,x)[0]

dudy = tf.gradients(u,y)[0]

du2dx2 = tf.gradients(dudx,x)[0]

du2dy2 = tf.gradients(dudy,y)[0]

dvdx = tf.gradients(v,x)[0]

dvdy = tf.gradients(v,y)[0]

dv2dx2 = tf.gradients(dvdx,x)[0]

dv2dy2 = tf.gradients(dvdy,y)[0]


dpdx = tf.gradients(p,x)[0]

dpdy = tf.gradients(p,y)[0]


# normalized NS equation
#res_mom_u = u*dudx + v*dudy + dpdx - (du2dx2 + du2dy2)/re
#res_mom_v = u*dvdx + v*dvdy + dpdy - (dv2dx2 + dv2dy2)/re
#res_cont  = dudx + dvdy

# 
res_mom_u = u*dudx + v*dudy + 1/rho*dpdx - (du2dx2 + du2dy2)*nu
res_mom_v = u*dvdx + v*dvdy + 1/rho*dpdy - (dv2dx2 + dv2dy2)*nu
res_cont  = dudx + dvdy



loss = tf.reduce_mean(tf.pow(res_true - res_mom_u, 2)) + \
       tf.reduce_mean(tf.pow(res_true - res_mom_v, 2)) + \
       tf.reduce_mean(tf.pow(res_true - res_cont, 2)) 

# train_step = tf.contrib.opt.ScipyOptimizerInterface(
                # loss,
                # method='L-BFGS-B',
                # options={'maxiter': 100})

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, int(num_steps+1)):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y, batch_nu = dataGenerator.next_batch(batch_size)

        res_batch = np.zeros(batch_x.shape)


        # Run optimization op (backprop) and cost op (to get loss value)
        # train_step.minimize(sess, feed_dict={x: batch_x, y: batch_y, res_true: res_batch})

        # compute loss
        # l = sess.run(loss, feed_dict={x: batch_x, y: batch_y, res_true: res_batch})
        
        _, l = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y, nu: batch_nu, res_true: res_batch})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    u_pred_2d_xy_list = sess.run([u,v,p], feed_dict={x: data_2d_xy[:,0:1], y: data_2d_xy[:,1:2], 
    												 nu: data_2d_xy[:,2:3], res_true: np.zeros(data_2d_xy[:,0:1].shape)})


	# test normal distribution of maxvelocity
    #N_pTest = 200
    N_pTest = 500
    data_1d_nuDist = np.random.normal(nuMean, 0.2*nuMean, N_pTest)
    data_2d_xy_before_test = np.array(np.meshgrid((xStart-xEnd)/2., 0, data_1d_nuDist))
    data_2d_xy_before_test_reshape = data_2d_xy_before_test.reshape(3, -1)
    data_2d_xy_test = data_2d_xy_before_test_reshape.T
    data_2d_xy_test = data_2d_xy_before_test_reshape.T
    uMax_pred_list = sess.run([u,v,p], feed_dict={x: data_2d_xy_test[:,0:1], y: data_2d_xy_test[:,1:2], 
												 nu: data_2d_xy_test[:,2:3], res_true: np.zeros(data_2d_xy_test[:,0:1].shape)})

#print('shape of uMax_pred',uMax_pred.shape)

uMax_pred = uMax_pred_list[0].T
print('uMax_pred is',uMax_pred)
print('shape of uMax_pred is',uMax_pred.shape)


u_pred_2d_xy = u_pred_2d_xy_list[0].T
v_pred_2d_xy = u_pred_2d_xy_list[1].T
p_pred_2d_xy = u_pred_2d_xy_list[2].T

u_pred_2d_xy_mesh = u_pred_2d_xy.reshape(N_y, N_x, N_p)
v_pred_2d_xy_mesh = v_pred_2d_xy.reshape(N_y, N_x, N_p)
p_pred_2d_xy_mesh = p_pred_2d_xy.reshape(N_y, N_x, N_p)

# analytical solution
uSolaM = np.zeros([N_y, N_x, N_p])
for i in range(N_p):
	uy = (R**2 - data_1d_y**2)*dP/(2*L*data_1d_nu[i]*rho)
	uSolaM[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

uMax_a = np.zeros([1, N_pTest])
for i in range(N_pTest):
	uMax_a[0, i] = (R**2)*dP/(2*L*data_1d_nuDist[i]*rho)

print (data_2d_xy_before.shape)

print (u_pred_2d_xy_mesh.shape)


# plt.figure()
# plt.contourf(data_2d_xy_before_full[0,:,:], data_2d_xy_before_full[1,:,:], u_pred_2d_xy_mesh)
# plt.contourf(data_2d_xt_before[0])
# plt.colorbar()
# plt.savefig('./u_2d_lid_cavity.png')
# plt.close()


np.savez('pred_poiseuille_para', mesh=data_2d_xy_before, u=u_pred_2d_xy_mesh, 
							v=v_pred_2d_xy_mesh, p=p_pred_2d_xy_mesh, ut=uSolaM,
							uMaxP=uMax_pred, uMaxA=uMax_a)




















