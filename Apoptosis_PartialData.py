import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
import time

class HiddenPathways:
    # Initialize the class
    def __init__(self, t, S, layers):
        
        self.D = S.shape[1]
        
        self.t_min = t.min(0)
        self.t_max = t.max(0)
        
        self.S_mean = S.mean(0)
        self.S_std = S.std(0)
        
        # data on velocity (inside the domain)
        self.t = t
        self.S = S
                
        # layers
        self.layers = layers
        
        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

#        self.k1 = tf.Variable(2.67e-9, dtype=tf.float32, trainable=False)
#        self.kd1 = tf.Variable(1e-2, dtype=tf.float32, trainable=False)
#        self.kd2 = tf.Variable(8e-3, dtype=tf.float32, trainable=False)
#        self.k3 = tf.Variable(6.8e-8, dtype=tf.float32, trainable=False)
#        self.kd3 = tf.Variable(5e-2, dtype=tf.float32, trainable=False)
#        self.kd4 = tf.Variable(1e-3, dtype=tf.float32, trainable=False)
#        self.k5 = tf.Variable(7e-5, dtype=tf.float32, trainable=False)
#        self.kd5 = tf.Variable(1.67e-5, dtype=tf.float32, trainable=False)
#        self.kd6 = tf.Variable(1.67e-4, dtype=tf.float32, trainable=False)

        self.logk1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd2 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk3 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd3 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd4 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk5 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd5 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logkd6 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        
        self.var_list_eqns = [self.logk1, self.logkd1, self.logkd2, self.logk3, self.logkd3,
                              self.logkd4, self.logk5, self.logkd5, self.logkd6]
  
        self.k1 = tf.exp(self.logk1)
        self.kd1 = tf.exp(self.logkd1)
        self.kd2 = tf.exp(self.logkd2)
        self.k3 = tf.exp(self.logk3)
        self.kd3 = tf.exp(self.logkd3)
        self.kd4 = tf.exp(self.logkd4)
        self.k5 = tf.exp(self.logk5)
        self.kd5 = tf.exp(self.logkd5)
        self.kd6 = tf.exp(self.logkd6)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.global_step = tf.Variable(0, trainable=False)
        self.sess.run(self.global_step.initializer)
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.S_tf = tf.placeholder(tf.float32, shape=[None, self.D])
        
        # placeholders for forward differentian
        self.dummy_tf = tf.placeholder(tf.float32, shape=(None, self.D)) # dummy variable for fwd_gradients (D outputs)
        
        # physics informed neural networks
        (self.S_pred,
         self.E_pred) = self.net_HiddenPathways(self.t_tf)

        # loss
        self.loss_data = tf.reduce_mean(tf.square((self.S_tf[:,3:6] - self.S_pred[:,3:6])/self.S_std[3:6]))
        self.loss_eqns = tf.reduce_mean(tf.square(self.E_pred/self.S_std))
        self.loss_auxl = tf.reduce_mean(tf.square((self.S_tf[-1,:]-self.S_pred[-1,:])/self.S_std))
        self.loss = 0.89*self.loss_data + 0.1*self.loss_eqns + 0.01*self.loss_auxl
        
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.optimizer_para = tf.train.AdamOptimizer(learning_rate = 0.001)#self.custom_decay(10000, 0.002, 0.001))
        
        self.train_op = self.optimizer.minimize(self.loss, var_list = self.weights + self.biases,
                                                global_step=self.global_step)
        self.trainpara_op = self.optimizer_para.minimize(self.loss, var_list = self.var_list_eqns)
#        self.grads, self.vars = zip(*self.optimizer_para.compute_gradients(self.loss,
#                                                                           var_list = self.var_list_eqns))
#        self.gradsnorm, _ = tf.clip_by_global_norm(self.grads, 1.0)
#        self.trainpara_op = self.optimizer_para.apply_gradients(zip(self.gradsnorm, self.vars))
        self.weightnorm, _ = tf.clip_by_global_norm(self.weights, 1.0)
    
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def custom_decay(self, decay_steps, lr_amp, lr_min):
        global_step = self.global_step.eval(session=self.sess)
        return lr_amp*np.abs(np.sin(np.pi*np.float32(global_step/decay_steps))) + lr_min

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.matmul(H, W) + b
            H = H*tf.sigmoid(H)
        W = weights[-1]
        b = biases[-1]
        Y = tf.matmul(H, W) + b
        return Y
    
    def fwd_gradients(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_tf)[0]
        return tf.gradients(g, self.dummy_tf)[0]
    
    def net_HiddenPathways(self, t):
        H = 2.0*(t - self.t_min)/(self.t_max - self.t_min) - 1.0
        S_tilde = self.neural_net(H, self.weights, self.biases)
#        S = self.S_mean + self.S_std*S_tilde
        S = self.S[0,:] + self.S_std*(H + 1.0)*S_tilde

        F1 = -self.k1*S[:,3:4]*S[:,0:1] + self.kd1*S[:,4:5]
        F2 = self.kd2*S[:,4:5] - self.k3*S[:,1:2]*S[:,2:3] + self.kd3*S[:,5:6] + self.kd4*S[:,5:6]
        F3 = -self.k3*S[:,1:2]*S[:,2:3] + self.kd3*S[:,5:6]
        F4 = self.kd4*S[:,5:6] - self.k1*S[:,3:4]*S[:,0:1] + self.kd1*S[:,4:5] - \
            self.k5*S[:,6:7]*S[:,3:4] + self.kd5*S[:,7:8] + self.kd2*S[:,4:5]
        F5 = -self.kd2*S[:,4:5] + self.k1*S[:,3:4]*S[:,0:1] - self.kd1*S[:,4:5]
        F6 = -self.kd4*S[:,5:6] + self.k3*S[:,1:2]*S[:,2:3] - self.kd3*S[:,5:6]
        F7 = -self.k5*S[:,6:7]*S[:,3:4] + self.kd5*S[:,7:8] + self.kd6*S[:,7:8]
        F8 = self.k5*S[:,6:7]*S[:,3:4] - self.kd5*S[:,7:8] - self.kd6*S[:,7:8]

        F = tf.concat([F1, F2, F3, F4, F5, F6, F7, F8], 1)

        S_t = self.fwd_gradients(S, t)
        
        E = S_t - F
        
        return S, E
    
    def train(self, num_epochs, batch_size, learning_rate):

        for epoch in range(num_epochs):
            
            N = self.t.shape[0]
            perm = np.concatenate( (np.array([0]), np.random.permutation(np.arange(1,N)),
                                    np.array([N])) )
            
            start_time = time.time()
            for it in range(0, N, batch_size):
                idx = perm[np.arange(it,it+batch_size)]
                (t_batch,
                 S_batch) = (self.t[idx,:],
                             self.S[idx,:])

                tf_dict = {self.t_tf: t_batch, self.S_tf: S_batch,
                           self.dummy_tf: np.ones((batch_size, self.D)),
                           self.learning_rate: learning_rate}

#                self.weights = self.sess.run(self.weightnorm)
                self.sess.run([self.train_op, self.trainpara_op], tf_dict)
#                               (self.grads, self.vars), self.gradsnorm, self.trainpara_op], tf_dict)
                
                # Print
                if it % batch_size == 0:
                    elapsed = time.time() - start_time
                    [loss_data_value,
                     loss_eqns_value,
                     loss_auxl_value,
                     learning_rate_value] = self.sess.run([self.loss_data,
                                                           self.loss_eqns,
                                                           self.loss_auxl,
                                                           self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss Data: %.3e, Loss Eqns: %.3e, Loss Auxl: %.3e, Time: %.3f, Learning Rate: %.1e'
                          %(epoch, it/batch_size, loss_data_value, loss_eqns_value, loss_auxl_value, elapsed, learning_rate_value))
                    start_time = time.time()

    def predict(self, t_star):
        
        tf_dict = {self.t_tf: t_star, self.dummy_tf: np.ones((t_star.shape[0], self.D))}
        
        S_star = self.sess.run(self.S_pred, tf_dict)
        
        return S_star

if __name__ == "__main__": 
    
    layers = [1] + 5*[8*30] + [8]
    
    t_scale = 3600
    c_scale = 1e5
    # function that returns dx/dt
    def f(x,t): # x is 8 x 1
        k1 = 2.67e-9*c_scale*t_scale
        kd1 = 1e-2*t_scale
        kd2 = 8e-3*t_scale
        k3 = 6.8e-8*c_scale*t_scale
        kd3 = 5e-2*t_scale
        kd4 = 1e-3*t_scale
        k5 = 7e-5*c_scale*t_scale
        kd5 = 1.67e-5*t_scale
        kd6 = 1.67e-4*t_scale
      
        f1 = -k1*x[3]*x[0] + kd1*x[4]
        f2 = kd2*x[4] - k3*x[1]*x[2] + kd3*x[5] + kd4*x[5]
        f3 = -k3*x[1]*x[2] + kd3*x[5]
        f4 = kd4*x[5] - k1*x[3]*x[0] + kd1*x[4] - k5*x[6]*x[3] + kd5*x[7] + kd2*x[4]
        f5 = -kd2*x[4] + k1*x[3]*x[0] - kd1*x[4]
        f6 = -kd4*x[5] + k3*x[1]*x[2] - kd3*x[5]
        f7 = -k5*x[6]*x[3] + kd5*x[7] + kd6*x[7]
        f8 = k5*x[6]*x[3] - kd5*x[7] - kd6*x[7]
        
        f = np.array([f1,f2,f3,f4,f5,f6,f7,f8])
        return f
        
    def addNoise(S):
        std = (0.05*S).mean(0)
        S[1:,3:6] += np.random.normal(0.0, std[3:6], (S.shape[0]-1,3))
        return S
    
    # time points
    t_star = np.concatenate((np.arange(0,60,0.1), np.arange(60,60,0.5)))
    
    S1 = 1.34e5/c_scale
    S2 = 1e5/c_scale
    S3 = 2.67e5/c_scale
    S4 = 0.0
    S5 = 0.0
    S6 = 0.0
    S7 = 2.9e3/c_scale
    S8 = 0.0
    
    # initial condition
    x0 = np.array([S1,S2,S3,S4,S5,S6,S7,S8]).flatten()
    
    # solve ODE
    S_star = odeint(f, x0, t_star)

    t_train = t_star[:,None]
    add_noise = True
    if add_noise:
        S_train = addNoise(S_star)
    else:
        S_train = S_star[:]
    N_train = t_train.shape[0]
    N_perm = np.int32(N_train/5)
    perm = np.concatenate( (np.array([0]), np.random.randint(1, high=N_train-1, size=N_perm),
                            np.array([N_train-1])) )
    
    model = HiddenPathways(t_train[perm], S_train[perm,:], layers)

    model.train(num_epochs = 25000, batch_size = perm.shape[0], learning_rate = 1e-3)
    model.train(num_epochs = 25000, batch_size = perm.shape[0], learning_rate = 1e-4)
#    model.train(num_epochs = 60000, batch_size = perm.shape[0], learning_rate = 1e-5)
#    model.train(num_epochs = 40000, batch_size = perm.shape[0], learning_rate = 1e-6)

    S_pred = model.predict(t_star[:,None])
    
    ####### Plotting ##################

    fig, ax = newfig(2.0, 0.7)
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.5, wspace=0.3)

    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,S_star[:,3],'C1',linewidth=2,label='input data')
    ax.scatter(t_star[perm],S_star[perm,3],marker='o',s=50,label='sampled input')
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_4 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)
    ax.legend(fontsize='large')
    
    ####################################
    
    fig, ax = newfig(3.5, 0.4)
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)    
    ax = plt.subplot(gs1[0:1, 0:1])
    ax.plot(t_star,S_star[:,0],'C1',linewidth=2,label='exact')
    ax.plot(t_star,S_pred[:,0],'g-.',linewidth=3,label='learned')
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_1 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)
    ax.legend(fontsize='large')
    
    ax = plt.subplot(gs1[0:1, 1:2])
    ax.plot(t_star,S_star[:,1],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,1],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_2 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)

    ax = plt.subplot(gs1[0:1, 2:3])
    ax.plot(t_star,S_star[:,2],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,2],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_3 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)

    fig, ax = newfig(3.5, 0.4)
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)    
    ax = plt.subplot(gs2[0:1, 0:1])    
    ax.scatter(t_star[perm],S_star[perm,3],marker='o',c='C1',s=30)
    ax.plot(t_star,S_pred[:,3],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_4 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)
    
    ax = plt.subplot(gs2[0:1, 1:2])
    ax.plot(t_star,S_star[:,4],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,4],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_5 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)
    
    ax = plt.subplot(gs2[0:1, 2:3])
    ax.plot(t_star,S_star[:,5],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,5],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_6 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)

    fig, ax = newfig(1.8, 0.75)
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    ax = plt.subplot(gs3[0:1, 0:1])
    ax.plot(t_star,S_star[:,6],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,6],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_7 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)
    
    ax = plt.subplot(gs3[0:1, 1:2])
    ax.plot(t_star,S_star[:,7],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,7],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (hours)$', fontsize=18)
    ax.set_ylabel('$x_8 \ \mathrm{x} \ 10^5 \ (molecules/cell)$', fontsize=18)

    print('k1 = %.4e' % ( model.sess.run(model.k1)/t_scale/c_scale ) )
    print('kd1 = %.4e' % ( model.sess.run(model.kd1)/t_scale ) )
    print('kd2 = %.4e' % ( model.sess.run(model.kd2)/t_scale ) )
    print('k3 = %.4e' % ( model.sess.run(model.k3)/t_scale/c_scale ) )
    print('kd3 = %.4e' % ( model.sess.run(model.kd3)/t_scale ) )
    print('kd4 = %.4e' % ( model.sess.run(model.kd4)/t_scale ) )
    print('k5 = %.4e' % ( model.sess.run(model.k5)/t_scale/c_scale ) )
    print('kd5 = %.4e' % ( model.sess.run(model.kd5)/t_scale ) )
    print('kd6 = %.4e' % ( model.sess.run(model.kd6)/t_scale ) )
    
    # savefig('./figures/Glycolytic', crop = False)