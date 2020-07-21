import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
import time

from utilities import neural_net, fwd_gradients,\
                      tf_session, mean_squared_error, relative_error

class HiddenPathways:
    # Initialize the class
    def __init__(self, t_data, S_data, t_eqns, layers):
        
        self.D = S_data.shape[1]
        
        self.t_min = t_data.min(0)
        self.t_max = t_data.max(0)
        
        self.S_scale = S_data.std(0) # tf.Variable(tf.ones(self.D, tf.float32), trainable=False)
        
        # data on all the species (only some are used as input)
        self.t_data, self.S_data = t_data, S_data
        self.t_eqns = t_eqns
                
        # layers
        self.layers = layers

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
        
        # placeholders for data
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.S_data_tf = tf.placeholder(tf.float32, shape=[None, self.D])
        self.t_eqns_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # physics uninformed neural networks
        self.net_sysbio = neural_net(layers=self.layers)
        
        self.H_data = 2.0*(self.t_data_tf - self.t_min)/(self.t_max - self.t_min) - 1.0
        self.S_data_pred = self.S_data[0,:] + self.S_scale*(self.H_data+1.0)*self.net_sysbio(self.H_data)

        # physics informed neural networks
        self.H_eqns = 2.0*(self.t_eqns_tf - self.t_min)/(self.t_max - self.t_min) - 1.0
        self.S_eqns_pred = self.S_data[0,:] + self.S_scale*(self.H_eqns+1.0)*self.net_sysbio(self.H_eqns)

        self.E_eqns_pred = self.SysODE(self.S_eqns_pred, self.t_eqns_tf)

#        self.S_scale = 0.9*self.S_scale + 0.1*tf.math.reduce_std(self.S_eqns_pred, 0)
#        self.S_scale = tf.map_fn(lambda x: x, self.S_data_std[3:4])
        
        #loss
        self.loss_data = mean_squared_error(self.S_data_tf[:,3:5]/self.S_scale[3:5], self.S_data_pred[:,3:5]/self.S_scale[3:5])
        self.loss_eqns = mean_squared_error(self.E_eqns_pred/self.S_scale, 0.0)
        self.loss_auxl = mean_squared_error(self.S_data_tf[-1,:]/self.S_scale, self.S_data_pred[-1,:]/self.S_scale)
        self.loss = 0.95*self.loss_data + 0.05*self.loss_eqns + 0.05*self.loss_auxl

        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.optimizer_para = tf.train.AdamOptimizer(learning_rate = 0.001)

        self.train_op = self.optimizer.minimize(self.loss,
                                                var_list=[self.net_sysbio.weights,
                                                          self.net_sysbio.biases,
                                                          self.net_sysbio.gammas])
        self.trainpara_op = self.optimizer_para.minimize(self.loss,
                                                         var_list = self.var_list_eqns)
        self.sess = tf_session()
        
    def SysODE(self, S, t):
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

        S_t = fwd_gradients(S, t)
        
        E = S_t - F
        return E
    
    def train(self, num_epochs, batch_size, learning_rate):

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]

        for epoch in range(num_epochs):
            start_time = time.time()
            for it in range(N_eqns//batch_size):
                idx_data = np.concatenate([np.array([0]),
                                          np.random.choice(np.arange(1, N_data-1), min(batch_size, N_data)-2),
                                          np.array([N_data-1])])
                idx_eqns = np.random.choice(N_eqns, batch_size)

                t_data_batch, S_data_batch = self.t_data[idx_data,:], self.S_data[idx_data,:]
                t_eqns_batch = self.t_eqns[idx_eqns,:]
    
                tf_dict = {self.t_data_tf: t_data_batch,
                           self.S_data_tf: S_data_batch,
                           self.t_eqns_tf: t_eqns_batch,
                           self.learning_rate: learning_rate}
                
                self.sess.run([self.train_op, self.trainpara_op], tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    [loss_data_value,
                     loss_eqns_value,
                     loss_auxl_value,
                     learning_rate_value] = self.sess.run([self.loss_data,
                                                           self.loss_eqns,
                                                           self.loss_auxl,
                                                           self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss Data: %.3e, Loss Eqns: %.3e, Loss Aux: %.3e, Time: %.3f, Learning Rate: %.1e'
                          %(epoch, it, loss_data_value, loss_eqns_value, loss_auxl_value, elapsed, learning_rate_value))
                    start_time = time.time()

    def predict(self, t_star):
        
        tf_dict = {self.t_data_tf: t_star}
        
        S_star = self.sess.run(self.S_data_pred, tf_dict)
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
        
    def addNoise(S, noise):
        std = noise*S.std(0)
        S[1:,:] += np.random.normal(0.0, std, (S.shape[0]-1, S.shape[1]))
        return S
    
    # time points
    t_star = np.concatenate((np.arange(0,60,0.1), np.arange(60,60,0.5)))
    N = t_star.shape[0]
    N_eqns = N
    N_data = N // 5
    
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

    noise = 0.0
    t_train = t_star[:,None]
    S_train = addNoise(S_star, noise)

    N0 = 0
    N1 = N // 2
    idx_data = np.concatenate([np.array([N0]),
                               np.random.choice(np.arange(1, N-1), size=N_data, replace=False),
                               np.array([N-1]),
                               np.array([N1])])
    idx_eqns = np.concatenate([np.array([N0]),
                               np.random.choice(np.arange(1, N-1), size=N_eqns-2, replace=False),
                               np.array([N-1])])

    model = HiddenPathways(t_train[idx_data],
                           S_train[idx_data,:],
                           t_train[idx_eqns],
                           layers)

    model.train(num_epochs=15000, batch_size=N_eqns, learning_rate=1e-3)
    model.train(num_epochs=15000, batch_size=N_eqns, learning_rate=1e-4)
    model.train(num_epochs=10000, batch_size=N_eqns, learning_rate=1e-5)

    S_pred = model.predict(t_star[:,None])
    
    ####### Plotting ##################

    fig, ax = newfig(2.0, 0.7)
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.5, wspace=0.3)

    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,S_star[:,3],'C1',linewidth=2,label='input data')
    ax.scatter(t_star[idx_data],S_star[idx_data,3],marker='o',s=50,label='sampled input')
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
    ax.scatter(t_star[idx_data],S_star[idx_data,3],marker='o',c='C1',s=30)
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