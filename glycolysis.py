import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
import time

from utilities import neural_net, fwd_gradients,\
                      tf_session, mean_squared_error, relative_error

class HiddenPathways(object):
    # Initialize the class
    def __init__(self, t_data, S_data, t_eqns, layers):
        
        self.D = S_data.shape[1]
        
        self.t_min = t_data.min(0)
        self.t_max = t_data.max(0)
        
        self.S_scale = tf.Variable(S_data.std(0), dtype=tf.float32, trainable=False)
        
        # data on all the species (only some are used as input)
        self.t_data, self.S_data = t_data, S_data
        self.t_eqns = t_eqns
                
        # layers
        self.layers = layers
        
#        self.J0 = tf.Variable(2.5, dtype=tf.float32, trainable=False)
#        self.k1 = tf.Variable(100.0, dtype=tf.float32, trainable=False)
#        self.k2 = tf.Variable(6.0, dtype=tf.float32, trainable=False)
#        self.k3 = tf.Variable(16.0, dtype=tf.float32, trainable=False)
#        self.k4 = tf.Variable(100.0, dtype=tf.float32, trainable=False)
#        self.k5 = tf.Variable(1.28, dtype=tf.float32, trainable=False)
#        self.k6 = tf.Variable(12.0, dtype=tf.float32, trainable=False)
#        self.k = tf.Variable(1.8, dtype=tf.float32, trainable=False)
#        self.kappa = tf.Variable(13.0, dtype=tf.float32, trainable=False)
#        self.q = tf.Variable(4.0, dtype=tf.float32, trainable=False)
#        self.K1 = tf.Variable(0.52, dtype=tf.float32, trainable=False)
#        self.psi = tf.Variable(0.1, dtype=tf.float32, trainable=False)
#        self.N = tf.Variable(1.0, dtype=tf.float32, trainable=False)
#        self.A = tf.Variable(4.0, dtype=tf.float32, trainable=False)

        self.logJ0 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk2 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk3 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk4 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk5 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk6 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logk = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.logkappa = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logq = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logK1 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.logpsi = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logN = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logA = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        
        self.var_list_eqns = [self.logJ0, self.logk1, self.logk2, self.logk3, self.logk4,
                              self.logk5, self.logk6, self.logk, self.logkappa,
                              self.logq, self.logK1, self.logpsi, self.logN, self.logA]
  
        self.J0 = tf.exp(self.logJ0)
        self.k1 = tf.exp(self.logk1)
        self.k2 = tf.exp(self.logk2)
        self.k3 = tf.exp(self.logk3)
        self.k4 = tf.exp(self.logk4)
        self.k5 = tf.exp(self.logk5)
        self.k6 = tf.exp(self.logk6)
        self.k = tf.exp(self.logk)
        self.kappa = tf.exp(self.logkappa)
        self.q = tf.exp(self.logq)
        self.K1 = tf.exp(self.logK1)
        self.psi = tf.exp(self.logpsi)
        self.N = tf.exp(self.logN)
        self.A = tf.exp(self.logA)

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
#        scale_list = tf.unstack(self.S_scale)
#        scale_list[4:6] = self.S_data.std(0)[4:6]
#        self.S_scale = tf.stack(scale_list)
        
        # loss
        self.loss_data = mean_squared_error(self.S_data_tf[:,4:6]/self.S_scale[4:6], self.S_data_pred[:,4:6]/self.S_scale[4:6])
        self.loss_eqns = mean_squared_error(0.0, self.E_eqns_pred/self.S_scale)
        self.loss_auxl = mean_squared_error(self.S_data_tf[-1,:]/self.S_scale[:], self.S_data_pred[-1,:]/self.S_scale[:])
        self.loss = 0.95*self.loss_data + 0.05*self.loss_eqns + 0.05*self.loss_auxl
        
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer_para = tf.train.AdamOptimizer(learning_rate=0.001)

        self.train_op = self.optimizer.minimize(self.loss,
                                                var_list=[self.net_sysbio.weights,
                                                          self.net_sysbio.biases,
                                                          self.net_sysbio.gammas])
        self.trainpara_op = self.optimizer_para.minimize(self.loss,
                                                         var_list=self.var_list_eqns)
        self.sess = tf_session()

    def SysODE(self, S, t):
        F1 = self.J0 - (self.k1*S[:,0:1]*S[:,5:6])/(1+(S[:,5:6]/self.K1)**self.q)
        F2 = 2*(self.k1*S[:,0:1]*S[:,5:6])/(1+(S[:,5:6]/self.K1)**self.q) - self.k2*S[:,1:2]*(self.N-S[:,4:5]) - self.k6*S[:,1:2]*S[:,4:5]
        F3 = self.k2*S[:,1:2]*(self.N-S[:,4:5]) - self.k3*S[:,2:3]*(self.A-S[:,5:6])
        F4 = self.k3*S[:,2:3]*(self.A-S[:,5:6]) - self.k4*S[:,3:4]*S[:,4:5] - self.kappa*(S[:,3:4]-S[:,6:7])
        F5 = self.k2*S[:,1:2]*(self.N-S[:,4:5]) - self.k4*S[:,3:4]*S[:,4:5] - self.k6*S[:,1:2]*S[:,4:5]
        F6 = -2*(self.k1*S[:,0:1]*S[:,5:6])/(1+(S[:,5:6]/self.K1)**self.q) + 2*self.k3*S[:,2:3]*(self.A-S[:,5:6]) - self.k5*S[:,5:6]
        F7 = self.psi*self.kappa*(S[:,3:4]-S[:,6:7]) - self.k*S[:,6:7]
        
        F = tf.concat([F1, F2, F3, F4, F5, F6, F7], 1)

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
        
        tf_dict = {self.t_eqns_tf: t_star}
        
        S_star, S_scale = self.sess.run([self.S_eqns_pred, self.S_scale], tf_dict)
        return S_star, S_scale

if __name__ == "__main__": 
    
    layers = [1] + 7*[7*40] + [7]
    
    # function that returns dx/dt
    def f(x, t): # x is 7 x 1
        J0 = 2.5
        k1 = 100.0
        k2 = 6.0
        k3 = 16.0
        k4 = 100.0
        k5 = 1.28
        k6 = 12.0
        k = 1.8
        kappa = 13.0
        q = 4.0
        K1 = 0.52
        psi = 0.1
        N = 1.0
        A = 4.0
        
        f1 = J0 - (k1*x[0]*x[5])/(1+(x[5]/K1)**q)
        f2 = 2*(k1*x[0]*x[5])/(1+(x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]
        f3 = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])
        f4 = k3*x[2]*(A-x[5]) - k4*x[3]*x[4] - kappa*(x[3]-x[6])
        f5 = k2*x[1]*(N-x[4]) - k4*x[3]*x[4] - k6*x[1]*x[4]
        f6 = -2*(k1*x[0]*x[5])/(1+(x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]
        f7 = psi*kappa*(x[3]-x[6]) - k*x[6]
        
        f = np.array([f1, f2, f3, f4, f5, f6, f7])
        return f

    def addNoise(S, noise):
        std = noise*S.std(0)
        S[1:,:] += np.random.normal(0.0, std, (S.shape[0]-1, S.shape[1]))
        return S
        
    # time points
    t_star = np.arange(0, 10, 0.005)
    N = t_star.shape[0]
    N_eqns = N
    N_data = N // 4
    
    S1 = np.random.uniform(0.15, 1.60, 1)
    S2 = np.random.uniform(0.19, 2.16, 1)
    S3 = np.random.uniform(0.04, 0.20, 1)
    S4 = np.random.uniform(0.10, 0.35, 1)
    S5 = np.random.uniform(0.08, 0.30, 1)
    S6 = np.random.uniform(0.14, 2.67, 1)
    S7 = np.random.uniform(0.05, 0.10, 1)
    
    # initial condition
#    x0 = np.array([S1, S2, S3, S4, S5, S6, S7]).flatten()
    x0 = np.array([0.50144272, 1.95478666, 0.19788759, 0.14769148, 0.16059078,
                   0.16127341, 0.06404702]).flatten()
    
    # solve ODE
    S_star = odeint(f, x0, t_star)
    
    noise = 0.1
    t_train = t_star[:,None]
    S_train = addNoise(S_star, noise)

    N0 = 0
    N1 = N - 1 
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

    model.train(num_epochs=20000, batch_size=N_eqns, learning_rate=1e-3)
    model.train(num_epochs=40000, batch_size=N_eqns, learning_rate=1e-4)
    model.train(num_epochs=20000, batch_size=N_eqns, learning_rate=1e-5)
    
    S_pred, S_pred_std = model.predict(t_star[:,None])
    
    ####### Plotting ##################
    
    fig, ax = newfig(3.0, 0.3)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.5, wspace=0.3)
    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,S_star[:,4],'C1',linewidth=2,label='input data')
    ax.scatter(t_star[idx_data],S_star[idx_data,4],marker='o',s=50,label='sampled input')
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_5\ (mM)$', fontsize=18)
    ax.legend(fontsize='large')
    
    ax = plt.subplot(gs0[0:1, 1:2])
    ax.plot(t_star,S_star[:,5],'C1',linewidth=2)
    ax.scatter(t_star[idx_data],S_star[idx_data,5],marker='o',s=50)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_6\ (mM)$', fontsize=18)

    ####################################

    fig, ax = newfig(3.0, 0.4)
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    ax = plt.subplot(gs1[0:1, 0:1])
    ax.plot(t_star,S_star[:,0],'C1',linewidth=2,label='exact')
    ax.plot(t_star,S_pred[:,0],'g-.',linewidth=3,label='learned')
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_1\ (mM)$', fontsize=18)
    ax.legend(fontsize='large')
    
    ax = plt.subplot(gs1[0:1, 1:2])
    ax.plot(t_star,S_star[:,1],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,1],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_2\ (mM)$', fontsize=18)

    ax = plt.subplot(gs1[0:1, 2:3])
    ax.plot(t_star,S_star[:,2],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,2],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_3\ (mM)$', fontsize=18)
    
    fig, ax = newfig(3.5, 0.4)
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    ax = plt.subplot(gs2[0:1, 0:1])
    ax.plot(t_star,S_star[:,3],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,3],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_4\ (mM)$', fontsize=18)

    ax = plt.subplot(gs2[0:1, 1:2])
    ax.scatter(t_star[idx_data],S_star[idx_data,4],marker='o',c='C1',s=30)
    ax.plot(t_star,S_pred[:,4],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_5\ (mM)$', fontsize=18)
    
    ax = plt.subplot(gs2[0:1, 2:3])
    ax.scatter(t_star[idx_data],S_star[idx_data,5],marker='o',c='C1',s=30)
    ax.plot(t_star,S_pred[:,5],'g--',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_6\ (mM)$', fontsize=18)

    fig, ax = newfig(1, 1.5)
    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(top=0.95, bottom=0.15, left=0.15, right=0.95, hspace=0.3, wspace=0.3)
    ax = plt.subplot(gs3[0:1, 0:1])
    ax.plot(t_star,S_star[:,6],'C1',linewidth=2)
    ax.plot(t_star,S_pred[:,6],'g-.',linewidth=3)
    ax.set_xlabel('$t\ (min)$', fontsize=18)
    ax.set_ylabel('$S_7\ (mM)$', fontsize=18)

    print('J0 = %.6f' % ( model.sess.run(model.J0) ) )
    print('k1 = %.6f' % ( model.sess.run(model.k1) ) )
    print('k2 = %.6f' % ( model.sess.run(model.k2) ) )
    print('k3 = %.6f' % ( model.sess.run(model.k3) ) )
    print('k4 = %.6f' % ( model.sess.run(model.k4) ) )
    print('k5 = %.6f' % ( model.sess.run(model.k5) ) )
    print('k6 = %.6f' % ( model.sess.run(model.k6) ) )
    print('k = %.6f' % ( model.sess.run(model.k) ) )
    print('kappa = %.6f' % ( model.sess.run(model.kappa) ) )
    print('q = %.6f' % ( model.sess.run(model.q) ) )
    print('K1 = %.6f' % ( model.sess.run(model.K1) ) )
    print('psi = %.6f' % ( model.sess.run(model.psi) ) )
    print('N = %.6f' % ( model.sess.run(model.N) ) )
    print('A = %.6f' % ( model.sess.run(model.A) ) )
    
    # savefig('./figures/Glycolytic', crop = False)
