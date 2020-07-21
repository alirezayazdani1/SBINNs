import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
import seaborn as sns
import time

from utilities import neural_net, fwd_gradients, heaviside, \
                      tf_session, mean_squared_error, relative_error

class HiddenPathways:
    # Initialize the class
    def __init__(self, t_data, S_data, t_eqns, layers, meal_tq):
        
        self.D = S_data.shape[1]
        
        self.t_min = t_data.min(0)
        self.t_max = t_data.max(0)
        
#        self.S_scale = tf.Variable(np.array(self.D*[1.0]), dtype=tf.float32, trainable=False)
        self.S_scale = S_data.std(0)
        
        # data on all the species (only some are used as input)
        self.t_data, self.S_data = t_data, S_data
        self.t_eqns = t_eqns
                
        # layers
        self.layers = layers

        self.mt = 2.0*(meal_tq[0] - self.t_min)/(self.t_max - self.t_min) - 1.0
        self.mq = meal_tq[1]
        
#        self.k = tf.Variable(1.0/120.0, dtype=tf.float32, trainable=False)
        self.Rm = tf.Variable(209.0/100.0, dtype=tf.float32, trainable=False)
        self.Vg = tf.Variable(10.0, dtype=tf.float32, trainable=False)
        self.C1 = tf.Variable(300.0/100.0, dtype=tf.float32, trainable=False)
        self.a1 = tf.Variable(6.6, dtype=tf.float32, trainable=False)
#        self.Ub = tf.Variable(72.0/100.0, dtype=tf.float32, trainable=False)
#        self.C2 = tf.Variable(144.0/100.0, dtype=tf.float32, trainable=False)
#        self.U0 = tf.Variable(4.0/100.0, dtype=tf.float32, trainable=False)
#        self.Um = tf.Variable(90.0/100.0, dtype=tf.float32, trainable=False)
#        self.C3 = tf.Variable(100.0/100.0, dtype=tf.float32, trainable=False)
#        self.C4 = tf.Variable(80.0/100.0, dtype=tf.float32, trainable=False)
        self.Vi = tf.Variable(11.0, dtype=tf.float32, trainable=False)
        self.E = tf.Variable(0.2, dtype=tf.float32, trainable=False)
        self.ti = tf.Variable(100.0, dtype=tf.float32, trainable=False)
#        self.beta = tf.Variable(1.772, dtype=tf.float32, trainable=False)
#        self.Rg = tf.Variable(180.0/100.0, dtype=tf.float32, trainable=False)
#        self.alpha = tf.Variable(7.5, dtype=tf.float32, trainable=False)
        self.Vp = tf.Variable(3.0, dtype=tf.float32, trainable=False)
#        self.C5 = tf.Variable(26.0/100.0, dtype=tf.float32, trainable=False)
        self.tp = tf.Variable(6.0, dtype=tf.float32, trainable=False)
#        self.td = tf.Variable(12.0, dtype=tf.float32, trainable=False)

        self.logk = tf.Variable(-6.0, dtype=tf.float32, trainable=True)
#        self.logRm = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logVg = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logC1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.loga1 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logUb = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logC2 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logU0 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logUm = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logC3 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logC4 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logVi = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logE = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logti = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logbeta = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logRg = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logalpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logVp = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logC5 = tf.Variable(0.0, dtype=tf.float32, trainable=True)
#        self.logtp = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.logtd = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.var_list_eqns = [self.logk, self.logUb, 
                              self.logC2, self.logU0, self.logUm, self.logC3, self.logC4, 
                              self.logbeta, self.logRg, self.logalpha, self.logC5,
                              self.logtd]
  
        self.k = tf.exp(self.logk)
#        self.Rm = tf.exp(self.logRm)
#        self.Vg = tf.exp(self.logVg)
#        self.C1 = tf.exp(self.logC1)
#        self.a1 = tf.exp(self.loga1)
        self.Ub = tf.exp(self.logUb)
        self.C2 = tf.exp(self.logC2)
        self.U0 = tf.exp(self.logU0)
        self.Um = tf.exp(self.logUm)
        self.C3 = tf.exp(self.logC3)
        self.C4 = tf.exp(self.logC4)
#        self.Vi = tf.exp(self.logVi)
#        self.E = tf.exp(self.logE)
#        self.ti = tf.exp(self.logti)
        self.beta = tf.exp(self.logbeta)
        self.Rg = tf.exp(self.logRg)
        self.alpha = tf.exp(self.logalpha)
#        self.Vp = tf.exp(self.logVp)
        self.C5 = tf.exp(self.logC5)
#        self.tp = tf.exp(self.logtp)
        self.td = tf.exp(self.logtd)

        # placeholders for data
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.S_data_tf = tf.placeholder(tf.float32, shape=[None, self.D])
        self.t_eqns_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.mt_tf = tf.placeholder(tf.float32, shape=[None, self.mt.shape[1]])
        self.mq_tf = tf.placeholder(tf.float32, shape=[None, self.mq.shape[1]])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # physics uninformed neural networks
        self.net_sysbio = neural_net(layers=self.layers)
        
        self.H_data = 2.0*(self.t_data_tf - self.t_min)/(self.t_max - self.t_min) - 1.0
        self.S_data_pred = self.S_data[0,:] + self.S_scale*(self.H_data+1.0)*self.net_sysbio(self.H_data)

        # physics informed neural networks
        self.H_eqns = 2.0*(self.t_eqns_tf - self.t_min)/(self.t_max - self.t_min) - 1.0
        self.S_eqns_pred = self.S_data[0,:] + self.S_scale*(self.H_eqns+1.0)*self.net_sysbio(self.H_eqns)

        self.E_eqns_pred, self.IG = self.SysODE(self.S_eqns_pred, self.t_eqns_tf,
                                                self.H_eqns, self.mt_tf, self.mq_tf)

        # Adaptive S_scale
#        self.S_scale = 0.9*self.S_scale + 0.1*tf.math.reduce_std(self.S_eqns_pred, 0)
#        scale_list = tf.unstack(self.S_scale)
#        scale_list[2] = self.S_data.std(0)[2]
#        self.S_scale = tf.stack(scale_list)
        
        # loss
        self.loss_data = mean_squared_error(self.S_data_tf[:,2:3]/self.S_scale[2:3], self.S_data_pred[:,2:3]/self.S_scale[2:3])
        self.loss_eqns = mean_squared_error(0.0, self.E_eqns_pred/self.S_scale)
        self.loss_auxl = mean_squared_error(self.S_data_tf[-1,:]/self.S_scale, self.S_data_pred[-1,:]/self.S_scale)
        self.loss = 0.99*self.loss_data + 0.01*self.loss_eqns + 0.01*self.loss_auxl

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
        
    def SysODE(self, S, t, H, mt, mq):
        intake = self.k * mq * heaviside(H-mt) * tf.exp(self.k*(mt-H)*(self.t_max-self.t_min)/2.0)
        IG = tf.reduce_sum(intake, axis=1, keepdims=True)
        kappa = 1.0/self.Vi + 1.0/(self.E*self.ti)
        f1 = self.Rm * tf.sigmoid(S[:,2:3]/(self.Vg*self.C1) - self.a1)
        f2 = self.Ub * (1.0 - tf.exp(-S[:,2:3]/(self.Vg*self.C2)))
        safe_log = tf.where(S[:,1:2] <= 0.0, tf.ones_like(S[:,1:2]), S[:,1:2])
        f3 = (self.U0 + self.Um*tf.sigmoid(self.beta*tf.log(kappa*safe_log/self.C4))) / (self.Vg*self.C3)
        f4 = self.Rg * tf.sigmoid(-self.alpha*(S[:,5:6]/(self.Vp*self.C5)-1.0))
                               
        F0 = f1 - self.E*(S[:,0:1]/self.Vp-S[:,1:2]/self.Vi) - S[:,0:1]/self.tp
        F1 = self.E*(S[:,0:1]/self.Vp-S[:,1:2]/self.Vi) - S[:,1:2]/self.ti
        F2 = f4 + IG - f2 - f3*S[:,2:3]
        F3 = (S[:,0:1] - S[:,3:4]) / self.td
        F4 = (S[:,3:4] - S[:,4:5]) / self.td
        F5 = (S[:,4:5] - S[:,5:6]) / self.td
        
        F = tf.concat([F0, F1, F2, F3, F4, F5], 1)

        S_t = fwd_gradients(S, t)
        
        E = S_t - F
        return E, IG
    
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
                mt_batch, mq_batch = self.mt[idx_eqns,:], self.mq[idx_eqns,:]
    
                tf_dict = {self.t_data_tf: t_data_batch,
                           self.S_data_tf: S_data_batch,
                           self.t_eqns_tf: t_eqns_batch,
                           self.mt_tf: mt_batch, self.mq_tf: mq_batch,
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
                    
    def predict(self, t_star, meal_tq):
        
        meal_tq[0] = 2.0*(meal_tq[0] - self.t_min)/(self.t_max - self.t_min) - 1.0
        tf_dict = {self.t_eqns_tf: t_star,
                   self.mt_tf: meal_tq[0], self.mq_tf: meal_tq[1]}
        
        S_star, IG = self.sess.run([self.S_eqns_pred, self.IG], tf_dict)
        S_star = np.append(S_star[:,:], IG[:], axis=1)
        return S_star

if __name__ == "__main__": 
    
    layers = [1] + 6*[6*30] + [6]

    meal_t = [300., 650., 1100., 2000.]
    meal_q = [60e3, 40e3, 50e3, 100e3]
    
    def intake(tn, k):
        def s(mjtj):
            return k*mjtj[1]*np.heaviside(tn-mjtj[0], 0.5)*np.exp(k*(mjtj[0]-tn))
        IG = np.array([s(mjtj) for mjtj in list(zip(meal_t, meal_q))]).sum()
        return IG
    
    # function that returns dx/dt
    def f(x, t): # x is 6 x 1
        k = 1./120.        
        Rm = 209.
        Vg = 10.
        C1 = 300.
        a1 = 6.6
        Ub = 72.
        C2 = 144.
        U0 = 4.
        Um = 90.
        C3 = 100.
        C4 = 80.
        Vi = 11.
        E = 0.2
        ti = 100.
        beta = 1.772
        Rg = 180.
        alpha = 7.5
        Vp = 3.
        C5 = 26.
        tp = 6.
        td = 12.
        
        kappa = 1.0/Vi + 1.0/E/ti
        f1 = Rm / (1.0 + np.exp(-x[2]/Vg/C1 + a1))
        f2 = Ub * (1.0 - np.exp(-x[2]/Vg/C2))
        f3 = (U0 + Um/(1.0+np.exp(-beta*np.log(kappa*x[1]/C4)))) / Vg / C3
        f4 = Rg / (1.0 + np.exp(alpha*(x[5]/Vp/C5-1.0)))
        
        x0 = f1 - E*(x[0]/Vp-x[1]/Vi) - x[0]/tp
        x1 = E*(x[0]/Vp-x[1]/Vi) - x[1]/ti
        x2 = f4 + intake(t, k) - f2 - f3*x[2]
        x3 = (x[0] - x[3]) / td
        x4 = (x[3] - x[4]) / td
        x5 = (x[4] - x[5]) / td
        
        X = np.array([x0, x1, x2, x3, x4, x5])
        return X
    
    # function that returns dx/dt
    def f_pred(x, t): # x is 6 x 1
        k =  0.007751
        Rm = 73.858517
        Vg = 10.000000
        C1 = 319.160032
        a1 = 6.253946
        Ub = 86.824888
        C2 = 152.637362
        U0 = 19.412358
        Um = 141.051173
        C3 = 235.955381
        C4 = 251.580667
        Vi = 2.689281
        E = 0.147199
        ti = 36.766254
        beta = 2.475349
        Rg = 212.777472
        alpha = 7.182466
        Vp = 0.707807
        C5 = 101.811242
        tp = 139.384628
        td = 7.417875
        
        kappa = 1.0/Vi + 1.0/E/ti
        f1 = Rm / (1.0 + np.exp(-x[2]/Vg/C1 + a1))
        f2 = Ub * (1.0 - np.exp(-x[2]/Vg/C2))
        f3 = (U0 + Um/(1.0+np.exp(-beta*np.log(kappa*x[1]/C4)))) / Vg / C3
        f4 = Rg / (1.0 + np.exp(alpha*(x[5]/Vp/C5-1.0)))
        
        x0 = f1 - E*(x[0]/Vp-x[1]/Vi) - x[0]/tp
        x1 = E*(x[0]/Vp-x[1]/Vi) - x[1]/ti
        x2 = f4 + intake(t, k) - f2 - f3*x[2]
        x3 = (x[0] - x[3]) / td
        x4 = (x[3] - x[4]) / td
        x5 = (x[4] - x[5]) / td
        
        X = np.array([x0, x1, x2, x3, x4, x5])
        return X

    def plotting(t_star, S_star, S_pred, perm, Vg2, forecast=False):
        sns.set()
    
        fig, ax = newfig(2.0, 0.7)
        gs0 = gridspec.GridSpec(1, 1)
        gs0.update(top=0.9, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
        ax = plt.subplot(gs0[0:1, 0:1])
        ax.plot(t_star,S_star[:,2],'C1',linewidth=2,label='input data')
        ax.scatter(t_star[perm],S_star[perm,2],marker='o',s=40,label='sampled input')
        ax.set_xlabel('$t\ (min)$', fontsize=18)
        ax.set_ylabel('$G\ (mg/dl) $', fontsize=18)
        ax.legend(fontsize='large')
        
        ####################################    
        fig, ax = newfig(1.8, 0.75)
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
        ax = plt.subplot(gs1[0:1, 0:1])
        ax.plot(t_star,S_star[:,0]*Vg2,'C1',linewidth=2,label='exact')
        ax.plot(t_star,S_pred[:,0]*Vg2,'g-.',linewidth=3,label='learned')
        ax.set_xlabel('$t\ (min)$', fontsize=18)
        ax.set_ylabel('$I_p\ (\mu U/ml)$', fontsize=18)
        ax.legend(fontsize='large')
        
        ax = plt.subplot(gs1[0:1, 1:2])
        ax.plot(t_star,S_star[:,1]*Vg2,'C1',linewidth=2)
        ax.plot(t_star,S_pred[:,1]*Vg2,'g-.',linewidth=3)
        ax.set_xlabel('$t\ (min)$', fontsize=18)
        ax.set_ylabel('$I_i\ (\mu U/ml)$', fontsize=18)
    
        fig, ax = newfig(1.8, 0.75)
        gs2 = gridspec.GridSpec(1, 2)
        gs2.update(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
        ax = plt.subplot(gs2[0:1, 0:1])
        if not forecast:
            ax.scatter(t_star[perm],S_star[perm,2],marker='o',c='C1',s=30)
        else:
            ax.plot(t_star,S_star[:,2],'C1',linewidth=2)
        ax.plot(t_star,S_pred[:,2],'g-.',linewidth=3)
        ax.set_xlabel('$t\ (min)$', fontsize=18)
        ax.set_ylabel('$G\ (mg/dl)$', fontsize=18)
        
        ax = plt.subplot(gs2[0:1, 1:2])
        ax.plot(t_star,IG_star*Vg2,'C1',linewidth=2)
        ax.plot(t_star,S_pred[:,6]*Vg2,'g-.',linewidth=3)
        ax.set_xlabel('$t\ (min)$', fontsize=18)
        ax.set_ylabel('$I_G\ (mg/min)$', fontsize=18)
        

    # time points
    t_star = np.arange(0, 3000, 1.0)
    N = t_star.shape[0]
    N_eqns = N
    N_data = N // 5

    k = 1./120.
    Vp = 3.0
    Vi = 11.0
    Vg2 = 10.0*10.0
    S0 = 12.0*Vp
    S1 = 4.0*Vi
    S2 = 110.0*Vg2
    S3 = 0.0
    S4 = 0.0
    S5 = 0.0
    
    # initial condition
    x0 = np.array([S0, S1, S2, S3, S4, S5]).flatten()
    
    # solve ODE
    S_star = odeint(f, x0, t_star)
    S_star /= Vg2 # scaling by Vg^2
    IG_star = np.array([intake(t, k) for t in t_star]) / Vg2

    t_train = t_star[:,None]
    S_train = S_star

    # two-point data must be given for all the species
    # 1st: initial at t=0; 2nd: any point between (0,T]
    N0 = 0
    N1 = N - 1
    idx_data = np.concatenate([np.array([N0]),
                               np.random.choice(np.arange(1, N-1), size=N_data, replace=False),
                               np.array([N-1]),
                               np.array([N1])])
    idx_eqns = np.concatenate([np.array([N0]),
                               np.random.choice(np.arange(1, N-1), size=N_eqns-2, replace=False),
                               np.array([N-1])])
    meal_tq = [np.array([N_eqns*[x] for x in meal_t]).T,
               np.array([N_eqns*[x/Vg2] for x in meal_q]).T]

    model = HiddenPathways(t_train[idx_data],
                           S_train[idx_data,:],
                           t_train[idx_eqns],
                           layers,
                           meal_tq)

    model.train(num_epochs=25000, batch_size=N_eqns, learning_rate=1e-3)
    model.train(num_epochs=25000, batch_size=N_eqns, learning_rate=1e-4)
    model.train(num_epochs=10000, batch_size=N_eqns, learning_rate=1e-5)

    # NN prediction
    meal_tq = [np.array([N*[x] for x in meal_t]).T,
               np.array([N*[x/Vg2] for x in meal_q]).T] 
    
    S_pred = model.predict(t_star[:,None], meal_tq)
    plotting(t_star, S_star, S_pred, idx_data, Vg2)

    print('k =  %.6f' % ( model.sess.run(model.k) ) )
    print('Rm = %.6f' % ( model.sess.run(model.Rm)*Vg2 ) )
    print('Vg = %.6f' % ( model.sess.run(model.Vg) ) )
    print('C1 = %.6f' % ( model.sess.run(model.C1)*Vg2 ) )
    print('a1 = %.6f' % ( model.sess.run(model.a1) ) )
    print('Ub = %.6f' % ( model.sess.run(model.Ub)*Vg2 ) )
    print('C2 = %.6f' % ( model.sess.run(model.C2)*Vg2 ) )
    print('U0 = %.6f' % ( model.sess.run(model.U0)*Vg2 ) )
    print('Um = %.6f' % ( model.sess.run(model.Um)*Vg2 ) )
    print('C3 = %.6f' % ( model.sess.run(model.C3)*Vg2 ) )
    print('C4 = %.6f' % ( model.sess.run(model.C4)*Vg2 ) )
    print('Vi = %.6f' % ( model.sess.run(model.Vi) ) )
    print('E = %.6f' % ( model.sess.run(model.E) ) )
    print('ti = %.6f' % ( model.sess.run(model.ti) ) )
    print('beta = %.6f' % ( model.sess.run(model.beta) ) )
    print('Rg = %.6f' % ( model.sess.run(model.Rg)*Vg2 ) )
    print('alpha = %.6f' % ( model.sess.run(model.alpha) ) )
    print('Vp = %.6f' % ( model.sess.run(model.Vp) ) )
    print('C5 = %.6f' % ( model.sess.run(model.C5)*Vg2 ) )
    print('tp = %.6f' % ( model.sess.run(model.tp) ) )
    print('td = %.6f' % ( model.sess.run(model.td) ) )
    
    # Prediction based on inferred parameters
#    k =  0.007751
#    Vp = 0.707807
#    Vi = 2.689281
#    S0 = 12.0*Vp
#    S1 = 4.0*Vi
#    S2 = 110.0*Vg2
#    S3 = 0.0
#    S4 = 0.0
#    S5 = 0.0
#    x0 = np.array([S0, S1, S2, S3, S4, S5]).flatten()    
#    S_pred = odeint(f_pred, x0, t_star)
#    S_pred /= Vg2
#    IG_pred = np.array([intake(t, k) for t in t_star]) / Vg2
#    S_pred = np.append(S_pred[:,:], IG_pred[:,None], axis=1)
#    plotting(t_star, S_star, S_pred, idx_data, Vg2, forecast=True)
    
    # savefig('./figures/Glycolytic', crop = False)