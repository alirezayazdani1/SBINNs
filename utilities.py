"""
@author: Alireza Yazdani
"""

import tensorflow as tf
import numpy as np
import random
import uuid

def heaviside(x: tf.Tensor, g: tf.Graph = tf.get_default_graph()):
    # Generate random name in order to avoid conflicts with inbuilt names
    rnd_name = 'HeavisideGrad-' + '%0x' % random.getrandbits(30 * 4)
    
    @tf.RegisterGradient(rnd_name)
    def _heaviside_grad(unused_op: tf.Operation, grad: tf.Tensor):
        return tf.maximum(0.0, 1.0 - tf.abs(unused_op.inputs[0])) * grad
    
    custom_grads = {
        'Identity': rnd_name
    }
    
    with g.gradient_override_map(custom_grads):
        i = tf.identity(x, name='identity_' + str(uuid.uuid1()))
        ge = tf.greater_equal(x, 0, name='ge_' + str(uuid.uuid1()))
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
        return step_func

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess

def relative_error(exact, pred):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(exact, pred):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

class neural_net(object):
    def __init__(self, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0, self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = self.xavier_init(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)
            
    def __call__(self, *inputs):
                
        H = tf.concat(inputs, 1)
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
        return H