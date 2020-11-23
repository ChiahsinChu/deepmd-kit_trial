import numpy as np

from deepmd.env import tf
from deepmd.RunOptions import global_tf_float_precision

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              precision = global_tf_float_precision, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False):
    with tf.variable_scope(name, reuse=reuse):
        # variable_scope: construct a new variable or layer
        # reuse: share the variable
        shape = inputs.get_shape().as_list()
        # tf.shape(): return the shape of tensors
        # x.get_shape(): x should be the tensor defined in TF, return the shape of x (in tuple)
        # x.get_shape().as_list(): transfer the returned tuple into list
        w = tf.get_variable('matrix', 
                            [shape[1], outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed), 
                            trainable = trainable)
        # define the weight matrix
        b = tf.get_variable('bias', 
                            [outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                            trainable = trainable)
        # define the bias matrix
        hidden = tf.matmul(inputs, w) + b
        # define the hidden layer
        if activation_fn != None and use_timestep :
            idt = tf.get_variable('idt',
                                  [outputs_size],
                                  precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed), 
                                  trainable = trainable)
        if activation_fn != None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    return activation_fn(hidden) * idt
                else :
                    return activation_fn(hidden)                    
        else:
            if useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
