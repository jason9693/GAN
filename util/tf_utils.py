import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import random

def LSTMlayer(input, hidden_size, batch_size,cycle=0,gpu_mode=False):

    lstm_cell = contrib.rnn.LSTMBlockFusedCell(hidden_size,name='lstm'+str(cycle))

    outs,state = lstm_cell(input,dtype=tf.float32)

    return (outs, state)

def Dense(input, output_shape, name='dense' ,activation=None):
    W = tf.get_variable(
        name=name+'_W',
        shape=[input.shape[1], output_shape],
        initializer = contrib.layers.xavier_initializer()
    )
    b = tf.get_variable(
        name=name+'_b',
        shape=output_shape,
        initializer=contrib.layers.xavier_initializer()
    )

    h = tf.matmul(input, W) + b
    if activation is None:
        return h
    else:
        return activation(h)


def rand_batch(data, batch_size):
    x,y = data
    x = np.array(x,np.float32)
    y = np.array(y,np.int32)
    range_list = np.array(range(x.__len__()))
    batch_index = random.sample(range_list.tolist(), k=batch_size)

    return (x[batch_index], y[batch_index])


def set_device_mode(is_gpu = False):
    gpu_str = 'CPU/:0'
    if is_gpu:
        gpu_str = 'GPU/:0'
    return tf.device(gpu_str)