import tensorflow as tf
from tensorflow.contrib.layers import flatten


def weights_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape=shape,stddev=stddev)
    return tf.Variable(initial,name=name)

def bias_variable(shape,bias=0.1, name=None):
     initial = tf.constant(bias,shape=shape)
     return tf.Variable(initial, name=None)

def network(x):
    conv1_w = weights_variable(shape=(8,8,4,32),name='conv1_w')
    conv1_b = bias_variable(shape=[32],name='conv1_b')
    conv1 = tf.nn.relu( tf.nn.conv2d(x, conv1_w, [1,4,4,1], padding='VALID')+conv1_b,name='conv1' )

    conv2_w = weights_variable(shape=(4,4,32,64),name='conv2_w')
    conv2_b = bias_variable(shape=[64],name='conv2_b')
    conv2 = tf.nn.relu( tf.nn.conv2d(conv1, conv2_w, [1,2,2,1], padding='VALID')+conv2_b,name='conv2' )

    conv3_w = weights_variable(shape=(3,3,64,64),name='conv3_w')
    conv3_b = bias_variable(shape=[64],name='conv3_b')
    conv3 = tf.nn.relu( tf.nn.conv2d(conv2, conv3_w, [1,1,1,1], padding='VALID')+conv3_b,name='conv3' )

    flat = flatten(conv3)
    fc4_w = weights_variable(shape=(3136,512),name='fc4_w')
    fc4_b = bias_variable(shape=[512],name='fc4_b')
    fc4 = tf.nn.relu( tf.matmul(flat,fc4_w)+fc4_b,name='fc4' )

    fc5_w = weights_variable(shape=(512,2),name='fc5_w')
    fc5_b = bias_variable(shape=[2],name='fc5_b')
    prediction = tf.add(tf.matmul(fc4,fc5_w),fc5_b,name='prediction')

    return prediction