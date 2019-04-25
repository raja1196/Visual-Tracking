import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16 
#from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sys
import os.path
import numpy as np
from tracker.shufflenet_utils import block
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from re3_utils.tensorflow_util import tf_util
from re3_utils.tensorflow_util.CaffeLSTMCell import CaffeLSTMCell

from constants import LSTM_SIZE
from layers import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout
#from tensorflow.keras.applications.mobilenet import MobileNet as mnet
IMAGENET_MEAN = [123.151630838, 115.902882574, 103.062623801]

msra_initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.zeros_initializer()
prelu_initializer = tf.constant_initializer(0.25)

def mobilenet_conv_layers2(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    with tf.variable_scope('conv_1'):
        conv1_1 = conv2d('conv_1', input, num_filters=int(round(32 * 1)),
                             kernel_size=(3, 3),
                             padding='SAME', stride=(2, 2), activation=tf.nn.relu6,
                             batchnorm_enabled=False,
                             is_training=True, l2_strength=0.0, bias=0.0)
    #self.__add_to_nodes([conv1_1])
    ############################################################################################
    with tf.variable_scope('conv_ds_2'):
        conv2_1_dw, conv2_1_pw = depthwise_separable_conv2d('conv_ds_2', conv1_1,
                                                        width_multiplier=1,
                                                        num_filters=64, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv2_1_dw, conv2_1_pw])
    with tf.variable_scope('conv_ds_3'):
        conv2_2_dw, conv2_2_pw = depthwise_separable_conv2d('conv_ds_3', conv2_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv2_2_dw, conv2_2_pw])
    ############################################################################################
    #with tf.variable_scope('conv1_skip'):
    
    if 0:
        pool2 = tf.nn.max_pool(conv2_2_pw, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2,alpha=2e-5, beta=0.75, bias=1.0, name='norm2')
    
        prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,initializer=prelu_initializer)
        conv1_skip = tf_util.prelu(tf_util.conv_layer(lrn2, 16, 1, activation=None),prelu_skip)
        conv1_skip = tf.transpose(conv1_skip, perm=[0,3,1,2])
        conv1_skip_flat = tf_util.remove_axis(conv1_skip, [2,3])

    with tf.variable_scope('conv_ds_4'):
        conv3_1_dw, conv3_1_pw = depthwise_separable_conv2d('conv_ds_4', conv2_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv3_1_dw, conv3_1_pw])
    with tf.variable_scope('conv_ds_5'):
        conv3_2_dw, conv3_2_pw = depthwise_separable_conv2d('conv_ds_5', conv3_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv3_2_dw, conv3_2_pw])
    ############################################################################################
    with tf.variable_scope('conv_ds_6'):
        conv4_1_dw, conv4_1_pw = depthwise_separable_conv2d('conv_ds_6', conv3_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv4_1_dw, conv4_1_pw])
    with tf.variable_scope('conv_ds_7'):
        conv4_2_dw, conv4_2_pw = depthwise_separable_conv2d('conv_ds_7', conv4_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv4_2_dw, conv4_2_pw])
    ############################################################################################
    if 0:
        pool3 = tf.nn.max_pool(conv4_2_pw, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool2')
        lrn3 = tf.nn.local_response_normalization(pool2, depth_radius=2,alpha=2e-5, beta=0.75, bias=1.0, name='norm2')
        
        
        with tf.variable_scope('conv2_skip'):
            prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,initializer=prelu_initializer)
            conv2_skip = tf_util.prelu(tf_util.conv_layer(lrn3, 16, 1, activation=None),prelu_skip)
            conv2_skip = tf.transpose(conv2_skip, perm=[0,3,1,2])
            conv2_skip_flat = tf_util.remove_axis(conv2_skip, [2,3])
    with tf.variable_scope('conv_ds_8'):
        conv5_1_dw, conv5_1_pw = depthwise_separable_conv2d('conv_ds_8', conv4_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_1_dw, conv5_1_pw])
    with tf.variable_scope('conv_ds_9'):
        conv5_2_dw, conv5_2_pw = depthwise_separable_conv2d('conv_ds_9', conv5_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_2_dw, conv5_2_pw])
    with tf.variable_scope('conv_ds_10'):
        conv5_3_dw, conv5_3_pw = depthwise_separable_conv2d('conv_ds_10', conv5_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_3_dw, conv5_3_pw])
    with tf.variable_scope('conv_ds_11'):
        conv5_4_dw, conv5_4_pw = depthwise_separable_conv2d('conv_ds_11', conv5_3_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_4_dw, conv5_4_pw])
    with tf.variable_scope('conv_ds_12'):
        conv5_5_dw, conv5_5_pw = depthwise_separable_conv2d('conv_ds_12', conv5_4_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_5_dw, conv5_5_pw])
    with tf.variable_scope('conv_ds_13'):
        conv5_6_dw, conv5_6_pw = depthwise_separable_conv2d('conv_ds_13', conv5_5_pw,
                                                        width_multiplier=1,
                                                        num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_6_dw, conv5_6_pw])
    ############################################################################################
    with tf.variable_scope('conv_ds_14'):
        conv6_1_dw, conv6_1_pw = depthwise_separable_conv2d('conv_ds_14', conv5_6_pw,
                                                        width_multiplier=1,
                                                        num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv6_1_dw, conv6_1_pw])
    ############################################################################################
        avg_pool = avg_pool_2d(conv6_1_pw, size=(7, 7), stride=(1, 1))
        dropped = dropout(avg_pool, -1, True)
    #print("dropout:shape:")
    #print(dropped.get_shape())
    with tf.variable_scope('fc'):
        if 1:
            logits = flatten(conv2d('fc', dropped, kernel_size=(1, 1), num_filters=32,
                                        l2_strength=0.0,
                                        bias=0.0,padding='SAME'))
        else:
            logits = (conv2d('fc', dropped, kernel_size=(1, 1), num_filters=32,
                                    l2_strength=0.0,
                                    bias=0.0,padding='SAME'))
            logits = tf_util.remove_axis(logits, [2,3])
    with tf.variable_scope('big_concat'):
        if 1:
            logits_shape = logits.get_shape().as_list()
            pool5_reshape = tf.reshape(logits, [batch_size, num_unrolls, 2, logits_shape[-1]])
        
        else:
            skip_concat = tf.concat([conv1_skip_flat, logits], 1)
            skip_concat_shape = skip_concat.get_shape().as_list()
            pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        #skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, logits], 1)
    #
    #print("logitss:shape:")
    #print(logits_shape)
    #
    
        # 
        # print("Ship_concat shape")
        # print(skip_concat_shape)
        # Split and merge image pairs
        # (BxTx2)xHxWxC
        #
        # (BxT)x(2xHxWxC)
        reshaped = tf_util.remove_axis(pool5_reshape, [1,3])
        return reshaped
    #self.__add_to_nodes([avg_pool, dropped, self.logits])





    # with tf.variable_scope('big_concat'):
    #     # Concat all skip layers.
    #     skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat], 1)
    #     skip_concat_shape = skip_concat.get_shape().as_list()
    #     print(skip_concat_shape)
    #     # Split and merge image pairs
    #     # (BxTx2)xHxWxC
    #     pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
    #     # (BxT)x(2xHxWxC)
    #     reshaped = tf_util.remove_axis(pool5_reshape, [1,3])

    #     return reshaped

def mobilenet_conv_layers(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    conv1_1 = conv2d('conv_1', input, num_filters=int(round(32 * 1)),
                             kernel_size=(3, 3),
                             padding='SAME', stride=(2, 2), activation=tf.nn.relu6,
                             batchnorm_enabled=False,
                             is_training=True, l2_strength=0.0, bias=0.0)
    #self.__add_to_nodes([conv1_1])
    ############################################################################################
    conv2_1_dw, conv2_1_pw = depthwise_separable_conv2d('conv_ds_2', conv1_1,
                                                        width_multiplier=1,
                                                        num_filters=64, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv2_1_dw, conv2_1_pw])
    conv2_2_dw, conv2_2_pw = depthwise_separable_conv2d('conv_ds_3', conv2_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv2_2_dw, conv2_2_pw])
    ############################################################################################
    #with tf.variable_scope('conv1_skip'):
    
    if 0:
        pool2 = tf.nn.max_pool(conv2_2_pw, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2,alpha=2e-5, beta=0.75, bias=1.0, name='norm2')
    
        prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,initializer=prelu_initializer)
        conv1_skip = tf_util.prelu(tf_util.conv_layer(lrn2, 16, 1, activation=None),prelu_skip)
        conv1_skip = tf.transpose(conv1_skip, perm=[0,3,1,2])
        conv1_skip_flat = tf_util.remove_axis(conv1_skip, [2,3])

    
    conv3_1_dw, conv3_1_pw = depthwise_separable_conv2d('conv_ds_4', conv2_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv3_1_dw, conv3_1_pw])

    conv3_2_dw, conv3_2_pw = depthwise_separable_conv2d('conv_ds_5', conv3_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv3_2_dw, conv3_2_pw])
    ############################################################################################
    
    conv4_1_dw, conv4_1_pw = depthwise_separable_conv2d('conv_ds_6', conv3_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv4_1_dw, conv4_1_pw])

    conv4_2_dw, conv4_2_pw = depthwise_separable_conv2d('conv_ds_7', conv4_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv4_2_dw, conv4_2_pw])
    ############################################################################################
    if 0:
        pool3 = tf.nn.max_pool(conv4_2_pw, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool2')
        lrn3 = tf.nn.local_response_normalization(pool2, depth_radius=2,alpha=2e-5, beta=0.75, bias=1.0, name='norm2')
        
        
        with tf.variable_scope('conv2_skip'):
            prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,initializer=prelu_initializer)
            conv2_skip = tf_util.prelu(tf_util.conv_layer(lrn3, 16, 1, activation=None),prelu_skip)
            conv2_skip = tf.transpose(conv2_skip, perm=[0,3,1,2])
            conv2_skip_flat = tf_util.remove_axis(conv2_skip, [2,3])

    conv5_1_dw, conv5_1_pw = depthwise_separable_conv2d('conv_ds_8', conv4_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_1_dw, conv5_1_pw])

    conv5_2_dw, conv5_2_pw = depthwise_separable_conv2d('conv_ds_9', conv5_1_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_2_dw, conv5_2_pw])

    conv5_3_dw, conv5_3_pw = depthwise_separable_conv2d('conv_ds_10', conv5_2_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_3_dw, conv5_3_pw])

    conv5_4_dw, conv5_4_pw = depthwise_separable_conv2d('conv_ds_11', conv5_3_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_4_dw, conv5_4_pw])

    conv5_5_dw, conv5_5_pw = depthwise_separable_conv2d('conv_ds_12', conv5_4_pw,
                                                        width_multiplier=1,
                                                        num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_5_dw, conv5_5_pw])

    conv5_6_dw, conv5_6_pw = depthwise_separable_conv2d('conv_ds_13', conv5_5_pw,
                                                        width_multiplier=1,
                                                        num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                        stride=(2, 2),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv5_6_dw, conv5_6_pw])
    ############################################################################################
    conv6_1_dw, conv6_1_pw = depthwise_separable_conv2d('conv_ds_14', conv5_6_pw,
                                                        width_multiplier=1,
                                                        num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=False,
                                                        activation=tf.nn.relu6,
                                                        is_training=True,
                                                        l2_strength=0.0,
                                                        biases=(0.0, 0.0))
    #self.__add_to_nodes([conv6_1_dw, conv6_1_pw])
    ############################################################################################
    avg_pool = avg_pool_2d(conv6_1_pw, size=(7, 7), stride=(1, 1))
    dropped = dropout(avg_pool, -1, True)
    #print("dropout:shape:")
    #print(dropped.get_shape())
    if 1:
        logits = flatten(conv2d('fc', dropped, kernel_size=(1, 1), num_filters=32,
                                    l2_strength=0.0,
                                    bias=0.0,padding='SAME'))
    else:
        logits = (conv2d('fc', dropped, kernel_size=(1, 1), num_filters=32,
                                    l2_strength=0.0,
                                    bias=0.0,padding='SAME'))
        logits = tf_util.remove_axis(logits, [2,3])
    if 1:
        logits_shape = logits.get_shape().as_list()
        pool5_reshape = tf.reshape(logits, [batch_size, num_unrolls, 2, logits_shape[-1]])
        
    else:
        skip_concat = tf.concat([conv1_skip_flat, logits], 1)
        #skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, logits], 1)
    #
    #print("logitss:shape:")
    #print(logits_shape)
    #
        skip_concat_shape = skip_concat.get_shape().as_list()
        print("Ship_concat shape")
        print(skip_concat_shape)
        # Split and merge image pairs
        # (BxTx2)xHxWxC
        pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
    reshaped = tf_util.remove_axis(pool5_reshape, [1,3])
    return reshaped
    #self.__add_to_nodes([avg_pool, dropped, self.logits])





    # with tf.variable_scope('big_concat'):
    #     # Concat all skip layers.
    #     skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat], 1)
    #     skip_concat_shape = skip_concat.get_shape().as_list()
    #     print(skip_concat_shape)
    #     # Split and merge image pairs
    #     # (BxTx2)xHxWxC
    #     pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
    #     # (BxT)x(2xHxWxC)
    #     reshaped = tf_util.remove_axis(pool5_reshape, [1,3])

    #     return reshaped

def mobilenet_conv_layers3(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    model = MobileNet(weights='imagenet', include_top=False, input_tensor=input,input_shape=(224,224,3)) 
    skip_concat = model.output
    skip_concat_flat = tf_util.remove_axis(skip_concat, [2,3])
    skip_concat_shape = skip_concat_flat.get_shape().as_list()
    print(skip_concat_shape)
    pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
    conv_layers = tf_util.remove_axis(pool5_reshape, [1,3])
    return conv_layers

def VGG_conv_layers(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    model=VGG16(include_top=False, weights='imagenet', input_tensor=input, input_shape=(224,224,3))
    skip_concat = model.output
    skip_concat_flat = tf_util.remove_axis(skip_concat, [2,3])
    skip_concat_shape = skip_concat_flat.get_shape().as_list()
    print(skip_concat_shape)
    pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
    conv_layers = tf_util.remove_axis(pool5_reshape, [1,3])
    return conv_layers

def alexnet_conv_layers(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    print(tf.shape(input))
    with tf.variable_scope('conv1'):
        conv1 = tf_util.conv_layer(input, 96, 11, 4, padding='VALID')
        pool1 = tf.nn.max_pool(
                conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2,
                alpha=2e-5, beta=0.75, bias=1.0, name='norm1')

    with tf.variable_scope('conv1_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,
                initializer=prelu_initializer)

        conv1_skip = tf_util.prelu(tf_util.conv_layer(lrn1, 16, 1, activation=None),
               prelu_skip)
       # conv1_skip = tf_util.prelu(tf_util.conv_layer(pool1, 16, 1, activation=None),
               # prelu_skip)
        conv1_skip = tf.transpose(conv1_skip, perm=[0,3,1,2])
        conv1_skip_flat = tf_util.remove_axis(conv1_skip, [2,3])

    with tf.variable_scope('conv2'):
        conv2 = tf_util.conv_layer(lrn1, 256, 5, num_groups=2, padding='SAME')
        pool2 = tf.nn.max_pool(
                conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2,
                alpha=2e-5, beta=0.75, bias=1.0, name='norm2')

    with tf.variable_scope('conv2_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[32], dtype=tf.float32,
                initializer=prelu_initializer)
        conv2_skip = tf_util.prelu(tf_util.conv_layer(lrn2, 32, 1, activation=None),
                prelu_skip)
        #conv2_skip = tf_util.prelu(tf_util.conv_layer(pool2, 32, 1, activation=None),
               # prelu_skip)
        conv2_skip = tf.transpose(conv2_skip, perm=[0,3,1,2])
        conv2_skip_flat = tf_util.remove_axis(conv2_skip, [2,3])

    with tf.variable_scope('conv3'):
        conv3 = tf_util.conv_layer(lrn2, 384, 3, padding='SAME')
        #conv3 = tf_util.conv_layer(pool2, 384, 3, padding='SAME')
    with tf.variable_scope('conv4'):
        conv4 = tf_util.conv_layer(conv3, 384, 3, num_groups=2, padding='SAME')

    with tf.variable_scope('conv5'):
        conv5 = tf_util.conv_layer(conv4, 256, 3, num_groups=2, padding='SAME')
        pool5 = tf.nn.max_pool(
                conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool5')
        pool5 = tf.transpose(pool5, perm=[0,3,1,2])
        pool5_flat = tf_util.remove_axis(pool5, [2,3])

    with tf.variable_scope('conv5_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[64], dtype=tf.float32,
                initializer=prelu_initializer)

        conv5_skip = tf_util.prelu(tf_util.conv_layer(conv5, 64, 1, activation=None),
                prelu_skip)
        conv5_skip = tf.transpose(conv5_skip, perm=[0,3,1,2])
        conv5_skip_flat = tf_util.remove_axis(conv5_skip, [2,3])

    with tf.variable_scope('big_concat'):
        # Concat all skip layers.
        #skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat], 1)
        skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, pool5_flat], 1)
        skip_concat_shape = skip_concat.get_shape().as_list()
        print("Ship_concat shape")
        print(skip_concat_shape)
        # Split and merge image pairs
        # (BxTx2)xHxWxC
        pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
        reshaped = tf_util.remove_axis(pool5_reshape, [1,3])
        print(reshaped.get_shape())
        print("----------------------------------------------")
        return reshaped

def ShuffleNetV2(input, batch_size,num_unrolls,include_top=True, scale_factor=1.0, pooling='max',input_shape=(224,224,3), load_model=None,num_shuffle_units=[3,7,3],bottleneck_ratio=1,classes=1000):
    input = tf.to_float(input) - IMAGENET_MEAN
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    
    img_input = Input(tensor=input, shape=input_shape)
    #if input is None:
       # img_input = Input(shape=input_shape)
    #else:
        #if not K.is_keras_tensor(input):
         #  img_input = Input(tensor=input, shape=input_shape)
        #else:
         #   img_input = input
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    #if input:
     #   inputs = get_source_inputs(input)
    #else:
     #   inputs = input

    model = Model(input, x)
    
    skip_concat = model.output
    if len(skip_concat.shape) == 2:
        skip_concat_flat = skip_concat
    else:
        skip_concat_flat = tf_util.remove_axis(skip_concat, [2,3])
    skip_concat_shape = skip_concat_flat.get_shape().as_list()
    print(skip_concat_shape)
    pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
    conv_layers = tf_util.remove_axis(pool5_reshape, [1,3])
    
    return conv_layers


def inference(inputs, num_unrolls, train, batch_size=None, prevLstmState=None, reuse=None):
    print(inputs)
    # Data should be in order BxTx2xHxWxC where T is the number of unrolls
    # Mean subtraction
    if batch_size is None:
        batch_size = int(inputs.get_shape().as_list()[0] / (num_unrolls * 2))

    variable_list = []

    if reuse is not None and not reuse:
        reuse = None

    with tf.variable_scope('re3', reuse=reuse):
        conv_layers = alexnet_conv_layers(inputs, batch_size, num_unrolls)
        #conv_layers = mobilenet_conv_layers3(inputs, batch_size, num_unrolls)
        #conv_layers = ShuffleNetV2(inputs,batch_size,num_unrolls, include_top=True, input_shape=(224, 224, 3), bottleneck_ratio=1)
        #conv_layers=VGG_conv_layers(inputs,batch_size,num_unrolls)
        #print(conv_layers)
        #print(conv_layers.summary())   
        
        
        # Embed Fully Connected Layer
        with tf.variable_scope('fc6'):
            fc6_out = tf_util.fc_layer(conv_layers, 1024)

            # (BxT)xC
            fc6_reshape = tf.reshape(fc6_out, tf.stack([batch_size, num_unrolls, fc6_out.get_shape().as_list()[-1]]))

        # LSTM stuff
        swap_memory = num_unrolls > 1
        with tf.variable_scope('lstm1'):
            #lstm1 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            lstm1 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, use_peepholes=True, initializer=msra_initializer, reuse=reuse)
            if prevLstmState is not None:
                state1 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[0], prevLstmState[1])
            else:
                state1 = lstm1.zero_state(batch_size, dtype=tf.float32)
            lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, fc6_reshape, initial_state=state1, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm1' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])

        with tf.variable_scope('lstm2'):
            #lstm2 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            lstm2 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, use_peepholes=True, initializer=msra_initializer, reuse=reuse)
            if prevLstmState is not None:
                state2 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[2], prevLstmState[3])
            else:
                state2 = lstm2.zero_state(batch_size, dtype=tf.float32)
            lstm2_inputs = tf.concat([fc6_reshape, lstm1_outputs], 2)
            lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm2' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])
            # (BxT)xC
            outputs_reshape = tf_util.remove_axis(lstm2_outputs, 1)

        # Final FC layer.
        with tf.variable_scope('fc_output'):
            fc_output_out = tf_util.fc_layer(outputs_reshape, 4, activation=None)

    if prevLstmState is not None:
        return fc_output_out, state1, state2
    else:
        return fc_output_out

def get_var_list():
    return tf.trainable_variables()

def loss(outputs, labels):
    with tf.variable_scope('loss'):
        diff = tf.reduce_sum(tf.abs(outputs - labels, name='diff'), axis=1)
        loss = tf.reduce_mean(diff, name='loss')

    # L2 Loss on variables.
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.0005 * tf.add_n([tf.nn.l2_loss(v)
            for v in get_var_list()])

    full_loss = loss + l2_weight_penalty

    return full_loss, loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()
    train_op = optimizer.minimize(loss, var_list=get_var_list(), global_step=global_step,
        colocate_gradients_with_ops=True)
    return train_op

