# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.keras import metrics
from tensorflow.python.keras import backend
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.backend import concatenate
import numpy as np
from tensorflow.keras import applications
# from model_profiler import model_profiler #https://pypi.org/project/model-profiler/
from tensorflow.keras import backend as K
# from epinet_func import evaluation_metric
# import keras_flops
import math
import tensorflow as tf
import os.path
from keras_flops import get_flops
from tensorflow.keras import backend as K
from model_profiler import model_profiler  # https://pypi.org/project/model-profiler/
from epinet_func.mixibn import MixNetConvInitializer, GroupedIBN2D
from epinet_func.evaluation_metric import bpr, PSNR

printStat = False
paddingType = 'same'  # valid

def layer1_multistream(input_dim1, input_dim2, input_dim3, filt_num, kernel_size):
    k = get_k_size(kernel_size)
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3))
    x = inputs
    for i in range(3):
        x = GroupedIBN2D(filt_num,kernel_size=k, type='dwsc', strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding='same',use_bias=False,name='S1_mixconv1_%d' % (i))(x)
        x = Activation('relu', name='S1_relu1%d' % (i))(x)
        x = GroupedIBN2D(filt_num, kernel_size=k, type='dwsc',  strides=[1, 1], kernel_initializer=MixNetConvInitializer(),
                      padding='same', use_bias=False, name='S1_mixconv2_%d' % (i))(x)
        x = BatchNormalization(axis=-1, name='S1_BN%d' % (i))(x)
        x = Activation('relu', name='S1_relu2%d' % (i))(x)
    outputs = x
    return Model(inputs=inputs, outputs=outputs)

def layer2_merged(input_dim1, input_dim2, input_dim3, filt_num, conv_depth, kernel_size):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    k = get_k_size(kernel_size)
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3))
    x = inputs
    for i in range(conv_depth):
        x = GroupedIBN2D(filt_num,kernel_size=k, type='dwsc', strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding='same',use_bias=False, name='S2_mixconv1_%d' % (i))(x)
        x = Activation('relu', name='S2_relu1%d' % (i))(x)
        x = GroupedIBN2D(filt_num,kernel_size=k, type='dwsc', strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding='same',use_bias=False, name='S2_mixconv2_%d' % (i))(x)
        x = BatchNormalization(axis=-1, name='S2_BN%d' % (i))(x)
        x = Activation('relu', name='S2_relu2%d' % (i))(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


def layer3_last(input_dim1, input_dim2, input_dim3, filt_num, kernel_size):
    ''' last layer : Conv - Relu - Conv '''
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3))
    x = inputs
    k = get_k_size(kernel_size)
    for i in range(1):
        x = GroupedIBN2D(filt_num,kernel_size=k, type='dwsc', strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding='same',use_bias=False,name='S3_mixconv1_%d' % (i))(x)
        x = Activation('relu', name='S3_relu1%d' % (i))(x)

    outputs = Conv2D(1, (kernel_size, kernel_size), padding=paddingType, use_bias=False, name='S3_last')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def define_epinet(sz_input, sz_input2, view_n, conv_depth, filt_num, learning_rate, kernel_size):
    tFlops = 0
    tMem = 0
    tParam = 0
    tMem_req = 0

    ''' 4-Input : Conv - Relu - Conv - BN - Relu '''
    input_stack_90d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_90d')
    input_stack_0d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_0d')
    input_stack_45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_45d')
    input_stack_M45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_M45d')

    ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu '''
    mid_90d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size)(input_stack_90d)
    mid_0d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size)(input_stack_0d)
    mid_45d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size)(input_stack_45d)
    mid_M45d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size)(input_stack_M45d)

    ''' Merge layers '''
    mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d])

    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    mid_merged_ = layer2_merged(sz_input, sz_input2, int(4 * filt_num), int(4 * filt_num), conv_depth, kernel_size)(
        mid_merged)

    ''' Last Conv layer : Conv - Relu - Conv '''
    output = layer3_last(sz_input, sz_input2, int(4 * filt_num), int(4 * filt_num), kernel_size)(mid_merged_)

    model_512 = Model(inputs=[input_stack_90d, input_stack_0d,
                              input_stack_45d, input_stack_M45d], outputs=[output])
    opt = RMSprop(learning_rate=learning_rate)
    model_512.compile(optimizer=opt, loss='mae', metrics=[metrics.MeanSquaredError(), bpr])
    return model_512


def evaluateEPINet_dwsc():
    image_w = 512
    image_h = 512
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 70
    model_learning_rate = 0.1 ** 4
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )
    model = define_epinet(image_w, image_h,
                                  Setting02_AngualrViews,
                                  model_conv_depth,
                                  model_filt_num,
                                  model_learning_rate)
    # flops1 = evaluation_metric.net_flops(model, table=True)
    # print('FLOPS1: ', flops1)
    # flops2 = keras_flops.get_flops(model,batch_size=1)
    # print('FLOPS2: ',flops2/10**9)
    # use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
    # profile, values = model_profiler(model, Batch_size=1, use_units=use_units)
    # model.summary(expand_nested=True)
    # print(profile)

# if __name__ == '__main__':
#    evaluateEPINet_dwsc()

def get_k_size(kernel_size):
    k = [2]
    if kernel_size == 2:
        k = [2]
    elif kernel_size == 3:
        k = [2, 3]
    elif kernel_size == 4:
        k = [2, 3, 4]
    elif kernel_size == 5:
        k = [2, 3, 4, 5]
    elif kernel_size == 6:
        k = [2, 3, 4, 5, 6]
    elif kernel_size == 7:
        k = [2, 3, 4, 5, 6, 7]
    elif kernel_size == 8:
        k = [2, 3, 4, 5, 6, 7, 8]
    elif kernel_size == 9:
        k = [2, 3, 4, 5, 6, 7, 8, 9]
    return k


def _inverted_res_block(inputs, stride, filters, prefix, kernel_size):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    alpha = 1.0
    expansion = 1.0
    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = pointwise_conv_filters
    x = inputs
    prefix = 'block_{}_'.format(prefix)
    x = layers.Conv2D(expansion * in_channels,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'expand')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'expand_BN')(x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x