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
import tensorflow as tf
import os.path
from keras_flops import get_flops
from tensorflow.keras import backend as K
from model_profiler import model_profiler  # https://pypi.org/project/model-profiler/
from epinet_func.mixpconv import MixNetConvInitializer, GroupedConv2D
from epinet_func.evaluation_metric import bpr, PSNR
from tensorflow.keras.layers import Activation, LeakyReLU, PReLU
from collections import Counter

printStat = False
paddingType = 'same'  # valid
split_t = 'equ' # exp OR equ
conv_type = 'dwsc'

ACTIVATION_FN = 'leakyrelu' # choose from ['relu', 'relu6', 'clipped_linear', 'leakyrelu', 'prelu', 'tanh']
GLOBAL_ATTENTION_TYPE = 'sqex'  # choose from ['se', 'eca', 'cbam', 'coord', 'sqex']

def print_layer_names(model, prefix=''):
    for layer in model.layers:
        print(prefix + layer.name)
        if hasattr(layer, 'layers'):
            print_layer_names(layer, prefix + '  ')

def get_activation(name='relu', layer_name='', **kwargs):
    """
    Returns an activation layer based on the specified name.

    Parameters:
        name (str): Name of the activation function.
        kwargs: Additional keyword arguments passed to the activation layer.

    Supported: 'relu', 'relu6', 'clipped_linear', 'leakyrelu', 'prelu', 'tanh'
    """
    if name == 'relu':
        return Activation('relu', name=layer_name, **kwargs)
    elif name == 'relu6':
        return Activation(tf.nn.relu6, name=layer_name, **kwargs)
    elif name == 'clipped_linear':
        return Activation(lambda x: tf.clip_by_value(x, -6.0, 6.0), name=layer_name, **kwargs)
    elif name == 'leakyrelu':
        return LeakyReLU(alpha=0.1, name=layer_name, **kwargs)
    elif name == 'prelu':
        return PReLU(name=layer_name, **kwargs)
    elif name == 'tanh':
        return Activation('tanh', name=layer_name, **kwargs)
    else:
        raise ValueError(f"Unsupported activation function: {name}")


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

def layer1_multistream(input_dim1, input_dim2, input_dim3, filt_num, kernel_size, msname):
    k = get_k_size(kernel_size)
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3), name='S'+msname+'input')
    x = inputs
    for i in range(3):
        x = GroupedConv2D(filt_num,kernel_size=k, type=conv_type, split_type=split_t,strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding=paddingType,use_bias=False,name='S'+msname+'_mixconv1_%d' % (i))(x)
        #x = Conv2D(int(filt_num),(1,1), padding=paddingType, name='S'+msname+'_p1%d' %(i))(x)
        x = get_activation(ACTIVATION_FN, layer_name='S'+msname+'_act1%d' % (i))(x)
        x = GroupedConv2D(filt_num, kernel_size=k, type=conv_type, split_type=split_t,  strides=[1, 1], kernel_initializer=MixNetConvInitializer(),
                      padding=paddingType, use_bias=False, name='S'+msname+'_mixconv2_%d' % (i))(x)
        #x = Conv2D(int(filt_num),(1,1), padding=paddingType, name='S'+msname+'_p2%d' %(i))(x)
        x = BatchNormalization(axis=-1, name='S'+msname+'_BN%d' % (i))(x)
        x = get_activation(ACTIVATION_FN, layer_name='S'+msname+'_act2%d' % (i))(x)
        #x = apply_attention(x, GLOBAL_ATTENTION_TYPE, name='S'+msname+'_att%d' % (i))
    outputs = x
    return Model(inputs=inputs, outputs=outputs)

def layer2_merged(input_dim1, input_dim2, input_dim3, filt_num, conv_depth, kernel_size):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    k = get_k_size(kernel_size)
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3), name='S_mid_input')
    x = inputs
    for i in range(conv_depth):
        x = GroupedConv2D(filt_num,kernel_size=k, type=conv_type, split_type=split_t,strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding=paddingType,use_bias=False,name='S2_mixconv1_%d' % (i))(x)
        #x = Conv2D(int(filt_num),(1,1), padding=paddingType, name='S2_p1_%d' %(i))(x)
        x = get_activation(ACTIVATION_FN, layer_name='S2_act1%d' % (i))(x)
        x = GroupedConv2D(filt_num,kernel_size=k, type=conv_type, split_type=split_t,strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding=paddingType,use_bias=False,name='S2_mixconv2_%d' % (i))(x)
        #x = Conv2D(int(filt_num),(1,1), padding=paddingType, name='S2_p2_%d' %(i))(x)
        x = BatchNormalization(axis=-1, name='S2_BN%d' % (i))(x)
        x = get_activation(ACTIVATION_FN, layer_name='S2_mixconv2_%d' % (i))(x)
        #x = apply_attention(x, GLOBAL_ATTENTION_TYPE, name='S2_mixconv2_att%d' % (i))
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


def layer3_last(input_dim1, input_dim2, input_dim3, filt_num, kernel_size):
    ''' last layer : Conv - Relu - Conv '''
    inputs = Input(shape=(input_dim1, input_dim2, input_dim3), name='S_last_input')
    x = inputs
    k = get_k_size(kernel_size)
    for i in range(1):
        x = GroupedConv2D(filt_num,kernel_size=k, type=conv_type, split_type=split_t,strides=[1, 1],kernel_initializer=MixNetConvInitializer(),
                          padding=paddingType,use_bias=False,name='S3_mixconv1_%d' % (i))(x)
        #x = Conv2D(int(filt_num),(1,1), padding=paddingType, name='S3_p1%d' %(i))(x)
        x = get_activation(ACTIVATION_FN, layer_name='S3_act1%d' % (i))(x)
        #x = apply_attention(x, GLOBAL_ATTENTION_TYPE, name='S3_att%d' % (i))

    #x = Conv2D(64, (1, 1), padding=paddingType, use_bias=False, name='S3_64_last')(x)
    #x = Conv2D(32, (1, 1), padding=paddingType, use_bias=False, name='S3_32_last')(x)
    outputs = Conv2D(1, (1, 1), padding=paddingType, use_bias=False, name='S3_last')(x)
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
    mid_90d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, '90d')(input_stack_90d)
    mid_0d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, '0d')(input_stack_0d)
    mid_45d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, '45d')(input_stack_45d)
    mid_M45d = layer1_multistream(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'm45d')(input_stack_M45d)

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

def ensure_unique_layer_names(model):
    existing_names = set()
    for layer in model.layers:
        original_name = layer.name
        if original_name in existing_names:
            count = 1
            new_name = f"{original_name}_{count}"
            while new_name in existing_names:
                count += 1
                new_name = f"{original_name}_{count}"
            layer._name = new_name  # Update the layer name
        existing_names.add(layer.name)

def evaluateEPINet_dwsc():
    image_w = 512
    image_h = 512
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 64
    model_learning_rate = 0.1 ** 4
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )
    model = define_epinet(image_w, image_h,
                                  Setting02_AngualrViews,
                                  model_conv_depth,
                                  model_filt_num,
                                  model_learning_rate,
                                  3)

    model.summary(expand_nested=True)
    model.save('attempt.hdf5')
    #print_layer_names(model)

    # Check for duplicate layer names
    layer_names = [layer.name for layer in model.layers]
    duplicates = [item for item, count in Counter(layer_names).items() if count > 1]

    if duplicates:
        print(f"Duplicate layer names found: {duplicates}")
        ensure_unique_layer_names(model)
        print("Layer names updated to be unique.")
    else:
        print('Layer names are unique')

    # Create dummy data for each input
    dummy_input_90d = np.random.rand(1, 512, 512, 9).astype(np.float32)
    dummy_input_0d = np.random.rand(1, 512, 512, 9).astype(np.float32)
    dummy_input_45d = np.random.rand(1, 512, 512, 9).astype(np.float32)
    dummy_input_M45d = np.random.rand(1, 512, 512, 9).astype(np.float32)
    output = model.predict([
        dummy_input_90d,
        dummy_input_0d,
        dummy_input_45d,
        dummy_input_M45d
    ])
    print("Output shape:", output.shape)

    # flops1 = evaluation_metric.net_flops(model, table=True)
    # print('FLOPS1: ', flops1)
    # flops2 = keras_flops.get_flops(model,batch_size=1)
    # print('FLOPS2: ',flops2/10**9)
    # use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
    # profile, values = model_profiler(model, Batch_size=1, use_units=use_units)
    # model.summary(expand_nested=True)
    # print(profile)

if __name__ == '__main__':
   evaluateEPINet_dwsc()
