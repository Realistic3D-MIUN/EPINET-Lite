# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation, Layer, Add, Lambda
from tensorflow.keras.layers import Conv2D, Reshape, DepthwiseConv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU
from tensorflow.keras.backend import concatenate
import numpy as np
from tensorflow.keras import applications
from epinet_func import evaluation_metric
#from epinet_func.func_feature_extraction_cell import FECell, FECellL, FECell_PDDARTS
from epinet_func.func_feature_extraction_cell import FECellA as FECell
import math
import tensorflow as tf
import os.path
#from keras_flops import get_flops
from tensorflow.keras import backend as K
from model_profiler import model_profiler  # https://pypi.org/project/model-profiler/

import tensorflow as tf

class MixedNormLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, **kwargs):
        super(MixedNormLoss, self).__init__(**kwargs)
        #alpha = 0.35
        self.alpha = alpha

    def call(self, y_true, y_pred):
        l1 = tf.abs(y_pred - y_true)
        l2 = tf.square(y_pred - y_true)
        loss = l1 + ((self.alpha * l2) ** 2)
        return tf.reduce_mean(loss, axis=-1)  # Compute mean loss across the batch axis


def bpr(y_true, y_pred):
    threshold = tf.constant(0.07, dtype=tf.float32)
    mask = tf.cast(tf.less(y_pred, threshold), dtype=tf.float32)
    incorrect_pixels = tf.reduce_sum(mask * (1 - y_true))
    total_pixels = tf.reduce_sum(1 - y_true)
    bpr = incorrect_pixels / total_pixels
    return bpr


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    # return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
                                                                                   y_true))))


vList = []
printStat = False
padding = 'same'  # 'valid


def pre_process(input_dim1, input_dim2, input_dim3, filt_num, kernel_size, prefix):
    seq = Sequential()
    seq.add(Conv2D(int(filt_num), (kernel_size, kernel_size), input_shape=(input_dim1, input_dim2, input_dim3),
                   padding=padding, name=prefix + '_conv2d'))
    seq.add(BatchNormalization(axis=-1, name=prefix + '_bn'))
    #seq.add(Lambda(lambda x: tf.nn.relu6(x)))
    seq.add(Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-4.0, clip_value_max=4.0)))
    #seq.add(Activation('relu', name=prefix + '_act'))
    return seq

class conv_bn(Layer):
    def __init__(self, out_filter, res):
        super(conv_bn, self).__init__()
        self._convbn = None
        self.res = res
        self.ou_filter = out_filter

    def build(self, input_shape):
        # Define layers here using input_shape
        self._convbn = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding, input_shape=(self.res, self.res, self.out_filter)),
            BatchNormalization(),
            #Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-4.0, clip_value_max=4.0))
            #ReLU()
        ])

    def call(self, inputs):
        return self._convbn(inputs)




def single_stream_features_extraction(p1, p2, sz_input, features, prefix):
    ss1 = FECell(features, sz_input, prefix + '_01_')
    ss2 = FECell(features, sz_input, prefix + '_02_')
    ss3 = FECell(features, sz_input, prefix + '_03_')
    #ss4 = FECell(features, sz_input, prefix + '_04_')
    # Output, Input_H-1 = FECell(Input_H-1, Input_H-2)
    p1_mid_1, p2_mid_1 = ss1(p1, p2)
    p1_mid_2, p2_mid_2 = ss2(p1_mid_1, p2_mid_1)
    p1_mid_3, p2_mid_3 = ss3(p1_mid_2, p2_mid_2)
    #p1_mid_4, p2_mid_4 = ss4(p1_mid_3, p2_mid_3)

    return p1_mid_3, p2_mid_3
    #return p1_mid_4, p2_mid_4


def multi_stream_features_extraction(p1_90d, p2_90d, sz_input, features, prefix):
    ss1 = FECell(features, sz_input, prefix + '_01_')
    ss2 = FECell(features, sz_input, prefix + '_02_')
    ss3 = FECell(features, sz_input, prefix + '_03_')
    ss4 = FECell(features, sz_input, prefix + '_04_')
    ss5 = FECell(features, sz_input, prefix + '_05_')
    ss6 = FECell(features, sz_input, prefix + '_06_')
    ss7 = FECell(features, sz_input, prefix + '_07_')
    ss8 = FECell(features, sz_input, prefix + '_08_')

    p1_mid_1, p2_mid_1 = ss1(p1_90d, p2_90d)
    p1_mid_2, p2_mid_2 = ss2(p1_mid_1, p2_mid_1)
    p1_mid_3, p2_mid_3 = ss3(p1_mid_2, p2_mid_2)
    p1_mid_4, p2_mid_4 = ss4(p1_mid_3, p2_mid_3)
    p1_mid_5, p2_mid_5 = ss5(p1_mid_4, p2_mid_4)
    p1_mid_6, p2_mid_6 = ss6(p1_mid_5, p2_mid_5)
    p1_mid_7, p2_mid_7 = ss7(p1_mid_6, p2_mid_6)
    p1_mid_8, _ = ss8(p1_mid_7, p2_mid_7)
    return p1_mid_8
    #return p1_mid_6, p2_mid_6

def layer3_last(input_dim1, input_dim2, input_dim3, kernel_size):
    ''' last layer : Conv - Relu - Conv '''
    kernel_size = 1
    seq = Sequential()
    seq.add(Conv2D(128, (kernel_size, kernel_size), padding=padding, input_shape=(input_dim1, input_dim2, input_dim3), name='S3_last_c1'))
    seq.add(Conv2D(64, (kernel_size, kernel_size), padding=padding, name='S3_last_c2'))
    seq.add(Conv2D(1, (kernel_size, kernel_size), padding=padding, name='S3_last_c3'))
    return seq


def define_epinet(sz_input, sz_input2, view_n, conv_depth, filt_num, learning_rate, kernel_size):
    tFlops = 0
    tMem = 0
    tParam = 0
    tMem_req = 0

    # Define the input layers
    input_stack_90d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_90d')
    input_stack_0d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_0d')
    input_stack_45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_45d')
    input_stack_M45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_M45d')
    #input_stack_sobel = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_sobel')

    ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu '''
    p1_90d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input90d_1')(input_stack_90d)
    p2_90d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input90d_2')(input_stack_90d)

    p1_0d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input0d_1')(input_stack_0d)
    p2_0d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input0d_2')(input_stack_0d)

    p1_45d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input45d_1')(input_stack_45d)
    p2_45d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'input45d_2')(input_stack_45d)

    p1_M45d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'inputM45d_1')(input_stack_M45d)
    p2_M45d = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'inputM45d_2')(input_stack_M45d)

    #p1_sobel = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'inputsobel_1')(input_stack_sobel)
    #p2_sobel = pre_process(sz_input, sz_input2, len(view_n), int(filt_num), kernel_size, 'inputsobel_2')(input_stack_sobel)

    # ss(tf.zeros([1, 512, 512, 70]),tf.zeros([1, 512, 512, 70]))

    p1_mid_90d_4, p2_mid_90d_4 = single_stream_features_extraction(p1_90d, p2_90d, sz_input, int(filt_num), 'ssfe90d')
    p1_mid_0d_4, p2_mid_0d_4 = single_stream_features_extraction(p1_0d, p2_0d, sz_input, int(filt_num), 'ssfe0d')
    p1_mid_45d_4, p2_mid_45d_4 = single_stream_features_extraction(p1_45d, p2_45d, sz_input, int(filt_num), 'ssfe45d')
    p1_mid_M45d_4, p2_mid_M45d_4 = single_stream_features_extraction(p1_M45d, p2_M45d, sz_input, int(filt_num), 'ssfeM45d')
    #p1_mid_sobel_4, p2_mid_sobel_4 = single_stream_features_extraction(p1_sobel, p2_sobel, sz_input, int(filt_num), 'ssfeSobel')

    p1_mid_merged = concatenate([p1_mid_90d_4, p1_mid_0d_4, p1_mid_45d_4, p1_mid_M45d_4])
    p2_mid_merged = concatenate([p2_mid_90d_4, p2_mid_0d_4, p2_mid_45d_4, p2_mid_M45d_4])
    # Merge the layers

    p1_mid_merged_1 = multi_stream_features_extraction(p1_mid_merged, p2_mid_merged, sz_input,
                                                                        int(4 * filt_num), 'msfe')

    #p_all = Add(name='addfinal')([p1_mid_merged_1, p2_mid_merged_2])

    output = layer3_last(sz_input, sz_input2, int(4 * filt_num), kernel_size)(p1_mid_merged_1)

    model_512 = Model(inputs=[input_stack_90d, input_stack_0d,
                              input_stack_45d, input_stack_M45d], outputs=[output])
    opt = RMSprop(learning_rate=learning_rate)
    model_512.compile(optimizer=opt, loss='mae', metrics=[metrics.MeanSquaredError(), bpr])
    return model_512


def evaluateEPINet2():
    image_w = 512
    image_h = 512
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 64
    model_learning_rate = 1e-5
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )
    model = define_epinet(image_w, image_h,
                          Setting02_AngualrViews,
                          model_conv_depth,
                          model_filt_num,
                          model_learning_rate,
                          3)

    print(model)
    model.summary(expand_nested=True)
    #evaluation_metric.get_flops(model)
    # Check for unique layer names
    layer_names = [layer.name for layer in model.layers]
    unique_layer_names = set(layer_names)
    if len(layer_names) == len(unique_layer_names):
        print("All layer names are unique.")
    else:
        print("There are duplicate layer names. Please ensure all layer names are unique.")

    model.save('attempt.hdf5')

def evaluateEPINet():
    image_w = 512
    image_h = 512
    model_conv_depth = 7 # 7 convolutional blocks for second layer
    model_filt_num = 64
    model_learning_rate = 0.1**4
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )
    model = define_epinet(image_w, image_h,
                              Setting02_AngualrViews,
                              model_conv_depth,
                              model_filt_num,
                              model_learning_rate, 3)
    model.summary(expand_nested=True)
    model.save('attempt.hdf5')
    evaluation_metric.get_flops(model)

if __name__ == '__main__':
   evaluateEPINet()

