# -*- coding: utf-8 -*-
"""
Created on April 2025

@author: AliHassan
Author Implementation of EPINET-Lite - MMSP 2025
"""

from __future__ import print_function

from epinet_func.func_makeinput import make_multiinput
from epinet_func.func_generate_traindata import generate_traindata_for_train
from epinet_func.func_generate_traindata import data_augmentation_for_train
from epinet_func.func_generate_traindata import generate_traindata512
from epinet_func.func_epinetmodel_mixpdwc import define_epinet
from epinet_func.func_pfm import read_pfm
from epinet_func.func_savedata import display_current_output
from epinet_func.util import load_LFdata

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os
import time
import imageio
import datetime
import threading
import sys
from epinet_func.func_pfm import write_pfm
import numpy as np
import tensorflow as tf
import h5py
import os
import time
import datetime
import threading
import pandas as pd
import imageio
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from epinet_func.func_savedata import measurePerformance
from tensorflow.keras.preprocessing.image import array_to_img, save_img

if __name__ == '__main__':
    kernel_size = 3#int(sys.argv[1])

    save_checkpoint = False

    ffname = 'epinet_lite_g' + str(kernel_size) + '_relu_rmsprop_100e'
    # kernel_size = 3
    print('Name: ', ffname)
    networkname = ffname

    ''' 
    We use fit_generator to train EPINET, 
    so here we defined a generator function.
    '''
    display_status_ratio = 10000  # 10000
    t_epoches = 1000 # 10000 * 1000 = 10000000


    class threadsafe_iter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """

        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __len__(self):
            global display_status_ratio
            return display_status_ratio

        def __iter__(self):
            return self

        def __next__(self):
            with self.lock:
                return self.it.__next__()


    def threadsafe_generator(f):
        """A decorator that takes a generator function and makes it thread-safe.
        """

        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))

        return g


    @threadsafe_generator
    def myGenerator(traindata_all, traindata_label,
                    input_size, label_size, batch_size,
                    Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15):
        while 1:
            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d,
             traindata_label_batchNxN) = generate_traindata_for_train(traindata_all, traindata_label,
                                                                      input_size, label_size, batch_size,
                                                                      Setting02_AngualrViews, boolmask_img4,
                                                                      boolmask_img6, boolmask_img15)

            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d,
             traindata_label_batchNxN) = data_augmentation_for_train(traindata_batch_90d,
                                                                     traindata_batch_0d,
                                                                     traindata_batch_45d,
                                                                     traindata_batch_m45d,
                                                                     traindata_label_batchNxN,
                                                                     batch_size)

            traindata_label_batchNxN = traindata_label_batchNxN[:, :, :, np.newaxis]

            x = [traindata_batch_90d,
                 traindata_batch_0d,
                 traindata_batch_45d,
                 traindata_batch_m45d]
            # x = np.concatenate(x, axis=-1)

            yield (x,
                   traindata_label_batchNxN)


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    If_trian_is = True;

    ''' 
    GPU setting ( Our setting: gtx 1080ti,  
                               gpu number = 0 ) 
    '''
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    iter00 = 0;

    ''' 
    Define Model parameters    
        first layer:  3 convolutional blocks, 
        second layer: 7 convolutional blocks, 
        last layer:   1 convolutional block
    '''
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 64
    model_learning_rate = 1e-3

    ''' 
    Define Patch-wise training parameters
    '''
    input_size = 25  # Input size should be greater than or equal to 23
    label_size = input_size  # - 22  Since label_size should be greater than or equal to 1
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )

    batch_size = 16
    workers_num = 2  # number of threads

    ''' 
    Define directory for saving checkpoint file & disparity output image
    '''
    directory_ckp = "epinet_checkpoints/%s_ckp" % (networkname)
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)

    if not os.path.exists('epinet_output/'):
        os.makedirs('epinet_output/')
    directory_t = 'epinet_output/%s' % (networkname)
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)

    txt_name = 'epinet_checkpoints/lf_%s.txt' % (networkname)

    ''' 
    Load Train data from LF .png files
    '''
    print('Load training data...')
    dir_LFimages = [
        'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
        'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
        'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
        'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl']

    data_path = '../full_data/'
    data_path = 'C:/Users/alihas/MyData/AliHassan/Python/lf_dataset/hci_dataset/'
    traindata_all, traindata_label = load_LFdata(dir_LFimages, data_path)

    traindata_90d, traindata_0d, traindata_45d, traindata_m45d, _ = generate_traindata512(traindata_all,
                                                                                          traindata_label,
                                                                                          Setting02_AngualrViews)
    # (traindata_90d, 0d, 45d, m45d) to validation or test
    # traindata_90d, 0d, 45d, m45d:  16x512x512x9  float32

    print('Load training data... Complete')

    '''load invalid regions from training data (ex. reflective region)'''
    boolmask_img4 = imageio.v2.imread(data_path + 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
    boolmask_img6 = imageio.v2.imread(data_path + 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
    boolmask_img15 = imageio.v2.imread(data_path + 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')

    boolmask_img4 = 1.0 * boolmask_img4[:, :, 3] > 0
    boolmask_img6 = 1.0 * boolmask_img6[:, :, 3] > 0
    boolmask_img15 = 1.0 * boolmask_img15[:, :, 3] > 0

    ''' 
    Load Test data from LF .png files
    '''
    print('Load test data...')
    dir_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']

    valdata_all, valdata_label = load_LFdata(dir_LFimages, data_path)

    valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(valdata_all,
                                                                                              valdata_label,
                                                                                              Setting02_AngualrViews)

    train_generator = myGenerator(traindata_all, traindata_label, input_size, label_size, batch_size,
                                  Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15)

    valid_generator = myGenerator(valdata_all, valdata_label, input_size, label_size, batch_size,
                                  Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15)

    # (valdata_90d, 0d, 45d, m45d) to validation or test
    print('Load test data... Complete')
    model_25 = define_epinet(input_size, input_size,
                             Setting02_AngualrViews,
                             model_conv_depth,
                             model_filt_num,
                             model_learning_rate,
                             kernel_size)

    model_25.summary()

    """ 
    load latest_checkpoint
    """
    load_weight_is = False
    if load_weight_is:
        model_25.load_weights(directory_ckp + '/modelLast_' + networkname + '.hdf5')
        print("Network weights will be loaded from previous checkpoints")

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                  patience=10, verbose=1, mode='min', min_lr=1e-12)

    checkpoint = ModelCheckpoint(directory_ckp + '/modelBest_' + networkname + '.hdf5',
                                 verbose=1,
                                 monitor='val_mean_squared_error',
                                 save_best_only=True,
                                 mode='min'
                                 )
    postfix = '_best'
    pfm_output = './' + networkname + postfix + '/disp_maps/'
    if not os.path.exists(pfm_output):
        os.makedirs(pfm_output)
    runtime_output = './' + networkname + postfix + '/runtimes/'
    if not os.path.exists(runtime_output):
        os.makedirs(runtime_output)
    #custom_eval_callback = CustomEvaluationCallback(dir_LFimages, data_path, 512, 512, Setting02_AngualrViews,
    #                                                pfm_output, runtime_output)

    hist = model_25.fit(train_generator, steps_per_epoch=int(display_status_ratio),
                        validation_data=valid_generator, validation_steps=int(display_status_ratio),
                        epochs=iter00 + t_epoches, class_weight=None, max_queue_size=10,
                        initial_epoch=iter00, verbose=2, workers=workers_num,
                        callbacks=[reduce_lr])  # , callbacks=[custom_eval_callback, checkpoint, reduce_lr]

    if save_checkpoint:
        # Check if the file exists and remove it
        save_dir = directory_ckp + '/modelLast_' + networkname + '.hdf5'
        if os.path.exists(directory_ckp + '/modelLast_' + networkname + '.hdf5'):
            os.remove(directory_ckp + '/modelLast_' + networkname + '.hdf5')

        model_25.save(directory_ckp + '/modelLast_' + networkname + '.hdf5')
    hist_df = pd.DataFrame(hist.history)
    hist_csv_file = 'epinet_output/history_' + str(networkname) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    ''' 
    Model for predicting full-size LF images  
    '''
    print('Evaluating Last Epoch Trained Model!')
    postfix = '_last'
    image_w = 512
    image_h = 512
    model = define_epinet(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                          model_learning_rate, kernel_size)
    weight_tmp1 = model_25.get_weights()
    model.set_weights(weight_tmp1)
    LFname = []
    mseList = []
    bp7List = []
    bp3List = []
    bp1List = []
    psnrList = []
    timeList = []
    ''' 
    Load Validation data from LF .png files
    '''
    dir_LFimages = ['stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
                    'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
    pfm_output = './' + networkname + postfix + '/disp_maps/'
    if not os.path.exists(pfm_output):
        os.makedirs(pfm_output)
    runtime_output = './' + networkname + postfix + '/runtimes/'
    if not os.path.exists(runtime_output):
        os.makedirs(runtime_output)
    for currentLFimages in dir_LFimages:
        curLFname = currentLFimages.split('/')[-1]
        LFname.append(currentLFimages)
        LFimages_list = [currentLFimages]
        valdata_all, valdata_label = load_LFdata(LFimages_list, data_path)
        valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(valdata_all,
                                                                                                  valdata_label,
                                                                                                  Setting02_AngualrViews)
        # (valdata_90d, 0d, 45d, m45d) to validation or test
        # print('Result of :',currentLFimages)
        start = time.time()
        model_output = model.predict([valdata_90d, valdata_0d,
                                      valdata_45d, valdata_m45d], batch_size=1)
        end = time.time()
        # save .pfm file
        outDisp = (np.array(model_output[0, :, :, 0])).reshape([512, 512, 1])
        # outDisp = array_to_img(outDisp)
        save_img(pfm_output + curLFname + '.png', outDisp, file_format='png')
        write_pfm(model_output[0, :, :, 0], pfm_output + curLFname + '.pfm')
        with open(runtime_output + curLFname + '.txt', 'w') as f:
            f.write(str(end - start))
        timeList.append(end - start)
        valid_error, valid_bp7, valid_bp3, valid_bp1, valid_psnr = measurePerformance(model_output, valdata_label,
                                                                                      networkname, currentLFimages)
        valid_mean_squared_error = np.average(np.square(valid_error))  # Multiplied by 100
        valid_bad_pixel_ratio7 = np.average(valid_bp7)  # Multiplied by 100
        valid_bad_pixel_ratio3 = np.average(valid_bp3)  # Multiplied by 100
        valid_bad_pixel_ratio1 = np.average(valid_bp1)  # Multiplied by 100
        print('Name: ', currentLFimages, ' Time: ', end - start, ' MSE: ', valid_mean_squared_error, ' BPR7: ',
              valid_bad_pixel_ratio7, ' BPR3: ', valid_bad_pixel_ratio3, ' BPR1: ', valid_bad_pixel_ratio1, ' PSNR: ',
              valid_psnr)
        mseList.append(valid_mean_squared_error)
        bp7List.append(valid_bad_pixel_ratio7)
        bp3List.append(valid_bad_pixel_ratio3)
        bp1List.append(valid_bad_pixel_ratio1)
        psnrList.append(valid_psnr)
    r = np.array(
        [np.array(LFname).astype('str').T, np.array(mseList).astype('float32').T, np.array(bp7List).astype('float32').T,
         np.array(bp3List).astype('float32').T, np.array(bp1List).astype('float32').T,
         np.array(psnrList).astype('float32').T, np.array(timeList).astype('float32').T])
    pd.DataFrame(r.T).to_csv('./epinet_output/' + networkname + '.csv',
                             header=['Name', 'MSE', 'BP7', 'BP3', 'BP1', 'PSNR', 'Time'], index=False)
    print('Average MSE: ', np.average(mseList))
    print('Average BPR7: ', np.average(bp7List))
    print('Average BPR3: ', np.average(bp3List))
    print('Average BPR1: ', np.average(bp1List))
    print('Average PSNR: ', np.average(psnrList))
    print('Average Inference Time: ', np.average(timeList))
    print('Done!')
    ''' 
    Load Test data from LF .png files
    '''
    # print('Load test data...')
    dir_LFimages = ['test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
    for currentLFimages in dir_LFimages:
        curLFname = currentLFimages.split('/')[-1]
        LFname.append(currentLFimages)
        LFimages_list = [currentLFimages]
        (val_90d, val_0d, val_45d, val_m45d) = make_multiinput(data_path + currentLFimages,
                                                               image_h,
                                                               image_w,
                                                               Setting02_AngualrViews)
        start = time.time()
        model_output = model.predict([val_90d, val_0d,
                                      val_45d, val_m45d], batch_size=1)
        end = time.time()
        # save .pfm file
        outDisp = (np.array(model_output[0, :, :, 0])).reshape([512, 512, 1])
        # outDisp = array_to_img(outDisp)
        save_img(pfm_output + curLFname + '.png', outDisp, file_format='png')
        write_pfm(model_output[0, :, :, 0], pfm_output + curLFname + '.pfm')
        with open(runtime_output + curLFname + '.txt', 'w') as f:
            f.write(str(end - start))
        timeList.append(end - start)
        print('Processed: ', currentLFimages)
    print('Average Inference Time: ', np.average(timeList))
    print('Done!')

    if save_checkpoint:
        # Best Model Predictions
        model_25.load_weights(directory_ckp + '/modelBest_' + networkname + '.hdf5')
        ''' 
        Model for predicting full-size LF images  
        '''
        print('Evaluating Best Trained Model!')
        postfix = '_best'
        image_w = 512
        image_h = 512
        model = define_epinet(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                              model_learning_rate, kernel_size)
        weight_tmp2 = model_25.get_weights()
        model.set_weights(weight_tmp2)
        LFname = []
        mseList = []
        bp7List = []
        bp3List = []
        bp1List = []
        psnrList = []
        timeList = []
        ''' 
        Load Validation data from LF .png files
        '''
        dir_LFimages = ['stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
                        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
        pfm_output = './' + networkname + postfix + '/disp_maps/'
        if not os.path.exists(pfm_output):
            os.makedirs(pfm_output)
        runtime_output = './' + networkname + postfix + '/runtimes/'
        if not os.path.exists(runtime_output):
            os.makedirs(runtime_output)
        for currentLFimages in dir_LFimages:
            curLFname = currentLFimages.split('/')[-1]
            LFname.append(currentLFimages)
            LFimages_list = [currentLFimages]
            valdata_all, valdata_label = load_LFdata(LFimages_list, data_path)
            valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(valdata_all,
                                                                                                      valdata_label,
                                                                                                      Setting02_AngualrViews)
            # (valdata_90d, 0d, 45d, m45d) to validation or test
            # print('Result of :',currentLFimages)
            start = time.time()
            model_output = model.predict([valdata_90d, valdata_0d,
                                          valdata_45d, valdata_m45d], batch_size=1)
            end = time.time()
            # save .pfm file
            outDisp = (np.array(model_output[0, :, :, 0])).reshape([512, 512, 1])
            # outDisp = array_to_img(outDisp)
            save_img(pfm_output + curLFname + '.png', outDisp, file_format='png')
            write_pfm(model_output[0, :, :, 0], pfm_output + curLFname + '.pfm')
            with open(runtime_output + curLFname + '.txt', 'w') as f:
                f.write(str(end - start))
            timeList.append(end - start)
            valid_error, valid_bp7, valid_bp3, valid_bp1, valid_psnr = measurePerformance(model_output, valdata_label,
                                                                                          networkname, currentLFimages)
            valid_mean_squared_error = np.average(np.square(valid_error))  # Multiplied by 100
            valid_bad_pixel_ratio7 = np.average(valid_bp7)  # Multiplied by 100
            valid_bad_pixel_ratio3 = np.average(valid_bp3)  # Multiplied by 100
            valid_bad_pixel_ratio1 = np.average(valid_bp1)  # Multiplied by 100
            print('Name: ', currentLFimages, ' Time: ', end - start, ' MSE: ', valid_mean_squared_error, ' BPR7: ',
                  valid_bad_pixel_ratio7, ' BPR3: ', valid_bad_pixel_ratio3, ' BPR1: ', valid_bad_pixel_ratio1, ' PSNR: ',
                  valid_psnr)
            mseList.append(valid_mean_squared_error)
            bp7List.append(valid_bad_pixel_ratio7)
            bp3List.append(valid_bad_pixel_ratio3)
            bp1List.append(valid_bad_pixel_ratio1)
            psnrList.append(valid_psnr)
        r = np.array(
            [np.array(LFname).astype('str').T, np.array(mseList).astype('float32').T, np.array(bp7List).astype('float32').T,
             np.array(bp3List).astype('float32').T, np.array(bp1List).astype('float32').T,
             np.array(psnrList).astype('float32').T, np.array(timeList).astype('float32').T])
        pd.DataFrame(r.T).to_csv('./epinet_output/' + networkname + '.csv',
                                 header=['Name', 'MSE', 'BP7', 'BP3', 'BP1', 'PSNR', 'Time'], index=False)
        print('Average MSE: ', np.average(mseList))
        print('Average BPR7: ', np.average(bp7List))
        print('Average BPR3: ', np.average(bp3List))
        print('Average BPR1: ', np.average(bp1List))
        print('Average PSNR: ', np.average(psnrList))
        print('Average Inference Time: ', np.average(timeList))
        print('Done!')
        ''' 
        Load Test data from LF .png files
        '''
        # print('Load test data...')
        dir_LFimages = ['test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
        for currentLFimages in dir_LFimages:
            curLFname = currentLFimages.split('/')[-1]
            LFname.append(currentLFimages)
            LFimages_list = [currentLFimages]
            (val_90d, val_0d, val_45d, val_m45d) = make_multiinput(data_path + currentLFimages,
                                                                   image_h,
                                                                   image_w,
                                                                   Setting02_AngualrViews)
            start = time.time()
            model_output = model.predict([val_90d, val_0d,
                                          val_45d, val_m45d], batch_size=1)
            end = time.time()
            # save .pfm file
            outDisp = (np.array(model_output[0, :, :, 0])).reshape([512, 512, 1])
            # outDisp = array_to_img(outDisp)
            save_img(pfm_output + curLFname + '.png', outDisp, file_format='png')
            write_pfm(model_output[0, :, :, 0], pfm_output + curLFname + '.pfm')
            with open(runtime_output + curLFname + '.txt', 'w') as f:
                f.write(str(end - start))
            timeList.append(end - start)
            print('Processed: ', currentLFimages)
        print('Average Inference Time: ', np.average(timeList))
        print('Done!')

