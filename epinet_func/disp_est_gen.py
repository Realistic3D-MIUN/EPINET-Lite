from __future__ import print_function


from epinet_func.func_generate_traindata import generate_traindata_for_train
from epinet_func.func_generate_traindata import data_augmentation_for_train
from epinet_func.func_generate_traindata import generate_traindata512
from epinet_func.util import load_LFdata
from epinet_func.func_epinetmodel_dwsc import define_epinet_dwsc
import numpy as np

import h5py
import os
import time
import datetime
import threading

def load_gen(batchSize):


    class threadsafe_iter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """

        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __len__(self):
            #print('__length_hint__ called')
            return 10000

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
                    Setting02_AngualrViews):#,boolmask_img4, boolmask_img6, boolmask_img15):
        while 1:
            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d,
             traindata_label_batchNxN) = generate_traindata_for_train(traindata_all, traindata_label,
                                                                      input_size, label_size, batch_size,
                                                                      Setting02_AngualrViews)
            # ,boolmask_img4, boolmask_img6, boolmask_img15)

            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d,
             traindata_label_batchNxN) = data_augmentation_for_train(traindata_batch_90d,
                                                                     traindata_batch_0d,
                                                                     traindata_batch_45d,
                                                                     traindata_batch_m45d,
                                                                     traindata_label_batchNxN,
                                                                     batch_size)

            traindata_label_batchNxN = traindata_label_batchNxN[:, :, :, np.newaxis]

            x = np.concatenate([traindata_batch_90d,
                    traindata_batch_0d,
                    traindata_batch_45d,
                    traindata_batch_m45d], axis=-1)

            yield (x,
                   traindata_label_batchNxN)


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ''' 
    Define Patch-wise training parameters
    '''
    input_size = 25  # Input size should be greater than or equal to 23
    label_size = input_size # - 22  Since label_size should be greater than or equal to 1
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )

    ''' 
    Load Train data from LF .png files
    '''
    print('Load training data...')
    dir_LFimages = [
        'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
        'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
        'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
        'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl']

    data_path = 'C:/Users/alihas/MyData/AliHassan/Python/lf_dataset/hci_dataset/full_data/'
    traindata_all, traindata_label = load_LFdata(dir_LFimages, data_path)

    #traindata_90d, traindata_0d, traindata_45d, traindata_m45d, _ = generate_traindata512(traindata_all,
    #                                                                                      traindata_label,
    #                                                                                      Setting02_AngualrViews)
    # (traindata_90d, 0d, 45d, m45d) to validation or test
    # traindata_90d, 0d, 45d, m45d:  16x512x512x9  float32

    print('Load training data... Complete')

    ''' 
    Load Test data from LF .png files
    '''
    print('Load test data...')
    dir_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']

    valdata_all, valdata_label = load_LFdata(dir_LFimages, data_path)

    #valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(valdata_all,
    #                                                                                          valdata_label,
    #                                                                                          Setting02_AngualrViews)

    t_generator = myGenerator(traindata_all, traindata_label, input_size, label_size, batchSize,
                               Setting02_AngualrViews)#, boolmask_img4, boolmask_img6, boolmask_img15)

    v_generator = myGenerator(valdata_all, valdata_label, input_size, label_size, batchSize,
                               Setting02_AngualrViews)#, boolmask_img4, boolmask_img6, boolmask_img15)

    return t_generator, v_generator