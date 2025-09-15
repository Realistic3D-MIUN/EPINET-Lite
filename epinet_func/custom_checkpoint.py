# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:58 2017

@author: shinyonsei2
"""

from __future__ import print_function

from epinet_func.func_makeinput import make_multiinput
from epinet_func.func_pfm import write_pfm
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.preprocessing.image import save_img
from epinet_func.func_savedata import measurePerformance
from epinet_func.util import load_LFdata
from epinet_func.func_epinetmodel_mixpdwc import define_epinet
from epinet_func.func_generate_traindata import generate_traindata512

class CustomEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dir_LFimages, data_path, image_h, image_w, Setting02_AngualrViews, pfm_output, runtime_output):
        super(CustomEvaluationCallback, self).__init__()
        self.dir_LFimages = dir_LFimages
        self.data_path = data_path
        self.image_h = image_h
        self.image_w = image_w
        self.Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views (0~8 for 9x9)
        self.model_conv_depth = 7
        self.model_filt_num = 70
        self.model_learning_rate = 1e-3
        self.kernel_size = 3
        self.pfm_output = pfm_output
        self.runtime_output = runtime_output
        self.timeList = []
        self.LFname = []
        self.prev_avg_mse = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        model_full = define_epinet(
            self.image_w,
            self.image_h,
            self.Setting02_AngualrViews,
            self.model_conv_depth,
            self.model_filt_num,
            self.model_learning_rate,
            self.kernel_size
        )
        model_full.set_weights(self.model.get_weights())

        mse_list = []
        bp7_list = []
        prediction_cache = {}

        # Predict and store results for training images
        for currentLFimages in self.dir_LFimages:
            curLFname = currentLFimages.split('/')[-1]
            LFimages_list = [currentLFimages]
            valdata_all, valdata_label = load_LFdata(LFimages_list, self.data_path)
            valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(
                valdata_all, valdata_label, self.Setting02_AngualrViews)

            start = time.time()
            model_output = model_full.predict(
                [valdata_90d, valdata_0d, valdata_45d, valdata_m45d],
                batch_size=1, verbose=0
            )
            end = time.time()

            valid_error, valid_bp7, *_ = measurePerformance(model_output, valdata_label, None, currentLFimages)
            mse = np.average(np.square(valid_error))
            bp7 = np.average(valid_bp7)

            mse_list.append(mse)
            bp7_list.append(bp7)

            prediction_cache[curLFname] = {
                "output": model_output,
                "time": end - start
            }

        avg_mse = np.average(mse_list)
        avg_bpr = np.average(bp7_list)
        print(
            f"\nEpoch {epoch + 1}: Validation MSE = {avg_mse:.6f}, Validation BPR7 = {avg_mse:.6f}")

        if avg_mse < self.prev_avg_mse:
            print("\nSaving outputs...")
            self.prev_avg_mse = avg_mse

            # Save predictions for training images from cache
            for currentLFimages in self.dir_LFimages:
                curLFname = currentLFimages.split('/')[-1]
                model_output = prediction_cache[curLFname]["output"]
                runtime = prediction_cache[curLFname]["time"]

                outDisp = np.array(model_output[0, :, :, 0]).reshape([512, 512, 1])
                save_img(self.pfm_output + curLFname + '.png', outDisp, file_format='png')
                write_pfm(model_output[0, :, :, 0], self.pfm_output + curLFname + '.pfm')
                with open(self.runtime_output + curLFname + '.txt', 'w') as f:
                    f.write(str(runtime))
                self.timeList.append(runtime)

            # Predict and save for test images (not in cache)
            test_images = ['test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
            for currentLFimages in test_images:
                curLFname = currentLFimages.split('/')[-1]
                val_90d, val_0d, val_45d, val_m45d = make_multiinput(
                    self.data_path + currentLFimages,
                    self.image_h,
                    self.image_w,
                    self.Setting02_AngualrViews
                )
                start = time.time()
                model_output = model_full.predict(
                    [val_90d, val_0d, val_45d, val_m45d],
                    batch_size=1, verbose=0
                )
                end = time.time()

                outDisp = np.array(model_output[0, :, :, 0]).reshape([512, 512, 1])
                save_img(self.pfm_output + curLFname + '.png', outDisp, file_format='png')
                write_pfm(model_output[0, :, :, 0], self.pfm_output + curLFname + '.pfm')
                with open(self.runtime_output + curLFname + '.txt', 'w') as f:
                    f.write(str(end - start))
                self.timeList.append(end - start)
