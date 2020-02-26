# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:27:28 2018

@author: rahul.ghosh
"""

import numpy as np
import cv2
import os
import config

gray_scale = np.load('/content/input/l/gray_scale.npy')
print(gray_scale.shape)  # (25000, 224, 224) from 0 to 255

class DATA():

    def __init__(self, dirname):

        if(dirname == 'test'):
          ab_scale3 = np.load('/content/input/ab/ab/ab3.npy')
          self.ab_scale = ab_scale3[3000:3020]
          self.gray_scale = gray_scale[23000:23020]
          del ab_scale3
        elif(dirname == 'train'):
          # ab_scale1 = np.load('/content/input/ab/ab/ab1.npy')
          # ab_scale2 = np.load('/content/input/ab/ab/ab2.npy')
          # self.ab_scale = np.concatenate((ab_scale1, ab_scale2), 0)
          # del ab_scale1
          # del ab_scale2
          self.ab_scale = np.load('/content/input/ab/ab/ab1.npy')
          self.gray_scale = gray_scale[:10000]
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = range(config.NUM_EPOCHS*config.BATCH_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.size = self.ab_scale.shape[0]
        self.data_index = 0

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        return np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), labimg[:, :, 1:]

    def generate_batch(self):
        batch = np.reshape(self.gray_scale[self.data_index:self.data_index+self.batch_size]/255, (self.batch_size, config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
        labels = self.ab_scale[self.data_index:self.data_index+self.batch_size]/255
        filelist = range(self.data_index, self.data_index+self.batch_size)
        self.data_index = self.data_index + self.batch_size
        return batch, labels, filelist