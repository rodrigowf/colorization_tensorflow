# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:16:29 2018

@author: rahul.ghosh
"""

import os

# DIRECTORY INFORMATION
DATASET = "flickr"
ROOT_DIR = '/content/colorization_tensorflow/'
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 2

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
PRETRAINED = "Flickrmodel2_1_10000"
NUM_EPOCHS = 1
