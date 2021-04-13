# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:16:14 2020

@author: Bruno
"""

import numpy as np
import cv2
import os

CATG = ['normal', 'covid', 'pneumonia']

def create_data(DIR, num_samples):
  IMG_SIZE = 50 #resize size
  train_data = [] # array of images

  for category in CATG:
      class_num = CATG.index(category) # classifica como 0, 1 ou 2
      path = os.path.join(DIR, category) #path to dir
      for i, img in enumerate(os.listdir(path)): #iterate over files in folder
          if(i>num_samples):  #dont get all samples
            break
          try:
            img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #read image in gray
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #resize
            train_data.append([img, class_num]) #add to array
          except:
            pass

  return train_data

# train = create_data(r'D:\Downloads\dataset_fastai\train', 144)

# test = create_data(r'D:\Downloads\dataset_fastai\test', 83)

train = create_data('chestxray-dataset-cnn/train', 800)
test = create_data('chestxray-dataset-cnn/test', 285)

np.save('train.npy', train) 
np.save('test.npy', test)