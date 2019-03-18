#!/usr/bin/python
from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import keras
import random
import os
import unet
from keras import backend as K

from data import load_train_data, load_valid_data, generate_arrays_from_file


smooth = 1.


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1

        self.valid.append(logs.get('val_dice_coef'))

        with open("runs/history.txt" , "a") as fout:
                fout.write("train\t%d\t%.4f\n" % (self.lastiter, self.losses[-1]))
                fout.write("valid\t%d\t%.4f\n" % (self.lastiter, self.valid[-1]))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1
        with open("runs/history.txt", "a") as fout:
                fout.write("train\t%d\t%.4f\n" % (self.lastiter, self.losses[-1]))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def train(epochs):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet.get_unet()
    model.load_weights("deeplearning/new.hdf5", by_name=True)
    lr = 1e-5
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load("prepdata/npyarrays/imgs_train.npy")
    imgs_mask_train = np.load("prepdata/npyarrays/imgs_mask_train.npy")

    imgs_valid = np.load("prepdata/npyarrays/imgs_valid.npy")
    imgs_mask_valid = np.load("prepdata/npyarrays/imgs_mask_valid.npy")

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_valid = imgs_valid.astype('float32')
    imgs_mask_valid = imgs_mask_valid.astype('float32')

    imgs_train -= 28.991758347
    imgs_train /= 46.875888824

    imgs_valid -= 28.991758347
    imgs_valid /= 46.875888824

    imgs_mask_train /= 255.
    imgs_mask_valid /= 255.

    print('-'*30)
    print('Evaluating transfer learning.')
    print('-'*30)

    valloss = model.evaluate(x = imgs_valid, y = imgs_mask_valid, batch_size=10, verbose=1)
    with open("runs/history.txt" , "a") as fout:
	fout.write("valid\t%d\t%.4f\n" % (0, valloss[1]))
    filepath="runs/weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor="val_dice_coef", mode="max")

    history = LossHistory()

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(x = imgs_train, y = imgs_mask_train, validation_data = (imgs_valid, imgs_mask_valid), 
        batch_size=10, epochs=epochs, verbose=1, callbacks=[history, checkpoint])

if __name__ == '__main__':
    train()
