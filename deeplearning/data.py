#!/usr/bin/python
from __future__ import print_function

import os
import numpy as np
import random

import cv2

data_path = '/data/octirfseg/round3/final/'

image_rows = 432
image_cols = 32

batchsize = 149*8


def create_valid_data():
    valid_data_path = os.path.join(data_path, 'valid')
    images = os.listdir(valid_data_path)
    total = len(images) / 2
    print(total)
    total = batchsize * int(total/batchsize)


    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating validing images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        if i == total:
            break
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = cv2.imread(os.path.join(valid_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(valid_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('/data/octirfseg/round3/imgs_valid.npy', imgs)
    np.save('/data/octirfseg/round3/imgs_mask_valid.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_valid_data():
    imgs_valid = np.load('/data/octirfseg/imgs_valid.npy')
    imgs_mask_valid = np.load('/data/octirfseg/imgs_mask_valid.npy')
    imgs_valid = np.reshape(imgs_valid, (imgs_valid.shape[0], imgs_valid.shape[2], imgs_valid.shape[3], 1))
    imgs_mask_valid = np.reshape(imgs_mask_valid, (imgs_mask_valid.shape[0], imgs_mask_valid.shape[2], imgs_mask_valid.shape[3], 1))
    return imgs_valid, imgs_mask_valid


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2
    total = batchsize * int(total/batchsize)
    maxvol = batchsize*100


    imgs = np.ndarray((maxvol, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((maxvol, 1, image_rows, image_cols), dtype=np.uint8)

    gi = 0
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        if i == maxvol:
            print('Saving into shard...')
            np.save("/data/octirfseg/round3/npyarrays/imgs_train-%03d.npy" % gi, imgs)
            np.save("/data/octirfseg/round3/npyarrays/imgs_mask_train-%03d.npy" % gi, imgs_mask)
            i = 0
            gi += 1
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images, {2} vols'.format(gi*maxvol + i, total, gi))
        i += 1
    print('Loading done.')

def find_max_vol_id():
    maxid = 0
    for f in os.listdir("/data/octirfseg/npyarrays/"):
        if "train-" in f:
            volid = int(f.split("-")[-1].split(".")[0])
            if maxid < volid:
                maxid = volid
    return maxid

def generate_arrays_from_file(mean,std,batchsize):
    maxvolid = find_max_vol_id()
    while True:
        vols = range(0, maxvolid)
        random.shuffle(vols)
        for volid in vols:
            imgs_train, imgs_mask_train = load_train_data(volid)
            imgs_train -= mean
            imgs_train /= std
            for i in range(0, imgs_train.shape[0], batchsize):
                batch_x = imgs_train[i:i+batchsize]
                batch_y = imgs_mask_train[i:i+batchsize]
                batch_x = np.reshape(batch_x, (batch_x.shape[0], batch_x.shape[2], batch_x.shape[3], 1))
                batch_y = np.reshape(batch_y, (batch_y.shape[0], batch_y.shape[2], batch_y.shape[3], 1))
                yield (batch_x, batch_y)

def load_train_data(vol):
    imgs_train = np.load("/data/octirfseg/npyarrays/imgs_train-%03d.npy" % vol)
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = np.load("/data/octirfseg/npyarrays/imgs_mask_train-%03d.npy" % vol)
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    return imgs_train, imgs_mask_train


if __name__ == '__main__':
    create_valid_data()
    create_train_data()
