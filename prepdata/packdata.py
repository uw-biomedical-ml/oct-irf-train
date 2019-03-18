#!/usr/bin/python
from __future__ import print_function

import os, sys, random, tqdm
import numpy as np

from PIL import Image

def pack_grp(grp, originals):
    data_path = 'prepdata/datacrop/'
    grp_data_path = os.path.join(data_path, grp)
    images = os.listdir(grp_data_path)
    imgs = []
    imgs_mask = []

    print('-'*30)
    print("Packing %s images..." % grp)
    print('-'*30)
    
    for image_name in tqdm.tqdm(images):
	if not '.png' in image_name:
	    continue
	arrf = image_name.split("-")
	exi = arrf.pop(0)
	rootf = "-".join(arrf)
	if not rootf in originals:
	  continue
	image_name = os.path.join(grp_data_path, image_name)
	image_mask_name = os.path.join(grp_data_path, "%s-%s" % (exi, originals[rootf]))
	img = np.array(Image.open(image_name))
	img_mask = np.array(Image.open(image_mask_name))

	imgs.append(img)
	imgs_mask.append(img_mask)

    print("Reshaping and saving")
    imgs = np.array(imgs).astype(np.uint8)
    imgs = np.reshape(imgs, (imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]))

    imgs_mask = np.array(imgs_mask).astype(np.uint8)
    imgs_mask = np.reshape(imgs_mask, (imgs_mask.shape[0], 1, imgs_mask.shape[1], imgs_mask.shape[2]))

    np.save("prepdata/npyarrays/imgs_%s.npy" % grp, imgs)
    np.save("prepdata/npyarrays/imgs_mask_%s.npy" % grp, imgs_mask)



def packdata(csvfiles):
	if not os.path.exists("prepdata/npyarrays"):
	  os.mkdir("prepdata/npyarrays")



	originals = {}
	with open(csvfiles) as fin:
	  first = True
	  for l in fin:
	    if first:
	      first = False
	      continue
	    arr = l.rstrip().split(",")
	    originals[arr[1]] = arr[2]
	pack_grp("valid", originals)
	pack_grp("train", originals)

