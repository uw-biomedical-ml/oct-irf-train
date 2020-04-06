#!/usr/bin/env python
from PIL import Image
import sys, glob, tqdm, os
import numpy as np
from colour import Color

def usage():
    print("./evaluate.py <imgdir> <outdir> <mode>")
    print("")
    print("\timgdir = folder of OCT B scans")
    print("\toutdir = EMPTY folder to output segmentation masks")
    print("\tmode = mask, mask_blend, blend")

    sys.exit(-1)

if len(sys.argv) != 4:
    usage()

import deeplearning.unet
(_, indir, outdir, mode) = sys.argv

if not os.path.isdir(indir):
    print("ERROR: %s is not a directory" % indir)
    sys.exit(-1)

if not os.path.isdir(outdir):
    print("ERROR: %s is not a directory" % outdir)
    sys.exit(-1)

if len(glob.glob("%s/*" % outdir)) != 0:
    print("ERROR: %s is not empty" % outdir)
    sys.exit(-1)

imgs = []
for f in glob.glob("%s/*" % indir):
    (_, ext) = os.path.splitext(f)
    if ext in [".jpg", ".png", ".jpeg"]:
        imgs.append(f)

if len(imgs) == 0:
    print("ERROR: %s has no images!" % indir)
    sys.exit(-1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = deeplearning.unet.get_unet()

model.load_weights("runs/weights.hdf5", by_name=True)
image_rows = 432
image_cols = 32
my_cm = []
colors = list(Color("yellow").range_to(Color("red"),1001))
for c in colors:
	my_cm.append((255 * np.array(c.rgb)).astype(np.uint8))
my_cm = np.array(my_cm)

for f in tqdm.tqdm(imgs):
    ji = Image.open(f)
    img = np.array(ji)
    img = img.astype(np.float)
    img -= 28.991758347
    img /= 46.875888824

    totaloutput = np.zeros((img.shape[0], img.shape[1], 32))
    ym = np.argmax(np.sum(img, axis=1))
    y0 = int(ym - image_rows / 2)
    y1 = int(ym + image_rows / 2)
    if y0 < 0:
      y0 = 0
    if y1 >= img.shape[0]:
      y1 = img.shape[0] - 1
    for dx in tqdm.tqdm(range(0, img.shape[1] - 32)):
	sliori = np.zeros((image_rows, image_cols), dtype=np.float)
	sliori[0:y1-y0, :]  = img[y0:y1, dx:dx+image_cols]
        imgsbatch = sliori.reshape((1,  1, image_rows,image_cols))

        output = model.predict(imgsbatch, batch_size=1)
        totaloutput[y0:y1,dx:dx+image_cols,dx % 32] = output[0,0,0:y1-y0,:]

    totaloutput = np.mean(totaloutput, 2)

    if (mode == "mask"):
        # for binary masks
        mask = (totaloutput > 0.5)
        mask = np.uint8(mask)
        mask *= 255
        mask = Image.fromarray(mask)
        mask.save(f.replace(indir,outdir))
    elif (mode == "mask_blend"):
        # for masked heatmap overlay
        mask = (totaloutput < 0.5)
        mask = np.uint8(mask)
        mask *= 255
        mask = Image.fromarray(mask)
	mapped_data = np.zeros((totaloutput.shape[0], totaloutput.shape[1],3), dtype="uint8")
	totalint = (1000 * totaloutput).astype(np.uint16)
	mapped_data = my_cm[totalint]
        j = Image.fromarray(mapped_data).convert('RGBA')
        ji = ji.convert("RGBA")
        Image.composite(ji, j,mask).save(f.replace(indir,outdir))
    elif (mode == "blend"):
        # for blend overlay
	totalint = (1000 * totaloutput).astype(np.uint16)
	mapped_data = my_cm[totalint]
        j = Image.fromarray(mapped_data).convert('RGBA')
        ji = ji.convert("RGBA")
        Image.blend(ji, j,0.5).save(f.replace(indir,outdir))

print("\n\nFinished.")
