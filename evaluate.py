#!/usr/bin/env python
import deeplearning.unet
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
    img = img.reshape((1, img.shape[0], img.shape[1])).astype(np.float)
    img -= 28.991758347
    img /= 46.875888824

    totaloutput = np.zeros((img.shape[1], img.shape[2], 32))
    for dx in tqdm.tqdm(range(0, img.shape[2] - 32)):
        imgs = img[0, 0:image_rows, dx:image_cols+dx]
        imgs = imgs.reshape((  1, image_rows,image_cols))
        imgsbatch = np.zeros((1, 1, image_rows,image_cols))
        imgsbatch[0] = imgs

        output = model.predict(imgsbatch, batch_size=1) # inference step
        totaloutput[0:image_rows,dx:dx+image_cols,dx % 32] = output[0,0,:,:]

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
