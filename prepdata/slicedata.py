#!/usr/bin/env python

import glob, random, os, tqdm
from PIL import Image
import numpy as np

def slice(inf, df):
	corph = 432
	slicewidth = 32
	normalprob = 0.4
	counts = {"train":{"feat":0, "norm":0}, "valid": {"feat":0, "norm":0}}

	ptgrp = {}

	os.system("rm -rf prepdata/datacrop ; mkdir -p prepdata/datacrop/train prepdata/datacrop/valid")

	allfiles = []
	with open("runs/ptgrp.csv", "w") as fout:
	  fout.write("ptid,group\n")
	  with open(inf) as fin:
	    first = True
	    for l in fin:
	      if first:
		first = False
		continue
	      arr = l.rstrip().split(",")
	      if not arr[0] in ptgrp:
		grp = "train"
		if random.random() < 0.4:
		  grp = "valid"
		ptgrp[arr[0]] = grp
		fout.write("%s,%s\n" % (arr[0], grp))
	      arr.append(ptgrp[arr[0]])
	      allfiles.append(arr)

	alli = 0
	for (pt, orif, maskf, grp) in tqdm.tqdm(allfiles):
	  stride = 8
	  if grp == "valid":
	    stride = slicewidth
	  ori = np.array(Image.open(df + "/" + orif))
	  mask = np.array(Image.open(df + "/" + maskf))
	  if len(ori.shape) == 3:
	    ori = ori[:,:,0]
	  if len(mask.shape) == 3:
	    mask = mask[:,:,0]
	  ym = np.argmax(np.sum(ori, axis=1))
	  y0 = int(ym - corph / 2)
	  y1 = int(ym + corph / 2)
	  if y0 < 0:
	    y0 = 0
	  if y1 >= ori.shape[0]:
	    y1 = ori.shape[0] - 1
	  for xi in range(0, ori.shape[1]-slicewidth, stride):
	    sliori = np.zeros((corph, slicewidth), dtype="uint8")
	    slimask = np.zeros((corph, slicewidth), dtype="uint8")
	    sliori[0:y1-y0, :]  = ori[y0:y1, xi:xi+slicewidth]
	    slimask[0:y1-y0, :] = 255 * mask[y0:y1, xi:xi+slicewidth]
	    found = np.sum(slimask > 0)
	    if found > 0 or random.random() < normalprob:
	      g2 = "norm"
	      if found > 0:
		g2 = "feat"
	      Image.fromarray(sliori).save("prepdata/datacrop/%s/%09d-%s" % (grp, alli, orif))
	      Image.fromarray(slimask).save("prepdata/datacrop/%s/%09d-%s" % (grp, alli, maskf))
	      alli += 1
	      counts[grp][g2] += 1

	print(counts)
