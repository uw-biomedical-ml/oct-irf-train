#!/usr/bin/env python

import sys, json, os, subprocess

def usage():
	print("Usage: ./run.py <image folder path> <CSV file>")
	print("\n\tCSV file should have three columns in the following order:")
	print("\t\tPtID - should be coded")
	print("\t\tOriginal - Original B scan image")
	print("\t\tMask - Binary segmentation mask where features of interest are set to 255 and everything else is 0")
	print("\n\tExample:")
	print("""ptid,original,mask
038883,038883-0.png,038883-0_mask.png
027134,027134-0.png,027134-0_mask.png
029719,029719-0.png,029719-0_mask.png
053971,053971-0.png,053971-0_mask.png
029058,029058-0.png,029058-0_mask.png""")

opts = {}

print("="*30)
print("Checking inputs")

if len(sys.argv) != 3:
	usage()
	sys.exit(-1)

(_, pdir, csvfile) = sys.argv
if not os.path.isdir(pdir):
	usage()
	print("%s is not a directory" % pdir)
	sys.exit(-1)

if not os.path.isfile(csvfile):
	usage()
	print("%s is not a file" % csvfile)
	sys.exit(-1)

with open(csvfile) as fin:
  first = True
  for l in fin:
    if first:
      first = False
      continue
    arr = l.strip().split(",")
    if len(arr) != 3:
      usage()
      print("Wrong number of columns in your CSV file")
      sys.exit(-1)
    for i in range(1,3):
      if not os.path.isfile("%s/%s" % (pdir, arr[i])):
        print("%s/%s is not a file!" % (pdir, arr[i]))
	sys.exit(-1)
    
opts["csv"] = csvfile
opts["dir"] = pdir
print("\tVerified that all images exist and that CSV layout looks reasonable")

print("="*30)
print("Verifying environment")
#os.system("pip install --user -r requirements.txt")

try:
	import pkg_resources
except ImportError:
	print("Please install pkg_resources. Ubuntu: sudo apt-get install python-setuptools")

try:
	import keras
except ImportError:
	print("Please install keras: pip install keras=='2.2.0'")
	sys.exit(-1)

if keras.__version__ != '2.2.0':
	print("Wrong keras version. Expected %s, but %s is installed." % ("2.2.0", keras.__version__))
	sys.exit(-1)

print("\tkeras: %s" % keras.__version__)

try:
	import tensorflow
except ImportError:
	print("Please install tensorflow: pip install tensorflow-gpu=='1.8.0'")
	sys.exit(-1)

if tensorflow.__version__ != '1.8.0':
	print("Wrong tensorflow version. Expected %s, but %s is installed." % ("1.8.0", tensorflow.__version__))
	sys.exit(-1)


l = next(str(i) for i in pkg_resources.working_set if 'tensorflow-gpu' in str(i))
if l == "" or l == None:
  print("\tWARNING Tensorflow-gpu is not installed. Will be CPU accelerated only")

print("\ttensorflow: %s" % tensorflow.__version__)
print("\t%s" % l)

try:
	import numpy as np
except ImportError:
	print("Please install numpy: pip install numpy")
	sys.exit(-1)


print("\tnumpy: %s" % np.__version__)

try:
	import PIL
except ImportError:
	print("Please install pillow: pip install pillow")
	sys.exit(-1)

print("\tpillow: %s" % PIL.__version__)

try:
	import GPUtil
except ImportError:
	print("Please install gputil: pip install gputil")
	sys.exit(-1)
print("\tgputil: %s" % GPUtil.__version__)

try:
	import tqdm
except ImportError:
	print("Please install tqdm. pip install tqdm")
	sys.exit(-1)


try:
	import altair
except ImportError:
	print("Please install altair. pip install altair")
	sys.exit(-1)

print("\n\tVerified environment")

print("="*30)
print("Checking GPU cards, will use first GPU")

gpus = GPUtil.getGPUs()
for i, d in enumerate(gpus):
	print("\tid: %r" % d.id)
	print("\tmemoryFree: %r" % d.memoryFree)
	print("\tmemoryTotal: %r" % d.memoryTotal)
	print("\tdriver: %r" % d.driver)
	print("\tname: %r" % d.name)
	if i == 0:
		opts["memtot"] =  d.memoryTotal
		opts["memfree"] =  d.memoryFree


print("Saving options. Ready to preprocess data")
os.system("rm -rf runs/ ; mkdir runs")
json.dump(opts, open("runs/options.json", "w"))

print("="*30)
print("Starting slicing of data and packing")

import prepdata.slicedata, prepdata.packdata
prepdata.slicedata.slice(csvfile, pdir)
prepdata.packdata.packdata(csvfile)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from deeplearning.train import train
train(30)
from deeplearning.render import render
render("runs/history.txt", "runs/learning.html")
