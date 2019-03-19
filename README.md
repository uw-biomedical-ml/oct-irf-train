# oct-irf-train

Retrain our published IRF segmentation algorithm with your own data. 

# Installation and setup

```
git clone https://github.com/uw-biomedical-ml/oct-irf-train.git

./run.py <image folder> <csv file>
```

For example if you added a folder called "data" inside the cloned project folder, and a file called segmentations.csv which should look like:

```
ptid,original,mask
038883,038883-0.png,038883-0_mask.png
027134,027134-0.png,027134-0_mask.png
029719,029719-0.png,029719-0_mask.png
053971,053971-0.png,053971-0_mask.png
029058,029058-0.png,029058-0_mask.png
```

Then you would run the command `./run.py data segmentations.csv`

This will check your environment and install any packages that are missing. It will also setup the data files into two sets of images: 80% for training and 20% for validation. The files will be packaged and then stored as Numpy arrays inside `prepdata/nyparrays`. Training will beginning using transfer learning and process for 10 epochs. 

The format of the images are expected to be as follows:
- original OCT B scans = 8-bit grayscale images, ideally PNG.
- segmentation masks = 8-bit grayscale images with 255 for the feature of interest and 0 for the background. 

# Output

After training is finished, there will be a folder called `run` inside the project folder. This folder will also contain the following files:

- learning.html = This is a HTML file that you should open using a web browser (ideally Chrome) which will show the learning curves.
- history.txt = This is a text file for the training and validation losses. This is used to generate the learning curves.
- options.json = The options that were set when this run was initiated.
- ptgrps.csv = This is how the patients were split into training and validation.
- weights.hdf5 = This is the weights file for the U-net with the best validation loss during training. 

# Running your own test set to evaluate the new retrained model

This repositiory also contains an evaluation program which can be run using the following syntax:

```
./evaluate.py <img folder> <out folder> <mode>
```

- img folder = this is where the new images from the test set should be placed. The script will batch process all the images in this folder.
- out folder = this folder is where to save the resulting images. This folder must be an *empty folder* and not contain any files inside of it.
- mode = This should be one of three options:
  - mask = only output the binary mask. This is useful for calculating Dice and IOU metrics.
  - blend = Alpha-blended mask on top of the original OCT B scan image
  - mask_blend = *Recommended* Using a cutoff of 0.5, a binary mask will be applied and the resulting probabilities will be shown as a color graident overlayed on top of the original B-scan.
  
