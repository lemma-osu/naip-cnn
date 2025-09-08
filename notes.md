# Notes

## Datasets 

### NAIP Acquisitions

Because NAIP was flown north-south in the Malheur, there are lots of artifacts between columns of imagery. Training with LiDAR acquisitions that don't encompass that east-west variety (e.g. Crow 2017) seem to generalize worse than LiDAR acquisitions that do (e.g. Canyon Creek 2016), even if the model performance looks better.

### Train-Validation-Test Splitting

Just do it once at the beginning and write to separate files. It's too easy to accidentally leak data when you do it on-the-fly.

### Caching

Much faster training, but obviously requires more RAM. Same large dataset with and without caching of the training set: 5.6m/epoch vs. 11.8m/epoch.

## Model Parameters

### Resizing

I found that models using 50x50 NAIP chips (1m resolution resampled to 60cm) performed better than the native 30x30 NAIP chips (1m resolution), so I added a resizing layer to the model. 64x64 seems to consistently improve performance. 128x128 also worked well, but took longer to train.

### Filter Size

So far most of my tests have used 3x3 filters. [One run](https://wandb.ai/aazuspan-team/naip-cnn/runs/x7prngt6/overview?nw=nwuseraazuspan) with a 5x5 filter did marginally worse on canopy cover, but not enough data to make any conclusions. Filter size will also interact with the resizing layer.

### Atrous Convolution

My [one dilated kernel run](https://wandb.ai/aazuspan-team/naip-cnn/runs/tl8w2480?nw=nwuseraazuspan) did marginally worse than other comparable cover runs.

### Convolutions Per Block

[1 convolution per block](https://wandb.ai/aazuspan-team/naip-cnn/runs/y4wablxq?nw=nwuseraazuspan) seemed to perform pretty poorly. [3 convolutions per block](https://wandb.ai/aazuspan-team/naip-cnn/runs/awjywygz/workspace?nw=nwuseraazuspan) performs marginally better than 2 convolutions per block with RH95.

### Learning Rate

#### Fixed

0.001 seems to be too high, so I've used 0.0001 for most runs.

#### Reduce on Plateau

TODO

### Activation

#### Leaky ReLU vs. ReLU

My [Leaky ReLU run](https://wandb.ai/aazuspan-team/naip-cnn/runs/jrqf2t29/overview?nw=nwuseraazuspan) had slightly better RH95 MAE than a comparable [ReLU run](https://wandb.ai/aazuspan-team/naip-cnn/runs/muyp1bbr/overview?nw=nwuseraazuspan), but the histogram showed that it did worse at predicting near-zeros.

### Kernel Regularization

L2 regularization seemed to help substantially, while L1+L2 seemed to be overkill.

## Pre-processing

### Normalization

I've only tried normalizing 0-1 so far.

### Augmentation

I started out applying lots of augmentation (flip, contrast, brightness), but found that models improved when I removed it. Need to test more.

### Vegetation Indices

Including vegetation indices in addition to or instead of spectral bands has no effect at best, and hurts performance at worst. It's possible this is a normalization issue, since I haven't rescaled the indices.

Visually, the hypothesis that band ratios would be more invariant to sensor variability doesn't seem to hold up