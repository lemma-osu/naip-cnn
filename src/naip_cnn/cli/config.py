import tensorflow as tf

from naip_cnn import models

# Data parameters
DATASET_NAME = "MAL2016_CanyonCreek-1-30-30x30-30"
BANDS = ("R", "G", "B", "N")
VEG_INDICES = tuple()
LABEL = "cover"
AUGMENT = None

# Training parameters
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 30

# Model parameters
LEARN_RATE = 0.0001
MODEL_CLS = models.CNN_resized
MODEL_PARAMS = dict(
    encoder_blocks=5,
    initial_filters=16,
    resize_shape=(128, 128),
    convolutions_per_block=2,
    filter_size=(3, 3),
    pool_size=(2, 2),
    dropout=0.3,
    regularization=tf.keras.regularizers.l2(0.01),
)

# Derived parameters
ALL_BANDS = BANDS + VEG_INDICES
