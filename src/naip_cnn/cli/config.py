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
    encoder_blocks=4,
    initial_filters=16,
    resize_shape=(64, 64),
    convolutions_per_block=2,
    filter_size=(3, 3),
    pool_size=(2, 2),
    dropout=0.3,
    dilation_rate=(1, 1),
    regularization=tf.keras.regularizers.l2(),
    activation="relu",
)

# Derived parameters
ALL_BANDS = BANDS + VEG_INDICES
