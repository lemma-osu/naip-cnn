from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def build_cnn(
    kernel_dim=5,
    filter_no=32,
    Dense1_no=256,
    Dense2_no=32,
    learn_rate=0.001,
    shape: tuple[int, int, int] = (30, 30, 4),
):
    kernel_size = (kernel_dim, kernel_dim, 4)

    model = tf.keras.models.Sequential(
        [
            Conv3D(
                filters=filter_no,
                kernel_size=kernel_size,
                input_shape=(*shape, 1),
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            Flatten(),
            Dense(Dense1_no, activation="relu"),
            Dropout(0.4),
            Dense(Dense2_no, activation="relu"),
            Dropout(0.4),
            Dense(units=1, activation="linear"),
        ]
    )

    optimizer = Adam(learning_rate=learn_rate)

    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model
