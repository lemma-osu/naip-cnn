import h5py
import numpy as np


def count_duplicate_images(train_path, val_path, key: str = "image") -> int:
    """
    Count the number of duplicate images between a train and validation HDF dataset.

    This is useful for ensuring that data leakage didn't occur during data splitting.

    References
    ----------
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    with h5py.File(val_path, "r") as f:
        val_image = f[key][:]
    with h5py.File(train_path, "r") as f:
        train_image = f[key][:]

    # Flatten the arrays
    val_image = val_image.reshape(val_image.shape[0], -1)
    train_image = train_image.reshape(train_image.shape[0], -1)

    val_dtype = {
        "names": [f"f{i}" for i in range(val_image.shape[1])],
        "formats": val_image.shape[1] * [val_image.dtype],
    }
    train_dtype = {
        "names": [f"f{i}" for i in range(train_image.shape[1])],
        "formats": train_image.shape[1] * [train_image.dtype],
    }

    return np.intersect1d(val_image.view(val_dtype), train_image.view(train_dtype))
