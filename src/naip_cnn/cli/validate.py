from naip_cnn.utils.validation import count_duplicate_images


def validate_data_split(dataset_name: str) -> None:
    """Check a split set of data for duplicate images."""
    train_path = f"./data/training/{dataset_name}_train.h5"
    val_path = f"./data/training/{dataset_name}_val.h5"
    test_path = f"./data/training/{dataset_name}_test.h5"

    print("Checking for duplicate images in the validation set...")
    val_duplicates = count_duplicate_images(train_path, val_path)
    if val_duplicates:
        msg = f"Found {val_duplicates} duplicate images in the validation set!"
        raise ValueError(msg)

    print("Checking for duplicate images in the test set...")
    test_duplicates = count_duplicate_images(train_path, test_path)
    if test_duplicates:
        msg = f"Found {test_duplicates} duplicate images in the test set!"
        raise ValueError(msg)

    print("No duplicate images found!")
