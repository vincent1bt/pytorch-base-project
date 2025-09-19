import pandas as pd
from hparameters import data_config

df = pd.read_csv(data_config.TRAIN_INFORMATION_FILE)

def obtain_train_dataset():
    train_set = df[df["is_valid"] == False]
    val_set = df[df["is_valid"] == True]

    train_img_paths = train_set["path"].values
    train_labels = train_set[data_config.NOISE_LEVEL].values

    val_img_paths = val_set["path"].values
    val_labels = val_set["noisy_labels_0"].values

    return (train_img_paths, train_labels), (val_img_paths, val_labels)

def obtain_dry_dataset():
    labels_name = 'noisy_labels_0'
    img_per_class = 1

    dry_set = pd.concat(
        [df[df[labels_name] == label][:img_per_class] for label in data_config.LABELS.keys()]
    )

    dry_img_paths = dry_set['path'].values
    dry_labels = dry_set[labels_name].values

    return (dry_img_paths, dry_labels), (dry_img_paths, dry_labels)