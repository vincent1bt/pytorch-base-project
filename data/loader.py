import torch
from torchvision.io import decode_image
from torchvision.transforms import v2
from pathlib import Path

from hparameters import data_config

labels = data_config.LABELS
label_mapping = dict(zip(labels.keys(), list(range(len(labels)))))
text_mapping = dict(zip(list(range(len(labels))), labels.values()))

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_paths,
            image_labels,
            folder_path,
            transforms
        ):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.folder_path = folder_path
        self.transforms = transforms
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = decode_image(str(Path(self.folder_path) / self.image_paths[idx]), mode='RGB') # Tensor uint8 0,255
        image = self.transforms(image) # Tensor float 0,1

        label = self.label_mapping[self.image_labels[idx]]

        return image, label, self.image_paths[idx]

train_transforms = v2.Compose([
    v2.RandomResizedCrop(
        size=(data_config.IMAGE_WIDTH, data_config.IMAGE_HEIGHT), antialias=True
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

val_transforms = v2.Compose([
    v2.RandomResizedCrop(
        size=(data_config.IMAGE_WIDTH, data_config.IMAGE_HEIGHT), antialias=True
    ),
    v2.ToDtype(torch.float32, scale=True),
])