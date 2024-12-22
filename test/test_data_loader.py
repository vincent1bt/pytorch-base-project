import unittest
import builtins
import pandas as pd
import torch

from data.loader import ImageDataset, train_transforms
from hparameters import data_config

using_notebook = getattr(builtins, "__IPYTHON__", False)

# change for a test dataset
df = pd.read_csv(data_config.TRAIN_INFORMATION_FILE)

train_set = df[df["is_valid"] == False]

train_img_paths = train_set["path"].values
train_labels = train_set[data_config.NOISE_LEVEL].values

class TestDataLoader(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.folder_path = data_config.TRAIN_FOLDER_PATH

    cls.dataset = ImageDataset(
        image_paths=train_img_paths,
        image_labels=train_labels,
        folder_path=cls.folder_path,
        transforms=train_transforms
    )

  def test_output_dtype(self):
    image, label, _ = self.dataset[0]
    self.assertIsInstance(image, torch.Tensor)
    self.assertIsInstance(label, int)
    self.assertEqual(image.dtype, torch.float32)

  def test_output_shape(self):
    image, label, _ = self.dataset[0]
    self.assertEqual(
        image.shape,
        (data_config.IMAGE_CHANNELS, data_config.IMAGE_WIDTH, data_config.IMAGE_HEIGHT)
    )

  def test_output_min_max_range(self):
    image, _, _ = self.dataset[0]
    self.assertGreaterEqual(image.min(), 0)
    self.assertLessEqual(image.max(), 1)

if __name__ == "__main__" and not using_notebook:
  unittest.main()

