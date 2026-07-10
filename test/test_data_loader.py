import unittest
import builtins

import tempfile
import shutil
import os
import random

from PIL import Image
import pandas as pd
import torch

from data.loader import ImageDataset, train_transforms
from hparameters import data_config

using_notebook = getattr(builtins, "__IPYTHON__", False)

class TestDataLoader(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # create a dummy dataset
    test_dir = tempfile.mkdtemp()
    cls.folder_path = test_dir
    image_paths = []
    labels = []

    image_shape = (
      data_config.IMAGE_WIDTH,
      data_config.IMAGE_HEIGHT
    )

    cls.image_to_generate = 15
    cls.batch_size = 5

    for i in range(cls.image_to_generate):
      img_path = f"test_image_{i}.JPEG"
      Image.new('RGB', image_shape, color='red').save(
        os.path.join(cls.folder_path, img_path)
      )

      image_paths.append(img_path)
      labels.append(
        random.choice(
          list(data_config.LABELS.keys())
        )
      )

    cls.dataset = ImageDataset(
        image_paths=image_paths,
        image_labels=labels,
        folder_path=cls.folder_path,
        transforms=train_transforms
    )
  
  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.folder_path)

  def test_output_dtype(self):
    image, label, _ = self.dataset[0]
    self.assertIsInstance(image, torch.Tensor)
    self.assertIsInstance(label, int)
    self.assertEqual(image.dtype, torch.float32)

  def test_output_shape(self):
    image, label, _ = self.dataset[0]
    self.assertEqual(
        image.shape,
        (
          data_config.IMAGE_CHANNELS, 
          data_config.IMAGE_WIDTH, 
          data_config.IMAGE_HEIGHT
        )
    )

  def test_output_min_max_range(self):
    image, _, _ = self.dataset[0]
    self.assertGreaterEqual(image.min(), 0)
    self.assertLessEqual(image.max(), 1)
  
  def test_dataset_len(self):
    self.assertEqual(self.image_to_generate, len(self.dataset))
  
  def test_dataloader_wrapper(self):
    data_loader = torch.utils.data.DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      shuffle=False
    )

    try:
      image, labels, _ = next(iter(data_loader))
      self.assertEqual(
        image.shape,
        (
          self.batch_size,
          data_config.IMAGE_CHANNELS, 
          data_config.IMAGE_WIDTH,
          data_config.IMAGE_HEIGHT
        )
      )
      self.assertEqual(
        labels.numel(),
        self.batch_size
      )
    except Exception as error:
      print(f"DataLoader test found error: {error}")
  

if __name__ == "__main__" and not using_notebook:
  unittest.main()

