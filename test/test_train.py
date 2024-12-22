import torch
import unittest
import builtins
import logging
from unittest import TestCase
import pandas as pd

from utils.train import EpochTrainer, StepTrainer
from utils.manage import DecoyLogger, MetricsManager
from hparameters import model_config, train_config, data_config
from data.loader import ImageDataset, train_transforms
from utils.train import create_optimizer, create_model, get_loss_function

using_notebook = getattr(builtins, "__IPYTHON__", False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# change for a test dataset
df = pd.read_csv(data_config.TRAIN_INFORMATION_FILE)

train_set = df[df["is_valid"] == False]

train_img_paths = train_set["path"][10:31].values
train_labels = train_set[data_config.NOISE_LEVEL][10:31].values

class TrainTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.folder_path = data_config.TRAIN_FOLDER_PATH

        if torch.cuda.is_available():
           cls.device = 'cuda'
        elif torch.backends.mps.is_available():
           cls.device = 'mps'
        else:
           cls.device = 'cpu'

        logger.info(f' Using device: {cls.device}')

        model = create_model(model_config)

        logger.info(f' Using model config: {model_config().__dict__}')
        
        model = model.to(cls.device)

        optimizer = create_optimizer(model, train_config)
        loss_fn = get_loss_function()

        cls.step_trainer = StepTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )

        decoy_logger = DecoyLogger()

        dataset = ImageDataset(
            image_paths=train_img_paths,
            image_labels=train_labels,
            folder_path=cls.folder_path,
            transforms=train_transforms
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cls.batch_size,
            shuffle=False,
            drop_last=True
        )

        train_metrics = MetricsManager()
        train_metrics.to(cls.device)

        cls.epoch_trainer = EpochTrainer(
            train_loader=train_loader,
            trainer=cls.step_trainer,
            device=cls.device,
            wandb=decoy_logger,
            ckp_manager=None,
            metrics=train_metrics,
        )

        small_batch_size = 4

        cls.output_shape = (small_batch_size, data_config.NUM_CLASSES)

        cls.inputs = torch.randn(
            small_batch_size, 3, 224, 224
        )

        cls.targets = torch.rand(
            small_batch_size
        )

    def test_step_trainer(self):
      inputs, targets = self.inputs.to(self.device), self.targets.to(self.device)
      predictions, loss = self.step_trainer.train_step(inputs, targets)

      self.assertIsInstance(loss, float)
      self.assertEqual(predictions.shape, self.output_shape)

    def test_epoch_trainer(self):
      rec, acc, prec = self.epoch_trainer.train(
        epoch=0,
        epochs=2,
        display=False
      )

      self.assertIsInstance(rec, float)
      self.assertIsInstance(acc, float)
      self.assertIsInstance(prec, float)
    #   self.assertIsInstance(precision_curve, list)
    #   self.assertIsInstance(recall_curve, list)

if __name__ == "__main__" and not using_notebook:
  unittest.main()

