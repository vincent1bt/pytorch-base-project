import torch
import unittest
import builtins
from unittest import TestCase
from unittest.mock import Mock

from utils.train import EpochTrainer, BatchTrainer, obtain_device
from utils.test import MetricsManager
from hparameters import model_config, train_config, data_config
from utils.train import create_optimizer, create_model, get_loss_function

using_notebook = getattr(builtins, "__IPYTHON__", False)

class FakeTestDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            num_samples=64,
        ):
        self.num_samples = num_samples
        self.image_paths = [f"fake/path/image_{i}.JPEG" for i in range(num_samples)]
        self.labels = torch.randint(0, data_config.NUM_CLASSES, (num_samples,)).tolist()

    def __getitem__(self, index):
        return torch.randn(
            data_config.IMAGE_CHANNELS,
            data_config.IMAGE_WIDTH,
            data_config.IMAGE_HEIGHT
        ), self.labels[index], self.image_paths[index]

    def __len__(self):
        return self.num_samples

class TrainTest(TestCase):
    @classmethod
    def setUpClass(cls):
        device = obtain_device()

        model = create_model(model_config)
        model = model.to(device)

        optimizer = create_optimizer(model, train_config)
        loss_fn = get_loss_function()

        cls.batch_trainer = BatchTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )

        dataset = FakeTestDataset(num_samples=64)
        batch_size = 8

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )

        train_metrics = MetricsManager()
        train_metrics.to(device)

        cls.mock_checkpoint_manager = Mock()
        cls.mock_wandb_manager = Mock()

        epoch_trainer = EpochTrainer(
            train_loader=train_loader,
            batch_trainer=cls.batch_trainer,
            device=device,
            wandb=cls.mock_wandb_manager,
            ckp_manager=cls.mock_checkpoint_manager,
            metrics=train_metrics,
        )

        cls.metrics = epoch_trainer.step(
            current_epoch=0,
            total_epochs=2,
        )

        cls.output_shape = (
           batch_size, 
           data_config.NUM_CLASSES
        )

        cls.inputs = torch.randn(
            2, 
            data_config.IMAGE_CHANNELS, 
            data_config.IMAGE_WIDTH, data_config.IMAGE_HEIGHT
        ).to(device)

        cls.targets = torch.rand(
            2, data_config.NUM_CLASSES
        ).to(device)

    def test_batch_trainer(self):
        predictions, loss = self.batch_trainer.step(
            self.inputs, self.targets
        )

        self.assertIsInstance(loss, float)
        self.assertEqual(
            predictions.shape,
            self.targets.shape
        )

    def test_epoch_trainer(self):
        for _, metric in self.metrics.log_metrics:
            with self.subTest(metric=metric):
                self.assertIsInstance(metric.item(), float)

        for _, metric in self.metrics.graph_metrics:
            with self.subTest(metric=metric):
                self.assertIsInstance(metric[0], list)
                self.assertIsInstance(metric[1], list)

                self.assertEqual(len(metric[0]), data_config.NUM_CLASSES)
                self.assertEqual(len(metric[1]), data_config.NUM_CLASSES)
    
    def test_weights_updates(self):
        # first Conv2d layer
        initial_weights = self.batch_trainer.model.get_submodule("model.0").weight.clone()

        self.batch_trainer.step(
            self.inputs,
            self.targets
        )

        updated_weights = self.batch_trainer.model.get_submodule("model.0").weight

        self.assertFalse(torch.equal(initial_weights, updated_weights))
    
    def test_metrics_manager(self):
        metrics = MetricsManager(
            num_classes=4
        )
        
        predictions = torch.tensor([
            [1.0, 0.1, 0.1, 0.1], # Argmax is 0
            [0.1, 1.0, 0.1, 0.1], # Argmax is 1
            [0.1, 0.1, 1.0, 0.1], # Argmax is 2
            [0.1, 0.1, 0.1, 1.0], # Argmax is 3
        ])

        targets = torch.tensor([0, 1, 2, 3])
        metrics.update_metrics(predictions, targets)

        for _, metric in metrics.log_metrics:
            with self.subTest(metric=metric):
                self.assertEqual(metric.item(), 1.0)
    
    def test_checkpoint_being_called(self):
        self.mock_checkpoint_manager.save.assert_called()

    def test_wandlog_being_called(self):
        self.mock_wandb_manager.log.assert_called()

if __name__ == "__main__" and not using_notebook:
  unittest.main()

