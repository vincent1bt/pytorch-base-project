from model.model import CNNet
from utils.train import obtain_device
from hparameters import data_config

import torch
import unittest
import builtins
from unittest import TestCase

using_notebook = getattr(builtins, "__IPYTHON__", False)

class ModelTests(TestCase):
    @classmethod
    def setUpClass(cls): # runs once before all the tests
        cls.batch_size = 2
        cls.channels = 64
        cls.feat_size = 224 # width and height
        cls.num_layers = 4
        cls.image_channels = 3

        device = obtain_device()

        cls.output_shape = (cls.batch_size, data_config.NUM_CLASSES)

        cls.shape = (
            cls.batch_size, 
            cls.image_channels, 
            cls.feat_size, cls.feat_size
        )

        cls.inputs = torch.randn(
            cls.shape
        ).to(device)

        cls.model = CNNet(
            num_layers=cls.num_layers,
            initial_channels=cls.channels,
            feat_size=cls.feat_size
        ).to(device)

    def test_model_output_shape(self):
        outputs = self.model(self.inputs)
        self.assertEqual(
            outputs.shape,
            self.output_shape
        )
    @unittest.skip("To do")
    def test_eval_mode(self):
        self.model.eval()
        # You would need a way to check if dropout is not applied.
        # This could involve comparing outputs or checking the state of the dropout layer.
        # For this example, we'll just check if the model runs.
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, self.output_shape)

    @unittest.skip("To do")
    def test_train_mode(self):
        self.model.train()
        # Again, asserting the effect of dropout can be complex.
        # A simple test is to ensure it runs without error.
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, self.output_shape)

if __name__ == "__main__" and not using_notebook:
  unittest.main()

