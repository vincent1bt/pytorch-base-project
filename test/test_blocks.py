import torch
import unittest
import builtins
from unittest import TestCase

import re

from model.blocks import ResBlock, ResPoolBlock

using_notebook = getattr(builtins, "__IPYTHON__", False)

class ResBlockTests(TestCase):
    @classmethod
    def setUpClass(cls): # runs once before all the tests
        cls.batch_size = 2
        cls.channels = 32
        cls.feat_size = 64 # width and height

        cls.shape = (
            cls.batch_size, cls.channels, cls.feat_size, cls.feat_size
        )

        cls.inputs = torch.randn(
            cls.shape
        )
        cls.res_block = ResBlock(
            channels=cls.channels,
            feat_size=cls.feat_size
        )

    def escape_regex(self, s):
      return re.escape(str(s))

    def test_same_output_shape(self):
        outputs = self.res_block(self.inputs)
        self.assertEqual(
            outputs.shape,
            self.shape
        )

    def test_raises_diff_channels(self):
        inputs = torch.randint(
            0, 1,
            (self.batch_size, self.channels * 2, self.feat_size, self.feat_size)
        )

        message_pattern = (
            rf"expected input\[{self.escape_regex(self.batch_size)},\s*"
            rf"{self.escape_regex(self.channels * 2)},\s*"
            rf"{self.escape_regex(self.feat_size)},\s*"
            rf"{self.escape_regex(self.feat_size)}\]\s*"
            rf"to have {self.escape_regex(self.channels)} channels,\s*"
            rf"but got {self.escape_regex(self.channels * 2)} channels instead"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            message_pattern
        ):
            _ = self.res_block(inputs)

class ResPoolBlockTests(TestCase):
    @classmethod
    def setUpClass(cls): # runs once before all the tests
        cls.batch_size = 2
        cls.in_channels = 32
        cls.out_channels = 64
        cls.feat_size = 64 # width and height

        cls.shape = (cls.batch_size, cls.in_channels, cls.feat_size, cls.feat_size)

        cls.inputs = torch.randn(
            cls.shape
        )
        cls.res_block = ResPoolBlock(
            in_channels=cls.in_channels,
            out_channels=cls.out_channels,
            feat_size=cls.feat_size // 2
        )

    def test_correct_output_shape(self):
        outputs = self.res_block(self.inputs)
        self.assertEqual(
            outputs.shape,
            (self.batch_size, self.out_channels, self.feat_size // 2, self.feat_size // 2)
        )

if __name__ == "__main__" and not using_notebook:
  unittest.main()

