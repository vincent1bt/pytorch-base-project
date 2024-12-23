# from pyparsing.helpers import List, Tuple, Dict
from dataclasses import dataclass, field
from typing import Dict

labels = {
    "n01440764": "Tench",
    "n02102040": "English Springer",
    "n02979186": "Cassette Player",
    "n03000684": "Chain Saw",
    "n03028079": "Church",
    "n03394916": "French Horn",
    "n03417042": "Garbage Truck",
    "n03425413": "Gas Pump",
    "n03445777": "Golf Ball",
    "n03888257": "Parachute",
}

@dataclass
class data_config():
  IMAGE_WIDTH: int = 224
  IMAGE_HEIGHT: int = 224
  IMAGE_CHANNELS: int = 3
  NUM_CLASSES: int = 10
  NOISE_LEVEL: str = "noisy_labels_5"
  TRAIN_FOLDER_PATH: str = "imagenette2-320"
  TRAIN_INFORMATION_FILE: str = "imagenette2-320/noisy_imagenette.csv"
  LABELS = labels

@dataclass
class train_config():
  BATCH_SIZE: int = 64
  LEARNING_RATE: float = 1.5e-3
  SHUFFLE: bool = True
  BETA1: float = 0.9
  BETA2: float = 0.98
  EPSILON: float = 1e-6
  WEIGHT_DECAY = 0.01
  PROJECT_NAME: str = "Base CNN Train"

@dataclass
class model_config():
  DROPOUT: float = 0.2
  NUM_LAYERS: int = 5
  INITIAL_CHANNELS: int = 32
  FEAT_SIZE: int = data_config.IMAGE_WIDTH

