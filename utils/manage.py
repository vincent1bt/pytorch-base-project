import os
import torch
from torchmetrics import Accuracy, Precision, Recall, PrecisionRecallCurve
import pandas as pd

from hparameters import data_config

class MetricsManager(torch.nn.Module):
  def __init__(self):
    super(MetricsManager, self).__init__()

    self.accuracy = Accuracy(
      task="multiclass",
      num_classes=data_config.NUM_CLASSES
    )
    self.precision = Precision(
      task="multiclass",
      num_classes=data_config.NUM_CLASSES,
      average="macro"
    )
    self.recall = Recall(
      task="multiclass",
      num_classes=data_config.NUM_CLASSES,
      average="macro"
    )
    self.precision_recall_curve = PrecisionRecallCurve(
      task="multiclass",
      num_classes=data_config.NUM_CLASSES
    )

    self.metrics = [
      self.accuracy,
      self.precision,
      self.recall,
      self.precision_recall_curve
    ]

  def reset_metrics(self):
    for metric in self.metrics:
      metric.reset()

  def update_metrics(self, predictions, targets):
    for metric in self.metrics:
      metric.update(predictions, targets)

  def compute_metrics(self):
    acc = self.accuracy.compute().item()
    prec = self.precision.compute().item()
    rec = self.recall.compute().item()
    # precision_curve, recall_curve, thresholds = self.precision_recall_curve.compute()

    return rec, acc, prec #, precision_curve, recall_curve
  
class DebugMetricsManager(MetricsManager):
  def __init__(self):
    super().__init__()

  def reset_metrics(self):
    super().reset_metrics()

    self.predictions = []
    self.scores = []
    self.targets = []
    self.img_paths = []

  def update_metrics(self, predictions, targets, img_paths):
    super().update_metrics(predictions, targets)

    predictions = predictions.max(dim=-1).item()
    maxarg = predictions.indices.tolist()
    maxval = predictions.values.tolist()

    self.predictions.extend(maxarg)
    self.scores.extend(maxval)
    self.targets.extend(targets)
    self.img_paths.extend(img_paths)
  
  def compute_metrics(self):
    rec, acc, prec = super().compute_metrics()

    data_frame = pd.DataFrame({
        "scores": self.scores,
        "predictions": self.predictions,
        "targets": self.targets,
        "img_paths": self.img_paths
    })

    return data_frame, rec, acc, prec

class MeanLossMetric(torch.nn.Module):
    def __init__(self):
        super(MeanLossMetric, self).__init__()
        self.loss_sum = torch.tensor(0.0)
        self.count = 0

    def forward(self, loss):
        self.loss_sum += loss
        self.count += 1
        return loss

    def reset(self):
        self.loss_sum = torch.tensor(0.0)
        self.count = 0

    @property
    def mean(self):
        if self.count == 0:
            return 0.0
        return self.loss_sum / self.count

class CheckpointManager():
    def __init__(
            self,
            optimizer,
            model,
            folder="training_checkpoints"
        ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.folder = folder

    def save(self, epoch, batch_step):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'batch_step': batch_step,
        }

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        filename = f"{self.folder}/checkpoint-{epoch}-{batch_step}.pt"
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(f"{self.folder}/{filename}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded from epoch {checkpoint.get('epoch')} and step {checkpoint.get('batch_step')}")

    def latest_checkpoint(self):
        files = os.listdir(self.folder)
        files.sort(key=self._get_key)

        return files

    def _get_key(file):
        filename_parts = file.split("-")

        epoch = int(filename_parts[1]) if len(filename_parts) > 1 else 0
        batch = int(filename_parts[2].split(".")[0]) if len(filename_parts) > 2 else 0

        return (-epoch, -batch)

class DecoyRunner():
    def __init__(self) -> None:
        pass

    def finish(self):
        pass

class DecoyLogger():
    def __init__(self) -> None:
        self.id = -1

    def init(self, *args, **kargs):
        return DecoyRunner()

    def log(self, *args, **kargs):
        pass

