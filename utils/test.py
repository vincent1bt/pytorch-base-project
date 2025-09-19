import torch
from torchmetrics import Accuracy, Precision, Recall, PrecisionRecallCurve
import pandas as pd
from itertools import chain
from hparameters import data_config

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

class MetricsManager(torch.nn.Module):
  def __init__(
      self, 
      name='Training',
      num_classes=data_config.NUM_CLASSES,
    ):
    super().__init__()

    self.accuracy = Accuracy(
      task="multiclass",
      num_classes=num_classes
    )
    self.precision = Precision(
      task="multiclass",
      num_classes=num_classes,
      average="macro"
    )
    self.recall = Recall(
      task="multiclass",
      num_classes=num_classes,
      average="macro"
    )
    self.precision_recall_curve = PrecisionRecallCurve(
      task="multiclass",
      num_classes=num_classes,
    )

    self._metrics = {
      f"{name} accuracy": self.accuracy,
      f"{name} precision": self.precision,
      f"{name} recall": self.recall,
    }

    self._graph = {
      f"{name} precision_recall_curve": self.precision_recall_curve,
    }

  def reset_metrics(self):
    for metric in chain(self._metrics.values(), self._graph.values()):
      metric.reset()

  def update_metrics(self, predictions, targets):
    for metric in chain(self._metrics.values(), self._graph.values()):
      metric.update(predictions, targets)

  @property
  def log_metrics(self):
    for name, metric in self._metrics.items():
      yield name, metric.compute()
  
  @property
  def graph_metrics(self):
    for name, metric in self._graph.items():
      precision_list, recall_list, threshold = metric.compute() 
      yield name, (precision_list, recall_list, threshold)

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

    predictions = predictions.max(dim=-1)
    # tolist moves data to CPU and creates a new python list
    maxarg = predictions.indices.tolist()
    maxval = predictions.values.tolist()

    self.predictions.extend(maxarg)
    self.scores.extend(maxval)
    self.targets.extend(targets.tolist())
    self.img_paths.extend(img_paths)

  @property
  def metrics(self):
    data_frame = pd.DataFrame({
        "scores": self.scores,
        "predictions": self.predictions,
        "targets": self.targets,
        "img_paths": self.img_paths
    })

    return self._metrics, data_frame

class TestManager():
  def __init__(
      self,
      model,
      device,
      metrics,
      wandb,
      loader,
    ):
    self.wandb = wandb
    self.model = model
    self.device = device
    self.metrics = metrics
    self.loader = loader

  def test(self, epoch):
    self.metrics.reset_metrics()
    self.model.eval()

    with torch.no_grad():
      for inputs, targets, _ in self.loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        predictions = self.model(inputs).softmax(dim=-1)

        self.metrics.update_metrics(predictions, targets)
        
    log_string = []

    for name, metric in self.metrics.log_metrics:
      metric_value = metric.item()
      log_string.append(f"{name}: {metric_value:.2f}| ")
      self.wandb.log({f"{name}": metric_value}, commit=False)

    self.wandb.log({"Val Epoch": epoch})

    log_metrics = "".join(log_string)
    print(log_metrics)

    return self.metrics.metrics