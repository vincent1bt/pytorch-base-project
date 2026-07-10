import time
from utils.test import MeanLossMetric
import torch
from model.model import CNNet

class BatchTrainer():
  def __init__(
      self,
      model,
      optimizer,
      loss_fn
    ):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def step(self, inputs, targets):
    predictions = self.model(inputs)
    loss = self.loss_fn(predictions, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return predictions.softmax(dim=-1), loss.item()

class EpochTrainer():
  def __init__(
      self,
      train_loader,
      batch_trainer,
      device,
      wandb,
      metrics,
      ckp_manager=None,
    ):
    self.train_loader = train_loader
    self.wandb = wandb

    self.batch_trainer = batch_trainer
    self.loss_metric = MeanLossMetric()
    self.metrics = metrics

    self.ckp_manager = ckp_manager
    self.device = device

    self.train_steps = len(train_loader)
  
  def _log_info(
      self,
      loss, log_metrics
    ):

    steps = f"{self.current_step}/{self.train_steps}"
    batch_info = f"Epoch: {self.epoch_count}| Step: {steps}| Loss: {loss:.3f}| "
    step_info = f"Step Time: {time.time() - self.batch_time:.2f}"

    print('\r', f"{batch_info}{log_metrics}{step_info}", end=" ")

  def _run_batch_step(
      self,
      inputs,
      targets,
    ):
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    predictions, loss = self.batch_trainer.step(inputs, targets)

    self.metrics.update_metrics(predictions, targets)

    self.loss_metric(loss)
    train_loss = self.loss_metric.mean
    log_string = []

    self.wandb.log({"Loss": train_loss}, commit=False)

    for name, metric in self.metrics.log_metrics:
      metric_value = metric.item()
      log_string.append(f"{name.split(' ')[1]}: {metric_value:.3f}| ")
      self.wandb.log({f"{name}": metric_value}, commit=False)

    self.wandb.log({"Step": self.current_step}, commit=False)
    self.wandb.log({"Epoch": self.epoch})

    log_metrics = "".join(log_string)
    self._log_info(train_loss, log_metrics)

    if self.ckp_manager:
      self.ckp_manager.save(
        epoch=self.epoch,
        batch_step=0
      )
    
    self.current_step += 1
    self.batch_time = time.time()

  def step(self, current_epoch, total_epochs):
    epoch_time = time.time()

    self.batch_trainer.model.train()
    self.loss_metric.reset()
    self.metrics.reset_metrics()
    
    self.batch_time = time.time()
    self.current_step = 1
    self.epoch = current_epoch

    self.epoch_count = f"00{current_epoch}/{total_epochs}"

    if current_epoch > 99:
      self.epoch_count = f"{current_epoch}/{total_epochs}"

    for inputs, targets, _ in self.train_loader:
      self._run_batch_step(
        inputs,
        targets,
      )
    
    print(f"Epoch Time: {time.time() - epoch_time:.2f}", end="\n")

    return self.metrics

class DebugTrainManager():
  def __init__(
      self,
      model,
      device,
      metrics,
      loader,
    ):
    self.model = model
    self.device = device
    self.metrics = metrics
    self.loader = loader

  def test(self):
    self.metrics.reset_metrics()
    self.model.eval()

    with torch.no_grad():
      for inputs, targets, img_paths in self.loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        predictions = self.model(inputs)
        predictions = predictions.softmax(dim=-1)

        self.metrics.update_metrics(predictions, targets, img_paths)

    return self.metrics

def obtain_device():
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    print(f"Deviced Used: {device}")

    return torch.device(
        device
    )

def create_model(config):
  return CNNet(
      num_layers=config.NUM_LAYERS,
      initial_channels=config.INITIAL_CHANNELS,
      feat_size=config.FEAT_SIZE,
  )

def create_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2),
        eps=config.EPSILON,
    )

def get_loss_function():
  return torch.nn.CrossEntropyLoss()

