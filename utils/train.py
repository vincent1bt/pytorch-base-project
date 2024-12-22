import time
from utils.manage import MeanLossMetric
import torch
from model.model import CNNet

class StepTrainer():
  def __init__(
      self,
      model,
      optimizer,
      loss_fn
    ):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def train_step(self, inputs, targets):
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
      trainer,
      device,
      wandb,
      metrics=None,
      ckp_manager=None,
    ):
    self.train_loader = train_loader
    self.wandb = wandb

    self.trainer = trainer
    self.loss_metric = MeanLossMetric()
    self.metrics = metrics

    self.ckp_manager = ckp_manager
    self.device = device

    self.train_steps = len(train_loader)
  
  def _print_metrics(
      self, 
      loss, 
      rec, 
      acc, 
      prec,
      end=''
    ):
    print(
      '\r',
      'Epoch', self.epoch_count, 
      '| Step', f"{self.step}/{self.train_steps}",
      '| Loss:', f"{loss:.5f}",
      '| Recall:', f"{rec:.5f}",
      '| Precision:', f"{prec:.5f}",
      '| Accuracy:', f"{acc:.5f}",
      '| Epoch Time:', f"{time.time() - self.epoch_time:.2f}" if end 
      else '| Step Time:', f"{time.time() - self.batch_time:.2f}", 
      end=end
    )

  def _train_step(
      self,
      inputs,
      targets,
    ):
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    predictions, loss = self.trainer.train_step(inputs, targets)

    self.metrics.update_metrics(predictions, targets)

    rec, acc, prec = self.metrics.compute_metrics()

    self.loss_metric(loss)
    train_loss = self.loss_metric.mean

    self.wandb.log({"Loss": train_loss}, commit=False)
    self.wandb.log({"Recall": rec}, commit=False)
    self.wandb.log({"Precision": prec}, commit=False)
    self.wandb.log({"Accuracy": acc}, commit=False)
    self.wandb.log({"Step": self.step}, commit=False)
    self.wandb.log({"Epoch": self.epoch})

    self._print_metrics(train_loss, rec, acc, prec)

    if self.ckp_manager:
      self.ckp_manager.save(
        epoch=self.epoch,
        batch_step=0
      )
    
    self.step += 1
    self.batch_time = time.time()

  def train(self, epoch, epochs):
    self.trainer.model.train()
    self.loss_metric.reset()
    self.metrics.reset_metrics()
    self.epoch_time = time.time()

    self.batch_time = time.time()
    self.step = 1
    self.epoch = epoch

    self.epoch_count = f"00{epoch}/{epochs}" if epoch < 99 else f"{epoch}/{epochs}"

    for inputs, targets, _ in self.train_loader:
      self._train_step(
        inputs,
        targets,
      )
    
    rec, acc, prec = self.metrics.compute_metrics()
    self._print_metrics(self.loss_metric.mean, rec, acc, prec, end='\n')

    return rec, acc, prec

class TestManager():
  def __init__(
      self,
      model,
      device,
      metrics,
      wandb,
      val_loader,
    ):
    self.wandb = wandb
    self.model = model
    self.device = device
    self.metrics = metrics
    self.val_loader = val_loader

  def _print_metrics(
      self, 
      rec, 
      acc, 
      prec,
    ):
    print(
      '| Val Recall:', f"{rec:.5f}",
      '| Val Precision:', f"{prec:.5f}",
      '| Val Accuracy:', f"{acc:.5f}",
    )

  def test(self, epoch):
    self.metrics.reset_metrics()
    self.model.eval()

    with torch.no_grad():
      for inputs, targets, _ in self.val_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        predictions = self.model(inputs).softmax(dim=-1)

        self.metrics.update_metrics(predictions, targets)

    rec, acc, prec = self.metrics.compute_metrics()
    self._print_metrics(rec, acc, prec)

    self.wandb.log({"Val Recall": rec}, commit=False)
    self.wandb.log({"Val Precision": prec}, commit=False)
    self.wandb.log({"Val Accuracy": acc}, commit=False)
    self.wandb.log({"Val Epoch": epoch})

    return rec, acc, prec

class DebugTrainManager():
  def __init__(
      self,
      model,
      device,
      metrics,
      val_loader,
    ):
    self.model = model
    self.device = device
    self.metrics = metrics
    self.val_loader = val_loader

  def test(self):
    self.metrics.reset_metrics()
    self.model.eval()

    with torch.no_grad():
      for inputs, targets, img_paths in self.val_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        predictions = self.model(inputs).softmax(dim=-1)

        self.metrics.update_metrics(predictions, targets, img_paths)

    return self.metrics.compute_metrics()

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

