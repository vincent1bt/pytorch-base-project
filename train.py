from data.loader import ImageDataset, train_transforms, val_transforms
from data.loader import label_mapping, text_mapping

from utils.train import StepTrainer, EpochTrainer, TestManager
from utils.train import create_optimizer, create_model, get_loss_function
from utils.manage import CheckpointManager, DecoyLogger, MetricsManager
from hparameters import data_config, model_config, train_config

import torch
from torch import nn
import pandas as pd
import argparse
import builtins

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--start_epoch', type=int, default=1, help='Initial epoch number')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')
parser.add_argument('--use_wandb', default=False, action='store_true', help='Log the training using wandb')
parser.add_argument('--wandb_id', type=str, default=False, help='id run for wandb')

using_notebook = getattr(builtins, "__IPYTHON__", False)
opts = parser.parse_args([]) if using_notebook else parser.parse_args()
run_id = opts.wandb_id

num_workers = 4

try:
  import wandb
except ImportError:
  if opts.use_wandb:
    print("Warning wandb library not installed but flag 'use_wandb' set as true")
    exit()

finally:
  if not opts.use_wandb:
    wandb = DecoyLogger()
    run_id = wandb.id
  else:
    pass
    # wandb.login()

if using_notebook:
  opts.continue_training = True

df = pd.read_csv(data_config.TRAIN_INFORMATION_FILE)

train_set = df[df["is_valid"] == False]
val_set = df[df["is_valid"] == True]

train_img_paths = train_set["path"].values
train_labels = train_set[data_config.NOISE_LEVEL].values
# train_labels = [label_mapping[label] for label in train_labels]

val_img_paths = val_set["path"].values
val_labels = val_set["noisy_labels_0"].values
# val_labels = [label_mapping[label] for label in val_labels]

train_dataset = ImageDataset(
    image_paths=train_img_paths,
    image_labels=train_labels,
    folder_path=data_config.FOLDER_PATH,
    transforms=train_transforms,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_config.BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)

val_dataset = ImageDataset(
    image_paths=val_img_paths,
    image_labels=val_labels,
    folder_path=data_config.FOLDER_PATH,
    transforms=val_transforms,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=train_config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = create_model(model_config)
model = model.to(device)

optimizer = create_optimizer(model, train_config)

loss_fn = get_loss_function()

train_metrics = MetricsManager()
train_metrics.to(device)

val_metrics = MetricsManager()
val_metrics.to(device)

step_trainer = StepTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)

ckp_manager = CheckpointManager(
    optimizer=optimizer,
    model=model,
)

epoch_trainer = EpochTrainer(
    train_loader=train_loader,
    trainer=step_trainer,
    device=device,
    wandb=wandb,
    ckp_manager=ckp_manager,
    metrics=train_metrics,
)

test_manager = TestManager(
    model=model,
    device=device,
    metrics=val_metrics,
    wandb=wandb,
    val_loader=val_loader,
)

if opts.continue_training:
  print("loading training checkpoints: ")
  latest_checkpoint = ckp_manager.latest_checkpoint()

  if latest_checkpoint:
    ckp_manager.load(latest_checkpoint[-1])

    print(f"Checkpoints from: {latest_checkpoint}")
  else:
    print("No checkpoints found")

train_config.PYTORCH_VERSION = torch.__version__

if device:
    train_config.GPU = torch.cuda.get_device_name(device)

config = {
  **train_config().__dict__,
  **model_config().__dict__,
  **data_config().__dict__
}

if not run_id:
    run_id = wandb.util.generate_id()

print(f"wandb run id: {run_id}")

run = wandb.init(project=train_config.PROJECT_NAME, tags=["train"], config=config, id=run_id, resume="allow")

epochs = opts.epochs + opts.start_epoch

for epoch in range(opts.start_epoch, epochs):
    epoch_trainer.train(
        epoch,
        epochs
    )

    test_manager.test(
      epoch
    )

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")

run.finish()