import torch
import argparse
import builtins

from data.loader import ImageDataset, train_transforms, val_transforms
from data.loader import label_mapping, text_mapping
from data.dataset import obtain_dry_dataset, obtain_train_dataset

from utils.train import BatchTrainer, EpochTrainer, obtain_device
from utils.train import create_optimizer, create_model, get_loss_function
from utils.test import TestManager, MetricsManager
from utils.manage import CheckpointManager, DecoyLogger
from hparameters import data_config, model_config, train_config

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='Initial epoch number')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')
parser.add_argument('--use_wandb', default=False, action='store_true', help='Log the training using wandb')
parser.add_argument('--wandb_id', type=str, default='', help='id run for wandb')
parser.add_argument('--dry_run', default=False, action='store_true', help='Run a dry test of the training')

using_notebook = getattr(builtins, "__IPYTHON__", False)
opts = parser.parse_args([]) if using_notebook else parser.parse_args()
run_id = opts.wandb_id

if opts.dry_run:
  print('Running a dry train')

num_workers = 2

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

if opts.dry_run:
  train_set, validation_set = obtain_dry_dataset()
else:
  train_set, validation_set = obtain_train_dataset()

train_dataset = ImageDataset(
    image_paths=train_set[0],
    image_labels=train_set[1],
    folder_path=data_config.TRAIN_FOLDER_PATH,
    transforms=train_transforms,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_config.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    # num_workers=num_workers,
    # pin_memory=True,
)

val_dataset = ImageDataset(
    image_paths=validation_set[0],
    image_labels=validation_set[1],
    folder_path=data_config.TRAIN_FOLDER_PATH,
    transforms=val_transforms,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=train_config.BATCH_SIZE,
    shuffle=False,
    # num_workers=num_workers,
    # pin_memory=True,
)

device = obtain_device()
print(device)

model = create_model(model_config)
model = model.to(device)

optimizer = create_optimizer(model, train_config)
loss_fn = get_loss_function()

train_metrics = MetricsManager(name="Training", num_classes=data_config.NUM_CLASSES)
train_metrics.to(device)

val_metrics = MetricsManager(name="Validation", num_classes=data_config.NUM_CLASSES)
val_metrics.to(device)

batch_trainer = BatchTrainer(
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
    batch_trainer=batch_trainer,
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
    loader=val_loader,
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

if device and device == 'cuda':
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
    epoch_trainer.step(
        epoch,
        epochs
    )

    test_manager.test(
      epoch
    )

run.finish()

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")

# new test comment