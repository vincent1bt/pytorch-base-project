import os
import torch
from hparameters import data_config

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
        if not os.path.isdir(self.folder):
           return False
        
        files = os.listdir(self.folder)
        files.sort(key=self._get_key)

        return files

    def _get_key(self, file):
        filename_parts = file.split("-")

        epoch = int(filename_parts[1]) if len(filename_parts) > 1 else 0
        batch = int(filename_parts[2].split(".")[0]) if len(filename_parts) > 2 else 0

        return (epoch, batch)

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

