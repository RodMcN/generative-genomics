import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from utils import get_logger

logger = get_logger(__name__)

class RNADataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data).share_memory_()
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Metrics:
    def __init__(self, early_stopping_rounds=5, early_stopping_warmup=5, desc="Train"):
        self.history = defaultdict(list)

        self.running_kl = 0
        self.running_recon = 0
        self.running_loss = 0
        self._epochs = 0
        self._steps = 0
        
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_warmup = early_stopping_warmup
        self.best_loss = float("inf")
        self.epochs_since_best = 0

        self.desc = desc

    def update(self, kl, recon, loss):
        self.running_kl += kl
        self.running_recon += recon
        self.running_loss += loss
        self._steps += 1

    def end_epoch(self, log=True):
        if self._steps == 0:
            raise Exception

        kl = self.running_kl / self._steps
        self.history["kl"].append(kl)

        recon = self.running_recon / self._steps
        self.history["recon"].append(recon)

        loss = self.running_loss / self._steps
        self.history["loss"].append(loss)

        self._steps = 0
        self.running_kl = 0
        self.running_recon = 0
        self.running_loss = 0

        self._epochs += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1

        if log:
            logger.info(f"{self.desc}: Epoch {self._epochs} | KL: {kl:.4f} | recon: {recon:.4f} | loss: {loss:.4f}")

    def early_stopping(self, verbose=True):
        if verbose:
            logger.info(f"Epochs epochs_since_best = {self.epochs_since_best}. Early Stopping rounds = {self.early_stopping_rounds}. Best loss = {self.best_loss}")
        if self.epochs_since_best > self.early_stopping_rounds:
            return True
        return False



def get_dataloader(dataset, config, weights=None, training=True):
    if weights is not None:
        sample_weights_tensor = torch.DoubleTensor(weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights_tensor, 
            num_samples=len(sample_weights_tensor), 
            replacement=True
        )
    else:
        sampler = None
    
    return DataLoader(dataset,
                         batch_size=config['batch_size'],
                         num_workers=config['train_workers'] if training else config['val_workers'],
                         pin_memory=training,
                         drop_last=True,
                         persistent_workers=training,
                         sampler=sampler # sampler handles shuffling
                       )
