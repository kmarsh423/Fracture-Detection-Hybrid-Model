import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import MURADataset
from model import FractureNet
import random
import numpy as np

from pathlib import Path

current_dir = Path.cwd()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0

    def step(self, val_loss):
        # returns True if we should stop
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (may slightly reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_eqoch(model:FractureNet, loader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0

  for imgs, labels in tqdm(loader, leave=False):
    imgs = imgs.to(device)
    labels = labels.to(device).float()

    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * imgs.size(0)
  
  return running_loss/ len(loader.dataset)

@torch.no_grad()
def evaluate(model: FractureNet, loader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  for imgs, labels in loader:
    imgs = imgs.to(device)
    labels = labels.to(device).float()

    logits = model(imgs)
    loss = criterion(logits, labels)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    running_loss += loss.item() * imgs.size(0)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
  
  return running_loss / total, correct / total


def main():
  set_seed()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load CSVs (no headers)
  train_df = pd.read_csv(f'{str(current_dir).replace('src','')}/data/train_image_paths.csv',
                       header=None,
                       names=['path'])
  
  valid_df = pd.read_csv(f'{str(current_dir).replace('src','')}/data/valid_image_paths.csv',
                       header=None,
                       names=['path'])
  
  # Create datasets
  train_dataset = MURADataset(df=train_df, data_root=DATA_ROOT)
  val_dataset = MURADataset(df=valid_df, data_root=DATA_ROOT)

  # Compute class imbalance
  pos_weight = (
    (train_dataset.df['label'] == 0).sum() / (train_dataset.df['label'] == 1).sum()
  )

  criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, device=device)
  )

  # DataLoaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
  )

  # Model
  model = FractureNet(backbone='resnet18', pretrained=True)
  model.to(device)

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=1e-4
  )

  num_epochs = 100
  early_stopper = EarlyStopping(patience=10, min_delta=1e-4)
  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    train_loss = train_one_eqoch(
      model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = evaluate(
      model, val_loader, criterion, device
    )

    print(
      f'Epoch {epoch+1}/{num_epochs} | '
      f'Train Loss: {train_loss:.4f} |'
      f'Val Loss: {val_loss:.4f} | '
      f'Val Acc: {val_acc:.4f}'
    )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), 'best_model.pt')

    if early_stopper.step(val_loss):
      print(f"Early stopping triggered. Best val loss: {best_val_loss:.4f}")
      break
  print('Training Complete.')

if __name__ == '__main__':

  main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import MURADataset
from model import FractureNet
import random
import numpy as np

from pathlib import Path

current_dir = Path.cwd()

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0

    def step(self, val_loss):
        # returns True if we should stop
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (may slightly reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_eqoch(model:FractureNet, loader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0

  for imgs, labels in tqdm(loader, leave=False):
    imgs = imgs.to(device)
    labels = labels.to(device).float()

    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * imgs.size(0)
  
  return running_loss/ len(loader.dataset)

@torch.no_grad()
def evaluate(model: FractureNet, loader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  for imgs, labels in loader:
    imgs = imgs.to(device)
    labels = labels.to(device).float()

    logits = model(imgs)
    loss = criterion(logits, labels)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    running_loss += loss.item() * imgs.size(0)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
  
  return running_loss / total, correct / total


def main():
  set_seed()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load CSVs (no headers)
  train_df = pd.read_csv(f'{str(current_dir).replace('src','')}/data/train_image_paths.csv',
                       header=None,
                       names=['path'])
  
  valid_df = pd.read_csv(f'{str(current_dir).replace('src','')}/data/valid_image_paths.csv',
                       header=None,
                       names=['path'])
  
  # Create datasets
  train_dataset = MURADataset(df=train_df, data_root='c:/Users/marzk/Documents/Coding/AI/imageClassification/data')
  val_dataset = MURADataset(df=valid_df, data_root='c:/Users/marzk/Documents/Coding/AI/imageClassification/data')

  # Compute class imbalance
  pos_weight = (
    (train_dataset.df['label'] == 0).sum() / (train_dataset.df['label'] == 1).sum()
  )

  criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, device=device)
  )

  # DataLoaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
  )

  # Model
  model = FractureNet(backbone='resnet18', pretrained=True)
  model.to(device)

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=1e-4
  )

  num_epochs = 100
  early_stopper = EarlyStopping(patience=10, min_delta=1e-4)
  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    train_loss = train_one_eqoch(
      model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = evaluate(
      model, val_loader, criterion, device
    )

    print(
      f'Epoch {epoch+1}/{num_epochs} | '
      f'Train Loss: {train_loss:.4f} |'
      f'Val Loss: {val_loss:.4f} | '
      f'Val Acc: {val_acc:.4f}'
    )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), 'best_model.pt')

    if early_stopper.step(val_loss):
      print(f"Early stopping triggered. Best val loss: {best_val_loss:.4f}")
      break
  print('Training Complete.')

if __name__ == '__main__':
  main()
