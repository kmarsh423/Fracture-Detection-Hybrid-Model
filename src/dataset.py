from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import re
import torch

# This makes sure that the images are normalized to stabilize training and to match the input distribution expected by pretrained Vision Transformer weights, which improves convergence and performance.

cnn_transform = transforms.Compose([
  transforms.Resize((224, 224)), # The ViT models were trained on 224x224 images
  transforms.ToTensor(), # Converts [0-255] PIL image to [0-1] float tensor
  transforms.Normalize(
    # These come from ImageNet
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

# Utility Functions

def extract_patient_id(path: str) -> str:
  '''Extract patientXXX form MURA path.'''
  match = re.search(r'(patient\d+)', path)
  if match:
    return match.group(1)
  raise ValueError(f'Patient ID not found in path: {path}')

def extract_label(path: str) -> int:
  '''
  Gets label from directory name
  study*_positive == 1
  study*_negative == 0
  '''
  if 'positive' in path:
    return 1
  if 'negative' in path:
    return 0
  raise ValueError(f'Label not found in path: {path}')



class MURADataset(Dataset):
  '''
  Dataset for MURA where CSV contains a single column of image paths.

  Example CSV row:
  MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png
  '''

  def __init__(self, data_root, df: pd.DataFrame, transform=cnn_transform):
    
    self.df = df.copy()
    self.data_root = data_root
    self.transform = transform
    self.df['label'] = self.df['path'].apply(extract_label)
    self.df['patient_id'] = self.df['path'].apply(extract_patient_id)
    self.df['abs_path'] = self.df['path'].apply(
      lambda p: p.replace('MURA-v1.1', data_root)
    )

    self.df['body_part'] = self.df['path'].apply(
      lambda p: p.split('/')[2].replace('XR_', '').lower()
    )

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    row = self.df.iloc[index]


    image = Image.open(row['abs_path']).convert('RGB')
    image = self.transform(image)


    label = torch.tensor(row['label'], dtype=torch.float32)


    return image, label


# Optional to prevent data leakage if there is no train test split. However the dataset already comes pre split so this can be ignored in most cases
def patient_level_split(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=42):
  """
  Split dataframe by patient ID to avoid data leakage.
  A patient is positive if ANY image is positive.
  """
  from sklearn.model_selection import train_test_split


  patient_labels = (
    df.groupby("patient_id")["label"]
      .max()
      .reset_index()
  )


  train_p, temp_p = train_test_split(
    patient_labels,
    test_size=val_ratio + test_ratio,
    stratify=patient_labels["label"],
    random_state=seed,
  )


  val_p, test_p = train_test_split(
    temp_p,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=temp_p["label"],
    random_state=seed,
  )


  def subset(patients):
    return df[df["patient_id"].isin(patients["patient_id"])]


  return subset(train_p), subset(val_p), subset(test_p)