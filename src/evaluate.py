import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from dataset import MURADataset
from model import FractureNet
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from pathlib import Path

current_dir = Path.cwd()

class GradCAM:
  '''
  Post-hoc Grad-CAM implementation for CNN fracture localization
  '''

  def __init__(self, model:FractureNet, target_layer):
    self.model = model
    self.target_layer = target_layer

    self.activations = None
    self.gradients = None

    self._register_hooks()

  def _register_hooks(self):
    def forward_hook(module, input, output):
      self.activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
      self.gradients = grad_out[0].detach()

    self.target_layer.register_forward_hook(forward_hook)
    self.target_layer.register_full_backward_hook(backward_hook)


  def generate(self, x):
    '''
    :param x: [1, 3, H, W]
    returns: Grad-CAM heatmap [H', W']
    '''
    
    self.model.zero_grad()

    logits = self.model(x)
    score = logits[0]
    score.backward()

    # Global average pool gradients
    weights = self.gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted sum of activations
    cam = (weights * self.activations).sum(dim=1)

    cam = F.relu(cam).squeeze()

    # Normalize
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam

def hybrid_predict(model: FractureNet, gradcam: GradCAM, x: torch.Tensor, p_low: float = .22, entropy_max: float = 1.50, area_max:float | None = None , alpha: float = .5):
  '''
    Returns:
      y_sys: int {0=NEG, 1=POS, 2=INC}
      p: float probability
      ent: float cam entropy (None if no cam computed)
      area: float cam area ratio (None if no cam computed)
  '''
  model.eval()


    # Compute probability WITHOUT gradients
  with torch.no_grad():
      logit = model(x)[0]
      p = torch.sigmoid(logit).item()

  if p < p_low:
      return 0, p, None, None

  # Compute Grad-CAM WITH gradients
  with torch.enable_grad():
      cam = gradcam.generate(x)

  ent = cam_entropy(cam)
  area = cam_area_ratio(cam, alpha)

  loc_ok = (ent <= entropy_max)
  if area_max is not None:
    loc_ok = loc_ok and (area <= area_max)

  if loc_ok:
    return 1, p, ent, area
  else:
    return 2, p, ent, area

@torch.no_grad()
def evaluate(model:FractureNet, loader, device, threshold=0.22):
  '''
  :param model: CNN
  :type model: FractureNet
  :param loader: Description
  :param device: Description
  :param threshold: Ranges form 0 - 1, the lower the threshold the higher the recall and vice versa
  I selected a classification threshold of 0.22 to bias the model toward sensitivity (91.6% recall) while maintaining sufficient confidence for Grad-CAM-based localization. 
  Lower thresholds degraded localization reliability without meaningful recall gains.
  '''
  model.eval()

  all_probs = []
  all_labels = []

  for imgs, labels in loader:
    imgs = imgs.to(device)
    labels = labels.to(device)

    logits = model(imgs)
    probs = torch.sigmoid(logits)
    
    all_probs.append(probs.cpu())
    all_labels.append(labels.cpu())

  y_prob = torch.cat(all_probs).numpy()
  y_true = torch.cat(all_labels).numpy()
  y_pred = (y_prob >= threshold).astype(int)

  metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, zero_division=0),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_prob)
  } 

  return metrics, y_prob, y_true
  
def evaluate_hybrid(model: FractureNet, dataset, device, p_low: float = .22, entropy_max: float = 1.5, area_max: float | None=None, max_items: int | None=None, alpha: float = .5):
  '''
    Runs hybrid system over dataset (batch_size=1 behavior).
    Returns:
      summary dict + raw lists
  '''
  model.eval()

  gradcam = GradCAM(model, model.target_layer)
  y_true = []
  y_sys = []
  probs = []
  ents = []
  areas = []

  n = len(dataset) if max_items is None else min(len(dataset), max_items)

  for i in range(n):
    img, label = dataset[i]
    x = img.unsqueeze(0).to(device)

    pred_sys, p, ent, area = hybrid_predict(model, gradcam, x, p_low, entropy_max, area_max, alpha)

    y_true.append(int(label))
    y_sys.append(int(pred_sys))
    probs.append(float(p))
    ents.append(None if ent is None else float(ent))
    areas.append(None if area is None else float(area))

  y_true = np.array(y_true)
  y_sys = np.array(y_sys)

  # Counts
  neg_count = int((y_sys == 0).sum())
  pos_count = int((y_sys == 1).sum())
  inc_count = int((y_sys == 2).sum())
  inconclusive_rate = inc_count / len(y_sys)
  total = len(y_sys)
  coverage = 1.0 - inconclusive_rate  

  # Metrics on confident cases only (exlude inconclusive)
  confident_mask = (y_sys != 2)
  y_true_conf = y_true[confident_mask]
  y_pred_conf = (y_sys[confident_mask] == 1).astype(int)

  # Including inconclusive
  # correctness for confident predictions only
  correct_conf = int((y_pred_conf == y_true_conf).sum()) if len(y_true_conf) > 0 else 0

  # overall accuracy: inconclusive counts as incorrect
  overall_accuracy = correct_conf / total

  # Handle edge case: if everything becomes inconclusive
  if len(y_true_conf) == 0:
    metrics_conf = {
      'accuracy_conf': None,
      'precision_conf': None,
      'recall_conf': None,
      'f1_conf': None
    }
  else:
    metrics_conf = {
      'accuracy_conf': accuracy_score(y_true_conf, y_pred_conf),
      'precision_conf': precision_score(y_true_conf, y_pred_conf, zero_division=0),
      'recall_conf': recall_score(y_true_conf, y_pred_conf, zero_division=0),
      'f1_conf': f1_score(y_true_conf, y_pred_conf, zero_division=0)
    }
  
  summary = {
    'p_low': p_low,
    'entropy_max': entropy_max,
    'area_max': area_max,
    'counts': {'negative': neg_count, 'positive': pos_count, 'inconclusive': inc_count},
    'inconclusive_rate': inconclusive_rate,
    "coverage": coverage,
    "overall_accuracy": overall_accuracy,
    **metrics_conf
  }
  
  return summary, (y_true.tolist(), y_sys.tolist(), probs, ents, areas)

def overlay_cam(cam, image):
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam_u8 = np.uint8(255 * cam)

    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.uint8(255 * img)

    overlay_bgr = cv2.addWeighted(heatmap_bgr, 0.6, img, 0.4, 0)

    # ✅ Convert to RGB for matplotlib
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def load_original_grayscale(dataset, index):
  abs_path = dataset.df.iloc[index]['abs_path']
  img = Image.open(abs_path).convert('L')
  return np.array(img)

def cam_entropy(cam: torch.Tensor) -> float:
    # cam is assumed normalized to [0,1]
    p = cam / (cam.sum() + 1e-8)
    return (-(p * torch.log(p + 1e-8)).sum()).item()

def cam_area_ratio(cam: torch.Tensor, frac: float = 0.5) -> float:
    # fraction of pixels above frac*max
    thr = frac * cam.max()
    return (cam > thr).float().mean().item()

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load validation data
  valid_df = pd.read_csv(f'{str(current_dir).replace('src','')}/data/valid_image_paths.csv',
                       header=None,
                       names=['path'])
  val_data = MURADataset(df=valid_df, data_root='c:/Users/marzk/Documents/Coding/AI/imageClassification/data')

  # Load model
  model = FractureNet(backbone='resnet18', pretrained=True)
  model.load_state_dict(torch.load('best_model.pt', map_location=device))
  model.to(device)

  # Metrics
  metrics, _= evaluate_hybrid(model, val_data, device, p_low=0.26, entropy_max=3.8, area_max=.3, alpha=.5)
  print(f'Validation Metrics:')
  for k, v in metrics.items():
    print(f'{k}: {v}')

  # Grad-CAM example
  gradcam = GradCAM(model, model.target_layer)

  img, label = val_data[0]
  img = img.unsqueeze(0).to(device)

  cam = gradcam.generate(img)
  overlay = overlay_cam(cam, img[0])
  og_img = load_original_grayscale(val_data, 0)
  print('Grad-CAM generated.')
  plt.figure(figsize=(12,4))

  plt.subplot(1,3,1)
  plt.imshow(og_img, cmap='gray')
  plt.title('Original X-ray')
  plt.axis('off')

  plt.subplot(1,3,2)
  plt.imshow(cam.cpu(), cmap='jet')
  plt.title('Grad-CAM Heatmap')
  plt.axis('off')

  plt.subplot(1,3,3)
  plt.imshow(overlay)
  plt.title('Overlay')
  plt.axis('off')

  plt.show()

  

if __name__ == '__main__':
  main()
