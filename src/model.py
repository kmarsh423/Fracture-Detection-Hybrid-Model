<<<<<<< HEAD
import torch.nn as nn
import torchvision.models as models

class FractureNet(nn.Module):
  '''
  CNN backbone for fracture classification
  Grad-CAM is applied post-hoc using the final convolutional layer.
  '''

  def __init__(self, backbone='resnet18', pretrained=True):
    super().__init__()

    if backbone == 'resnet18':
      base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
      feat_dim = 512
    elif backbone == 'resnet34':
      base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
      feat_dim = 512
    else:
      raise ValueError('Unsupported Backbone')
    
    # Remove classification head
    self.backbone = nn.Sequential(*list(base.children())[:-2])

    # Global pooling
    self.pool = nn.AdaptiveAvgPool2d(1)

    # Binary classifier
    self.classifier = nn.Linear(feat_dim, 1)

    # Target layer
    self.target_layer = base.layer4

  def forward(self, x):
    '''
    :param x: Tensor [B, 3, H, W]
    returns: logits [B]
    '''

    features = self.backbone(x) # [B, C, H`, W`]
    pooled = self.pool(features).flatten(1) # [B, C, 1, 1] -> [B, C]
    logits = self.classifier(pooled).squeeze(1) # [B, 1] -> [B]
    return logits
=======
import torch.nn as nn
import torchvision.models as models

class FractureNet(nn.Module):
  '''
  CNN backbone for fracture classification
  Grad-CAM is applied post-hoc using the final convolutional layer.
  '''

  def __init__(self, backbone='resnet18', pretrained=True):
    super().__init__()

    if backbone == 'resnet18':
      base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
      feat_dim = 512
    elif backbone == 'resnet34':
      base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
      feat_dim = 512
    else:
      raise ValueError('Unsupported Backbone')
    
    # Remove classification head
    self.backbone = nn.Sequential(*list(base.children())[:-2])

    # Global pooling
    self.pool = nn.AdaptiveAvgPool2d(1)

    # Binary classifier
    self.classifier = nn.Linear(feat_dim, 1)

    # Target layer
    self.target_layer = base.layer4

  def forward(self, x):
    '''
    :param x: Tensor [B, 3, H, W]
    returns: logits [B]
    '''

    features = self.backbone(x) # [B, C, H`, W`]
    pooled = self.pool(features).flatten(1) # [B, C, 1, 1] -> [B, C]
    logits = self.classifier(pooled).squeeze(1) # [B, 1] -> [B]
    return logits
>>>>>>> 37d3db4 (added some fixes)
