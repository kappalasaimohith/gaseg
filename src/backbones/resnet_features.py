import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        # Return a dictionary of feature maps
        return {'layer2': f2, 'layer3': f3, 'layer4': f4}
