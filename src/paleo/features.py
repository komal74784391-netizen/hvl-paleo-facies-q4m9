import torch
import torch.nn as nn
import torchvision


class CoreFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(512, embedding_dim)

    def forward(self, images):
        return self.projection(self.backbone(images))
