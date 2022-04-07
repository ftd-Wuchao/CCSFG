import torch
from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from fastreid.modeling.backbones.resnet import build_resnet_backbone
# from fastreid.modeling.backbones.osnet import build_osnet_backbone
from torch import nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.adaptive_param = self._make_layer(self.model)

    def _make_layer(self, model):
        layers = []

        for _name, _params in self.model.named_parameters():
            if "conv" in _name:
                if "weight" in _name:
                    layers.append(nn.Parameter(data=torch.tensor(0.0)))
                if "bias" in _name:
                    layers.append(nn.Parameter(data=torch.tensor(0.0)))
            if "bn" in _name:
                if "weight" in _name:
                    layers.append(nn.Parameter(data=torch.tensor(0.0)))
                if "bias" in _name:
                    layers.append(nn.Parameter(data=torch.tensor(0.0)))
        return nn.ParameterList(layers)

    def forward(self, x):
        x = self.model(x)
        return x

