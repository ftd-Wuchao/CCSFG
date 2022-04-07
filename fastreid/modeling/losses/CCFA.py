import torch
import torch.nn as nn
import torch.nn.functional as F


def D(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def ccfa_loss(feature, fake_features):
    loss_items = []
    mse_loss_fn = nn.MSELoss().cuda()
    for fake_feature in fake_features:
        loss_items.append(mse_loss_fn(feature, fake_feature))
    return torch.mean(torch.stack(loss_items))
