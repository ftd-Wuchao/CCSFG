import  torch
import torch.nn as nn
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class domain_classfier(nn.Module):
    def __init__(self, camids):
        super(domain_classfier, self).__init__()

        classifier = []
        classifier.append(nn.BatchNorm1d(2048))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Linear(2048, 100))
        classifier.append(nn.BatchNorm1d(100))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Linear(100, camids))
        self.classifier = nn.Sequential(*classifier)

        self.classifier.apply(weights_init_kaiming)

    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.classifier(reverse_feature)
        return domain_output




