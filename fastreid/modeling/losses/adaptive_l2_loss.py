import torch
from torch.autograd import Variable


def hard_sigmoid(x, c=2.5):
    if x < -c:
        return 0
    if x > c:
        return 1
    return x / (2 * c) + 0.5


def adaptive_l2_loss(model, kernel_regularization_factor=0.005, bias_regularization_factor=0.005):
    l2_reg = Variable(torch.tensor(0.0), requires_grad=True).cuda()
    i = 0
    for name, params in model.named_parameters():
        if "conv" in name:
            if "weight" in name:
                l2_reg = l2_reg + hard_sigmoid(kernel_regularization_factor * model.adaptive_param[i].cuda() * params.norm(2).cuda())
                i += 1
            if "bias" in name:
                l2_reg = l2_reg + hard_sigmoid(bias_regularization_factor * model.adaptive_param[i].cuda() * params.norm(2).cuda())
                i += 1

    return l2_reg.cuda()
