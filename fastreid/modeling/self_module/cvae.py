import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out.cuda()


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

class CVAE(nn.Module):
    def __init__(self, camids, pids, feat_size, latent_size):
        super().__init__()

        self.camids = camids
        self.pids = pids
        self.latent_size = latent_size

        self.camids_encoder = Encoder(
            camids, pids, feat_size, latent_size, "camids")
        self.pids_encoder = Encoder(
            camids, pids, feat_size, latent_size, "pids")

        self.pids_prior = Prior_net(
            camids, pids, latent_size, "pids")
        self.camids_prior = Prior_net(
            camids, pids, latent_size, "camids")

        self.decoder = Decoder(
            camids, pids, feat_size, latent_size)

    def forward(self, x, c, p, aphla):

        out = {}
        out['camids_means'], out['camids_log_var'] = self.camids_encoder(x, c, p)
        out['pids_means'], out['pids_log_var'] = self.pids_encoder(x, c, p)

        prior = {}
        prior['camids_means'], prior['camids_log_var'] = self.camids_prior(c, p)
        prior['pids_means'], prior['pids_log_var'] = self.pids_prior(c, p)

        camids_z = self.reparameterize(out['camids_means'], out['camids_log_var'])
        pids_z = self.reparameterize(out['pids_means'], out['pids_log_var'])
        prior_camids_z = self.reparameterize(prior['camids_means'], prior['camids_log_var'])
        prior_pids_z = self.reparameterize(prior['pids_means'], prior['pids_log_var'])
        out_x = self.decoder(camids_z, pids_z, c, p)
        prior_x = self.decoder(prior_camids_z, prior_pids_z, c, p)

        hybrid_loss = self.loss_fn(out_x, x, prior_x, out, prior, aphla)

        recon_x = aphla*out_x + (1-aphla)*prior_x

        return recon_x, camids_z, pids_z, prior, hybrid_loss

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, p, prior_dict):
        batch_size = p.size()[0]
        c_list = []
        for i in range(self.camids):
            c_list.append((torch.ones(batch_size) * i).to(p.device))
        recon_x_list = []
        self.decoder.eval()
        with torch.no_grad():
            for c in c_list:
                rand_camids_z = self.reparameterize(prior_dict['camids_means'], prior_dict['camids_log_var'])
                rand_pids_z = self.reparameterize(prior_dict['pids_means'], prior_dict['pids_log_var'])
                recon_x_list.append(self.decoder(rand_camids_z, rand_pids_z, c, p))
        self.decoder.train()
        return recon_x_list

    def caculate_kl(self, src_means, src_log_var, tar_means, tar_log_var):
        return 0.5 * torch.sum(
            (tar_log_var - src_log_var) + (src_log_var.exp() + (src_means - tar_means).pow(2)) / tar_log_var.exp() - 1)

    def loss_fn(self, out_x, x, prior_x, out, prior, aphla):

        mse_loss_fn = nn.MSELoss(reduction='sum').cuda()
        BCE = mse_loss_fn(out_x, x)
        camids_KLD = self.caculate_kl(out['camids_means'], out['camids_log_var'], prior['camids_means'], prior['camids_log_var'])
        pids_KLD = self.caculate_kl(out['pids_means'], out['pids_log_var'], prior['pids_means'], prior['pids_log_var'])
        L_cvae = BCE / x.size(0) + camids_KLD / x.size(0) + pids_KLD / x.size(0)

        L_gsnn = mse_loss_fn(prior_x, x) / x.size(0)

        hybrid_loss = aphla*L_cvae + (1-aphla)*L_gsnn
        return hybrid_loss


class Prior_net(nn.Module):

    def __init__(self, camids, pids, latent_size, flag):
        super().__init__()

        self.flag = flag

        self.camids = camids
        self.pids = pids

        layers = []
        if self.flag == "camids":
            layers.append(nn.Linear(self.camids, 32))
        else:
            layers.append(nn.Linear(self.pids, 32))

        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(32, 64))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)

        self.linear_means = nn.Sequential(nn.Linear(128, latent_size))
        self.linear_log_var = nn.Sequential(nn.Linear(128, latent_size))

        self.layers.apply(weights_init_kaiming)
        self.linear_means.apply(weights_init_kaiming)
        self.linear_log_var.apply(weights_init_kaiming)

    def forward(self, c, p):
        if self.flag == "camids":
            c = label2onehot(c, self.camids)
            x = c
        else:
            p = label2onehot(p, self.pids)
            x = p
        x = self.layers(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Encoder(nn.Module):

    def __init__(self, camids, pids, feat_size, latent_size, flag):
        super().__init__()
        self.flag = flag

        self.camids = camids
        self.pids = pids

        layers = []
        if self.flag == "camids":
            layers.append(nn.Linear(feat_size + camids, 1024))
        else:
            layers.append(nn.Linear(feat_size + pids, 1024))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(1024, 512))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(512, 256))
        layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)

        self.linear_means = nn.Sequential(nn.Linear(256, latent_size))
        self.linear_log_var = nn.Sequential(nn.Linear(256, latent_size))

        self.layers.apply(weights_init_kaiming)
        self.linear_means.apply(weights_init_kaiming)
        self.linear_log_var.apply(weights_init_kaiming)

        self.means_bn = nn.BatchNorm1d(latent_size)
        self.means_bn.weight.requires_grad = False
        self.means_bn.weight.fill_(0.5)
        nn.init.constant_(self.means_bn.bias, 0.0)

    def forward(self, x, c, p):
        c = label2onehot(c, self.camids)
        p = label2onehot(p, self.pids)
        if self.flag == "camids":
            x = torch.cat((x, c), dim=-1)
        else:
            x = torch.cat((x, p), dim=-1)
        x = self.layers(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, camids, pids, feat_size, latent_size):
        super().__init__()
        self.camids = camids
        self.pids = pids

        layers = []
        layers.append(nn.Linear(latent_size + latent_size + camids + pids, 512))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(512, 1024))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(1024, 2048))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(2048, feat_size))
        self.layers = nn.Sequential(*layers)

        self.layers.apply(weights_init_kaiming)

    def forward(self, camids_z, pids_z, c, p):
        c = label2onehot(c, self.camids)
        p = label2onehot(p, self.pids)
        z = torch.cat((camids_z, pids_z, c, p), dim=-1)

        x = self.layers(z)

        return x
