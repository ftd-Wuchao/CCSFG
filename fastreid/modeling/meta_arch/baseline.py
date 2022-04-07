# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.modeling.self_module import *
from fastreid.utils.my_tools import *
from fastreid.utils.misc import *
from fastreid.utils.weight_init import *


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        # backbone
        self.backbone = ModelWrapper(build_backbone(cfg))
        # head
        self.heads = build_heads(cfg)

        self.vae = CVAE(cfg.DATASETS.NUM_CAMERAS, cfg.DATASETS.NUM_ID, cfg.MODEL.HEADS.IN_FEAT, cfg.MODEL.CVAE.Latent_dim)
        self.local_vae = CVAE(cfg.DATASETS.NUM_CAMERAS, cfg.DATASETS.NUM_ID, cfg.MODEL.Transformer.hidden_dim * cfg.MODEL.Transformer.num_patch, 64)

        # transformer
        self.transformer = build_transformer(cfg)
        # positional encoder
        self.positional_encoder = build_position_encoding(cfg)
        # transformer_related
        self.class_embed = nn.Linear(cfg.MODEL.Transformer.hidden_dim, cfg.MODEL.Transformer.num_patch)
        self.query_embed = nn.Embedding(cfg.MODEL.Transformer.num_patch, cfg.MODEL.Transformer.hidden_dim)
        self.input_proj = nn.Conv2d(cfg.MODEL.HEADS.IN_FEAT, cfg.MODEL.Transformer.hidden_dim, kernel_size=1)
        self.local_bns = nn.ModuleList(torch.nn.BatchNorm1d(cfg.MODEL.Transformer.hidden_dim, affine=False) for _ in range(cfg.MODEL.Transformer.num_patch)) #

        self.distance_loss = build_distanceloss()

        self.ccfa_weight = cfg.MODEL.LOSSES.CCFA.SCALE
        self.cls_weight = cfg.MODEL.LOSSES.ID.SCALE
        self.mcnl_weight = cfg.MODEL.LOSSES.MCNL.SCALE
        self.aphla = cfg.MODEL.CVAE.ALPHA

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, class2cid=None, iter=None):

        images = self.preprocess_image(batched_inputs)

        if isinstance(images, (List, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        features = self.backbone(images.tensors)

        if not self.training:
            output, local_feat_map = self.heads(features)

            return output, local_feat_map

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            pids = batched_inputs["targets"].to(self.device)
            camids = batched_inputs['camids'].to(self.device)

            outputs = self.heads(features, targets=pids, camids=camids)

            g_recon_x, g_camids_z, g_pids_z, g_prior, g_hybrid_loss = self.vae.forward(outputs['bn_feat'].detach(), camids, pids, self.aphla)
            fake_features = self.vae.inference(pids, g_prior)


            patch_features_anchor, outp_class_anchor = forward_transformer(images, outputs['local_feat_map'],
                                                                           self.positional_encoder, self.transformer,
                                                                           self.input_proj, self.query_embed,
                                                                           self.class_embed)

            local_feat_list = []
            local_bn_feat_list = []

            for i in range(patch_features_anchor.size(1)):
                local_feat_list.append(patch_features_anchor[:, i, :])
                local_bn_feat_list.append(self.local_bns[i](patch_features_anchor[:, i, :]))

            outputs['local_feat'] = torch.cat(local_feat_list, dim=1)
            outputs['local_bn_feat'] = torch.cat(local_bn_feat_list, dim=1)

            l_recon_x, l_camids_z, l_pids_z, l_prior, l_hybrid_loss = self.local_vae.forward(outputs['local_bn_feat'].detach(), camids, pids, self.aphla)
            l_fake_features = self.local_vae.inference(pids, l_prior)

            losses = self.losses(outputs, pids, camids, iter, g_hybrid_loss, fake_features, l_hybrid_loss, l_fake_features)
            return losses

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, pids, camids, iter, g_hybrid_loss, fake_features, l_hybrid_loss, l_fake_features):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """

        pred_class_logits = outputs['pred_class_logits'].detach()

        # Log prediction accuracy
        log_accuracy(pred_class_logits, pids)
        loss_dict = {}
        loss_dict['g_cls_loss'] = cross_entropy_loss(
            outputs['cls_outputs'], pids,
            self._cfg.MODEL.LOSSES.CE.EPSILON,
            self._cfg.MODEL.LOSSES.CE.ALPHA)*self.cls_weight

        loss_dict["mcnl-G_loss"] = self.distance_loss(outputs['features'], pids, camids)[0]*self.mcnl_weight
        loss_dict['g_cvae'] = g_hybrid_loss

        loss_dict["mcnl-L_loss"] = self.distance_loss(outputs['local_feat'], pids, camids)[0]*self.mcnl_weight
        loss_dict['l_cvae'] = l_hybrid_loss

        loss_dict["adaptive_l2"] = adaptive_l2_loss(self.backbone)
        if iter > 6000:
            loss_dict['gccfa_loss'] = ccfa_loss(outputs['bn_feat'], fake_features)*self.ccfa_weight
            loss_dict['lccfa_loss'] = ccfa_loss(outputs['local_bn_feat'], l_fake_features)*self.ccfa_weight

        return loss_dict

