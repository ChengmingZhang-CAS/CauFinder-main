# -*- coding: utf-8 -*-
"""
Created by Chengming Zhang, Mar 31st, 2023
"""
from typing import Dict, Iterable, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from .layers import (
    FeatureSplit,
    FeatureScaler,
    Encoder,
    Decoder,
    DynamicPhenotypeDescriptor,
)


# DCDVAE model
class DCDVAE(nn.Module):
    """
    Dual Causal Decoupled Variational Autoencoder with Feature Selection Module for Causal Inference
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_causal: int = 2,
        # n_controls: int = 10,
        # scale: float = 1.0,
        n_layers_encoder: int = 0,
        n_layers_decoder: int = 0,
        n_layers_dpd: int = 0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_dpd: float = 0.1,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_batch_norm_dpd: bool = True,
        init_weight=None,
        init_thresh: float = 0.2,
        attention: bool = False,
        att_mean: bool = False,
        pdp_linear: bool = True,
    ):
        super(DCDVAE, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_spurious = n_latent - n_causal
        # self.n_controls = n_controls
        # self.warm_up = 0
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.feature_mapper = FeatureSplit(
            self.n_input, init_weight=init_weight, init_thresh=init_thresh, thresh_grad=True,
            attention=attention, att_mean=att_mean
        )
        self.encoder1 = Encoder(
            self.n_input,
            self.n_causal,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder1 = Decoder(
            self.n_causal,
            self.n_input,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.encoder2 = Encoder(
            self.n_input,
            self.n_spurious,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder2 = Decoder(
            self.n_spurious,
            self.n_input,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.dpd_model = DynamicPhenotypeDescriptor(
            self.n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_dpd,
            n_layers=n_layers_dpd,
            batch_norm=use_batch_norm_dpd,
            linear=pdp_linear,
        )

    def forward(self, x, use_mean=False):
        """
        Forward pass through the whole network.
        """
        x1, feat_w = self.feature_mapper(x, mode="causal")
        latent1 = self.encoder1(x1)
        latent1["z"] = latent1["qz_m"] if use_mean else latent1["z"]
        x_rec1 = self.decoder1(latent1["z"])

        x2, _ = self.feature_mapper(x, mode="spurious")
        latent2 = self.encoder2(x2)
        latent2["z"] = latent2["qz_m"] if use_mean else latent2["z"]
        x_rec2 = self.decoder2(latent2["z"])

        z = torch.cat((latent1["z"], latent2["z"]), dim=1)
        org_dpd = self.dpd_model(z)

        alpha_z = torch.zeros_like(z)
        alpha_z[:, : self.n_causal] = latent1["z"]
        alpha_z[:, self.n_causal:] = latent2["z"].mean(dim=0, keepdim=True)
        alpha_dpd = self.dpd_model(alpha_z)

        return dict(
            latent1=latent1,
            latent2=latent2,
            x_rec1=x_rec1,
            x_rec2=x_rec2,
            feat_w=feat_w,
            org_dpd=org_dpd,
            alpha_dpd=alpha_dpd,
        )

    @staticmethod
    def compute_loss(model_outputs, x, y, imb_factor=None):
        # get model
        latent1, latent2, x_rec1, x_rec2, feat_w, org_dpd, alpha_dpd, = model_outputs.values()

        qz1_m, qz1_v = latent1["qz_m"], latent1["qz_v"]
        qz2_m, qz2_v = latent2["qz_m"], latent2["qz_v"]
        qz_m = torch.cat((qz1_m, qz2_m), dim=1)
        qz_v = torch.cat((qz1_v, qz2_v), dim=1)
        org_logit, org_prob = org_dpd["logit"], org_dpd["prob"]
        alpha_logit, alpha_prob = alpha_dpd["logit"], alpha_dpd["prob"]

        # input feature reconstruction loss
        feat_w = feat_w.mean(dim=0) if feat_w.dim() == 2 else feat_w
        full_rec_loss1 = F.mse_loss(x_rec1, x, reduction="none") * feat_w
        rec_loss1 = full_rec_loss1.mean(dim=1)
        full_rec_loss2 = F.mse_loss(x_rec2, x, reduction="none") * (1 - feat_w)
        rec_loss2 = full_rec_loss2.mean(dim=1)
        # latent kl divergence loss
        z_kl_loss = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        # feature weight l1 loss
        # feat_w = feat_w.mean(dim=0) if feat_w.dim() == 2 else feat_w
        feat_l1_loss = torch.sum(torch.abs(feat_w))
        # DPD binary classification loss
        # Calculate pos_weight if im_factor is not None
        if imb_factor is not None:
            num_pos = y.sum().item()
            num_neg = y.size(0) - num_pos
            if num_pos == 0 or num_neg == 0:
                pos_weight = torch.tensor(1.0, dtype=torch.float32, device=y.device)
            else:
                pos_weight = (num_neg / num_pos) * imb_factor
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=y.device)
        else:
            pos_weight = torch.tensor(1.0, dtype=torch.float32, device=y.device)
        # dpd_loss = F.binary_cross_entropy(org_prob.squeeze(), y, reduction="none")
        dpd_loss = F.binary_cross_entropy_with_logits(org_logit.squeeze(), y, pos_weight=pos_weight, reduction='none')

        # fidelity kl divergence loss
        alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        org_probs = torch.cat((org_prob, 1 - org_prob), dim=1)
        fide_kl_loss = F.kl_div(torch.log(alpha_probs + 1e-8), org_probs + 1e-8, reduction="none").sum(dim=1)

        # Save each loss to the dictionary to return
        loss_dict = dict(
            rec_loss1=rec_loss1,
            rec_loss2=rec_loss2,
            z_kl_loss=z_kl_loss,
            feat_l1_loss=feat_l1_loss,
            dpd_loss=dpd_loss,
            fide_kl_loss=fide_kl_loss,
        )

        return loss_dict

    @staticmethod
    def update_loss_weights(current_epoch, max_epochs, scheme=None):
        # Update loss weights based on current epoch and maximum number of epochs
        loss_weights = {}
        if scheme is None:
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 1,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.01,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.01,
                }
            elif current_epoch < max_epochs * 0.70:
                # Third quarter of training: emphasize BCE loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 1.0,
                    "fide_kl_loss": 0.01,
                    "causal_loss": 0.01,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss1": 0.5,
                    "rec_loss2": 0.5,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 2.0,
                    "causal_loss": 2.0,
                }
        elif scheme == "sim":
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.01,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.01,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.01,
                }
            elif current_epoch < max_epochs * 0.70:
                # Third quarter of training: emphasize KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 0.01,
                    "causal_loss": 0.01,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss1": 0.5,
                    "rec_loss2": 0.5,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 2.0,
                    "causal_loss": 3.0,
                }
        elif scheme == "lusa":
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.01,
                    "causal_loss": 0.01,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.01,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.01,
                    "causal_loss": 0.01,
                }
            elif current_epoch < max_epochs * 0.70:
                # Third quarter of training: emphasize KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 1.0,
                    "fide_kl_loss": 0.1,
                    "causal_loss": 0.1,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss1": 0.5,
                    "rec_loss2": 0.5,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 2.0,
                    "causal_loss": 2.0,
                }
        elif scheme == "pc9":
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.00,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.00,
                    "feat_l1_loss": 0.01,
                    "dpd_loss": 0.01,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.9:
                # Third quarter of training: emphasize KL loss
                loss_weights = {
                    "rec_loss1": 1.0,
                    "rec_loss2": 1.0,
                    "z_kl_loss": 0.0,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 1.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss1": 0.5,
                    "rec_loss2": 0.5,
                    "z_kl_loss": 0.10,
                    "feat_l1_loss": 0.00,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }

        return loss_weights


# DCVAE model
class DCVAE(nn.Module):
    """
    Decoupled Causal Variational Autoencoder with SHAP module.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: int = 128,
            n_latent: int = 10,
            n_causal: int = 2,
            # n_controls: int = 10,
            scale: float = 1.0,
            n_layers_encoder: int = 1,
            n_layers_decoder: int = 1,
            n_layers_dpd: int = 1,
            dropout_rate_encoder: float = 0.1,
            dropout_rate_decoder: float = 0.0,
            dropout_rate_dpd: float = 0.1,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            use_batch_norm_dpd: bool = True,
            pdp_linear: bool = True,
    ):
        super(DCVAE, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_causal = n_causal
        # self.n_controls = n_controls
        # self.warm_up = 0
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.feature_mapper = FeatureScaler(self.n_input)
        self.encoder = Encoder(
            self.n_input,
            self.n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder = Decoder(
            self.n_latent,
            self.n_input,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.dpd_model = DynamicPhenotypeDescriptor(
            self.n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_dpd,
            n_layers=n_layers_dpd,
            batch_norm=use_batch_norm_dpd,
            linear=pdp_linear,
        )

    def forward(self, x):
        """
        Forward pass through the whole network.
        """
        x_w, feat_w = self.feature_mapper(x)
        latent = self.encoder(x_w)
        x_rec = self.decoder(latent["z"])
        # org_dpd, org_prob = self.dpd_model(latent['z'])
        org_dpd = self.dpd_model(latent["z"])

        alpha_z = torch.zeros_like(latent["z"])
        alpha_z[:, : self.n_causal] = latent["z"][:, : self.n_causal]
        alpha_z[:, self.n_causal:] = latent["z"][:, self.n_causal:].mean(
            dim=0, keepdim=True
        )
        # alpha_dpd, alpha_prob = self.dpd_model(alpha_z)
        alpha_dpd = self.dpd_model(alpha_z)

        # x_all, feat_w = self.feature_selector(x, keep_top=True, keep_not_top=True)
        # # x_all, feat_w = self.feature_selector(x, keep_top=True, keep_not_top=False)
        # latent_all = self.encoder(x_all*self.scale)
        # x_all_rec = self.decoder(latent_all['z'])
        return dict(
            latent=latent,
            x_rec=x_rec,
            feat_w=feat_w,
            org_dpd=org_dpd,
            alpha_dpd=alpha_dpd,
        )

    @staticmethod
    def compute_loss(model_outputs, x, y):
        # get model
        latent, x_rec, feat_w, org_dpd, alpha_dpd = model_outputs.values()
        qz_m, qz_v = latent["qz_m"], latent["qz_v"]
        org_logit, org_prob = org_dpd["logit"], org_dpd["prob"]
        alpha_logit, alpha_prob = alpha_dpd["logit"], alpha_dpd["prob"]

        # input feature reconstruction loss
        rec_loss = F.mse_loss(x_rec, x, reduction="none").mean(dim=1)
        # latent kl divergence loss
        z_kl_loss = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        # feature weight l1 loss
        feat_l1_loss = torch.sum(torch.abs(feat_w))
        # DPD binary classification loss
        dpd_loss = F.binary_cross_entropy(org_prob.squeeze(), y, reduction="none")
        # dpd_loss = F.binary_cross_entropy_with_logits(org_dpd.squeeze(), y, reduction='none')

        # fidelity kl divergence loss
        alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        org_probs = torch.cat((org_prob, 1 - org_prob), dim=1)
        fide_kl_loss = F.kl_div(
            torch.log(alpha_probs + 1e-8), org_probs, reduction="none"
        ).sum(dim=1)

        # Save each loss to the dictionary to return
        loss_dict = dict(
            rec_loss=rec_loss,
            z_kl_loss=z_kl_loss,
            feat_l1_loss=feat_l1_loss,
            dpd_loss=dpd_loss,
            fide_kl_loss=fide_kl_loss,
        )

        return loss_dict

    @staticmethod
    def update_loss_weights(current_epoch, max_epochs, scheme=None):
        # Update loss weights based on current epoch and maximum number of epochs
        loss_weights = {}
        if scheme is None:
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss": 2.0,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 0.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss": 1.0,
                    "z_kl_loss": 0.05,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 0.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.70:
                # Third quarter of training: emphasize KL loss
                loss_weights = {
                    "rec_loss": 0.5,
                    "z_kl_loss": 0.1,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 0.1,
                    "causal_loss": 0.1,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss": 0.2,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 2.0,
                    "causal_loss": 2.0,
                }
        elif scheme == "sc":
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {
                    "rec_loss": 2.0,
                    "z_kl_loss": 0.01,
                    "feat_l1_loss": 1.0,
                    "dpd_loss": 0.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {
                    "rec_loss": 1.0,
                    "z_kl_loss": 0.5,
                    "feat_l1_loss": 0.5,
                    "dpd_loss": 0.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            elif current_epoch < max_epochs * 0.70:
                # Third quarter of training: emphasize KL loss
                loss_weights = {
                    "rec_loss": 0.5,
                    "z_kl_loss": 0.2,
                    "feat_l1_loss": 0.2,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 0.0,
                    "causal_loss": 0.0,
                }
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {
                    "rec_loss": 0.2,
                    "z_kl_loss": 0.1,
                    "feat_l1_loss": 0.1,
                    "dpd_loss": 2.0,
                    "fide_kl_loss": 1.0,
                    "causal_loss": 2.0,
                }

        return loss_weights
