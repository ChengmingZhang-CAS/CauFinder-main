import collections
from typing import Callable, Iterable, List, Optional, Literal

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList


class FeatureSplit(nn.Module):
    def __init__(self, n_features, init_weight=None, init_thresh=0.2, thresh_grad=True, attention=False, att_mean=False):
        super(FeatureSplit, self).__init__()
        self.n_features = n_features
        self.weight = torch.nn.Parameter(torch.zeros(n_features))
        self.thresh = nn.Parameter(torch.tensor(init_thresh), requires_grad=thresh_grad)
        self.attention = attention
        self.att_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(),
            nn.Linear(n_features, n_features),
            # nn.ReLU(),
            # nn.Linear(n_features, n_features)
        )
        self.att_mean = att_mean
        if init_weight is not None:
            assert len(init_weight) == n_features, "The length of initial_weight should be equal to n_features"
            self.weight.data = torch.tensor(init_weight, dtype=torch.float32)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.weight, 0.999999, 0.9999999)
        # nn.init.constant_(self.weight, 0.5)
        nn.init.constant_(self.weight, 0.0)
        # sorted_weight, sorted_idx = torch.sort(self.weight.data, descending=True)
        # print(sorted_weight)
        # print(sorted_idx)

    def forward(self, x, mode='causal'):
        if self.attention:
            attention_scores = self.att_net(x)
            w = torch.sigmoid(attention_scores)
            w = torch.where(w.gt(self.thresh), w, torch.zeros_like(w))
            w_used = torch.mean(w, dim=0) if self.att_mean else w
        else:
            # use model weight
            w = torch.sigmoid(self.weight)
            w = torch.where(w.gt(self.thresh), w, torch.zeros_like(w))
            w_used = w

        x_mask = None
        if mode not in ['causal', 'spurious']:
            raise ValueError("Mode must be one of 'causal' or 'spurious'")
        elif mode == 'causal':
            x_mask = torch.mul(x, w_used)
        elif mode == 'spurious':
            x_mask = torch.mul(x, 1 - w_used)

        return x_mask, w


class FeatureScaler(nn.Module):
    def __init__(self, n_features, initial_weight=None):
        super(FeatureScaler, self).__init__()
        self.n_features = n_features
        self.weight = torch.nn.Parameter(torch.ones(n_features), requires_grad=True)

        if initial_weight is not None:
            assert len(initial_weight) == n_features, "The length of initial_weight should be equal to n_features"
            self.weight.data = torch.tensor(initial_weight)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0.999999, 0.9999999)

    def forward(self, x):
        w = torch.relu(self.weight)
        # x = torch.mul(x, w)
        return x, w


class MLP(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation='relu', batch_norm=False, layer_norm=False,
                 dropout_rate=0.0, init_type='kaiming'):
        super(MLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = []
        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'none':
                    pass  # no activation
                else:
                    raise ValueError("Invalid activation option")

                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))

        self.layers = nn.Sequential(*layers)

        self.init_weights(init_type)

    def init_weights(self, init_type='kaiming'):
        if init_type is None:
            return
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


# Encoder
class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            var_eps: float = 1e-4,
            **kwargs,
    ):
        super(Encoder, self).__init__()

        self.var_eps = var_eps
        self.encoder = MLP(
            input_dim=n_input,
            output_dim=n_hidden,
            hidden_dims=[n_hidden] * n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # Parameters for latent distribution
        qz = self.encoder(x)
        qz_m = self.mean_encoder(qz)
        qz_v = torch.exp(self.log_var_encoder(qz)) + self.var_eps
        z = Normal(qz_m, torch.clamp(qz_v, max=5).sqrt()).rsample()  # torch.clamp(qz_v, max=10)
        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
        )


# Decoder
class Decoder(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.0,
            **kwargs,
    ):
        super(Decoder, self).__init__()

        self.decoder1 = MLP(
            input_dim=n_input,
            output_dim=n_hidden,
            hidden_dims=[n_hidden] * n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.decoder2 = nn.Linear(n_hidden, n_output)

    def forward(self, z):
        x_rec = self.decoder1(z)
        x_rec = self.decoder2(x_rec)
        return x_rec


class DynamicPhenotypeDescriptor(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_output: int = 1,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.0,
            linear: bool = False,
            **kwargs,
    ):
        super(DynamicPhenotypeDescriptor, self).__init__()
        self.linear = linear
        if self.linear:
            self.dpd = nn.Linear(n_input, n_output)
        else:
            self.dpd1 = MLP(
                input_dim=n_input,
                output_dim=n_hidden,
                hidden_dims=[n_hidden] * n_layers,
                dropout_rate=dropout_rate,
                **kwargs,
            )
            self.dpd2 = nn.Linear(n_hidden, n_output)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        if self.linear:
            logit = self.dpd(x)
        else:
            x = self.dpd1(x)
            logit = self.dpd2(x)
        prob = self.activation(logit)
        return dict(
            logit=logit,
            prob=prob,
        )


