import logging
import os.path
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Literal,
)

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from anndata import AnnData
import scanpy as sc
from tqdm import tqdm
from torch.distributions import Normal, Poisson
from scipy.linalg import norm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import shap
import copy
import networkx as nx

from CauFinder.dataloader import data_splitter, batch_sampler
from CauFinder.models import DCDVAE, DCVAE
from CauFinder.causal_effect import (
    joint_uncond_v1,
    joint_uncond_single_dim_v1,
    beta_info_flow_v1,
)
from CauFinder.causal_effect import (
    joint_uncond_v2,
    joint_uncond_single_dim_v2,
    beta_info_flow_v2,
)
from CauFinder.driver_regulators import driver_regulators
from CauFinder.utils import set_seed, prepare_network, get_network, get_influence_score
from CauFinder.utils import plot_feature_boxplots


class CausalFinder(nn.Module):
    """
    Causal Disentanglement Model with Feature Selection Layer.
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            n_state: int = 2,  # Number of states (set to 2 for binary or 0-1 continuous states)
            # n_controls: int = 10,  # Number of upstream features
            **model_kwargs,
    ):
        super(CausalFinder, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_state = n_state
        # self.n_controls = n_controls
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = DCDVAE(
            n_input=adata.X.shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_state=n_state,
            # n_controls=n_controls,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            rec_loss1_weight: float = 1.0,
            rec_loss2_weight: float = 1.0,
            z_kl_weight: float = 0.1,
            feat_l1_weight: float = 0.01,
            dpd_weight: float = 2.0,
            fide_kl_weight: float = 0.5,
            causal_weight: float = 1.0,
            spurious_fold: float = 2.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            imb_factor: Optional[float] = None,
            fix_split: bool = False,
            drop_last: bool = False,
            **kwargs,
    ):
        """
        Trains the model using fractal variational autoencoder.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        set_seed(42) if fix_split else None
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            "N_alpha": n_alpha,
            "N_beta": n_beta,
            "K": self.n_causal,
            "L": self.n_latent - self.n_causal,
            "z_dim": self.n_latent,
            "M": self.n_state,
        }
        self.ce_params = ce_params
        loss_weights = {
            "rec_loss1": rec_loss1_weight,
            "rec_loss2": rec_loss2_weight,
            "z_kl_loss": z_kl_weight,
            "feat_l1_loss": feat_l1_weight,
            "dpd_loss": dpd_weight,
            "fide_kl_loss": fide_kl_weight,
            "causal_loss": causal_weight,
        }
        self.batch_size = batch_size
        # params_w = list(self.module.feature_selector.parameters())
        # params_net = list(self.module.encoder.parameters()) + list(self.module.decoder.parameters()) + list(
        #     self.module.dpd_model.parameters())
        optimizer = optim.Adam(
            self.module.parameters(), lr=lr, weight_decay=weight_decay
        )
        # optimizer1 = optim.Adam(params_w, lr=lr, weight_decay=weight_decay)
        # optimizer2 = optim.Adam(params_net, lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_losses = {
            "total_loss": [],
            "rec_loss1": [],
            "rec_loss2": [],
            "z_kl_loss": [],
            "feat_l1_loss": [],
            "dpd_loss": [],
            "fide_kl_loss": [],
            "causal_loss": [],
        }
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            # # print('Epoch: ', epoch)
            # if epoch == 4:
            #     print('stop')
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True, drop_last=drop_last)
            batch_losses = {
                "total_loss": [],
                "rec_loss1": [],
                "rec_loss2": [],
                "z_kl_loss": [],
                "feat_l1_loss": [],
                "dpd_loss": [],
                "fide_kl_loss": [],
                "causal_loss": [],
            }
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(
                    epoch, max_epochs, scheme=weight_scheme
                )
            for train_batch in train_adata_batch:
                inputs = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                labels = torch.tensor(
                    train_batch.obs["labels"], dtype=torch.float32, device=device
                )
                model_outputs = self.module(inputs)
                loss_dict = self.module.compute_loss(model_outputs, inputs, labels, imb_factor=imb_factor)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    if loss_weights["causal_loss"] == 0:
                        causal_loss_list = [torch.tensor(0.0, device=device)]
                        break
                    # _causal_loss1, _ = joint_uncond(ce_params, self.module, inputs, idx, device=device)
                    # _causal_loss2, _ = beta_info_flow(ce_params, self.module, inputs, idx, device=device)
                    _causal_loss1, _ = joint_uncond_v1(ce_params, self.module, inputs, idx, alpha_vi=True,
                                                       beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow_v1(ce_params, self.module, inputs, idx, alpha_vi=True,
                                                         beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 - spurious_fold * _causal_loss2
                    causal_loss_list += [_causal_loss]
                rec_loss1 = loss_dict["rec_loss1"].mean()
                rec_loss2 = loss_dict["rec_loss2"].mean()
                z_kl_loss = loss_dict["z_kl_loss"].mean()
                feat_l1_loss = loss_dict["feat_l1_loss"].mean()
                dpd_loss = loss_dict["dpd_loss"].mean()
                fide_kl_loss = loss_dict["fide_kl_loss"].mean()
                causal_loss = torch.stack(causal_loss_list).mean()
                if self.module.feature_mapper.attention:
                    loss_weights["feat_l1_loss"] = 0.001
                total_loss = (
                        loss_weights["rec_loss1"] * rec_loss1
                        + loss_weights["rec_loss2"] * rec_loss2
                        + loss_weights["z_kl_loss"] * z_kl_loss
                        + loss_weights["feat_l1_loss"] * feat_l1_loss
                        + loss_weights["dpd_loss"] * dpd_loss
                        + loss_weights["fide_kl_loss"] * fide_kl_loss
                        + loss_weights["causal_loss"] * causal_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses["total_loss"].append(total_loss.item())
                batch_losses["rec_loss1"].append(rec_loss1.item())
                batch_losses["rec_loss2"].append(rec_loss2.item())
                batch_losses["z_kl_loss"].append(z_kl_loss.item())
                batch_losses["feat_l1_loss"].append(feat_l1_loss.item())
                batch_losses["dpd_loss"].append(dpd_loss.item())
                batch_losses["fide_kl_loss"].append(fide_kl_loss.item())
                batch_losses["causal_loss"].append(causal_loss.item())

            # scheduler.step()
            # update epochs losses
            epoch_losses["total_loss"].append(np.mean(batch_losses["total_loss"]))
            epoch_losses["rec_loss1"].append(np.mean(batch_losses["rec_loss1"]))
            epoch_losses["rec_loss2"].append(np.mean(batch_losses["rec_loss2"]))
            epoch_losses["z_kl_loss"].append(np.mean(batch_losses["z_kl_loss"]))
            epoch_losses["feat_l1_loss"].append(np.mean(batch_losses["feat_l1_loss"]))
            epoch_losses["dpd_loss"].append(np.mean(batch_losses["dpd_loss"]))
            epoch_losses["fide_kl_loss"].append(np.mean(batch_losses["fide_kl_loss"]))
            epoch_losses["causal_loss"].append(np.mean(batch_losses["causal_loss"]))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses["total_loss"])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def pretrain_attention(
            self,
            prior_probs: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = 50,
            pretrain_lr: float = 1e-3,
            batch_size: int = 128,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None
    ):
        """
        Pretrain attention network.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, _ = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )

        if prior_probs is None:
            prior_probs = np.ones(self.module.feature_mapper_up.n_features) * 0.5
        elif not isinstance(prior_probs, np.ndarray):
            prior_probs = np.array(prior_probs)

        prior_probs_tensor = torch.tensor(prior_probs, dtype=torch.float32).view(1, -1).to(device)

        criterion = torch.nn.MSELoss()
        pretrain_optimizer = torch.optim.Adam(self.module.feature_mapper.att_net.parameters(), lr=pretrain_lr)

        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="pretraining", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            loss = None
            for train_batch in train_adata_batch:
                inputs = torch.tensor(train_batch.X, dtype=torch.float32, device=device)

                attention_scores = self.module.feature_mapper.att_net(inputs)
                # Repeat prior_probs_tensor to match the batch size
                repeated_prior_probs = prior_probs_tensor.repeat(attention_scores.size(0), 1)

                loss = criterion(torch.sigmoid(attention_scores), repeated_prior_probs)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
            if epoch % 10 == 0 or epoch == (max_epochs - 1):
                logging.info(f"Epoch {epoch} pretraining loss: {loss.item():.4f}")

        print("Pretraining attention net completed.")

    def plot_train_losses(self, fig_size=(8, 8), save_path=None):
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 3, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "train_losses.png"), format="png")
            plt.savefig(os.path.join(save_path, "train_losses.pdf"), format="pdf")
        # show figure
        # plt.show()
        plt.close()

    def get_feature_weights(
            self,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            grad_source: Optional[str] = "prob",
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = True,
    ):
        r"""
        Return the weights of features.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata_batch = batch_sampler(self.adata, self.batch_size, shuffle=False)

        def compute_shap_weights(key="prob"):
            # key = "prob" or "logit"
            shap_weights_full = []
            idx = np.random.permutation(self.adata.shape[0])[0:n_bg_samples]
            background_data = torch.tensor(self.adata.X[idx], dtype=torch.float32)
            background_data = background_data.to(device)

            model = ShapModel(self.module, key).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            for data in adata_batch:
                inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                shap_value = explainer.shap_values(inputs)
                shap_weights_full.append(shap_value)

            return np.concatenate(shap_weights_full, axis=0)

        def compute_grad_weights(grad_source="prob"):
            grad_weights_full = []
            for data in adata_batch:
                inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                labels = torch.tensor(data.obs["labels"], dtype=torch.float32, device=device)
                inputs.requires_grad = True
                model_outputs = self.module(inputs, use_mean=True)
                prob = model_outputs["alpha_dpd"]["prob"]
                logit = model_outputs["alpha_dpd"]["logit"]
                if grad_source == "loss":
                    loss_dict = self.module.compute_loss(model_outputs, inputs, labels)
                    dpd_loss = loss_dict["dpd_loss"]
                    dpd_loss.sum().backward()
                elif grad_source == "prob":
                    prob.sum().backward()
                elif grad_source == "logit":
                    logit.sum().backward()
                grad_weights_full.append(inputs.grad.cpu().numpy())

            return np.concatenate(grad_weights_full, axis=0)

        def compute_model_weights():
            if self.module.feature_mapper.attention:
                attention_weights_full = []
                for data in adata_batch:
                    inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                    model_outputs = self.module(inputs, use_mean=True)
                    att_w = model_outputs["feat_w"].cpu().detach().numpy()
                    attention_weights_full.append(att_w)
                weight_matrix = np.concatenate(attention_weights_full, axis=0)
            else:
                weight_vector = torch.sigmoid(self.module.feature_mapper.weight).cpu().detach().numpy()
                # Expand weight vector to a matrix with the same weight vector repeated for each sample in adata_batch
                weight_matrix = np.tile(weight_vector, (len(self.adata), 1))
            return weight_matrix

        weights_full = None
        if method == "Model":
            weights_full = compute_model_weights()
        elif method == "SHAP":
            weights_full = compute_shap_weights()
        elif method == "Grad":
            weights_full = compute_grad_weights(grad_source=grad_source)
        elif method == "Ensemble":
            model_weights = np.abs(compute_model_weights())
            shap_weights = np.abs(compute_shap_weights())
            grad_weights = np.abs(compute_grad_weights())

            # Normalize each set of weights
            model_sum = np.sum(model_weights, axis=1, keepdims=True)
            model_weights = np.where(model_sum != 0, model_weights / model_sum, 0)

            shap_sum = np.sum(shap_weights, axis=1, keepdims=True)
            shap_weights = np.where(shap_sum != 0, shap_weights / shap_sum, 0)

            grad_sum = np.sum(grad_weights, axis=1, keepdims=True)
            grad_weights = np.where(grad_sum != 0, grad_weights / grad_sum, 0)

            # Combine the weights
            weights_full = (model_weights + shap_weights + grad_weights) / 3

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)
        # weights = np.abs(np.mean(weights_full, axis=0))

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)

        # Create a new DataFrame with the weights
        weights_df = self.adata.var.copy()
        weights_df["weight"] = weights
        weights_df["weight_dir"] = np.mean(weights_full, axis=0)

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by="weight", ascending=False)

        # Ensure 2D before creating DataFrame
        weights_full = np.squeeze(weights_full)
        weights_full = pd.DataFrame(weights_full, index=self.adata.obs_names, columns=self.adata.var_names)

        return weights_df, weights_full

    def get_class_weights(self, weights_full, normalize=True, sort_by_weight=True):
        # Get class labels from adata.obs
        labels = self.adata.obs['labels']

        # Initialize DataFrames
        weights_class_0 = self.adata.var.copy()
        weights_class_1 = self.adata.var.copy()

        # Calculate weights for class 0
        class_0_mask = labels == 0
        weights_class_0['weight'] = weights_full.loc[class_0_mask].abs().mean(axis=0)  # Absolute mean
        weights_class_0['weight_dir'] = weights_full.loc[class_0_mask].mean(axis=0)  # Direct mean with direction

        # Calculate weights for class 1
        class_1_mask = labels == 1
        weights_class_1['weight'] = weights_full.loc[class_1_mask].abs().mean(axis=0)  # Absolute mean
        weights_class_1['weight_dir'] = weights_full.loc[class_1_mask].mean(axis=0)  # Direct mean with direction

        # Normalize the weights if required
        if normalize:
            weights_class_0['weight'] = weights_class_0['weight'] / weights_class_0['weight'].sum()
            weights_class_1['weight'] = weights_class_1['weight'] / weights_class_1['weight'].sum()
            # Note: We do not normalize 'weight_dir' as it contains directional information

        # Sort the DataFrame by absolute weight if required
        if sort_by_weight:
            weights_class_0 = weights_class_0.sort_values(by="weight", ascending=False)
            weights_class_1 = weights_class_1.sort_values(by="weight", ascending=False)

        # Return the results
        return weights_class_0, weights_class_1

    def get_driver_info(self, method, prior_network, save_path, topK=50, corr_cutoff=0.6, weight_degree=True, normalize=True, fig_name=None):
        """
        Get driver information for a given method (e.g., "SHAP" or "Grad") and plot weight boxplot.
        """
        # Generate the figure name
        fig_path = None
        if fig_name is not None:
            fig_path = os.path.join(save_path, fig_name)

        # Get feature weights for total and each class
        weight_total, weight_full = self.get_feature_weights(normalize=normalize, sort_by_weight=True, method=method)
        weight_0, weight_1 = self.get_class_weights(weight_full, normalize=normalize, sort_by_weight=True)

        # Identify overall drivers from the total combined classes
        driver_df = self.network_master_regulators(prior_network, weight_total, topK=topK, corr_cutoff=corr_cutoff, weight_degree=weight_degree, out_lam=1.0,
                                                   ILP_lam=0.5)
        driver_df = driver_df[driver_df['is_CauFVS_driver']]
        driver_total = driver_df.index.tolist()

        # Plot feature weight boxplots
        driver_weight = weight_full.loc[:, driver_df.index]
        plot_feature_boxplots(driver_weight, self.adata.obs['labels'], save_path=fig_path)

        # Identify class-specific drivers
        driver_0, driver_1 = [], []
        for weight, driver_list in zip([weight_0, weight_1], [driver_0, driver_1]):
            driver_df = self.network_master_regulators(prior_network, weight, corr_cutoff=corr_cutoff, out_lam=1.0, ILP_lam=0.5)
            driver_df = driver_df[driver_df['is_CauFVS_driver']]
            driver_list.extend(driver_df.index.tolist())

        return driver_total, driver_0, driver_1, weight_total, weight_0, weight_1


    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        probs = []
        preds = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)
            latent_z = torch.cat([model_outputs["latent1"]["z"], model_outputs["latent2"]["z"]], dim=1)
            latent.append(latent_z.cpu().numpy())
            logits.append(model_outputs["alpha_dpd"]["logit"].cpu().numpy())
            probs.append(model_outputs["alpha_dpd"]["prob"].cpu().numpy())
            preds.append(
                (model_outputs["alpha_dpd"]["prob"].cpu().numpy() > 0.5).astype(int)
            )

        output = dict(
            latent=np.concatenate(latent, axis=0),
            logits=np.concatenate(logits, axis=0),
            probs=np.concatenate(probs, axis=0),
            preds=np.concatenate(preds, axis=0),
        )

        return output

    @torch.no_grad()
    def plot_pca_with_probs(
            self,
            adata: Optional[AnnData] = None,
            color_by: str = 'condition',
            save_dir: Optional[str] = None,
            elev: int = 10,
            azim: int = 60,
    ):
        """
        Perform PCA on the adata and plot a 3D scatter plot with the PCA results
        and the provided probabilities.

        Parameters:
        - adata: Optional AnnData object containing the dataset. Defaults to self.adata if not provided.
        - color_by: Choose to color by 'condition' or 'labels'.
        - save_dir: Optional directory to save the plot as PDF and PNG.
        - elev: Elevation angle in the z plane.
        - azim: Azimuth angle in the x-y plane.
        """
        # Use the provided adata or default to self.adata
        adata = adata if adata is not None else self.adata

        # Get probabilities from the model
        probs = adata.obs['probs'] if 'probs' in adata.obs.columns else None
        if 'probs' not in adata.obs.columns:
            model_output = self.get_model_output(adata=adata)
            probs = model_output['probs']

        # Check if PCA has already been performed
        if 'X_pca' not in adata.obsm.keys():
            sc.tl.pca(adata, svd_solver='arpack')

        # Extract PCA components for the first two dimensions
        x = adata.obsm['X_pca'][:, 0]
        y = adata.obsm['X_pca'][:, 1]
        z = probs

        # Choose the color scheme
        if color_by == 'condition' and 'condition' in adata.obs.columns:
            color = adata.obs['condition']
        elif color_by == 'labels' and 'labels' in adata.obs.columns:
            color = adata.obs['labels']
        else:
            raise ValueError("Invalid color_by value or missing condition/labels data.")

        # Convert categorical labels to numbers
        if color.dtype == 'object' or isinstance(color, pd.Categorical):
            le = LabelEncoder()
            color = le.fit_transform(color)
            categories = le.classes_
        else:
            categories = sorted(set(color))

        # Apply additional stylistic adjustments for better aesthetics
        mpl.rcParams['axes.linewidth'] = 0.8  # Thinner axes
        mpl.rcParams['xtick.direction'] = 'in'  # Ticks pointing in
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.major.size'] = 5  # Tick length
        mpl.rcParams['ytick.major.size'] = 5

        # Create the 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with customized appearance using Paired colormap
        scatter = ax.scatter(x, y, z, c=color, cmap='Paired', s=30, edgecolor='k', linewidth=0.5, alpha=0.8)

        # Add legend for the categories
        handles, _ = scatter.legend_elements(prop="colors")
        ax.legend(handles, categories, title=color_by.capitalize(), loc='best')

        # Set axis labels with proper padding and size
        ax.set_xlabel('PCA 1', fontsize=14, labelpad=12)
        ax.set_ylabel('PCA 2', fontsize=14, labelpad=12)
        ax.set_zlabel('Probs', fontsize=14, labelpad=12)

        # Ensure z-axis limits match the data range
        ax.set_zlim(0, 1)

        # Adjust the view angle
        ax.view_init(elev=elev, azim=azim)

        # Save the plot as PDF and PNG if a save directory is provided
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pdf_path = os.path.join(save_dir, 'plot_pca_with_probs.pdf')
            png_path = os.path.join(save_dir, 'plot_pca_with_probs.png')
            plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

        # Display the plot
        plt.show()

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            plot_info_flow: Optional[bool] = True,
            save_fig: Optional[bool] = False,
            save_dir: Optional[str] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        self.module.eval() if self.module.training else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata.copy()
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow for each dimension
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim_v1(
                    ce_params, self.module, inputs, i, j, device=device
                )
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        info_flow = info_flow.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        # Calculate information flow for causal and spurious dimensions
        dims = ['causal', 'spurious']
        info_flow_cat = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            # Get the latent space of the current sample
            inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # Calculate the information flow
            info_c, _ = joint_uncond_v1(ce_params, self.module, inputs, i, alpha_vi=False, beta_vi=True, device=device)
            info_s, _ = beta_info_flow_v1(ce_params, self.module, inputs, i, alpha_vi=True, beta_vi=False,
                                          device=device)
            info_flow_cat.loc[i, 'causal'] = -info_c.item()
            info_flow_cat.loc[i, 'spurious'] = -info_s.item()
        info_flow_cat.set_index(adata.obs_names, inplace=True)
        info_flow_cat = info_flow_cat.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        # Set figures style
        sns.set(style="whitegrid")
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

        if plot_info_flow:
            # Plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette="muted")
            plt.xlabel("Dimensions", fontsize=14)
            plt.ylabel("Information Measurements", fontsize=14)
            plt.title("Information Flow across Dimensions", fontsize=16)
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow.png"), bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, "info_flow.pdf"), bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(5, 5))
            ax = sns.boxplot(data=info_flow_cat, palette="muted")
            plt.xlabel("Dimensions", fontsize=14)
            plt.ylabel("Information Measurements", fontsize=14)
            plt.title("Categorized Information Flow", fontsize=16)
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_cat.png"), bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, "info_flow_cat.pdf"), bbox_inches='tight')
            plt.show()

        return info_flow, info_flow_cat

    def network_master_regulators(self, prior_network, gene_info, adata=None, topK=50, add_edges_pct=0.01, corr_cutoff=0.6,
                                  uppercase=False, net_degree=10, weight_degree=True, save_net_path=None, out_lam=0.8, driver_union=True,
                                  ILP_lam=0.5, **kwargs):
        """
        Obtain master regulators from the constructed prior network.
        """
        adata = adata if adata is not None else self.adata
        network = prepare_network(adata, prior_network, add_edges_pct=add_edges_pct, corr_cutoff=corr_cutoff, uppercase=uppercase)
        network = get_network(adata, network, average_degree=net_degree, weight_degree=weight_degree, save_path=save_net_path)
        gene_score = get_influence_score(network, gene_info, lam=out_lam)
        driver_regulator = driver_regulators(network, gene_score, topK=topK, driver_union=driver_union, solver='GUROBI', lam=ILP_lam, **kwargs)

        return driver_regulator

    def perform_state_transition(
            self,
            adata=None,
            causal_features=None,
            causal_idx=None,  # Causal feature indices
            grad_source="prob",  # gradient source
            lr=0.01,  # learning rate
            max_iter=200,  # number of iterations
            min_iter=10,  # minimum number of iterations
            optimizer_type="Adam",  # optimizer type
            save_step=1,  # interval for saving the data
            stop_thresh=1e-8,  # early stopping threshold
            control_direction="increase",  # control direction
            num_sampling=1000,  # number of sampling
            verbose=False,  # print training process
    ):
        self.module.eval() if self.module.training else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata.copy() if adata is not None else self.adata.copy()
        # Determine causal indices from causal features if provided
        if causal_features is not None:
            causal_idx = [adata.var_names.get_loc(feat) for feat in causal_features]
        elif causal_idx is None:
            causal_idx = list(range(adata.shape[1]))
            print("Warning: No causal features or indices provided. Using all features.")

        causal_update = {}
        causal_sampling = {}  # causal sampling
        control_details = pd.DataFrame()

        for i, sample in enumerate(adata.X):
            orig_causal_sample = sample[causal_idx].copy()  # Original causal features
            causal_sample = sample[causal_idx]
            sample_update = []
            initial_prob = None
            last_prob = None  # last prob
            print(f"Processing sample {i}, Target direction: {control_direction}")

            tensor_sample = torch.tensor(sample, dtype=torch.float32, device=device)
            causal_tensor = torch.tensor(causal_sample, dtype=torch.float32, device=device, requires_grad=True)

            # Initialize optimizer for causal_tensor
            if optimizer_type == "Adam":  # default
                optimizer = optim.Adam([causal_tensor], lr=lr)
            elif optimizer_type == "SGD":  # not recommended
                optimizer = optim.SGD([causal_tensor], lr=lr)
            elif optimizer_type == "RMSprop":  # adaptive learning rate
                optimizer = optim.RMSprop([causal_tensor], lr=lr)
            # elif optimizer_type == "Adagrad":  # sparse data
            #     optimizer = optim.Adagrad([causal_tensor], lr=lr)
            # elif optimizer_type == "AdamW":  # adam with weight decay
            #     optimizer = optim.AdamW([causal_tensor], lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            # =================== causal feature update ===================
            prob = None
            for iter in range(max_iter):
                optimizer.zero_grad()
                tensor_sample = tensor_sample.clone().detach()  # Clone and detach tensor_sample
                tensor_sample[causal_idx] = causal_tensor

                # forward propagation
                outputs = self.module(tensor_sample.unsqueeze(0), use_mean=True)
                prob = outputs["alpha_dpd"]["prob"]
                logit = outputs["alpha_dpd"]["logit"]
                current_prob = prob.item()

                # initial_prob
                if iter == 0:
                    initial_prob = current_prob
                else:
                    prob_change = current_prob - last_prob
                    if iter > min_iter and abs(prob_change) < stop_thresh:
                        print(f"Early stopping at iteration {iter} for sample {i}")
                        break
                last_prob = current_prob  # update last prob

                # backward propagation
                target = logit if grad_source == "logit" else prob
                target = -target if control_direction == "increase" else target
                target.backward()

                # update causal features
                optimizer.step()

                # save updated sample and probability
                if iter % save_step == 0:
                    x_delta = np.linalg.norm(causal_tensor.detach().cpu().numpy() - orig_causal_sample)
                    record = {'iteration': iter, 'prob': prob.item(), 'x_delta': x_delta}
                    if verbose:
                        print(record)
                    for feature_name, feature_value in zip(adata.var_names[causal_idx],
                                                           tensor_sample[causal_idx].detach().cpu().numpy()):
                        record[feature_name] = feature_value
                    sample_update.append(record)

            # Convert updates to DataFrame and store
            update_data = pd.DataFrame(sample_update)
            causal_update[i] = update_data

            # ==================== calculate controllability score ====================
            causal_delta = np.linalg.norm(orig_causal_sample - causal_tensor.detach().cpu().numpy())
            prob_delta = abs(prob.item() - initial_prob)
            score = prob_delta / (max(np.log(iter), 1) * causal_delta)
            control_item = {
                'sample_idx': int(i),
                'sample_name': adata.obs_names[i],  # sample name
                'score': score,
                'prob_delta': prob_delta,
                'causal_delta': causal_delta,
                'n_iter': iter,
            }
            control_item_df = pd.DataFrame.from_dict(control_item, orient='index').T
            control_details = pd.concat([control_details, control_item_df], ignore_index=True)

            # causal sampling for surface plot
            feature_columns = update_data.columns[3:]  # causal feature columns

            # Sampling from the causal feature space
            sampled_points = np.zeros((num_sampling, len(feature_columns)))

            for j, feature in enumerate(feature_columns):
                min_value = adata.X[:, causal_idx[j]].min()
                max_value = adata.X[:, causal_idx[j]].max()
                # min_value = update_data[feature].min()
                # max_value = update_data[feature].max()
                sampled_points[:, j] = np.random.uniform(low=min_value, high=max_value, size=num_sampling)

            # =================== sampling from the causal feature space ===================
            batch_samples = np.tile(sample, (num_sampling, 1))  # repeat the sample
            batch_samples[:, causal_idx] = sampled_points  # replace causal features

            # get the probability of the sampled points
            tensor_batch_samples = torch.tensor(batch_samples, dtype=torch.float32).to(device)
            outputs = self.module(tensor_batch_samples, use_mean=True)
            probs = outputs["alpha_dpd"]["prob"].detach().cpu().numpy()

            # concat sampled points and probability
            sampled_data = pd.DataFrame(sampled_points, columns=feature_columns)
            sampled_data['prob'] = probs
            causal_sampling[i] = sampled_data

        # save updated data and control score
        adata.uns["causal_update"] = causal_update
        adata.uns["causal_sampling"] = causal_sampling
        adata.uns["control_details"] = control_details
        adata.uns["control_direction"] = control_direction

        return adata

    def guided_state_transition(
            self,
            adata=None,
            causal_features=None,
            causal_idx=None,  # Causal feature indices
            # grad_source="prob",  # gradient source
            lr=0.01,  # learning rate
            max_iter=200,  # number of iterations
            min_iter=10,  # minimum number of iterations
            optimizer_type="Adam",  # optimizer type
            save_step=1,  # interval for saving the data
            stop_thresh=1e-8,  # early stopping threshold
            num_sampling=1000,  # number of sampling
            target_state=0,  # New: target state (0 or 1)
            lambda_reg=1e-4,  # New: regularization coefficient
            iter_norm=True,  # New: normalize the iteration number
            verbose=False,  # print training process
    ):
        self.module.eval() if self.module.training else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata.copy() if adata is not None else self.adata.copy()
        # Determine causal indices from causal features if provided
        if causal_features is not None:
            causal_idx = [adata.var_names.get_loc(feat) for feat in causal_features]
        elif causal_idx is None:
            causal_idx = list(range(adata.shape[1]))
            print("Warning: No causal features or indices provided. Using all features.")

        causal_update = {}
        causal_sampling = {}  # causal sampling
        control_details = pd.DataFrame()
        control_direction = 'increase' if target_state == 1 else 'decrease'
        for i, sample in enumerate(adata.X):
            orig_causal_sample = sample[causal_idx].copy()  # Original causal features
            causal_sample = sample[causal_idx]
            sample_update = []
            initial_prob = None
            last_prob = None  # last prob
            print(f"Processing sample {i}, Target direction: {control_direction}")

            tensor_sample = torch.tensor(sample, dtype=torch.float32, device=device)
            causal_tensor = torch.tensor(causal_sample, dtype=torch.float32, device=device, requires_grad=True)

            # Initialize optimizer for causal_tensor
            if optimizer_type == "Adam":  # default
                optimizer = optim.Adam([causal_tensor], lr=lr)
            elif optimizer_type == "SGD":  # not recommended
                optimizer = optim.SGD([causal_tensor], lr=lr)
            elif optimizer_type == "RMSprop":  # adaptive learning rate
                optimizer = optim.RMSprop([causal_tensor], lr=lr)
            # elif optimizer_type == "Adagrad":  # sparse data
            #     optimizer = optim.Adagrad([causal_tensor], lr=lr)
            # elif optimizer_type == "AdamW":  # adam with weight decay
            #     optimizer = optim.AdamW([causal_tensor], lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            # =================== causal feature update ===================
            prob = None
            iter = 0
            for iter in range(max_iter):
                optimizer.zero_grad()
                tensor_sample = tensor_sample.clone().detach()  # Clone and detach tensor_sample
                tensor_sample[causal_idx] = causal_tensor

                # forward propagation
                outputs = self.module(tensor_sample.unsqueeze(0), use_mean=True)
                prob = outputs["alpha_dpd"]["prob"]
                # logit = outputs["alpha_dpd"]["logit"]
                current_prob = prob.item()

                # initial_prob
                if iter == 0:
                    initial_prob = current_prob
                else:
                    prob_change = current_prob - last_prob
                    if iter > min_iter and abs(prob_change) < stop_thresh:
                        print(f"Early stopping at iteration {iter} for sample {i}")
                        break
                last_prob = current_prob  # update last prob

                # Calculate the new loss function based on the optimization target
                loss = (prob - target_state) ** 2
                if lambda_reg != 0:  # Add regularization if lambda_reg is not zero
                    x_delta = causal_tensor - torch.tensor(orig_causal_sample, dtype=torch.float32, device=device)
                    regularization_loss = lambda_reg * torch.sum(x_delta ** 2)
                    loss += regularization_loss

                # backward propagation
                loss.backward()

                # update causal features
                optimizer.step()

                # save updated sample and probability
                if iter % save_step == 0:
                    x_delta = np.linalg.norm(causal_tensor.detach().cpu().numpy() - orig_causal_sample)
                    record = {'iteration': iter, 'prob': prob.item(), 'x_delta': x_delta}
                    if verbose:
                        print(record)
                    for feature_name, feature_value in zip(adata.var_names[causal_idx],
                                                           tensor_sample[causal_idx].detach().cpu().numpy()):
                        record[feature_name] = feature_value
                    sample_update.append(record)

            # Convert updates to DataFrame and store
            update_data = pd.DataFrame(sample_update)
            causal_update[i] = update_data

            # ==================== calculate controllability score ====================
            causal_delta = np.linalg.norm(orig_causal_sample - causal_tensor.detach().cpu().numpy())
            prob_delta = abs(prob.item() - initial_prob)
            # score = prob_delta / (max(np.log(iter), 1) * causal_delta)
            if iter_norm:
                denominator = max(np.log(iter), 1) * causal_delta
            else:
                denominator = causal_delta
            score = prob_delta / denominator

            control_item = {
                'sample_idx': int(i),
                'sample_name': adata.obs_names[i],  # sample name
                'score': score,
                'prob_delta': prob_delta,
                'causal_delta': causal_delta,
                'n_iter': iter,
            }
            control_item_df = pd.DataFrame.from_dict(control_item, orient='index').T
            control_details = pd.concat([control_details, control_item_df], ignore_index=True)

            # causal sampling for surface plot
            feature_columns = update_data.columns[3:]  # causal feature columns

            # Sampling from the causal feature space
            sampled_points = np.zeros((num_sampling, len(feature_columns)))

            for j, feature in enumerate(feature_columns):
                min_value = adata.X[:, causal_idx[j]].min()
                max_value = adata.X[:, causal_idx[j]].max()
                # min_value = update_data[feature].min()
                # max_value = update_data[feature].max()
                sampled_points[:, j] = np.random.uniform(low=min_value, high=max_value, size=num_sampling)

            # =================== sampling from the causal feature space ===================
            batch_samples = np.tile(sample, (num_sampling, 1))  # repeat the sample
            batch_samples[:, causal_idx] = sampled_points  # replace causal features

            # get the probability of the sampled points
            tensor_batch_samples = torch.tensor(batch_samples, dtype=torch.float32).to(device)
            outputs = self.module(tensor_batch_samples, use_mean=True)
            probs = outputs["alpha_dpd"]["prob"].detach().cpu().numpy()

            # concat sampled points and probability
            sampled_data = pd.DataFrame(sampled_points, columns=feature_columns)
            sampled_data['prob'] = probs
            causal_sampling[i] = sampled_data

        # save updated data and control score
        adata.uns["causal_update"] = causal_update
        adata.uns["causal_sampling"] = causal_sampling
        adata.uns["control_details"] = control_details
        adata.uns["control_direction"] = control_direction

        return adata


class ShapModel(nn.Module):
    def __init__(self, original_model, key="prob"):
        super().__init__()
        self.original_model = original_model
        self.key = key

    def forward(self, x):
        model_outputs = self.original_model(x, use_mean=True)
        output = model_outputs["alpha_dpd"][self.key]
        # alpha_prob = model_outputs["alpha_dpd"]["prob"]
        return output
        # return alpha_dpd['logit']

    # def forward(self, x):
    #     feature_mapper = self.original_model.feature_mapper
    #     w = torch.sigmoid(feature_mapper.weight)
    #     # w = (w > self.thresh).float() * w  # set values below threshold to 0
    #     w = torch.where(w > feature_mapper.thresh, w, torch.zeros_like(w))
    #     # w = torch.relu(self.weight)
    #
    #     x1 = torch.mul(x, w)
    #     latent1 = self.original_model.encoder1(x1)
    #     # x_rec1 = self.original_model.decoder1(latent1['z'])
    #
    #     x2 = torch.mul(x, 1 - w)
    #     latent2 = self.original_model.encoder2(x2)
    #     # x_rec2 = self.original_model.decoder2(latent2['z'])
    #
    #     z = torch.cat((latent1["z"], latent2["z"]), dim=1)
    #     # org_dpd = self.dpd_model(z)
    #
    #     alpha_z = torch.zeros_like(z)
    #     alpha_z[:, : self.original_model.n_causal] = latent1["z"]
    #     alpha_dpd = self.original_model.dpd_model(alpha_z)
    #
    #     return alpha_dpd["prob"]
    #     # return alpha_dpd['logit']