from CauFinder.dataloader_crn import generate_synthetic
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from CauFinder.caufinder_main import CausalFinder
from CauFinder.utils import set_seed
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
import time
from itertools import combinations
from scipy.stats import norm
import math
from typing import List
import networkx as nx
from typing import List, Optional
import pickle
import scanpy as sc


# %% Run CauFinder_shap (our model)
def run_caufinder(
    adata,
    n_latent=10,
    n_hidden=64,
    n_layers_encoder=1,
    n_layers_decoder=0,
    n_layers_dpd=0,
    dropout_rate_encoder=0.0,
    dropout_rate_decoder=0.0,
    dropout_rate_dpd=0.0,
    use_batch_norm="none",
    use_batch_norm_dpd=True,
    pdp_linear=True,
):
    model = CausalFinder(
        adata=adata,
        # n_controls=n_controls,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers_encoder=n_layers_encoder,
        n_layers_decoder=n_layers_decoder,
        n_layers_dpd=n_layers_dpd,
        dropout_rate_encoder=dropout_rate_encoder,
        dropout_rate_decoder=dropout_rate_decoder,
        dropout_rate_dpd=dropout_rate_dpd,
        use_batch_norm=use_batch_norm,
        use_batch_norm_dpd=use_batch_norm_dpd,
        pdp_linear=pdp_linear,
    )
    model.train(max_epochs=300, stage_training=True)
    shap_df = cumulative_weight_sum_rate(
        model.get_feature_weights(sort_by_weight=True, method="SHAP")
    )
    grad_df = cumulative_weight_sum_rate(
        model.get_feature_weights(sort_by_weight=True, method="Grad")
    )
    both_df = cumulative_weight_sum_rate(
        model.get_feature_weights(sort_by_weight=True, method="Both")
    )
    return shap_df, grad_df, both_df


# %% Run CauFinder (our model)
def run_caufinder121(
    adata,
    n_latent=10,
    n_hidden=64,
    n_layers_encoder=1,
    n_layers_decoder=0,
    n_layers_dpd=0,
    dropout_rate_encoder=0.0,
    dropout_rate_decoder=0.0,
    dropout_rate_dpd=0.0,
    use_batch_norm="none",
    use_batch_norm_dpd=True,
    pdp_linear=True,
):
    import scanpy as sc
    adata_cf = adata.copy()
    adata_cf.var['label'] = adata_cf.obs['soft_label']
    # sc.pp.normalize_total(adata_cf)
    # sc.pp.log1p(adata_cf)
    from scipy.stats import ttest_ind
    from sklearn.preprocessing import MinMaxScaler
    group1 = adata_cf[adata_cf.obs["hard_label"] == 0].X
    group2 = adata_cf[adata_cf.obs["hard_label"] == 1].X

    p_values = np.array([ttest_ind(group1[:, i], group2[:, i], equal_var=False)[1] for i in range(adata_cf.shape[1])])
    p_values[p_values == 0] = np.min(p_values[p_values > 0])
    log_p_values = -np.log10(p_values)
    scaler = MinMaxScaler()
    scaled_weights = scaler.fit_transform(log_p_values.reshape(-1, 1)).flatten()

    init_weight = np.zeros_like(scaled_weights)
    init_weight[p_values < 1e-3] = scaled_weights[p_values < 1e-3]

    adata_cf.var["init_weight"] = init_weight

    model = CausalFinder(
        adata=adata_cf,
        # n_controls=n_controls,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers_encoder=n_layers_encoder,
        n_layers_decoder=n_layers_decoder,
        n_layers_dpd=n_layers_dpd,
        dropout_rate_encoder=dropout_rate_encoder,
        use_batch_norm_dpd=use_batch_norm_dpd,
        dropout_rate_decoder=dropout_rate_decoder,
        pdp_linear=pdp_linear,
        dropout_rate_dpd=dropout_rate_dpd,
        use_batch_norm=use_batch_norm,
        init_weight=init_weight,
    )
    # print(model)
    model.train(max_epochs=300, stage_training=True)
    # epoch_losses = model.history
    # weights = model.module.feature_mapper.weight
    # top_feat, top_weight, w = model.get_top_features(normalize=False)
    # w_df = cumulative_weight_sum_rate(model.get_feature_weights(sort_by_weight=True))
    # feature_names = [int(index.split('_')[1]) for index in w.index]
    # print("weight mean max min:", w.mean(), w.max(), w.min())
    # print(adata.var['feature_types'][top_feat])
    # model_res = adata.var['feature_types'][top_feat]
    # feature_names = [i for i in range(adata.X.shape[1])]
    # res = pd.DataFrame({'features': feature_names, 'weights': w.iloc[:, 1]})
    # return w_df
    shap_df = cumulative_weight_sum_rate(
        model.get_feature_weights(sort_by_weight=True, method="SHAP")[0]
    )
    grad_df = cumulative_weight_sum_rate(
        model.get_feature_weights(sort_by_weight=True, method="Model")[0]
    )
    # print(shap_df[shap_df['w_cum_rate'] < 0.3])
    # print(grad_df[grad_df['w_cum_rate'] < 0.3])
    # both_df = cumulative_weight_sum_rate(
    #     model.get_feature_weights(sort_by_weight=True, method="Both")[0]
    # )
    # model_df = cumulative_weight_sum_rate(
    #     model.get_feature_weights(sort_by_weight=True, method="Model")[0]
    # )
    return shap_df, grad_df


# %% Run t test
def run_t_test(
    X, y, var_df, threshold=0.05, p_adjust_method="fdr", sort_by_pvalue=True
):
    control_group = X[y == 0]
    case_group = X[y == 1]
    p_values = []

    for gene_index in range(X.shape[1]):
        gene_values_group_0 = control_group[:, gene_index]
        gene_values_group_1 = case_group[:, gene_index]
        t_statistic, p_value = stats.ttest_ind(gene_values_group_0, gene_values_group_1)
        p_values.append(p_value)

    # Multiple testing correction
    if p_adjust_method == "bonferroni":
        adjusted_p_values = np.array(p_values) * X.shape[1]
        adjusted_p_values = np.minimum(
            adjusted_p_values, 1
        )  # Ensure p-values do not exceed 1
    elif p_adjust_method == "fdr":
        _, adjusted_p_values, _, _ = multitest.multipletests(p_values, method="fdr_bh")
    else:
        adjusted_p_values = np.array(p_values)

    # Add p-value and adjusted p-value to var_df
    var_df["p_value"] = p_values
    var_df["adjusted_p_value"] = adjusted_p_values

    # Filter significant features based on adjusted p-values
    # var_df["pred_label"] = (adjusted_p_values < threshold).astype(int)

    # var_df["weight"] = 1 - var_df["adjusted_p_value"]
    var_df["weight"] = - np.log(var_df["adjusted_p_value"])

    # Sort the result by p-value if required
    if sort_by_pvalue:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)

    return var_df


# %% Run random forest classifier
def run_rf(
    X,
    y,
    var_df,
    sort_by_weight=True,
):
    # rf_model = RandomForestClassifier(n_estimators=50, max_depth=2)
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=2)
    rf_model.fit(X, y)
    imp = rf_model.feature_importances_
    var_df["weight"] = imp

    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)
    return var_df


# %% Run multi-layer perceptron
def run_mlp(X, y, var_df, n_hidden=32, n_layers=2, epoch=100, sort_by_weight=True):
    # %% data load
    n_features = X.shape[1]
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # %% FCN model

    # class Net(nn.Module):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.fc1 = nn.Linear(n_features, n_hidden)
    #         self.fc2 = nn.Linear(n_hidden, n_hidden)
    #         self.fc3 = nn.Linear(n_hidden, 1)
    #
    #     def forward(self, x):
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = F.sigmoid(self.fc3(x))
    #         return x

    class Net(nn.Module):
        def __init__(self, n_feature):
            super(Net, self).__init__()
            self.n_layers = n_layers
            self.layers = nn.ModuleList()

            for _ in range(n_layers):
                self.layers.append(nn.Linear(n_feature, n_hidden))
                n_feature = n_hidden

            self.output_layer = nn.Linear(n_hidden, 1)

        def forward(self, x):
            for i in range(self.n_layers):
                x = F.relu(self.layers[i](x))
            x = F.sigmoid(self.output_layer(x))
            return x

    # %% Instantiation
    model = Net(n_features)

    # loss & optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    model.train()
    losses = []
    for epoch in range(epoch):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # loss plt
    # plt.plot(losses)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    # %% feature importance
    model.eval()
    # SHAP
    num_samples = 50
    sample_indices = np.random.choice(features.shape[0], num_samples, replace=False)
    sampled_features = features[sample_indices]
    explainer = shap.DeepExplainer(model, sampled_features)
    shap_values = explainer.shap_values(features)
    # shap.summary_plot(shap_values, features.numpy())
    # feature_names = [i for i in range(X.shape[1])]
    features_importance = np.abs(shap_values).mean(axis=0)
    shap_df = var_df.copy()
    shap_df["weight"] = features_importance

    # Grad
    features.requires_grad = True
    output = model(features)
    loss = criterion(output, labels)
    loss.backward()
    grads = features.grad.abs()
    features_importance = grads.mean(dim=0)
    grad_df = var_df.copy()
    grad_df["weight"] = features_importance.detach().numpy()
    if sort_by_weight:
        shap_df = shap_df.sort_values(by="weight", ascending=False)
        shap_df = cumulative_weight_sum_rate(shap_df)
        grad_df = grad_df.sort_values(by="weight", ascending=False)
        grad_df = cumulative_weight_sum_rate(grad_df)

    return shap_df, grad_df


# %% Run mutual information
def run_mut_info(X, y, var_df, sort_by_weight=True):
    # mi_scores = mutual_info_classif(X, y)
    mi_scores = mutual_info_regression(X, y)
    var_df["weight"] = mi_scores
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)
    return var_df


# %% Run logistic regression
def run_log_reg(X, y, var_df, sort_by_weight=True):
    # logistic = LogisticRegression(penalty='l1', solver='liblinear', C=10)
    # logistic = LogisticRegression()
    logistic = LinearRegression()
    logistic.fit(X, y)
    importance = np.abs(logistic.coef_)[0, :]
    var_df["weight"] = importance
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)
    return var_df


# %% SVM
def run_SVM(X, y, var_df, sort_by_weight=True):
    svm_classifier = SVC(kernel="linear")
    svm_classifier.fit(X, y)
    importance = svm_classifier.coef_[0, :]
    var_df["weight"] = importance
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)
    return var_df


# %% Pearson correlation coefficient
def run_PearsonCor(X, y, var_df, sort_by_weight=True):
    correlations = []
    for feature in X.T:
        correlation = np.corrcoef(feature, y)[0, 1]
        correlations.append(abs(correlation))

    var_df["weight"] = correlations
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)

    return var_df


# %% Spearman correlation coefficient
def run_Spearman_Cor(X, y, var_df, sort_by_weight=True):
    correlations = []
    for feature in X.T:
        rho, _ = spearmanr(feature, y)
        correlations.append(abs(rho))

    var_df["weight"] = correlations
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)

    return var_df


# %% VAE
def run_VAE(X, y, var_df, n_hidden, n_latent, sort_by_weight=True):
    # %% data load
    n_features = X.shape[1]
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    class VAE(nn.Module):
        def __init__(self, num_features):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(num_features, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 2 * n_latent),
            )

            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, num_features),
                # nn.Sigmoid()
            )

            self.DPD = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid(),
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, x):
            mu_logvar = self.encoder(x)
            mu = mu_logvar[:, :n_latent]
            logvar = mu_logvar[:, n_latent:]
            z = self.reparameterize(mu, logvar)
            y = self.DPD(z)
            reconstructed = self.decoder(z)
            return reconstructed, y, mu, logvar

    model = VAE(n_features)
    recon_criterion = nn.MSELoss()
    # dpd_criterion = nn.BCELoss()
    dpd_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    model.train()
    losses = []
    re_losses = []
    kl_losses = []
    dpd_losses = []
    for epoch in range(200):
        for data, targets in dataloader:
            optimizer.zero_grad()
            recon_batch, y_dpd, mu, logvar = model(data)
            # reconstructed loss
            re_loss = recon_criterion(recon_batch, data)
            re_losses.append(re_loss.item())

            # kl loss
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.shape[0]
            )
            kl_losses.append(kl_loss.item())

            # dpd loss
            dpd_loss = dpd_criterion(y_dpd, targets)
            dpd_losses.append(dpd_loss.item())

            # total loss
            if epoch <= 100:
                loss = re_loss + kl_loss * 0.1 + dpd_loss * 0.1
            else:
                loss = re_loss + kl_loss * 0.1 + dpd_loss * 0.1

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    #         if (epoch+1) % 10 == 0:
    #             # print(outputs[:10, :], targets[:10, :])
    #             print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    #             # print(y_dpd, targets)
    #
    # print("Training finished!")
    # # loss plt
    # plt.plot(losses)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('total loss')
    # plt.show()
    #
    # plt.plot(re_losses)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('reconstructed loss')
    # plt.show()
    #
    # plt.plot(kl_losses)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('kl loss')
    # plt.show()
    #
    # plt.plot(dpd_losses)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('dpd loss')
    # plt.show()

    # %% feature importance
    model.eval()

    class ShapModel(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model

        def forward(self, x):
            _, y_prob, _, _ = self.original_model(x)
            return y_prob

    # SHAP
    num_samples = 50
    sample_indices = np.random.choice(features.shape[0], num_samples, replace=False)
    sampled_features = features[sample_indices]
    shapmodel = ShapModel(model)
    explainer = shap.DeepExplainer(shapmodel, sampled_features)
    shap_values = explainer.shap_values(features)
    # shap.summary_plot(shap_values, features.numpy())
    # feature_names = [i for i in range(X.shape[1])]
    features_importance = np.abs(shap_values).mean(axis=0)
    shap_df = var_df.copy()
    shap_df["weight"] = features_importance

    # Grad
    features.requires_grad = True
    _, y_prob, _, _ = model(features)
    loss = dpd_criterion(y_prob, labels)
    loss.backward()
    grads = features.grad.abs()
    grad_features_importance = grads.mean(dim=0)
    grad_df = var_df.copy()
    grad_df["weight"] = grad_features_importance.detach().numpy()
    if sort_by_weight:
        shap_df = shap_df.sort_values(by="weight", ascending=False)
        shap_df = cumulative_weight_sum_rate(shap_df)
        grad_df = grad_df.sort_values(by="weight", ascending=False)
        grad_df = cumulative_weight_sum_rate(grad_df)

    return shap_df, grad_df


# %% Peter-Clark


def get_neighbors(G, x: int, y: int):
    return [i for i in range(len(G)) if G[x][i] == True and i != y]


def gauss_ci_test(suff_stat, x: int, y: int, K: List[int], cut_at: float = 0.9999999):
    """条件独立性检验"""
    C = suff_stat["C"]
    n = suff_stat["n"]

    # ------ 偏相关系数 ------
    if len(K) == 0:  # K 为空
        r = C[x, y]

    elif len(K) == 1:  # K 中只有一个点，即一阶偏相关系数
        k = K[0]
        r = (C[x, y] - C[x, k] * C[y, k]) / math.sqrt(
            (1 - C[y, k] ** 2) * (1 - C[x, k] ** 2)
        )

    else:  # 其实我没太明白这里是怎么求的，但 R 语言的 pcalg 包就是这样写的
        m = C[np.ix_([x] + [y] + K, [x] + [y] + K)]
        p = np.linalg.pinv(m)
        r = -p[0, 1] / math.sqrt(abs(p[0, 0] * p[1, 1]))

    r = min(cut_at, max(-cut_at, r))

    # Fisher's z-transform
    z = 0.5 * math.log1p((2 * r) / (1 - r))
    z_standard = z * math.sqrt(n - len(K) - 3)

    # Φ^{-1}(1-α/2)
    p_value = 2 * (1 - norm.cdf(abs(z_standard)))

    return p_value


def skeleton(suff_stat, alpha: float):
    p_value_mat = np.zeros_like(suff_stat["C"])
    n_nodes = suff_stat["C"].shape[0]

    # 分离集
    O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]

    # 完全无向图
    G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]

    # 点对（不包括 i -- i）
    pairs = [
        (i, (n_nodes - j - 1)) for i in range(n_nodes) for j in range(n_nodes - i - 1)
    ]

    done = False
    l = 0  # 节点数为 l 的子集

    while done != True and any(G):
        done = True

        # 遍历每个相邻点对
        for x, y in pairs:
            if G[x][y] == True:
                neighbors = get_neighbors(G, x, y)  # adj(C,x) \ {y}

                if len(neighbors) >= l:  # |adj(C, x) \ {y}| > l
                    done = False

                    # |adj(C, x) \ {y}| = l
                    for K in set(combinations(neighbors, l)):
                        # 节点 x, y 是否被节点数为 l 的子集 K d-seperation
                        # 条件独立性检验，返回 p-value
                        p_value = gauss_ci_test(suff_stat, x, y, list(K))
                        if p_value > p_value_mat[x][y]:
                            p_value_mat[x][y] = p_value_mat[y][x] = p_value
                        # 条件独立
                        if p_value >= alpha:
                            G[x][y] = G[y][x] = False  # 去掉边 x -- y
                            O[x][y] = O[y][x] = list(K)  # 把 K 加入分离集 O
                            break

        l += 1

    return np.asarray(G, dtype=int), O, p_value_mat


def extend_cpdag(G, O):
    n_nodes = G.shape[0]

    def rule1(g):
        """Rule 1: 如果存在链 i -> j - k ，且 i, k 不相邻，则变为 i -> j -> k"""
        pairs = [
            (i, j)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if g[i][j] == 1 and g[j][i] == 0
        ]  # 所有 i - j 点对

        for i, j in pairs:
            all_k = [
                k
                for k in range(n_nodes)
                if (g[j][k] == 1 and g[k][j] == 1) and (g[i][k] == 0 and g[k][i] == 0)
            ]

            if len(all_k) > 0:
                g[j][all_k] = 1
                g[all_k][0, j] = 0

        return g

    def rule2(g):
        """Rule 2: 如果存在链 i -> k -> j ，则把 i - j 变为 i -> j"""
        pairs = [
            (i, j)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if g[i][j] == 1 and g[j][i] == 1
        ]  # 所有 i - j 点对

        for i, j in pairs:
            all_k = [
                k
                for k in range(n_nodes)
                if (g[i][k] == 1 and g[k][i] == 0) and (g[k][j] == 1 and g[j][k] == 0)
            ]

            if len(all_k) > 0:
                g[i][j] = 1
                g[j][1] = 0

        return g

    def rule3(g):
        """Rule 3: 如果存在 i - k1 -> j 和 i - k2 -> j ，且 k1, k2 不相邻，则把 i - j 变为 i -> j"""
        pairs = [
            (i, j)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if g[i][j] == 1 and g[j][i] == 1
        ]  # 所有 i - j 点对

        for i, j in pairs:
            all_k = [
                k
                for k in range(n_nodes)
                if (g[i][k] == 1 and g[k][i] == 1) and (g[k][j] == 1 and g[j][k] == 0)
            ]

            if len(all_k) >= 2:
                for k1, k2 in combinations(all_k, 2):
                    if g[k1][k2] == 0 and g[k2][k1] == 0:
                        g[i][j] = 1
                        g[j][i] = 0
                        break

        return g

    # Rule 4: 如果存在链 i - k -> l 和 k -> l -> j，且 k 和 l 不相邻，把 i - j 改为 i -> j
    # 显然，这种情况不可能存在，所以不需要考虑 rule4

    # 相邻点对
    pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if G[i][j] == 1]

    # 把 x - y - z 变为 x -> y <- z
    for x, y in sorted(pairs, key=lambda x: (x[1], x[0])):
        all_z = [z for z in range(n_nodes) if G[y][z] == 1 and z != x]

        for z in all_z:
            if G[x][z] == 0 and y not in O[x][z]:
                G[x][y] = G[z][y] = 1
                G[y][x] = G[y][z] = 0

    # Orientation rule 1 - rule 3
    old_G = np.zeros((n_nodes, n_nodes))

    while not np.array_equal(old_G, G):
        old_G = G.copy()

        G = rule1(G)
        G = rule2(G)
        G = rule3(G)

    return np.array(G)


def pc(suff_stat, alpha: float = 0.5, verbose: bool = False):
    G, O, pvm = skeleton(suff_stat, alpha)  # 骨架
    cpdag = extend_cpdag(G, O)  # 扩展为 CPDAG

    if verbose:
        print(cpdag)

    return cpdag, pvm


def run_pc(X, y, var_df, sort_by_weight=True):
    alpha = 0.05
    data = pd.DataFrame(np.column_stack((X, y)))
    p, pvm = pc(
        suff_stat={"C": data.corr().values, "n": data.shape[0]},
        verbose=False,
        alpha=alpha,
    )
    pv = pvm[:-1, -1]
    mask = np.where((pv < alpha) & (p[:-1, -1] == 0))
    pv[mask] = alpha
    var_df["weight"] = 1 - pv
    var_df["pred_label"] = p[:-1, -1]
    if sort_by_weight:
        var_df = var_df.sort_values(by="weight", ascending=False)
        var_df = cumulative_weight_sum_rate(var_df)
    return var_df


# %% method main： Call all of the above methods
def run_benchmark(
    # simulation parameters
    n_dataset=10,
    # noise_level=0.1,
    # causal_strength=6.0,
    threshold_method='cum_pct',  # 'top_k' or 'cum_pct
    n_top_k=10,  # top k features with largest weight
    n_cum_pct=0.3,  # cumulative percentage of features with largest weight
    save_path=None,
    **model_kwargs,
):
    # caufinder_shap_result = []
    # caufinder_grad_result = []
    # caufinder121_shap_result = []
    caufinder121_grad_result = []
    pc_result = []
    # ttest_result = []
    rf_result = []
    # mlp_shap_result = []
    # mlp_grad_result = []
    vae_shap_result = []
    vae_grad_result = []
    mi_result = []
    lr_result = []
    # svm_result = []
    pearsonCor_result = []
    spearmamCor_result = []

    GENIE3_result = []
    velorama_result = []
    SCODE_result = []
    WMDS_result = []

    set_seed(44)
    for i in range(n_dataset):
        # %% Generate simulate data
        print(f"This is the {i + 1}/{n_dataset} dataset")
        # adata = generate_synthetic(n_samples=200, n_features=150, n_causal=10, n_X_confusing=25, n_y_confusing=25,
        #                            shuffle_feature=True,
        #                            shuffle_sample=True, noise=noise, is_linear=linear_mode, activation=activation,
        #                            causal_strength=causal_strength, seed=i)
        # adata = generate_synthetic6(
        #     noise_level=noise_level, causal_strength=causal_strength, is_linear=is_linear
        # )
        adata = generate_synthetic()
        # adata = generate_synthetic(
        #     noise_scale=noise_level, causal_strength=causal_strength, is_linear=is_linear
        # )
        X = adata.X
        y = adata.obs["labels"].values
        var_df = adata.var

        adata.obs["hard_label"] = 0  # 默认全是 0
        adata.obs.iloc[:100, adata.obs.columns.get_loc("hard_label")] = 1  # 前 100 设为 1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        adata.obs["soft_label"] = scaler.fit_transform(y.copy().reshape(-1, 1))
        y = adata.obs["soft_label"].values.copy()

        # if save_path is not None:
        #     sub_dir = os.path.join(save_path, f"data_seed_{n_dataset}")
        #     os.makedirs(sub_dir, exist_ok=True)  # 确保子目录存在
        #     adata_filename = os.path.join(sub_dir, f"adata_dataset_{i}.h5ad")
        #     adata.write(adata_filename)
        #     print(f"Saved adata to {adata_filename}")

        # normalization
        # X = zscore_normalization(X)

        # Get the feature importance of each method
        # Caufinder_shap&Caufinder_grad result
        # cf_shap_res, cf_grad_res, _ = run_caufinder(
        #     adata=adata,
        #     **model_kwargs
        # )
        # # caufinder_shap_result.append(cf_shap_res)
        # caufinder_grad_result.append(cf_grad_res)

        # Caufinder result
        cf121_shap_res, cf121_grad_res = run_caufinder121(
            adata=adata,
            **model_kwargs
        )
        # caufinder121_shap_result.append(cf121_shap_res)
        caufinder121_grad_result.append(cf121_grad_res)


        # # T-test result
        # tt_res = run_t_test(X, y, var_df.copy(), sort_by_pvalue=True)
        # # tt_res = tt_res.merge(label, on='features', how='left')
        # ttest_result.append(tt_res)

        # Random forest result
        rf_res = run_rf(X, y, var_df.copy(), sort_by_weight=True)
        # rf_res = rf_res.merge(label, on='features', how='left')
        rf_result.append(rf_res)

        # VAE_SHAP&Grad result
        vae_shap_res, vae_grad_res = run_VAE(
            X=X,
            y=y,
            var_df=var_df.copy(),
            n_latent=10,
            n_hidden=64,
            sort_by_weight=True,
        )
        vae_shap_result.append(vae_shap_res)
        vae_grad_result.append(vae_grad_res)

        # Mutual information result
        mi_res = run_mut_info(X, y, var_df.copy(), sort_by_weight=True)
        # mi_res = mi_res.merge(label, on='features', how='left')
        mi_result.append(mi_res)

        # Logistic regression result
        # lr_res = run_log_reg(X, y, var_df.copy(), sort_by_weight=True)
        # # lr_res = lr_res.merge(label, on='features', how='left')
        # lr_result.append(lr_res)

        # # SVM
        # svm_res = run_SVM(X, y, var_df.copy(), sort_by_weight=True)
        # svm_result.append(svm_res)

        # Pearson Cor
        pearson_res = run_PearsonCor(X, y, var_df.copy(), sort_by_weight=True)
        pearsonCor_result.append(pearson_res)

        # Spearman Cor
        spearmam_res = run_Spearman_Cor(X, y, var_df.copy(), sort_by_weight=True)
        spearmamCor_result.append(spearmam_res)

        # PC algorithm
        pc_res = run_pc(X, y, var_df.copy(), sort_by_weight=True)
        pc_result.append(pc_res)

        base_dir = r"E:\Project_Research\CauFinder_Project\CauFinder-master\CausalRegNet\output"
        print('GENIE3')
        #GENIE3_res = run_GENIE3(X, y, var_df, sort_by_weight=True)
        GENIE3_res= run_GENIE3_load(load_file=os.path.join(base_dir, "GENIE3_res.pkl"), time=i)
        GENIE3_result.append(GENIE3_res)
        
        print('velorama')
#         velorama_res = run_velorama(X, y, var_df.copy(), cell_names, sort_by_weight=True,seed=i, prefix='stimu',output_dir=save_path,
#                                     L = 5,hidden = 32,max_iter = 400 ,learning_rate = 0.05 ,
#                                    n_comps = 30, n_neighbors = 15, num_lambdas = 15)
        velorama_res= run_velorama_load(load_file=os.path.join(base_dir, "velorama_res.pkl"),time=i)
        velorama_result.append(velorama_res)
        
        print('SCODE')
        SCODE_res= run_SCODE_load(load_file=os.path.join(base_dir, "SCODE_res.pkl"),time=i)
        SCODE_result.append(SCODE_res)

        print('WMDS.net')
        wmds_ret = run_wmdsnet(os.path.join(base_dir, "WMDSnet"), var_df, time=i)
        WMDS_result.append(wmds_ret)
    result_dict = {
        # 'Caufinder': caufinder_shap_result,
        # 'CAUF': caufinder_grad_result,
        # 'Caufinder121_SHAP': caufinder121_shap_result,
        # "CAUF121": caufinder121_grad_result,
        "CauFinder": caufinder121_grad_result,
        "PC": pc_result,
        "VAE_SHAP": vae_shap_result,
        "VAE_Grad": vae_grad_result,
        "RF": rf_result,
        # "LR": lr_result,
        # 'SVM': svm_result,
        # 'MLP_SHAP': mlp_shap_result,
        # 'MLP_Grad': mlp_grad_result,
        "MI": mi_result,
        # "T-test": ttest_result,
        "PCC": pearsonCor_result,
        "SCC": spearmamCor_result,
        'GENIE3':GENIE3_result,
        'Velorama':velorama_result,
        'SCODE':SCODE_result,
        'WMDSnet': WMDS_result,
    }

    # Creating sub_folder
    # folder_name = f"noise_{noise_level}_causal_{causal_strength}"
    # folder_name = "CausalRegNet_Data"
    # full_folder_path = os.path.join(save_path, folder_name)
    full_folder_path = save_path
    os.makedirs(full_folder_path, exist_ok=True)

    auc_df = pd.DataFrame()
    acc_df = pd.DataFrame()
    mcc_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    specificity_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    f1_df = pd.DataFrame()
    num_w_cum_rate_df = pd.DataFrame()
    n = len(next(iter(result_dict.values())))
    plt_row = int(np.ceil(n / 4))
    fig, axs = plt.subplots(plt_row, 4, figsize=(20, plt_row * 5))
    for i, (key, value) in enumerate(result_dict.items()):
        auc_list = []
        acc_list = []
        mcc_list = []
        precision_list = []
        specificity_list = []
        recall_list = []
        f1_list = []
        num_w_cum_rate_list = []
        # Iterate over the result
        for j, df in enumerate(value):
            # Preprocessing labels
            df = df.sort_values(by="weight", ascending=False)
            if "pred_label" not in df.columns:
                if threshold_method == "top_k":
                    df["pred_label"] = 0
                    df.loc[df.index[0:n_top_k], "pred_label"] = 1
                elif threshold_method == "cum_pct":

                    def convert_rate_to_label(rate):
                        if rate < n_cum_pct:
                            return 1
                        else:
                            return 0

                    df["pred_label"] = df["w_cum_rate"].apply(convert_rate_to_label)
                else:
                    raise ValueError("no value for threshold_method")
            true_label = df["feat_label"].values
            model_score = df["weight"].values
            pred_label = df["pred_label"].values

            # AUC
            fpr, tpr, thresholds = roc_curve(true_label, model_score)
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)

            # ACC (TP+TN)/(TP+TN+FP+FN)
            acc = accuracy_score(true_label, pred_label)
            acc_list.append(acc)

            # MCC (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            mcc = matthews_corrcoef(true_label, pred_label)
            mcc_list.append(mcc)

            # Precision TP/(TP+FP)
            precision = precision_score(true_label, pred_label, pos_label=1)
            precision_list.append(precision)

            # Specificity TN/(TN+FP)
            cm = confusion_matrix(true_label, pred_label)
            TN = cm[0, 0]
            FP = cm[0, 1]
            specificity = TN / (TN + FP)
            specificity_list.append(specificity)

            # Recall TP/(TP+FN)
            recall = recall_score(true_label, pred_label, pos_label=1)
            recall_list.append(recall)

            # F1_score 2 * precision * recall / (precision + recall)
            f1 = f1_score(true_label, pred_label, pos_label=1)
            f1_list.append(f1)

            # number of cumulative_weight_sum_rate
            num_w_cum_rate = (df["w_cum_rate"] < n_cum_pct).sum()
            num_w_cum_rate_list.append(num_w_cum_rate)

            # AUC_ROC
            if plt_row == 1:
                axs[j].plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                axs[j].plot([0, 1], [0, 1], "k--")
                axs[j].set_title(f"ROC Curve of dataset:No.{j + 1}")
                axs[j].set_xlabel("False Positive Rate")
                axs[j].set_ylabel("True Positive Rate")
                axs[j].legend(loc="lower right")
            else:
                x = j // 4
                y = j % 4
                axs[x, y].plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                axs[x, y].plot([0, 1], [0, 1], "k--")
                axs[x, y].set_title(f"ROC Curve of dataset:No.{j + 1}")
                axs[x, y].set_xlabel("False Positive Rate")
                axs[x, y].set_ylabel("True Positive Rate")
                axs[x, y].legend(loc="lower right")
        auc_df[f"{key}"] = auc_list
        acc_df[f"{key}"] = acc_list
        mcc_df[f"{key}"] = mcc_list
        precision_df[f"{key}"] = precision_list
        specificity_df[f"{key}"] = specificity_list
        recall_df[f"{key}"] = recall_list
        f1_df[f"{key}"] = f1_list
        num_w_cum_rate_df[f"{key}"] = num_w_cum_rate_list

    # save AUC_ROC
    if n < plt_row * 4:
        for i in range(n, plt_row * 4):
            fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.savefig(os.path.join(full_folder_path, "AUC_roc.pdf"), format="pdf")
    # plt.show(block=False)
    # time.sleep(3)
    plt.close()

    # Pick the plot where the median auc is
    median_value = np.median(auc_df.iloc[:, 0])
    median_column_index = auc_df.iloc[:, 0].sub(median_value).abs().idxmin()
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (key, value) in enumerate(result_dict.items()):
        for j, df in enumerate(value):
            if j == median_column_index:
                df = df.sort_values(by="weight", ascending=False)
                df["pred_label"] = 0
                df.loc[df.index[0:n_top_k], "pred_label"] = 1

                true_label = df["feat_label"]
                model_score = df["weight"]

                # AUC(median)
                fpr, tpr, thresholds = roc_curve(true_label, model_score)
                roc_auc = auc(fpr, tpr)

                # AUC_ROC(median)
                ax.plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_title(f"ROC Curve of dataset:No.{j + 1}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
    # save AUC_ROC_median
    plt.tight_layout()
    plt.savefig(os.path.join(full_folder_path, "AUC_roc_median.pdf"), format="pdf")
    # plt.show()
    plt.close()

    auc_df = generate_boxplot_csv(auc_df, "AUC", subfolder_name=full_folder_path)
    acc_df = generate_boxplot_csv(acc_df, "ACC", subfolder_name=full_folder_path)
    mcc_df = generate_boxplot_csv(mcc_df, "MCC", subfolder_name=full_folder_path)
    precision_df = generate_boxplot_csv(precision_df, "Precision", subfolder_name=full_folder_path)
    specificity_df = generate_boxplot_csv(specificity_df, "Specificity", subfolder_name=full_folder_path)
    recall_df = generate_boxplot_csv(recall_df, "Recall", subfolder_name=full_folder_path)
    f1_df = generate_boxplot_csv(f1_df, "F1_score", subfolder_name=full_folder_path)
    num_w_cum_rate_df = generate_boxplot_csv(num_w_cum_rate_df, "num_w_cum_rate", subfolder_name=full_folder_path)

    score_dict = {
        "AUC": auc_df,
        "ACC": acc_df,
        "MCC": mcc_df,
        "Precision": precision_df,
        "Specificity": specificity_df,
        "Recall": recall_df,
        "F1_score": f1_df,
        "num_w_cum_rate": num_w_cum_rate_df,
    }

    return score_dict, result_dict


# %% function of generate boxplot&csv


def generate_boxplot_csv(
    df, file_start_name="", subfolder_name=""  # A dataframe  # A string
):
    # boxplot
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=df, palette="Set2")
    plt.xticks(rotation=45, ha="right", fontsize=14)  # 修改 X 轴字体大小
    plt.yticks(fontsize=14)  # 修改 Y 轴字体大小
    plt.xlabel("Methods", fontsize=16)  # 修改 X 轴标签字体大小
    plt.ylabel(f"{file_start_name}", fontsize=16)  # 修改 Y 轴标签字体大小
    plt.title(f"{file_start_name} Boxplot for Different Methods", fontsize=18)  # 修改标题字体大小
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 增加底部空间

    # save boxplot
    plt.savefig(
        os.path.join(subfolder_name, f"{file_start_name}_boxplot.pdf"), format="pdf"
    )
    # plt.show(block=False)
    # time.sleep(3)
    plt.close()

    df = df.T
    df["Mean"] = df.mean(axis=1)
    df["Max"] = df.max(axis=1)
    df["Min"] = df.min(axis=1)

    # save AUC_csv
    df.to_csv(os.path.join(subfolder_name, f"{file_start_name}_evaluation.csv"))

    return df


# %%  function for calculating the rate of cumulative_weight_sum
def cumulative_weight_sum_rate(df):
    cumulative_weight_sum = df["weight"].cumsum()
    total_weight_sum = df["weight"].sum()
    df["w_cum_rate"] = cumulative_weight_sum / total_weight_sum

    return df

def run_GENIE3_load(load_file='./CauRegNet/output/GENIE3_res.pkl',time=0):
    
    # 打开文件并加载数据
    with open(load_file, 'rb') as file:
        GENIE3_data = pickle.load(file)

    # 检查 time 是否在数据中
    if time <= len(GENIE3_data):
        var_df =  GENIE3_data[time]
        
        return var_df

    else:
        return None
        print(f"Error: An unexpected error occurred")
        
        
def run_velorama_load(load_file='./CauRegNet/output/velorama_res.pkl',time=0):
    
    # 打开文件并加载数据
    with open(load_file, 'rb') as file:
        velorama_data = pickle.load(file)

    # 检查 time 是否在数据中
    if time <= len(velorama_data):
        var_df =  velorama_data[time]
        
        return var_df

    else:
        return None
        print(f"Error: An unexpected error occurred")
        
        
def run_SCODE_load(load_file='./CauRegNet/output/SCODE_res.pkl',time=0):
    
    # 打开文件并加载数据
    with open(load_file, 'rb') as file:
        SCODE_data = pickle.load(file)

    # 检查 time 是否在数据中
    if time <= len(SCODE_data):
        var_df =  SCODE_data[time]
        
        return var_df

    else:
        return None
        print(f"Error: An unexpected error occurred")


def run_wmdsnet(folder_path, var_df, time=0):
    file_path = os.path.join(folder_path, f"wmds_{time}.csv")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    df = pd.read_csv(file_path, header=None, names=["gene", "weight", "driver_gene"])
    df_sorted = df.sort_values(by="weight", ascending=False).set_index("gene")
    df_sorted["w_cum_rate"] = df_sorted["weight"].cumsum() / df_sorted["weight"].sum()

    # 直接按索引合并，不再 `set_index`
    df_merged = df_sorted.merge(var_df, left_index=True, right_index=True, how="left")

    return df_merged



# %% method main： Call all of the above methods
def run_internal_benchmark(
    # simulation parameters
    n_dataset=10,
    noise_level=0.1,
    causal_strength=6.0,
    activation="relu",
    is_linear=False,
    threshold_method='cum_pct',  # 'top_pct' or 'cum_pct
    n_top_k=10,  # top k features with largest weight
    n_cum_pct=0.3,  # cumulative percentage of features with largest weight
    save_path=None,
    **model_kwargs,
):
    caufinder_shap_result = []
    caufinder_grad_result = []
    caufinder_both_result = []
    caufinder121_shap_result = []
    caufinder121_grad_result = []
    caufinder121_both_result = []
    caufinder121_model_result = []

    for i in range(n_dataset):
        # %% Generate simulate data
        print(f"This is the {i + 1}/{n_dataset} dataset")
        adata = generate_synthetic5(
            noise_scale=noise_level, causal_strength=causal_strength, is_linear=is_linear
        )
        X = adata.X
        y = adata.obs["labels"].values
        var_df = adata.var

        # normalization
        # X = zscore_normalization(X)

        # Get the feature importance of each method
        # Caufinder_shap&Caufinder_grad result
        cf_shap_res, cf_grad_res, cf_both_res = run_caufinder(
            adata=adata,
            **model_kwargs
        )
        caufinder_shap_result.append(cf_shap_res)
        caufinder_grad_result.append(cf_grad_res)
        caufinder_both_result.append(cf_both_res)

        # Caufinder result
        (
            cf121_shap_res,
            cf121_grad_res,
            cf121_both_res,
            cf121_model_res,
        ) = run_caufinder121(
            adata=adata,
            **model_kwargs
        )
        caufinder121_shap_result.append(cf121_shap_res)
        caufinder121_grad_result.append(cf121_grad_res)
        caufinder121_both_result.append(cf121_both_res)
        caufinder121_model_result.append(cf121_model_res)

    result_dict = {
        "CAUF_SHAP": caufinder_shap_result,
        "CAUF": caufinder_grad_result,
        "CAUF_BOTH": caufinder_both_result,
        "CAUF121_SHAP": caufinder121_shap_result,
        "CAUF121": caufinder121_grad_result,
        "CAUF121_BOTH": caufinder121_both_result,
        "CAUF121_MODEL": caufinder121_model_result,
    }

    # Creating sub_folder
    folder_name = f"noise_{noise_level}_causal_{causal_strength}"
    full_folder_path = os.path.join(save_path, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    auc_df = pd.DataFrame()
    acc_df = pd.DataFrame()
    mcc_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    specificity_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    f1_df = pd.DataFrame()
    num_w_cum_rate_df = pd.DataFrame()
    n = len(next(iter(result_dict.values())))
    plt_row = int(np.ceil(n / 4))
    fig, axs = plt.subplots(plt_row, 4, figsize=(20, plt_row * 5))
    for i, (key, value) in enumerate(result_dict.items()):
        auc_list = []
        acc_list = []
        mcc_list = []
        precision_list = []
        specificity_list = []
        recall_list = []
        f1_list = []
        num_w_cum_rate_list = []
        # Iterate over the result
        for j, df in enumerate(value):
            # Preprocessing labels
            df = df.sort_values(by="weight", ascending=False)
            if "pred_label" not in df.columns:
                if threshold_method == "top_k":
                    df["pred_label"] = 0
                    df.loc[df.index[0:n_top_k], "pred_label"] = 1
                elif threshold_method == "cum_pct":

                    def convert_rate_to_label(rate):
                        if rate < n_cum_pct:
                            return 1
                        else:
                            return 0

                    df["pred_label"] = df["w_cum_rate"].apply(convert_rate_to_label)
                else:
                    raise ValueError("no value for method_for_threshold")
            true_label = df["feat_label"].values
            model_score = df["weight"].values
            pred_label = df["pred_label"].values

            # AUC
            fpr, tpr, thresholds = roc_curve(true_label, model_score)
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)

            # ACC (TP+TN)/(TP+TN+FP+FN)
            acc = accuracy_score(true_label, pred_label)
            acc_list.append(acc)

            # MCC (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            mcc = matthews_corrcoef(true_label, pred_label)
            mcc_list.append(mcc)

            # Precision TP/(TP+FP)
            precision = precision_score(true_label, pred_label, pos_label=1)
            precision_list.append(precision)

            # Specificity TN/(TN+FP)
            cm = confusion_matrix(true_label, pred_label)
            TN = cm[0, 0]
            FP = cm[0, 1]
            specificity = TN / (TN + FP)
            specificity_list.append(specificity)

            # Recall TP/(TP+FN)
            recall = recall_score(true_label, pred_label, pos_label=1)
            recall_list.append(recall)

            # F1_score 2 * precision * recall / (precision + recall)
            f1 = f1_score(true_label, pred_label, pos_label=1)
            f1_list.append(f1)

            # number of cumulative_weight_sum_rate
            num_w_cum_rate = (df["w_cum_rate"] < n_cum_pct).sum()
            num_w_cum_rate_list.append(num_w_cum_rate)

            # AUC_ROC
            if plt_row == 1:
                axs[j].plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                axs[j].plot([0, 1], [0, 1], "k--")
                axs[j].set_title(f"ROC Curve of dataset:No.{j + 1}")
                axs[j].set_xlabel("False Positive Rate")
                axs[j].set_ylabel("True Positive Rate")
                axs[j].legend(loc="lower right")
            else:
                x = j // 4
                y = j % 4
                axs[x, y].plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                axs[x, y].plot([0, 1], [0, 1], "k--")
                axs[x, y].set_title(f"ROC Curve of dataset:No.{j + 1}")
                axs[x, y].set_xlabel("False Positive Rate")
                axs[x, y].set_ylabel("True Positive Rate")
                axs[x, y].legend(loc="lower right")
        auc_df[f"{key}"] = auc_list
        acc_df[f"{key}"] = acc_list
        mcc_df[f"{key}"] = mcc_list
        precision_df[f"{key}"] = precision_list
        specificity_df[f"{key}"] = specificity_list
        recall_df[f"{key}"] = recall_list
        f1_df[f"{key}"] = f1_list
        num_w_cum_rate_df[f"{key}"] = num_w_cum_rate_list

    # save AUC_ROC
    if n < plt_row * 4:
        for i in range(n, plt_row * 4):
            fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.savefig(os.path.join(full_folder_path, "AUC_roc.pdf"), format="pdf")
    # plt.show(block=False)
    # time.sleep(3)
    plt.close()

    # Pick the plot where the median auc is
    median_value = np.median(auc_df.iloc[:, 0])
    median_column_index = auc_df.iloc[:, 0].sub(median_value).abs().idxmin()
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (key, value) in enumerate(result_dict.items()):
        for j, df in enumerate(value):
            if j == median_column_index:
                df = df.sort_values(by="weight", ascending=False)
                df["pred_label"] = 0
                df.loc[df.index[0:n_top_k], "pred_label"] = 1

                true_label = df["feat_label"]
                model_score = df["weight"]

                # AUC(median)
                fpr, tpr, thresholds = roc_curve(true_label, model_score)
                roc_auc = auc(fpr, tpr)

                # AUC_ROC(median)
                ax.plot(fpr, tpr, label=f"{key} (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_title(f"ROC Curve of dataset:No.{j + 1}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
    # save AUC_ROC_median
    plt.tight_layout()
    plt.savefig(os.path.join(full_folder_path, "AUC_roc_median.pdf"), format="pdf")
    # plt.show()
    plt.close()

    auc_df = generate_boxplot_csv(auc_df, "AUC", subfolder_name=full_folder_path)
    acc_df = generate_boxplot_csv(acc_df, "ACC", subfolder_name=full_folder_path)
    mcc_df = generate_boxplot_csv(mcc_df, "MCC", subfolder_name=full_folder_path)
    precision_df = generate_boxplot_csv(precision_df, "Precision", subfolder_name=full_folder_path)
    specificity_df = generate_boxplot_csv(specificity_df, "Specificity", subfolder_name=full_folder_path)
    recall_df = generate_boxplot_csv(recall_df, "Recall", subfolder_name=full_folder_path)
    f1_df = generate_boxplot_csv(f1_df, "F1_score", subfolder_name=full_folder_path)
    num_w_cum_rate_df = generate_boxplot_csv(num_w_cum_rate_df, "num_w_cum_rate", subfolder_name=full_folder_path)

    score_dict = {
        "AUC": auc_df,
        "ACC": acc_df,
        "MCC": mcc_df,
        "Precision": precision_df,
        "Specificity": specificity_df,
        "Recall": recall_df,
        "F1_score": f1_df,
        "num_w_cum_rate": num_w_cum_rate_df,
    }

    return score_dict, result_dict


def zscore_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_score_X = (X - mean) / std

    return z_score_X
