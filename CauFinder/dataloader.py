import numpy as np
import pandas as pd
from scipy import linalg
import torch
import anndata
from anndata import AnnData
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from math import ceil, floor
from typing import Dict, List, Optional, Union


def generate_synthetic(
        n_samples=200,
        oversampling_factor=5,
        n_features=100,
        n_causal=10,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=5,
        is_linear=False,
        shuffle_features=True
):

    if not 0 <= causal_strength <= 10:
        raise ValueError("causal_strength must be between 0 and 10")

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples
    # 1. generate z and xc
    z = np.random.standard_normal(size=(total_samples, n_latent))

    mean_values1 = np.random.uniform(low=0.5, high=1.5, size=n_causal)
    std_devs1 = np.random.uniform(low=1, high=2, size=n_causal)
    Xc = np.random.normal(loc=mean_values1, scale=std_devs1, size=(total_samples, n_causal))
    # Xc = np.random.exponential(scale=mean_values1, size=(total_samples, n_causal))

    # 2. generate xs and y2

    # Xs
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, (n_features - n_causal)))
    # y2
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, 1))

    noise1 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    Xs = np.dot(np.dot(z, weights1), weights2)
    # y2 = np.dot(np.dot(z, weights3), weights4) + noise1
    if is_linear:
        y2 = np.dot(np.dot(z, weights3), weights4) + noise1
    else:
        y2 = apply_activation(np.dot(apply_activation(np.dot(z, weights3), "tanh"), weights4), "tanh") + noise1

    Xc2 = np.zeros((total_samples, n_causal))
    for i in range(n_causal):
        related_feature = Xs.dot(np.random.rand(int(n_features - n_causal), 1))
        Xc2[:, i] = related_feature.squeeze()
    Xc2 = zscore_normalization(Xc2)
    Xc += Xc2

    # 3. generate y1
    weights5 = np.random.standard_normal(size=(n_causal, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, 1))
    noise2 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    # y1 = np.dot(np.dot(Xc, weights5), weights6) + noise2
    if is_linear:
        y1 = np.dot(np.dot(Xc, weights5), weights6) + noise2
    else:
        y1 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights5), "tanh"), weights6), "tanh") + noise2

    # 4. generate final y
    data = np.hstack((Xc, Xs))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_features))
    data = zscore_normalization(data) + noise

    rate = (causal_strength / 10)

    y = rate * y1 + (1-rate) * y2

    sorted_indices = np.argsort(y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])  

    labels = np.where(y > np.median(y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)

    if shuffle_features:
        features_indices = np.random.permutation(n_features)
        causal_indices = np.where((features_indices >= 0) & (features_indices < n_causal))
        causal_indices = causal_indices[0].ravel()
        data = data[:, features_indices]
    else:
        causal_indices = np.arange(n_causal)

    # 5. generate labels 
    feature_names = ["f_" + str(i + 1) for i in range(n_features)]
    feature_types = np.full(n_features, 'spurious')
    feature_types[causal_indices] = 'causal'
    y_resized = y[indices].flatten()  

    adata = AnnData(data[indices, :], dtype=data.dtype)
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.obs["y_values"] = y_resized
    adata.var["feat_label"] = (feature_types == 'causal').astype(int)
    adata.var.index = feature_names

    return adata


def generate_causal_weights(n_features, n_units, n_causal, cau_idxs, loc_causal=6.0):
    weights = np.zeros((n_features, n_units))
    values = np.random.normal(loc=loc_causal, scale=1, size=(n_causal, n_units))
    signs = np.random.normal(loc=0.0, scale=1.0, size=(n_causal, n_units))
    cau_values = np.where(signs < 0, -values, values)
    weights[cau_idxs, :] = cau_values
    other_idxs = np.setdiff1d(np.arange(n_features), cau_idxs)
    # weights[other_idxs, :] = np.random.normal(loc=loc_other, scale=0.5, size=(n_features - n_causal, n_units))
    weights[other_idxs, :] = np.clip(np.random.normal(loc=0.0, scale=1.0, size=(n_features - n_causal, n_units)), -1, 1)
    return weights


def apply_activation(x, activation):
    if activation == "relu":
        return np.maximum(x, 0)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "log":
        return np.log(x + 1)
    return x


def make_low_rank_matrix(
        n_samples=100,
        n_features=100,
        effective_rank=10,
        tail_strength=0.5,
):
    """Generate a mostly low rank matrix with bell-shaped singular values.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The matrix.
    """
    n = min(n_samples, n_features)

    # Random (ortho normal) vectors
    u, _ = np.linalg.qr(np.random.standard_normal(size=(n_samples, n)))
    v, _ = np.linalg.qr(np.random.standard_normal(size=(n_features, n)))
    # u, _ = linalg.qr(
    #     np.random.standard_normal(size=(n_samples, n)),
    #     mode="economic",
    #     check_finite=False,
    # )
    #
    # v, _ = linalg.qr(
    #     np.random.standard_normal(size=(n_features, n)),
    #     mode="economic",
    #     check_finite=False,
    # )

    # Index of the singular values
    singular_ind = np.arange(n, dtype=np.float64)

    # Build the singular profile by assembling signal and noise components
    low_rank = (1 - tail_strength) * np.exp(-1.0 * (singular_ind / effective_rank) ** 2)
    tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
    s = np.identity(n) * (low_rank + tail)

    return np.dot(np.dot(u, s), v.T) * 100


def generate_sparse_matrix(n_rows, n_cols, k=2, replace=False):
    # Initialize sparse matrix
    M = np.zeros((n_rows, n_cols))

    # Generate random indices
    indices = np.random.choice(n_cols, size=k * n_rows, replace=replace)
    # indices = np.sort(indices)

    # Assign random values to matrix
    for i in range(n_rows):
        row_indices = indices[i * k: (i + 1) * k]
        row_values = np.random.randn(k)
        M[i, row_indices] = row_values

    return M


def generate_block_matrix(p, k=10, sparsity=0.1):
    """
    Generates a dim x p matrix consisting of k blocks.
    Each block is a randomly generated p/k x p/k matrix.
    """
    matrix = np.zeros((p, p))
    block_size = p // k

    for i in range(k):
        block = np.random.randn(block_size, block_size)
        matrix[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = block

    if sparsity > 0:
        matrix[np.random.rand(*matrix.shape) < sparsity] = 0
    return matrix


def generate_synthetic_v1(n_samples, n_feats=1000, n_up_feats=10, n_down_feats=990, n_latent=5, noise_scale=1.0,
                          seed=42):
    """Different latent variables"""
    np.random.seed(seed)
    total_samples = n_samples * 5
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    n_other_feats = n_feats - n_up_feats - n_down_feats

    # Generate random weights for upstream features
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_up_feats)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # Generate random weights for downstream features
    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_down_feats, n_latent, k=1, replace=True)
    b2 = np.random.randn(n_down_feats) * 1.0

    W3 = generate_sparse_matrix(n_latent, n_up_feats)
    b3 = np.random.randn(n_latent) * 1.0
    W4 = generate_sparse_matrix(n_other_feats, n_latent, k=1, replace=True)
    b4 = np.random.randn(n_other_feats) * 1.0

    # Generate random weights for state variable
    sparsity = 0.1
    V1 = generate_block_matrix(n_down_feats, k=10, sparsity=0.5)
    c1 = np.random.randn(n_down_feats) * 1.0
    V2 = np.random.randn(1, n_down_feats)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_up = np.random.randn(total_samples, n_up_feats) * 6.0
    # X_up = X_up - X_up.min()
    # X_up, labels = make_blobs(n_samples=total_samples, n_features=n_up_feats, centers=2, cluster_std=1.0)

    X_latent1 = np.dot(X_up, W1.T) + b1
    X_latent1 = np.tanh(X_latent1)
    X_latent1 += np.random.randn(*X_latent1.shape) * 0.5 * X_latent1.std()

    # Generate downstream features
    X_down = 1 * np.dot(X_latent1, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5 * X_down.std()
    # X_down = X_down - X_down.min()
    # V1 = np.identity(n_down_feat)

    # Generate other features
    X_latent2 = np.dot(X_up, W3.T) + b3
    X_latent2 = np.tanh(X_latent2)
    X_latent2 += np.random.randn(*X_latent2.shape) * 0.5 * X_latent2.std()
    X_other = 1 * np.dot(X_latent2, W4.T) + b4
    X_other += np.random.randn(*X_other.shape) * 0.5 * X_other.std()

    # Generate state variable
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Generate indices for positive and negative samples
    sorted_indices = np.argsort(Y, axis=0).flatten()
    pos_indices = sorted_indices[-pos_samples:]
    neg_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([pos_indices, neg_indices])
    # Generate state variable
    # print(Y.mean())
    labels = np.where(Y > Y.mean(), 1, 0)

    # Concatenate input features and labels
    X = np.concatenate([X_up, X_down, X_other], axis=1)
    if noise_scale > 0.0:
        n_samples, n_features = X.shape
        variances = np.var(X, axis=0)
        X = X + np.random.normal(loc=0, scale=np.sqrt(noise_scale * variances), size=(n_samples, n_features))
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(['upstream', 'downstream', 'others'], [n_up_feats, n_down_feats, n_other_feats])

    adata = AnnData(X[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feature_types"] = feat_types

    return adata


def generate_synthetic_v2(n_samples, n_up_feats=10, n_down_feat=990, n_latent=5, seed=42):
    """Generate synthetic data for testing. X_up has 2 clusters"""
    np.random.seed(seed)

    # Generate random weights for upstream features
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_up_feats, sparsity=0.4)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # Generate random weights for downstream features
    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_down_feat, n_latent, sparsity=0.2)
    b2 = np.random.randn(n_down_feat) * 1.0

    # Generate random weights for state variable
    sparsity = 0.2
    V1 = generate_block_matrix(n_down_feat, k=10, sparse=0.1)

    c1 = np.random.randn(n_down_feat) * 1.0
    V2 = np.random.randn(1, n_down_feat)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_up, labels = make_blobs(n_samples=n_samples, n_features=n_up_feats, centers=2, cluster_std=1.0)

    X_latent = np.dot(X_up, W1.T) + b1
    X_latent += np.random.randn(*X_latent.shape) * 0.5
    X_latent = np.maximum(X_latent, 0)

    X_down = np.dot(X_latent, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5
    # V1 = np.identity(n_down_feat)
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Concatenate input features and labels
    X = np.concatenate([X_up, X_down], axis=1)
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(['upstream', 'downstream'], [n_up_feats, n_down_feat])

    adata = AnnData(X)
    adata.obs["labels"] = labels
    adata.var["feature_types"] = feat_types

    return adata


def generate_synthetic_v3(n_samples, n_feats=200, n_causal=10, n_related=10, n_latent=5, seed=42):
    """Generate synthetic data for testing."""
    np.random.seed(seed)
    total_samples = n_samples * 5
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    n_other = n_feats - n_causal - n_related

    # Generate random weights for causal flow
    n_causal_down = 100
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_causal, k=2, replace=False)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_causal_down, n_latent, k=1, replace=True)
    b2 = np.random.randn(n_causal_down) * 1.0

    W1_related = generate_sparse_matrix(n_latent, n_causal, k=2, replace=True)
    b1_related = np.random.randn(n_latent) * 1.0
    W2_related = generate_sparse_matrix(n_related, n_latent, k=1, replace=True)
    b2_related = np.random.randn(n_related) * 1.0

    # Generate random weights for state variable
    sparsity = 0.1
    V1 = generate_block_matrix(n_causal_down, k=10, sparsity=0.5)
    c1 = np.random.randn(n_causal_down) * 1.0
    V2 = np.random.randn(1, n_causal_down)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_causal = np.random.randn(total_samples, n_causal) * 6.0
    # X_up = X_up - X_up.min()
    # X_up, labels = make_blobs(n_samples=total_samples, n_features=n_up_feats, centers=2, cluster_std=1.0)

    X_latent = np.dot(X_causal, W1.T) + b1
    X_latent = np.tanh(X_latent)
    X_latent += np.random.randn(*X_latent.shape) * 0.5 * X_latent.std()

    # Generate downstream features
    X_down = np.dot(X_latent, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5 * X_down.std()
    # X_down = X_down - X_down.min()
    # V1 = np.identity(n_down_feat)

    # Generate related features
    X_latent_related = np.dot(X_causal, W1_related.T) + b1_related
    X_latent_related = np.tanh(X_latent_related)
    X_latent_related += np.random.randn(*X_latent_related.shape) * 0.5 * X_latent_related.std()

    X_related = np.dot(X_latent_related, W2_related.T) + b2_related
    X_related += np.random.randn(*X_related.shape) * 0.5 * X_related.std()

    # Generate other features
    X_other, _ = make_blobs(n_samples=total_samples, n_features=n_other, centers=5, cluster_std=3.0)
    X_other = X_other * 0.5

    # Generate state variable
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Generate indices for positive and negative samples
    sorted_indices = np.argsort(Y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])
    # Generate state variable
    print(Y.mean())
    labels = np.where(Y > Y.mean(), 1, 0)

    # Concatenate input features and labels
    X = np.concatenate([X_causal, X_related, X_other], axis=1)
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(['causal', 'related', 'others'], [n_causal, n_related, n_other])

    adata = AnnData(X[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feature_types"] = feat_types

    return adata


def data_splitter(
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
):
    """Split indices in train/test/val sets."""
    n_train, n_val = validate_data_split(adata.n_obs, train_size, validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(adata.n_obs)
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]

    train_adata = adata[train_idx]
    val_adata = adata[val_idx]
    if test_idx.shape[0] == 0:
        return train_adata, val_adata
    else:
        test_adata = adata[test_idx]
        return train_adata, val_adata, test_adata


def validate_data_split(
        n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


def batch_sampler(
        adata: AnnData,
        batch_size: int,
        shuffle: bool = False,
        drop_last: Union[bool, int] = False, ):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.

    Parameters
    ----------
    adata
        adata to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """
    if drop_last > batch_size:
        raise ValueError(
            "drop_last can't be greater than batch_size. "
            + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
        )

    last_batch_len = adata.n_obs % batch_size
    if (drop_last is True) or (last_batch_len < drop_last):
        drop_last_n = last_batch_len
    elif (drop_last is False) or (last_batch_len >= drop_last):
        drop_last_n = 0
    else:
        raise ValueError("Invalid input for drop_last param. Must be bool or int.")

    if shuffle is True:
        idx = torch.randperm(adata.n_obs).tolist()
    else:
        idx = torch.arange(adata.n_obs).tolist()

    if drop_last_n != 0:
        idx = idx[: -drop_last_n]

    adata_iter = [
        adata[idx[i: i + batch_size]]
        for i in range(0, len(idx), batch_size)
    ]
    return adata_iter


def generate_synthetic_nb(
        batch_size: int = 500,
        n_genes: int = 2000,
        n_proteins: int = 20,
        n_batches: int = 2,
        n_labels: int = 2,
) -> AnnData:
    #  Here samples are drawn from a negative binomial distribution with specified parameters,
    # `n` failures and `p` probability of failure where `n` is > 0 and `p` is in the interval
    #  [0, 1], `n` is equal to diverse dispersion parameter.
    data = np.zeros(shape=(batch_size * n_batches, n_genes))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(5, 1 - p[i],
                                                                                    size=(batch_size, n_genes))
    data = np.random.negative_binomial(5, 0.3, size=(batch_size * n_batches, n_genes))
    mask = np.random.binomial(n=1, p=0.7, size=(batch_size * n_batches, n_genes))
    data = data * mask  # We put the batch index first
    labels = np.random.randint(0, n_labels, size=(batch_size * n_batches,))
    # labels = np.array(["label_%d" % i for i in labels])

    batch = []
    for i in range(n_batches):
        batch += ["batch_{}".format(i)] * batch_size

    adata = AnnData(data)
    batch = np.random.randint(high=n_batches, low=0, size=(batch_size * n_batches, 1)).astype(np.float32)
    # adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["batch"] = batch
    # adata.obs["labels"] = pd.Categorical(labels)
    adata.obs["labels"] = labels
    adata.uns['n_batch'] = n_batches

    # Protein measurements
    p_data = np.zeros(shape=(adata.shape[0], n_proteins))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        p_data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(5, 1 - p[i],
                                                                                      size=(batch_size, n_proteins))
    p_data = np.random.negative_binomial(5, 0.3, size=(adata.shape[0], n_proteins))
    adata.obsm["protein_expression"] = p_data
    adata.uns["protein_names"] = np.arange(n_proteins).astype(str)

    return adata


def generate_X_confusing_variables(
        X,
        n_causal,
        confusing_number=20,

):
    X_causal = X[:, n_causal]

    # matrix mapping & data shifted
    np.random.seed(42)
    transformation_matrix = np.random.randn(len(n_causal), confusing_number)
    confusing_var = X_causal.dot(transformation_matrix)
    min_value = np.min(confusing_var)
    # confusing_var = confusing_var - min_value

    # add noise
    noise_level = 1.0
    np.random.seed(42)

    def laplace_noise(shape, scale):
        return np.random.laplace(scale=scale, size=shape)

    noise = laplace_noise(confusing_var.shape, scale=noise_level)

    # confusing_var_shifted_log_noise = np.tanh(confusing_var + 1) + noise
    confusing_var_shifted_log_noise = zscore_normalization(confusing_var) + noise
    combined_X = np.concatenate((X, confusing_var_shifted_log_noise), axis=1)

    # # var_df
    # feature_names = ["f_" + str(i + X.shape[1] + 1) for i in range(confusing_number)]
    # feat_type = ["confusing" for i in range(confusing_number)]
    # feat_label = [0 for i in range(confusing_number)]
    # confusing_var_df = pd.DataFrame()
    # confusing_var_df["feat_type"] = feat_type
    # confusing_var_df["feat_label"] = feat_label
    # confusing_var_df.index = feature_names
    # combined_var_df = pd.concat([var_df, confusing_var_df], axis=0)

    return combined_X


def generate_y_confusing_variables(
        X,
        y,
        confusing_number=20
):
    n_samples = len(y)
    confusing_vars = np.zeros((n_samples, confusing_number))
    for i in range(confusing_number):
        transformation_matrix = np.random.randn(len(y), len(y))
        confusing_var = transformation_matrix.dot(y)
        min_value = np.min(confusing_var)
        # confusing_var = confusing_var - min_value

        def laplace_noise(shape, scale):
            return np.random.laplace(scale=scale, size=shape)

        noise = laplace_noise(confusing_var.shape, scale=1.0)

        # confusing_var_shifted_log_noise = np.tanh(confusing_var + 1) + noise
        confusing_var_shifted_log_noise = zscore_normalization(confusing_var) + noise
        confusing_vars[:, i] = confusing_var_shifted_log_noise.flatten()

    combined_X = np.concatenate((X, confusing_vars), axis=1)

    return combined_X


def zscore_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_score_X = (X - mean) / std

    return z_score_X

