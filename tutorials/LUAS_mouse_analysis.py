#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
from CauFinder.utils import load_luas_human_adata, load_luas_mouse_adata, human_all_adata, human_data_direction, calculate_w1_w2, find_index
from CauFinder.utils import plot_3d_state_transition, plot_causal_feature_transitions
import scanpy as sc
import pickle as pkl
import collections as ct
import warnings

# Set base and output directories
BASE_DIR = r'E:\Project_Research\CauFinder_Project\CauFinder-master'
case_dir = os.path.join(BASE_DIR, 'LUAS', 'mouse')
data_path = os.path.join(case_dir, 'data')
output_path = os.path.join(case_dir, 'output', 'final_model')
os.makedirs(output_path, exist_ok=True)

# Load necessary files and data
tf_path = os.path.join(BASE_DIR, 'resources', 'tf', 'mm_mgi_tfs.txt')
network_path = os.path.join(BASE_DIR, 'resources', 'network', 'NicheNet_mouse.csv')
tf_list = pd.read_csv(tf_path, header=None, squeeze=True).tolist()
prior_network = pd.read_csv(network_path, index_col=None, header=0)
# if you want retrains the model, reload the data
adata, adata_filter = load_luas_mouse_adata(data_dir=data_path, tf_list=tf_list)

# Check loaded data
print(adata_filter)

# Load driver information and the model (50 runs)
driver_info_path = os.path.join(output_path, 'driver_summary_shap_total.csv')
driver_info = pd.read_csv(driver_info_path, index_col=0)
driver_info = driver_info[driver_info['counts'] > 10]
drivers = driver_info.index.to_list()

# Load the model
model_path = os.path.join(data_path, 'mouse_model.pkl')
with open(model_path, 'rb') as file:
    model = pkl.load(file)

# Plot PCA with probabilities (state 0 and 1)
adata_filter = model.adata.copy()
adata_filter.obs['probs'] = model.get_model_output(adata_filter)['probs']
model.plot_pca_with_probs(adata_filter, save_dir=output_path, elev=20, azim=60)

# Run state transitions
adata_increase = model.guided_state_transition(
    adata=model.adata,
    causal_features=drivers,
    lambda_reg=1e-6,
    lr=0.2,
    max_iter=500,
    stop_thresh=0.0,
    target_state=1,
    iter_norm=False,
)

adata_decrease = model.guided_state_transition(
    adata=model.adata,
    causal_features=drivers,
    lambda_reg=1e-6,
    lr=0.2,
    max_iter=500,
    stop_thresh=0.0,
    target_state=0,
    iter_norm=False,
)

# Draw PCA plots
test_adata = adata_filter.copy()
sc.pp.highly_variable_genes(test_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(test_adata)

test_adata.raw = test_adata
test_adata = test_adata[:, test_adata.var.highly_variable]

sc.tl.pca(test_adata, svd_solver="arpack", n_comps=5)
sc.pl.pca(test_adata, color='time')

# Combine data for further plotting
df1 = test_adata.obs
df2 = adata_increase.uns['control_details']
df2.index = df1.index
test_adata.obs = pd.concat([df1, df2], axis=1)

# Plot PCA with scores and save plots
fig, ax = plt.subplots(figsize=(15, 15))
sc.pl.pca(test_adata, color='time', legend_loc="on data", size=1200, legend_fontsize=16, palette="Set2",
          frameon=False, add_outline=True, outline_width=(0.05, 0.05), return_fig=True)
plt.savefig(os.path.join(output_path, 'mouse_pca_time.pdf'), bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 15))
sc.pl.pca(test_adata, color='time', frameon=False, add_outline=True, outline_width=(0.05, 0.05), ax=ax, return_fig=False)
for i, txt in enumerate(test_adata.obs['sample_idx']):
    ax.annotate(txt, (test_adata.obsm['X_pca'][i, 0], test_adata.obsm['X_pca'][i, 1]),
                fontsize=20, color='black', ha='center', va='center')
plt.show()

test_adata.obs['norm_score'] = [(i - min(test_adata.obs['score'])) / (max(test_adata.obs['score']) - min(test_adata.obs['score'])) for i in test_adata.obs['score']]

fig, ax = plt.subplots(figsize=(15, 15))
sc.pl.pca(test_adata, color='norm_score', size=1200, legend_fontsize=16, palette="Set2",
          frameon=False, add_outline=True, outline_width=(0.05, 0.05), return_fig=True)
plt.savefig(os.path.join(output_path, 'mouse_pca_norm_score.pdf'), bbox_inches='tight')

# Further PCA plots for specific time points
test_adata2 = adata_filter.copy()
test_adata2 = test_adata2[[i in ['9W', '10W'] for i in test_adata2.obs['time']],]
sc.pp.highly_variable_genes(test_adata2, min_mean=0.0125, max_mean=3, min_disp=0.5)
test_adata2.raw = test_adata2
test_adata2 = test_adata2[:, test_adata2.var.highly_variable]
sc.tl.pca(test_adata2, svd_solver="arpack", n_comps=2)

df1 = test_adata2.obs
df2 = adata_decrease.uns['control_details'][12::]
df2.index = df1.index
test_adata2.obs = pd.concat([df1, df2], axis=1)

test_adata2.obs['norm_score'] = [(i - min(test_adata2.obs['score'])) / (max(test_adata2.obs['score']) - min(test_adata2.obs['score'])) for i in test_adata2.obs['score']]

fig, ax = plt.subplots(figsize=(15, 15))
sc.pl.pca(test_adata2, color='norm_score', legend_loc="on data", size=1200, legend_fontsize=16, palette="Set2",
          frameon=False, add_outline=True, outline_width=(0.05, 0.05), return_fig=True)
plt.savefig(os.path.join(output_path, 'mouse_decrease_pca_norm_score.pdf'), bbox_inches='tight')

# fig, ax = plt.subplots(figsize=(15, 15))
# sc.pl.pca(test_adata2, color='sample_name', legend_loc="on data", size=1200, legend_fontsize=16, palette="Set2",
#           frameon=False, add_outline=True, outline_width=(0.05, 0.05), return_fig=True)
# plt.savefig(os.path.join(output_path, 'mouse_decrease_name_norm_score.pdf'), bbox_inches='tight')

# Plot 3D state transitions
save_path = os.path.join(output_path, 'final_model')
os.makedirs(save_path, exist_ok=True)

plot_3d_state_transition(adata_increase, sample_indices=[9, 11, 8], use_pca=True, save_path=save_path, elev=20, azim=120)
plot_3d_state_transition(adata_increase, sample_indices=[9, 11, 8], use_pca=False, feature1='Smad4', feature2='Myc', add_key='Smad4_Myc', save_path=save_path, elev=20, azim=120)
plot_3d_state_transition(adata_increase, sample_indices=[9, 11, 8], use_pca=False, feature1='Sox2', feature2='Trp63', add_key='Sox2_Trp63', save_path=save_path, elev=20, azim=120)

plot_3d_state_transition(adata_decrease, sample_indices=[12, 15], use_pca=True, save_path=save_path, elev=20, azim=30)
plot_3d_state_transition(adata_decrease, sample_indices=[12, 15], use_pca=False, feature1='Smad4', feature2='Myc', add_key='Smad4_Myc', save_path=save_path, elev=20, azim=120)
plot_3d_state_transition(adata_decrease, sample_indices=[12, 15], use_pca=False, feature1='Sox2', feature2='Trp63', add_key='Sox2_Trp63', save_path=save_path, elev=20, azim=120)

# Additional plotting for a different set of samples
final_save_path = os.path.join(output_path, 'final_add_stat')
os.makedirs(final_save_path, exist_ok=True)
plot_3d_state_transition(adata_decrease, sample_indices=[13, 17], use_pca=False, feature1='Smad4', feature2='Myc', add_key='Sox2_Trp63', save_path=final_save_path, elev=30, azim=30)

# Add 8W time point data and plot
adata_8w = adata[adata.obs['time'] == '8W',]
adata_8w_f = adata_8w[:, [i for i in adata_8w.var.index.to_list() if i in model.adata.var.index.to_list()]]

adata_decrease_8w = model.guided_state_transition(
    adata=adata_8w_f,
    causal_features=drivers, lambda_reg=1e-6, lr=0.1, max_iter=10,
    target_state=0
)

adata_increase_8w = model.guided_state_transition(
    adata=adata_8w_f,
    causal_features=drivers, lambda_reg=1e-6, lr=0.1, max_iter=10,  # for more score
    target_state=1
)

# Combine data and plot PCA
test_adata8w = adata_8w_f.copy()
sc.tl.pca(test_adata8w, svd_solver="arpack", n_comps=2)

df1 = test_adata8w.obs
df2 = adata_decrease_8w.uns['control_details']
df2.index = df1.index
df2 = df2.add_prefix('decrease_')
df3 = adata_increase_8w.uns['control_details']
df3.index = df1.index
df3 = df3.add_prefix('increase_')
test_adata8w.obs = pd.concat([df1, df2, df3], axis=1)

all_score_in = test_adata8w.obs['increase_score'].to_list() + test_adata.obs['score'].to_list()
all_score = test_adata8w.obs['decrease_score'].to_list() + test_adata2.obs['score'].to_list()
all_all = all_score_in + all_score

test_adata8w.obs['decrease_norm_score'] = [(i - min(all_score)) / (max(all_score) - min(all_score)) for i in test_adata8w.obs['decrease_score']]
test_adata8w.obs['decrease_norm_score_all'] = [(i - min(all_all)) / (max(all_all) - min(all_all)) for i in test_adata8w.obs['decrease_score']]
test_adata8w.obs['increase_norm_score'] = [(i - min(all_score_in)) / (max(all_score_in) - min(all_score_in)) for i in test_adata8w.obs['increase_score']]
test_adata8w.obs['increase_norm_score_all'] = [(i - min(all_all)) / (max(all_all) - min(all_all)) for i in test_adata8w.obs['increase_score']]

sc.pl.pca(test_adata8w, color='decrease_norm_score', legend_loc="on data", size=1200, vmax=1, vmin=0)
sc.pl.pca(test_adata8w, color='increase_norm_score', legend_loc="on data", size=1200, vmax=1, vmin=0)
sc.pl.pca(test_adata8w, color='decrease_norm_score_all', legend_loc="on data", size=1200, vmax=1, vmin=0)
sc.pl.pca(test_adata8w, color='increase_norm_score_all', legend_loc="on data", size=1200, vmax=1, vmin=0)

