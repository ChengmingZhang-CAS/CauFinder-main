import h5py
import numpy as np
import pandas as pd
import random
import torch
import os
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy import sparse

from typing import Dict, List, Optional, Sequence, Tuple, Union
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.ndimage import gaussian_filter
import anndata
import scipy.io
import scanpy as sc
import gc
import scanpy.external as sce
import re
import networkx as nx

os.environ["OMP_NUM_THREADS"] = '1'
Number = Union[int, float]


# ======================================== data.utils ========================================
def load_luas_human_adata(data_dir, tf_list=None):
    if tf_list is None:
        tf_list = []
    # Load data
    luas_data = pd.read_csv(os.path.join(data_dir, "Hum_LUAS.csv"), index_col=0)
    class1_samples = pd.read_csv(os.path.join(data_dir, "class1_ordered.txt"), header=None).squeeze().tolist()
    class2_samples = pd.read_csv(os.path.join(data_dir, "class2_ordered.txt"), header=None).squeeze().tolist()
    class3_samples = pd.read_csv(os.path.join(data_dir, "class3_ordered.txt"), header=None).squeeze().tolist()

    # Assign labels
    condition = ['LUAD'] * len(class1_samples) + ['Intermediate'] * len(class2_samples) + ['LUSC'] * len(class3_samples)
    labels = [0] * len(class1_samples) + [np.nan] * len(class2_samples) + [1] * len(class3_samples)
    sample_names = class1_samples + class2_samples + class3_samples

    # Create AnnData object
    adata = AnnData(luas_data.loc[:, sample_names].T)
    adata.obs['condition'] = condition
    adata.obs['labels'] = labels

    # Calculate DEGs between LUAD and LUSC
    adata = calculate_deg(adata, group1_label='LUAD', group2_label='LUSC', alpha=0.05, fc_threshold=1.2)

    extra_genes = ["CXCL3", "CXCL5", "CXCL8", "KDM1A", "STK11", "AKT1", "RAC1", "ALK"]

    SOX2_target = ['KRT6B', 'PKP1', 'KRT6A', 'KRT14', 'PI3', 'TRIM29', 'S100A7', 'SERPINB5', 'S100A8', 'MIR205HG',
                   'CSTA', 'S100A2', 'TNS4', 'SPRR3', 'KRT6C', 'PTHLH', 'GPX2', 'PERP', 'FST', 'ITGA6',
                   'BMP7', 'SERPINB2', 'ADH7', 'JAG1', 'SFN', 'GRHL3', 'KRT1', 'HMGA2', 'UPK1B', 'CDK6',
                   'HAS3', 'KCNG1', 'SNAI2', 'ITGB8', 'KRT10', 'FGFR3', 'ID1', 'STON2', 'IVL', 'TP53AIP1',
                   'UGT1A7', 'SERPINE1', 'PRNP', 'WNT5A', 'ADM', 'CDH3', 'TINAGL1', 'ITGB4', 'EMP1', 'FOSL1',
                   'HBEGF', 'PARD6G', 'KIF23', 'ULBP2', 'DKK1', 'FGFR2', 'ITGA2', 'ALDH1A3', 'TWIST1', 'MYC',
                   'CDK1', 'ADA', 'NCS1', 'DLX1', 'MKI67', 'MCM10', 'RRM2', 'DST', 'ACTL6A', 'GNAI1',
                   'HK2', 'BLM', 'IL1RAP', 'MIR205', 'GAPDH', 'RIN1', 'RACGAP1', 'CCNE1', 'PIK3CA', 'FUT1',
                   'CKS2', 'LIMK2', 'MAD2L1', 'CDKN1A', 'CCNB1', 'PNPT1', 'RAD51', 'SMAD3', 'PTEN', 'NOTCH3',
                   'STMN1', 'PRPF4', 'CCNA2', 'PCNA', 'CFLAR', 'TSC2', 'ENG', 'CTSD', 'TGFBR2', 'DOK1',
                   'CDK18', 'STX1A', 'CITED2', 'BOK', 'TNFSF15', 'GLB1L2', 'FOXA2', 'WFDC2', 'MLPH', 'COL4A3',
                   'AGR2', 'CLDN3', 'KRT7', 'CXCL17']

    FOXA2_target = ['PPP1R14C', 'FABP5', 'WNT7A', 'EN1', 'PRNP', 'WNT5A', 'PAX6', 'RARG', 'UST', 'HK2',
                    'FOXO1', 'THBS3', 'PON1', 'DDC', 'UCP2', 'A2M', 'ONECUT1', 'TBXT', 'CHRD', 'C5',
                    'C3', 'TBXAS1', 'PIFO', 'CD55', 'RARA', 'AMY2B', 'SOD3', 'FOXA1', 'CHI3L1', 'SERPINA1',
                    'TMPRSS2', 'HNF1B', 'SFTPD', 'AGR2', 'ABCA3', 'HPGD', 'RORC', 'SFTPA1', 'MUC5B',
                    'SFTPB', 'NKX2-1']

    TP63_target = ['KRT6B', 'PKP1', 'KRT6A', 'KRT14', 'PI3', 'TRIM29', 'S100A7', 'SERPINB5', 'S100A8', 'MIR205HG',
                   'CSTA', 'S100A2', 'TNS4', 'SPRR3', 'KRT6C', 'PTHLH', 'GPX2', 'PERP', 'FST', 'ITGA6',
                   'BMP7', 'SERPINB2', 'ADH7', 'JAG1', 'SFN', 'GRHL3', 'KRT1', 'HMGA2', 'UPK1B', 'CDK6',
                   'HAS3', 'KCNG1', 'SNAI2', 'ITGB8', 'KRT10', 'FGFR3', 'ID1', 'STON2', 'IVL', 'TP53AIP1',
                   'UGT1A7', 'SERPINE1', 'PRNP', 'WNT5A', 'ADM', 'CDH3', 'TINAGL1', 'ITGB4', 'EMP1', 'FOSL1',
                   'HBEGF', 'PARD6G', 'KIF23', 'ULBP2', 'DKK1', 'FGFR2', 'ITGA2', 'ALDH1A3', 'TWIST1', 'MYC',
                   'CDK1', 'ADA', 'NCS1', 'DLX1', 'MKI67', 'MCM10', 'RRM2', 'DST', 'ACTL6A', 'GNAI1',
                   'HK2', 'BLM', 'IL1RAP', 'MIR205', 'GAPDH', 'RIN1', 'RACGAP1', 'CCNE1', 'PIK3CA', 'FUT1',
                   'CKS2', 'LIMK2', 'MAD2L1', 'CDKN1A', 'CCNB1', 'PNPT1', 'RAD51', 'SMAD3', 'PTEN', 'NOTCH3',
                   'STMN1', 'PRPF4', 'CCNA2', 'PCNA', 'CFLAR', 'TSC2', 'ENG', 'CTSD', 'TGFBR2', 'DOK1',
                   'CDK18', 'STX1A', 'CITED2', 'BOK', 'TNFSF15', 'GLB1L2', 'FOXA2', 'WFDC2', 'MLPH', 'COL4A3',
                   'AGR2', 'CLDN3', 'KRT7', 'CXCL17']

    NKX21_target = ['TP63', 'GJB3', 'FXYD3', 'SOX2', 'SNAI2', 'SERPINE1', 'PFN2', 'CKS1B', 'WNT5B', 'RAG1',
                    'SEC14L2', 'RRM2', 'PDPN', 'PRRX2', 'SCARA3', 'EPHB3', 'RAD51', 'HELLS', 'ESAM', 'PON1',
                    'SMAD7', 'HCK', 'QSOX1', 'ROR1', 'C5', 'CHAD', 'INMT', 'LRG1', 'SCNN1G', 'SCN7A',
                    'MYBPH', 'AGT', 'AQP1', 'TFF3', 'CD74', 'ADGRF5', 'SFTPD', 'LRP2', 'AGR2', 'DPP4',
                    'AQP5', 'ROS1', 'LMO3', 'SFTPA1', 'SFTPB', 'SLC34A2', 'NAPSA']
    driver = ["FOXA2", "NKX2-1", "SOX2", "TP63"]
    target = SOX2_target + FOXA2_target + TP63_target + NKX21_target
    adata.var['is_extra'] = adata.var_names.isin(extra_genes)
    adata.var['is_prior_driver'] = adata.var_names.isin(driver)
    adata.var['is_target'] = adata.var_names.isin(target)
    adata.var['is_tf'] = adata.var_names.isin(tf_list)

    features_mask = adata.var['is_deg'] | adata.var['is_extra']
    samples_mask = (adata.obs['condition'] == 'LUAD') | (adata.obs['condition'] == 'LUSC')
    adata_filter = adata[samples_mask, features_mask].copy()

    return adata, adata_filter


def load_luas_mouse_adata(
        data_dir=r"D:\sci_job\casual\causal3.0",
        tf_list=None
):
    if tf_list is None:
        tf_list = []

    mouse_luas_path = os.path.join(data_dir, r"Mous_LUAS.csv")
    mouse_driver = ["Foxa2", "Nkx2-1", "Sox2", "Trp63"]
    extra_genes = []

    mouse_luas = pd.read_csv(mouse_luas_path, index_col=0)
    mouse_luas = mouse_luas[mouse_luas.var(axis=1) > 0.1]

    adata = anndata.AnnData(X=mouse_luas.T)
    adata.obs['time'] = adata.obs_names.str.extract(r'(\d+W|NL)')[0].values
    conditions = {
        'NL': 'Normal',
        '4W': 'ADC', '5W': 'ADC', '6W': 'ADC', '7W': 'ADC_TP',
        '8W': 'ADC_SCC',
        '9W': 'SCC', '10W': 'SCC'
    }
    adata.obs['condition'] = adata.obs['time'].map(conditions)
    condition_to_label = {'ADC': 0, 'SCC': 1}
    adata.obs['labels'] = adata.obs['condition'].map(condition_to_label)
    # Calculate DEGs between LUAD and LUSC  # fold change maybe not accurate for log-scale data
    adata = calculate_deg(adata, group1_label='ADC', group2_label='SCC', alpha=0.05, fc_threshold=2)

    adata.var['is_extra'] = adata.var_names.isin(extra_genes)
    adata.var['is_prior_driver'] = adata.var_names.isin(mouse_driver)
    adata.var['is_tf'] = adata.var_names.isin(tf_list)

    features_mask = adata.var['is_deg'] | adata.var['is_extra']
    samples_mask = (adata.obs['condition'] == 'ADC') | (adata.obs['condition'] == 'SCC')
    adata_filter = adata[samples_mask, features_mask].copy()
    adata = adata[:, features_mask].copy()
    return adata, adata_filter


def human_adata(
        main_dir=r"D:\sci_job\casual\causal3.0",
        downstream_list=None

):
    supplementary_list = ["CXCL3", "CXCL5", "CXCL8", "KDM1A", "STK11", "AKT1", "RAC1", "ALK"]

    SOX2 = ['KRT6B', 'PKP1', 'KRT6A', 'KRT14', 'PI3', 'TRIM29', 'S100A7', 'SERPINB5', 'S100A8', 'MIR205HG',
            'CSTA', 'S100A2', 'TNS4', 'SPRR3', 'KRT6C', 'PTHLH', 'GPX2', 'PERP', 'FST', 'ITGA6',
            'BMP7', 'SERPINB2', 'ADH7', 'JAG1', 'SFN', 'GRHL3', 'KRT1', 'HMGA2', 'UPK1B', 'CDK6',
            'HAS3', 'KCNG1', 'SNAI2', 'ITGB8', 'KRT10', 'FGFR3', 'ID1', 'STON2', 'IVL', 'TP53AIP1',
            'UGT1A7', 'SERPINE1', 'PRNP', 'WNT5A', 'ADM', 'CDH3', 'TINAGL1', 'ITGB4', 'EMP1', 'FOSL1',
            'HBEGF', 'PARD6G', 'KIF23', 'ULBP2', 'DKK1', 'FGFR2', 'ITGA2', 'ALDH1A3', 'TWIST1', 'MYC',
            'CDK1', 'ADA', 'NCS1', 'DLX1', 'MKI67', 'MCM10', 'RRM2', 'DST', 'ACTL6A', 'GNAI1',
            'HK2', 'BLM', 'IL1RAP', 'MIR205', 'GAPDH', 'RIN1', 'RACGAP1', 'CCNE1', 'PIK3CA', 'FUT1',
            'CKS2', 'LIMK2', 'MAD2L1', 'CDKN1A', 'CCNB1', 'PNPT1', 'RAD51', 'SMAD3', 'PTEN', 'NOTCH3',
            'STMN1', 'PRPF4', 'CCNA2', 'PCNA', 'CFLAR', 'TSC2', 'ENG', 'CTSD', 'TGFBR2', 'DOK1',
            'CDK18', 'STX1A', 'CITED2', 'BOK', 'TNFSF15', 'GLB1L2', 'FOXA2', 'WFDC2', 'MLPH', 'COL4A3',
            'AGR2', 'CLDN3', 'KRT7', 'CXCL17']

    FOXA2 = ['PPP1R14C', 'FABP5', 'WNT7A', 'EN1', 'PRNP', 'WNT5A', 'PAX6', 'RARG', 'UST', 'HK2',
             'FOXO1', 'THBS3', 'PON1', 'DDC', 'UCP2', 'A2M', 'ONECUT1', 'TBXT', 'CHRD', 'C5',
             'C3', 'TBXAS1', 'PIFO', 'CD55', 'RARA', 'AMY2B', 'SOD3', 'FOXA1', 'CHI3L1', 'SERPINA1',
             'TMPRSS2', 'HNF1B', 'SFTPD', 'AGR2', 'ABCA3', 'HPGD', 'RORC', 'SFTPA1', 'MUC5B',
             'SFTPB', 'NKX2-1']

    TP63 = ['KRT6B', 'PKP1', 'KRT6A', 'KRT14', 'PI3', 'TRIM29', 'S100A7', 'SERPINB5', 'S100A8', 'MIR205HG',
            'CSTA', 'S100A2', 'TNS4', 'SPRR3', 'KRT6C', 'PTHLH', 'GPX2', 'PERP', 'FST', 'ITGA6',
            'BMP7', 'SERPINB2', 'ADH7', 'JAG1', 'SFN', 'GRHL3', 'KRT1', 'HMGA2', 'UPK1B', 'CDK6',
            'HAS3', 'KCNG1', 'SNAI2', 'ITGB8', 'KRT10', 'FGFR3', 'ID1', 'STON2', 'IVL', 'TP53AIP1',
            'UGT1A7', 'SERPINE1', 'PRNP', 'WNT5A', 'ADM', 'CDH3', 'TINAGL1', 'ITGB4', 'EMP1', 'FOSL1',
            'HBEGF', 'PARD6G', 'KIF23', 'ULBP2', 'DKK1', 'FGFR2', 'ITGA2', 'ALDH1A3', 'TWIST1', 'MYC',
            'CDK1', 'ADA', 'NCS1', 'DLX1', 'MKI67', 'MCM10', 'RRM2', 'DST', 'ACTL6A', 'GNAI1',
            'HK2', 'BLM', 'IL1RAP', 'MIR205', 'GAPDH', 'RIN1', 'RACGAP1', 'CCNE1', 'PIK3CA', 'FUT1',
            'CKS2', 'LIMK2', 'MAD2L1', 'CDKN1A', 'CCNB1', 'PNPT1', 'RAD51', 'SMAD3', 'PTEN', 'NOTCH3',
            'STMN1', 'PRPF4', 'CCNA2', 'PCNA', 'CFLAR', 'TSC2', 'ENG', 'CTSD', 'TGFBR2', 'DOK1',
            'CDK18', 'STX1A', 'CITED2', 'BOK', 'TNFSF15', 'GLB1L2', 'FOXA2', 'WFDC2', 'MLPH', 'COL4A3',
            'AGR2', 'CLDN3', 'KRT7', 'CXCL17']

    NKX21 = ['TP63', 'GJB3', 'FXYD3', 'SOX2', 'SNAI2', 'SERPINE1', 'PFN2', 'CKS1B', 'WNT5B', 'RAG1',
             'SEC14L2', 'RRM2', 'PDPN', 'PRRX2', 'SCARA3', 'EPHB3', 'RAD51', 'HELLS', 'ESAM', 'PON1',
             'SMAD7', 'HCK', 'QSOX1', 'ROR1', 'C5', 'CHAD', 'INMT', 'LRG1', 'SCNN1G', 'SCN7A',
             'MYBPH', 'AGT', 'AQP1', 'TFF3', 'CD74', 'ADGRF5', 'SFTPD', 'LRP2', 'AGR2', 'DPP4',
             'AQP5', 'ROS1', 'LMO3', 'SFTPA1', 'SFTPB', 'SLC34A2', 'NAPSA']

    if downstream_list is None:
        downstream_list = list(set(SOX2 + NKX21 + FOXA2 + TP63 + supplementary_list))

    humam_luas_path = os.path.join(main_dir, r"LUAS\data\Hum_LUAS.csv")
    human_class1_path = os.path.join(main_dir, r"LUAS\data\class1_ordered.txt")
    # human_class2_path = os.path.join(main_dir, r"real_data\LUAS\class2_ordered.txt")
    human_class3_path = os.path.join(main_dir, r"LUAS\data\class3_ordered.txt")
    driver = ["FOXA2", "NKX2-1", "SOX2", "TP63"]

    human_luas = pd.read_csv(humam_luas_path, index_col=0)
    human_class1 = pd.read_csv(human_class1_path, header=None, names=['class1'])
    # human_class2 = pd.read_csv(human_class2_path, header=None, names=['class2'])
    human_class3 = pd.read_csv(human_class3_path, header=None, names=['class3'])

    human_luas_AD_SC_data = pd.concat([human_luas[human_class1["class1"].tolist()],
                                       human_luas[human_class3["class3"].tolist()]], axis=1)

    human_label = np.concatenate((np.zeros(len(human_class1.index)), np.ones(len(human_class3.index))))

    tmp, human_deg_list = mannwhitneyu_func(human_luas_AD_SC_data, human_label)

    human_deg_list = human_deg_list + supplementary_list

    human_luas_AD_SC_filter_data = human_luas_AD_SC_data.loc[human_deg_list, :]

    feature_name = human_luas_AD_SC_filter_data.index

    feature_types = np.zeros(len(feature_name), dtype=int)
    for i, feature in enumerate(feature_name):
        if feature in driver:
            feature_types[i] = 1

    downstream_types = np.zeros(len(feature_name), dtype=int)
    for i, feature in enumerate(feature_name):
        if feature in downstream_list:
            downstream_types[i] = 1

    tf_types = np.zeros(len(feature_name), dtype=int)
    for i, feature in enumerate(feature_name):
        if feature in driver:
            tf_types[i] = 1

    human_adata = AnnData(human_luas_AD_SC_filter_data.values.T, dtype=human_luas_AD_SC_filter_data.values.T.dtype)
    human_adata.obs["labels"] = human_label
    human_adata.var["feat_type"] = feature_types
    human_adata.var["downstream_type"] = downstream_types
    human_adata.var["tf_type"] = tf_types
    human_adata.var.index = feature_name

    return human_adata, human_luas_AD_SC_filter_data


def mouse_adata(
        main_dir=r"D:\sci_job\casual\causal3.0",
        downstream_list=None
):
    supplementary_list = ["Stk11", "Napsa", "Krt5", "Ctnnb1"]

    TF_list = ["Nkx2-1", "Foxa2", "Nfe2", "Pbxip1", "Trp63", "Grhl3", "Sox2", "Foxm1", "Tfap2a", "Smarcb1", "Snai2"]

    Foxa2 = ["Ascl1", "Birc2", "Cbx5", "Ccl11", "Ccl20", "Cdc73", "Cpt1a", "Cul2", "Ferd3l", "Ferd3l", "G6pc", "G6pc2",
             "Gcg", "Gli2", "Hadh", "Hcrt", "Helt", "Hhex", "Hmgcs2", "Il13", "Il33", "Il4", "Lcn5", "Lcn5", "Lmx1a",
             "Lmx1b", "Map1b", "Mchr1", "Msx1", "Muc2", "Otx1", "Otx2", "Pdx1", "Pdx1", "Pgf", "Reg3g", "Sox17", "Sox2",
             "St18", "Tbx1", "Tle4", "Vtn", "Wnt7b"]

    Nkx21 = ["Abca3", "Foxj1", "Hcrtr2", "Pax8", "Prdm13", "Scgb1a1", "Sftpb", "Sftpb", "Sftpc", "Sftpc", "Shh",
             "Sox17",
             "Wnt7b"]

    Sox2 = ["Atoh1", "Birc5", "Ccnd1", "Cdkn1b", "Egfr", "Fgf4", "Fgf4", "Fgf4", "Gli3", "Jag1", "Lef1", "Lefty1",
            "Ly6a",
            "Mir302a", "Mycn", "Nanog", "Nanog", "Neurod1", "Neurog1", "Neurog1", "Nkx2-1", "Nr2e1", "Pou1f1", "Pou5f1",
            "Pou5f1", "Sema6a", "Shh", "Six3", "Sox2", "Utf1", "Utf1", "Xist", "Zscan10", "Zfp819"]

    Trp63 = ["Cdkn1a", "Evpl", "Irf6", "Krt14", "Ptk2", "Nectin1", "Satb1"]

    if downstream_list is None:
        downstream_list = list(set(Sox2 + Nkx21 + Foxa2 + Trp63 + supplementary_list))

    mouse_luas_path = os.path.join(main_dir, r"real_data\LUAS\Mous_LUAS.csv")
    mouse_class = [
        # "NL",
        "4W",
        "6W",
        "7W",
        # "8AW",
        # "8SW",
        "9W",
        "10W"
    ]
    mouse_driver = [
        "Foxa2",
        "Nkx2-1",
        # "Foxa1",
        "Sox2",
        "Trp63"
    ]
    mouse_luas = pd.read_csv(mouse_luas_path, index_col=0)
    mouse_luas_AD_SC_data = mouse_luas.filter(regex=f"({'|'.join(mouse_class)})", axis=1)
    mouse_label = np.concatenate((np.zeros(12), np.ones(8)))

    tmp, mouse_deg_list = mannwhitneyu_func(mouse_luas_AD_SC_data, mouse_label)
    mouse_deg_list = mouse_deg_list + ["Stk11", "Ctnnb1"]
    mouse_luas_AD_SC_filter_data = mouse_luas_AD_SC_data.loc[mouse_deg_list, :]

    feature_name = mouse_luas_AD_SC_filter_data.index

    feature_types = np.where(feature_name.isin(mouse_driver), 1, 0)

    downstream_types = np.where(feature_name.isin(downstream_list), 1, 0)

    tf_types = np.where(feature_name.isin(TF_list), 1, 0)

    mouse_adata = AnnData(mouse_luas_AD_SC_filter_data.values.T, dtype=mouse_luas_AD_SC_filter_data.values.T.dtype)
    mouse_adata.obs["labels"] = mouse_label
    mouse_adata.var["feat_type"] = feature_types
    mouse_adata.var["downstream_type"] = downstream_types
    mouse_adata.var["tf_type"] = tf_types
    mouse_adata.var.index = feature_name

    return mouse_adata


# ======================================== module.utils ========================================
def calculate_deg(adata, group1_label='LUAD', group2_label='LUSC', alpha=0.05, fc_threshold=1.2):
    """
    Calculate DEGs using Mann-Whitney U test, considering fold change threshold.
    """
    # Extract expression data for both groups
    group1_data = adata[adata.obs['condition'] == group1_label].X
    group2_data = adata[adata.obs['condition'] == group2_label].X

    # Initialize columns in adata.var for storing results
    adata.var['p_value'] = np.nan
    adata.var['fold_change'] = np.nan
    adata.var['is_deg'] = False

    for gene_idx, gene_name in enumerate(adata.var_names):
        expr1 = group1_data[:, gene_idx]
        expr2 = group2_data[:, gene_idx]

        # Mann-Whitney U test
        _, p_value = mannwhitneyu(expr1, expr2, alternative='two-sided')

        # Calculate Fold Change (mean of group 2 / mean of group 1)
        fc = np.mean(expr2) / np.mean(expr1 + 1e-6)

        # Update adata.var with the results
        adata.var.at[gene_name, 'p_value'] = p_value
        adata.var.at[gene_name, 'fold_change'] = fc

        # Determine if the gene is a DEG based on p-value and fold change threshold
        is_deg = (p_value < alpha) and ((fc >= fc_threshold) or (fc <= 1 / fc_threshold))
        adata.var.at[gene_name, 'is_deg'] = is_deg

    return adata


def mannwhitneyu_func(
        data,
        labels,
        p_value_threshold=0.05,
        fold_change_threshold=1
):
    mannwhitneyu_results = []

    # 对每个特征执行Wilcoxon秩和检验，并计算log2 fold change
    for feature in data.index:
        statistic, p_value = mannwhitneyu(data.loc[feature, labels == 0], data.loc[feature, labels == 1])

        # 计算fold change
        fold_change = data.loc[feature, labels == 0].mean() - data.loc[feature, labels == 1].mean()
        fold_change = round(fold_change, 2)

        mannwhitneyu_results.append({
            'Feature': feature,
            'Statistic': statistic,
            'P-Value': p_value,
            'Fold Change': fold_change
        })

    # 将结果转换为DataFrame
    mannwhitneyu_df = pd.DataFrame(mannwhitneyu_results)

    filtered_features = mannwhitneyu_df[(mannwhitneyu_df['P-Value'] < p_value_threshold) &
                                        ((mannwhitneyu_df['Fold Change'] >= fold_change_threshold) |
                                         (mannwhitneyu_df['Fold Change'] <= -fold_change_threshold))]

    # 获取特征名称并输出为列表
    selected_features = filtered_features['Feature'].tolist()

    return mannwhitneyu_df, selected_features


def find_index(column_names, adata):
    row_numbers = []

    for value in column_names:
        try:
            row_number = adata.var.index.get_loc(value)
            row_numbers.append(row_number)
        except KeyError:
            row_numbers.append(None)

    return row_numbers


def human_data_direction(weights_full, index, column_names):
    weights_full_filter = weights_full[:, index]

    tmp_df = pd.DataFrame(weights_full_filter, columns=column_names)
    AD_mean = tmp_df.head(44).mean()
    SC_mean = tmp_df.tail(26).mean()
    all_mean = tmp_df.mean()

    return AD_mean, SC_mean, all_mean


def mouse_data_direction(weights_full, index, column_names):
    weights_full_filter = weights_full[:, index]

    tmp_df = pd.DataFrame(weights_full_filter, columns=column_names)
    AD_mean = tmp_df.head(12).mean()
    SC_mean = tmp_df.tail(8).mean()
    all_mean = tmp_df.mean()

    return AD_mean, SC_mean, all_mean


def result_add_direction(weights_full, result, n_AD, n_SC):
    weights_full = weights_full.T
    supplementary = pd.DataFrame()
    supplementary['direction'] = weights_full.mean(axis=1)
    supplementary['AD_direction'] = np.mean(weights_full[:, :n_AD], axis=1)
    supplementary['SC_direction'] = np.mean(weights_full[:, -n_SC:], axis=1)
    supplementary['weight'] = np.mean(np.abs(weights_full), axis=1)
    supplementary['weight'] = supplementary['weight'] / np.sum(supplementary['weight'])
    supplementary = supplementary.sort_values(by="weight", ascending=False)
    supplementary.index = result.index
    result_df = pd.concat([result, supplementary.iloc[:, :3]], axis=1)
    return result_df


def result_add_direction2(weights_full, result, star, end):
    weights_full = weights_full.T
    supplementary = pd.DataFrame()
    supplementary['direction'] = weights_full.mean(axis=1)
    supplementary['star_direction'] = np.mean(weights_full[:, star], axis=1)
    supplementary['end_direction'] = np.mean(weights_full[:, end], axis=1)
    supplementary['weight'] = np.mean(np.abs(weights_full), axis=1)
    supplementary['weight'] = supplementary['weight'] / np.sum(supplementary['weight'])
    supplementary = supplementary.sort_values(by="weight", ascending=False)
    supplementary.index = result.index
    result_df = pd.concat([result, supplementary.iloc[:, :3]], axis=1)
    return result_df


def cumulative_weight_sum_rate(df):
    cumulative_weight_sum = df["weight"].cumsum()
    total_weight_sum = df["weight"].sum()
    df["w_cum_rate"] = cumulative_weight_sum / total_weight_sum

    return df


def plot_w_cum_rate(df, head_num):
    plt.figure(figsize=(10, 6))
    df_subset = df.iloc[:head_num]
    plt.plot(df_subset['gene_symbols'], df_subset['w_cum_rate'], marker='o')
    plt.title('Line Plot of features')
    plt.xlabel('gene Label')
    plt.ylabel('cum_rate')
    plt.xticks(rotation=90)
    plt.show()


def calculate_w1_w2(
        w1,
        w2,
        method='SHAP',
        save=False,
        save_dir=''
):
    fig, ax = plt.subplots()
    ax.boxplot([w1, w2], labels=['causal', 'spurious'])
    if method == 'SHAP':
        ax.set_title("SHAP Feature weights")
    elif method == 'Grad':
        ax.set_title("Grad Feature weights")
    elif method == 'Model':
        ax.set_title("Model Feature weights")
    if save:
        plt.savefig(os.path.join(save_dir, method + '_weight_box.png'))
    plt.show()

    # Perform a two-sample t-test assuming equal variance
    t_stat, p_value = ttest_ind(w1, w2, equal_var=True, alternative='greater')
    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # Perform a Mann-Whitney U test
    u_stat, u_p_value = mannwhitneyu(w1, w2, alternative='greater')
    print('U-statistic:', u_stat)
    print('p-value:', u_p_value)

    with open(os.path.join(save_dir, method + '_P_value.txt'), 'w') as f:
        f.write("T.test:\n")
        f.write(f't-statistic: {t_stat}\n')
        f.write(f'p-value: {p_value}\n')
        f.write('')
        f.write("Mann-Whitney U test:\n")
        f.write(f't-statistic: {u_stat}\n')
        f.write(f'p-value: {u_p_value}\n')


def human_all_adata(
        main_dir=r"D:\sci_job\casual\causal3.0",

):
    humam_luas_path = os.path.join(main_dir, r"real_data\LUAS\Hum_LUAS.csv")
    human_class1_path = os.path.join(main_dir, r"real_data\LUAS\class1_ordered.txt")
    human_class2_path = os.path.join(main_dir, r"real_data\LUAS\class2_ordered.txt")
    human_class3_path = os.path.join(main_dir, r"real_data\LUAS\class3_ordered.txt")
    driver = ["FOXA2", "NKX2-1", "SOX2", "TP63"]

    human_luas = pd.read_csv(humam_luas_path, index_col=0)
    human_class1 = pd.read_csv(human_class1_path, header=None, names=['class1'])
    human_class2 = pd.read_csv(human_class2_path, header=None, names=['class2'])
    human_class3 = pd.read_csv(human_class3_path, header=None, names=['class3'])

    human_luas_AD_SC_data = pd.concat([human_luas[human_class1["class1"].tolist()],
                                       human_luas[human_class3["class3"].tolist()]], axis=1)

    human_label = np.concatenate((np.zeros(len(human_class1.index)), np.ones(len(human_class3.index))))

    tmp, human_deg_list = mannwhitneyu_func(human_luas_AD_SC_data, human_label)

    human_luas_AD_SC_data = pd.concat(
        [human_luas_AD_SC_data, human_luas[human_class2["class2"].tolist()]], axis=1
    )

    human_label = np.zeros(len(human_class1.index) + len(human_class2.index) + len(human_class3.index))

    supplementary_list = ["CXCL3", "CXCL5", "CXCL8", "KDM1A", "STK11", "AKT1", "RAC1", "ALK"]

    human_deg_list = human_deg_list + supplementary_list

    human_luas_AD_SC_filter_data = human_luas_AD_SC_data.loc[human_deg_list, :]

    feature_name = human_luas_AD_SC_filter_data.index

    feature_types = np.zeros(len(feature_name), dtype=int)

    human_adata = AnnData(human_luas_AD_SC_filter_data.values.T, dtype=human_luas_AD_SC_filter_data.values.T.dtype)
    human_adata.obs["labels"] = human_label
    human_adata.var["feat_type"] = feature_types
    human_adata.var.index = feature_name

    return human_adata


def mouse_all_adata(
        main_dir=r"D:\sci_job\casual\causal3.0"
):
    mouse_luas_path = os.path.join(main_dir, r"real_data\LUAS\Mous_LUAS.csv")
    mouse_class = [
        # "NL",
        "4W",
        "6W",
        "7W",
        # "8AW",
        # "8SW",
        "9W",
        "10W"]
    mouse_driver = ["Foxa2", "Nkx2-1",
                    # "Foxa1",
                    "Sox2", "Trp63"]

    mouse_luas = pd.read_csv(mouse_luas_path, index_col=0)
    mouse_luas_AD_SC_data = mouse_luas.filter(regex=f"({'|'.join(mouse_class)})", axis=1)
    deg_label = np.concatenate((np.zeros(12), np.ones(8)))

    tmp, mouse_deg_list = mannwhitneyu_func(mouse_luas_AD_SC_data, deg_label)

    mouse_class = [
        # "NL",
        "4W",
        "6W",
        "7W",
        "8W",
        "9W",
        "10W"]
    mouse_luas_AD_SC_data = mouse_luas.filter(regex=f"({'|'.join(mouse_class)})", axis=1)
    mouse_label = np.zeros(28)
    mouse_deg_list = mouse_deg_list + ["Stk11", "Ctnnb1"]
    mouse_luas_AD_SC_filter_data = mouse_luas_AD_SC_data.loc[mouse_deg_list, :]

    feature_name = mouse_luas_AD_SC_filter_data.index
    feature_types = np.where(feature_name.isin(mouse_driver), 1, 0)

    mouse_adata = AnnData(mouse_luas_AD_SC_filter_data.values.T, dtype=mouse_luas_AD_SC_filter_data.values.T.dtype)
    mouse_adata.obs["labels"] = mouse_label
    mouse_adata.var["feat_type"] = feature_types
    mouse_adata.var.index = feature_name

    return mouse_adata


# ======================================== model.utils ========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_feature_boxplots(data, labels, features=None, save_path=None):
    """
    Plot boxplots for features, grouped by labels, with feature names displayed on a single plot.
    """
    # Use all features if none are specified
    if features is None:
        features = data.columns.tolist()

    # Create a new DataFrame with selected features' values and each sample's label
    data_for_plot = data.loc[:, features].copy()
    data_for_plot['Label'] = labels

    # Convert the DataFrame from wide to long format
    data_long_format = data_for_plot.melt(id_vars='Label', var_name='Feature', value_name='Weight')

    # Plot boxplots on a single plot
    plt.figure(figsize=(len(features) * 0.5, 6))  # Adjust figure size based on number of features
    sns.boxplot(x='Feature', y='Weight', hue='Label', data=data_long_format)
    plt.xticks(rotation=90)  # Rotate feature names for better visibility
    plt.tight_layout()  # Adjust layout to make room for feature names

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()
    # plt.show()


def merge_basic_driver(driver_total, driver_0, driver_1, weight_total, weight_0, weight_1):
    # Get the union of all drivers
    drivers = list(set(driver_total).union(driver_0).union(driver_1))

    # Create a copy of weight_total for manipulation
    merged_weight = weight_total.loc[drivers].copy()
    merged_weight.rename(columns={'weight': 'weight_total'}, inplace=True)
    merged_weight.rename(columns={'weight_dir': 'weight_total_dir'}, inplace=True)

    # Add new columns from weight_0 and weight_1 to merged_weight
    merged_weight['weight_0'] = weight_0.loc[drivers, 'weight']
    merged_weight['weight_0_dir'] = weight_0.loc[drivers, 'weight_dir']
    merged_weight['weight_1'] = weight_1.loc[drivers, 'weight']
    merged_weight['weight_1_dir'] = weight_1.loc[drivers, 'weight_dir']

    # Add new columns indicating if each driver is in the index of driver_total, driver_0, and driver_1
    merged_weight['is_driver_total'] = merged_weight.index.isin(driver_total)
    merged_weight['is_driver_0'] = merged_weight.index.isin(driver_0)
    merged_weight['is_driver_1'] = merged_weight.index.isin(driver_1)

    return merged_weight


def merge_complex_driver(driver_shap_total, driver_shap_0, driver_shap_1, weight_shap_total, weight_shap_0,
                         weight_shap_1,
                         driver_grad_total, driver_grad_0, driver_grad_1, weight_grad_total, weight_grad_0,
                         weight_grad_1):
    # get the union of all drivers
    drivers = list(set(driver_shap_total).union(driver_shap_0).union(driver_shap_1)
                   .union(driver_grad_total).union(driver_grad_0).union(driver_grad_1))

    # Create a copy of weight_shap_total for manipulation
    merged_weight = weight_shap_total.loc[drivers].copy()
    merged_weight.rename(columns={'weight': 'weight_shap_total'}, inplace=True)
    merged_weight.rename(columns={'weight_dir': 'weight_shap_total_dir'}, inplace=True)

    # Add new columns from weight_shap_0 and weight_shap_1 to merged_weight
    merged_weight['weight_shap_0'] = weight_shap_0.loc[drivers, 'weight']
    merged_weight['weight_shap_0_dir'] = weight_shap_0.loc[drivers, 'weight_dir']
    merged_weight['weight_shap_1'] = weight_shap_1.loc[drivers, 'weight']
    merged_weight['weight_shap_1_dir'] = weight_shap_1.loc[drivers, 'weight_dir']

    # Add new columns from weight_grad_total to merged_weight
    merged_weight['weight_grad_total'] = weight_grad_total.loc[drivers, 'weight']
    merged_weight['weight_grad_total_dir'] = weight_grad_total.loc[drivers, 'weight_dir']

    # Add new columns from weight_grad_0 and weight_grad_1 to merged_weight
    merged_weight['weight_grad_0'] = weight_grad_0.loc[drivers, 'weight']
    merged_weight['weight_grad_0_dir'] = weight_grad_0.loc[drivers, 'weight_dir']
    merged_weight['weight_grad_1'] = weight_grad_1.loc[drivers, 'weight']
    merged_weight['weight_grad_1_dir'] = weight_grad_1.loc[drivers, 'weight_dir']

    # Add new columns indicating if each driver is in index of driver_total, driver_0, and driver_1 for SHAP and Grad
    merged_weight['is_driver_shap_total'] = merged_weight.index.isin(driver_shap_total)
    merged_weight['is_driver_shap_0'] = merged_weight.index.isin(driver_shap_0)
    merged_weight['is_driver_shap_1'] = merged_weight.index.isin(driver_shap_1)
    merged_weight['is_driver_grad_total'] = merged_weight.index.isin(driver_grad_total)
    merged_weight['is_driver_grad_0'] = merged_weight.index.isin(driver_grad_0)
    merged_weight['is_driver_grad_1'] = merged_weight.index.isin(driver_grad_1)

    return merged_weight


def prepare_network(adata, prior_net, add_edges_pct=0.001, corr_cutoff=0.6, uppercase=True):
    # Ensure gene names are uppercase to match network data
    adata.var_names_make_unique()
    if uppercase:
        adata.var_names = adata.var_names.str.upper()
        prior_net['from'] = prior_net['from'].str.upper()
        prior_net['to'] = prior_net['to'].str.upper()

    # Filter the network to include only genes present in adata
    net_filtered = prior_net[prior_net['from'].isin(adata.var_names) & prior_net['to'].isin(adata.var_names)]
    net_filtered = net_filtered.drop_duplicates(subset=['from', 'to'])

    # Create a directed graph using networkx
    network = nx.from_pandas_edgelist(net_filtered, 'from', 'to', create_using=nx.DiGraph)

    # Only keep the genes that exist in both single cell data and the prior gene interaction network
    network_nodes = list(network.nodes())
    adata_filtered = adata[:, network_nodes]

    # Calculate expression correlation to add additional edges
    gene_exp = pd.DataFrame(adata_filtered.X.A if sparse.issparse(adata_filtered.X) else adata_filtered.X,
                            columns=adata_filtered.var_names)
    corr_matrix = gene_exp.corr(method='spearman').abs()
    np.fill_diagonal(corr_matrix.values, 0)

    # Identify additional edges based on correlation threshold
    additional_edges = corr_matrix.stack().reset_index()
    additional_edges.columns = ['from', 'to', 'weight']
    additional_edges = additional_edges[additional_edges['weight'] > corr_cutoff]

    # Limit the number of additional edges based on the percentage
    top_k = int(len(network.nodes()) * (len(network.nodes()) - 1) / 2 * add_edges_pct)
    additional_edges = additional_edges.nlargest(top_k, 'weight')[['from', 'to']]

    # Add additional edges to the network, ensuring no duplicates
    existing_edges = set(network.edges())
    new_edges = [(row['from'], row['to']) for _, row in additional_edges.iterrows() if
                 (row['from'], row['to']) not in existing_edges]
    print(f"Adding {len(new_edges)} additional edges to the network")
    network.add_edges_from(new_edges)

    return network


def get_network(adata, network, keep_self_loops=True, average_degree=10, weight_degree=True, save_path=None):
    """
    Further process a gene regulatory network by calculating edge weights based on Spearman correlation
    coefficients from gene expression data, removing self-loops, adjusting edge weights based on node degrees,
    filtering edges to meet an average degree threshold, and finding the largest connected subgraph.
    """
    from scipy.sparse import issparse
    from scipy.stats import spearmanr
    # Convert adata to DataFrame for easier manipulation
    gene_exp = pd.DataFrame(adata.X.A if issparse(adata.X) else adata.X, columns=adata.var_names)

    # Calculate Spearman correlation coefficient for all edges in the network
    for u, v in network.edges():
        if u in gene_exp.columns and v in gene_exp.columns:
            coef, _ = spearmanr(gene_exp[u], gene_exp[v])
            network[u][v]['weight'] = abs(coef)

    # Remove self-loops if not allowed
    if not keep_self_loops:
        network.remove_edges_from(nx.selfloop_edges(network))

    # Adjust edge weights based on the degree of connected nodes
    for u, v, d in network.edges(data=True):
        u_degree = network.degree(u)
        v_degree = network.degree(v)
        # Adjust the weight by the average degree of the two nodes
        if weight_degree:
            d['weight'] *= (u_degree + v_degree) / 2
        # print(u, v, d['weight'])

    # If average_degree is specified, select edges to keep based on their adjusted weights
    if average_degree is not None:
        # Calculate the total number of edges to keep based on the average_degree
        total_edges_to_keep = int(average_degree * len(network.nodes()))
        # Sort edges by their adjusted weight in descending order and keep the top ones
        all_edges_sorted_by_weight = sorted(network.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        edges_to_keep = all_edges_sorted_by_weight[:total_edges_to_keep]

        # Create a new graph with the selected edges and their attributes
        network_filtered = nx.DiGraph()
        for u, v, d in edges_to_keep:
            network_filtered.add_edge(u, v, **d)
        network = network_filtered

    # Find the largest connected subgraph
    largest_components = max(nx.weakly_connected_components(network), key=len)
    if (len(largest_components) / len(network.nodes())) < 0.5:
        print('Warning: the size of the maximal connected subgraph is less than half of the input whole graph!')
    network = network.subgraph(largest_components).copy()

    # Save the network to CSV if a save path is provided
    if save_path is not None:
        # Convert the network into a DataFrame, only including edges with a 'weight' attribute
        edge_list = [(u, v, d['weight']) for u, v, d in network.edges(data=True)]
        network_df = pd.DataFrame(edge_list, columns=['Source', 'Target', 'Weight'])
        # Save the DataFrame to CSV
        network_df.to_csv(save_path, index=False)
    print(f"Number of nodes: {len(network.nodes)}")
    print(f"Number of edges: {len(network.edges)}")
    return network


def get_influence_score(network, gene_info, lam=0.8, power=1.0):
    """
    Calculate gene influence scores based on in-coming and out-going network connectivity, and gene causal weights.
    """
    # Initialize influence score DataFrame
    influence_score = pd.DataFrame(np.zeros((len(network.nodes), 2)),
                                   index=sorted(network.nodes),
                                   columns=['score_out', 'score_in'])  # Adjusted column names

    # Calculate out and in influence scores based on network connectivity
    for i, v in enumerate(['in', 'out']):
        gene_corr_score = np.sum(nx.to_numpy_array(network,
                                                   nodelist=sorted(network.nodes),
                                                   weight='weight' if v == 'out' else None),
                                 axis=1 - i)
        influence_score.iloc[:, i] = np.log1p(gene_corr_score).flatten().tolist()
        # influence_score.iloc[:, i] = np.sqrt(gene_corr_score).flatten().tolist()
    feature_weights = gene_info['weight'] ** power
    feature_weights /= feature_weights.sum()
    # Adjust scores based on gene feature weights from gene_info
    for gene in influence_score.index:
        if gene in feature_weights.index:
            adjusted_weight = feature_weights[gene]
            influence_score.loc[gene, ['score_out', 'score_in']] *= adjusted_weight

    # Combine out and in scores to get the final influence score
    influence_score['score'] = lam * influence_score['score_out'] + (1 - lam) * influence_score[
        'score_in']  # Adjusted formula

    # Sort genes by their influence score
    influence_score = influence_score.sort_values(by='score', ascending=False)

    return influence_score


def plot_control_scores(probs, control_scores_inc, control_scores_dec, metric='score', sample_size=None, alpha=0.5):
    """
    Plot control scores and probabilities.
    """
    if isinstance(control_scores_inc, pd.DataFrame):
        control_scores_inc = control_scores_inc[metric].to_numpy()
    if isinstance(control_scores_dec, pd.DataFrame):
        control_scores_dec = control_scores_dec[metric].to_numpy()

    # Downsample if sample_size is specified and less than the total number of samples
    if sample_size is not None and sample_size < len(probs):
        indices = np.random.choice(len(probs), sample_size, replace=False)
        control_scores_inc = control_scores_inc[indices]
        control_scores_dec = control_scores_dec[indices]
        probs = probs[indices]

    # Find the maximum control score for scaling
    max_control_score = max(np.max(np.abs(control_scores_inc)), np.max(np.abs(control_scores_dec)))

    # Plotting
    plt.figure(figsize=(15, 10))

    # Scatter plot for probabilities on y-axis
    plt.scatter(np.zeros(len(probs)), probs, color='blue', alpha=alpha, label='Probability')

    for i, prob in enumerate(probs):
        # Scale control scores to the maximum control score
        scaled_inc_score = (control_scores_inc[i] / max_control_score) * 0.1
        scaled_dec_score = (control_scores_dec[i] / max_control_score) * 0.1

        # Bar plot for increase scores on the right (red)
        plt.barh(prob, scaled_inc_score, height=0.01, color='red', alpha=alpha, left=0)
        # Bar plot for decrease scores on the left (blue)
        plt.barh(prob, -scaled_dec_score, height=0.01, color='blue', alpha=alpha, left=0)

    plt.xlabel(f'Control {metric.capitalize()}')
    plt.ylabel(f'Probability')
    plt.title(f'Control {metric.capitalize()} and Probability for Each Sample')

    # Create custom legend
    red_patch = mpatches.Patch(color='red', label='Increase Control')
    blue_patch = mpatches.Patch(color='blue', label='Decrease Control')
    plt.legend(handles=[red_patch, blue_patch])

    plt.xlim(-0.15, 0.15)  # Adjust the x-axis limits for better visualization
    plt.show()


def plot_control_scores_by_category(adata, control_details_inc, control_details_dec, metric='score'):
    """
    Plot control scores by category (label) for increase and decrease conditions using a boxplot with specific color schemes.
    """
    # Extract the specific metric and sample categories
    scores_inc = control_details_inc[metric].to_numpy()
    scores_dec = control_details_dec[metric].to_numpy()
    labels = adata.obs['labels']

    # Create DataFrame and convert to long format
    df_scores = pd.DataFrame({'Increase': scores_inc, 'Decrease': scores_dec, 'Label': labels})
    df_long = pd.melt(df_scores.dropna(), id_vars='Label', value_vars=['Increase', 'Decrease'], var_name='Variable',
                      value_name='Score')

    # Plot boxplot with specific color scheme
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_long, x='Label', y='Score', hue='Variable',
                palette={'Increase': sns.color_palette("Reds")[2], 'Decrease': sns.color_palette("Blues")[2]},
                hue_order=['Decrease', 'Increase'])
    plt.title(f'Control {metric.capitalize()} for Increase vs Decrease Conditions by Label')
    plt.xlabel('Sample Label')
    plt.ylabel(f'Control {metric.capitalize()}')
    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.legend(title='Condition')
    plt.show()

    # # Plot scatter plot with specific color scheme
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df_scores.index, df_scores['Increase'], label='Increase', color='red', alpha=0.6)
    # plt.scatter(df_scores.index, df_scores['Decrease'], label='Decrease', color='blue', alpha=0.6)
    # plt.xlabel('Sample Index')
    # plt.ylabel(f'Control {metric.capitalize()}')
    # plt.title(f'Control {metric.capitalize()} for Each Sample by Label')
    # plt.legend()
    # plt.show()
    #
    # # Plot line plot with specific color scheme
    # plt.figure(figsize=(12, 6))
    # plt.plot(df_scores['Increase'], label='Increase', color='red', alpha=0.7)
    # plt.plot(df_scores['Decrease'], label='Decrease', color='blue', alpha=0.7)
    # plt.xlabel('Sample Index')
    # plt.ylabel(f'Control {metric.capitalize()}')
    # plt.title(f'Control {metric.capitalize()} for Each Sample by Label')
    # plt.legend()
    # plt.show()


def plot_3d_state_transition(adata, sample_indices=None, use_pca=True, concat_pca=True, feature1=None, feature2=None,
                             n_components=2, max_samples=10, smooth=1, draw_contours=True, save_path=None):
    """
    Plot the state transition for specified samples in the dataset.
    The transition can be visualized using PCA or specified causal features.
    Limits the number of samples plotted to 'max_samples'.
    """
    # Create a custom color map
    cmap = plt.cm.viridis  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # If no specific samples are provided, plot all
    if sample_indices is None:
        sample_indices = list(adata.uns['causal_update'].keys())

    # Limit the number of samples to plot
    if len(sample_indices) > max_samples:
        print(f"Too many samples to plot. Only plotting the first {max_samples} samples.")
        sample_indices = sample_indices[:max_samples]

    # Iterate over specified samples for plotting
    for sample_idx in sample_indices:
        sample_label = adata.obs['labels'].iloc[sample_idx]
        update_data = adata.uns['causal_update'][sample_idx]
        sampling_data = adata.uns['causal_sampling'][sample_idx]
        control_score = adata.uns["control_details"]['score'][sample_idx]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if use_pca:
            key = 'pca'
            pca = PCA(n_components=n_components)
            combined_data = np.vstack((sampling_data.iloc[:, :-1], update_data[update_data.columns[3:]]))
            combined_df = pd.DataFrame(data=combined_data, columns=sampling_data.columns[:-1])
            if concat_pca:
                pca.fit(combined_df)
            else:
                pca.fit(sampling_data.iloc[:, :-1])
                # pca.fit(update_data[update_data.columns[3:]])
            pca_result = pca.transform(sampling_data.iloc[:, :-1])  # Exclude 'prob' column
            x_surf, y_surf, z_surf = pca_result[:, 0], pca_result[:, 1], sampling_data['prob'].values
            pca_path = pca.transform(update_data[update_data.columns[3:]])
            x_path, y_path, z_path = pca_path[:, 0], pca_path[:, 1], update_data['prob'].values
        else:
            key = 'feature'
            if feature1 is None or feature2 is None:
                raise ValueError("Please specify both feature1 and feature2 for non-PCA plotting.")
            x_surf, y_surf, z_surf = sampling_data[feature1], sampling_data[feature2], sampling_data['prob']
            x_path, y_path, z_path = update_data[feature1].values, update_data[feature2].values, update_data[
                'prob'].values
        # Interpolation for smoother surface
        x_combined = np.concatenate((x_surf, x_path))
        y_combined = np.concatenate((y_surf, y_path))
        z_combined = np.concatenate((z_surf, z_path))
        x_min = x_combined.min()
        x_max = x_combined.max()
        y_min = y_combined.min()
        y_max = y_combined.max()
        x_range = np.linspace(x_min, x_max, 200)
        y_range = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x_range, y_range)

        rbf = Rbf(x_combined, y_combined, z_combined, function='linear', smooth=smooth)
        Z = rbf(X, Y)
        # Z = griddata((x_combined, y_combined), z_combined, (X, Y), method='linear')  # "cubic" or "nearest"
        z_path = np.clip(z_path, Z.min(), Z.max())

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        ax.plot_wireframe(X, Y, Z, color='black', alpha=0.5, linewidth=0.3)
        # scatter = ax.scatter(x_path, y_path, z_path, c=z_path, cmap=cmap, vmin=0, vmax=1, s=1, alpha=0.5)
        ax.plot(x_path, y_path, z_path, color='red', linewidth=5)  # Plot the path
        ax.view_init(elev=30, azim=30)  # Adjust the elevation and azimuth angles as needed
        if draw_contours:
            ax.contour(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], cmap=cmap)
        # Annotate axes
        ax.text(x_path[0], y_path[0], z_path[0], 'Start', color='black', fontsize=12)
        ax.text(x_path[-1], y_path[-1], z_path[-1], 'End', color='black', fontsize=12)
        ax.set_zlim(0, 1)
        ax.set_xlabel(feature1 if not use_pca else 'PC1')
        ax.set_ylabel(feature2 if not use_pca else 'PC2')
        ax.set_zlabel('Probability')
        plt.title(
            f'Sample {sample_idx} (Label: {sample_label}, Control Score: {control_score:.2f})')
        plt.colorbar(surf, label='Probability', shrink=0.5, pad=0.1)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'state_transition_{sample_idx}_{key}.png'), dpi=300)
        plt.show()


def plot_causal_feature_transitions(adata, sample_indices=None, features=None, max_features=10, save_path=None):
    """
    Plot the transitions of multiple features against probability for specified samples.
    Adjusted to handle overlapping x-axis labels.
    """
    # If no specific sample indices are provided, plot all
    if sample_indices is None:
        sample_indices = adata.uns['causal_update'].keys()

    # If no specific features are provided, use all causal features
    if features is None:
        features = adata.uns['causal_update'][list(sample_indices)[0]].columns[3:]

    # Check if the number of features exceeds the maximum limit
    if len(features) > max_features:
        print(f"Warning: Too many features to plot. Only plotting the first {max_features} features.")
        features = features[:max_features]

    # Iterate over specified sample indices for plotting
    for sample_idx in sample_indices:
        # Extract the update data for the sample
        update_data = adata.uns['causal_update'][sample_idx]
        control_score = adata.uns["control_details"]['score'][sample_idx]

        # Normalize probability values for color mapping
        norm = plt.Normalize(vmin=0, vmax=1)
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis')

        # Determine the layout of subplots
        n_features = len(features)
        n_cols = 3  # Adjust the number of columns
        n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the required number of rows

        # Adjust figure size to avoid crowding
        plt.figure(figsize=(n_cols * 6, n_rows * 5))  # Increase figure size
        for i, feature in enumerate(features):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            colors = cmap(norm(update_data['prob']))
            ax.scatter(update_data[feature], update_data['prob'], c=colors, marker='o', s=10)  # Adjust point size
            # Add start and end annotations
            ax.text(update_data[feature].iloc[0], update_data['prob'].iloc[0], 'Start', color='black')
            ax.text(update_data[feature].iloc[-1], update_data['prob'].iloc[-1], 'End', color='black')

            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
            # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='Probability')
            # ax.set_xlabel(f'Feature Value ({feature})', fontsize=9)  # Adjust font size
            # Adjust font size and label positions
            if i // n_cols == n_rows - 1 or (i // n_cols == n_rows - 2 and i >= len(features) - n_cols):
                ax.set_xlabel('Feature Value', fontsize=9)  # Generic label for x-axis
            else:
                ax.set_xlabel('')  # No label for other rows
            ax.set_ylabel('Probability', fontsize=9)  # Adjust font size
            ax.set_title(f'Feature {feature}', fontsize=10)  # Adjust font size
            ax.set_ylabel('Probability', fontsize=9)  # Adjust font size
            ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
            ax.set_title(f'Feature {feature}', fontsize=10)  # Adjust font size

            # Format x-axis to reduce decimal places
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(f'Transitions of Multiple Features for Sample {sample_idx} (Control Score: {control_score:.2f})',
                     fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to avoid overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust horizontal and vertical spacing
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'feature_transitions_{sample_idx}.png'), dpi=300)
        plt.show()


# ======================================== DrugRespond.utils ========================================


def read_ALKlung(
        Base_path: str,
        data_name: str
):
    paths = {
        'ts485tos': "ts485tos",
        'ts485toc': "ts485toc",
        'ts485toa': "ts485toa",
        'fa34os': "fa34os",
        'fa34oc': "fa34oc",
        'fa34oa': "fa34oa"
    }

    if data_name not in paths:
        raise ValueError(f"data_name '{data_name}' not found in available datasets.")

    dataset_folder = paths[data_name]
    folder_path = os.path.join(Base_path, f"DrugR/ALKlung/datasets/{dataset_folder}")

    # create file path
    barcodes_path = os.path.join(folder_path, "barcodes.tsv")
    features_path = os.path.join(folder_path, "features.tsv")
    matrix_path = os.path.join(folder_path, "matrix.mtx")

    # read GEM
    matrix = scipy.io.mmread(matrix_path).T.toarray()

    # read barcode & feature
    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
    features = pd.read_csv(features_path, header=None, sep='\t')
    barcodes.index = barcodes.index.astype(str)
    features.index = features.index.astype(str)

    # anndata
    adata = anndata.AnnData(X=matrix, obs=barcodes, var=features)
    adata.obs.columns = ['barcodes']
    adata.var.columns = ['gene_ids', 'gene_symbols', 'feature_types']

    return adata


def adata_ALKlung(
        Base_path: str,
        data1_name: str,
        data2_name: str,
        top_n=2000
):
    data1 = read_ALKlung(Base_path=Base_path, data_name=data1_name)
    data2 = read_ALKlung(Base_path=Base_path, data_name=data2_name)

    merge_data = anndata.concat([data1, data2], label='labels')
    merge_data.obs_names_make_unique()
    merge_data.var = data1.var

    sc.pp.filter_cells(merge_data, min_genes=200)
    sc.pp.filter_genes(merge_data, min_cells=3)
    sc.pp.normalize_total(merge_data, target_sum=1e4)
    sc.pp.log1p(merge_data)

    sc.tl.rank_genes_groups(merge_data, groupby='labels', groups=['1'], reference='0', method='t-test')
    rank_genes_groups = merge_data.uns['rank_genes_groups']
    diff_exp_genes = rank_genes_groups['names']['1']
    n = int(top_n / 2)
    combined_array1 = np.concatenate([diff_exp_genes[:n], diff_exp_genes[-n:]])
    valid_genes = [gene for gene in combined_array1 if gene in merge_data.var_names]
    selected_data = merge_data[:, valid_genes]

    adata = anndata.AnnData(X=selected_data.X.astype(float), obs=selected_data.obs, var=selected_data.var)
    adata.obs['labels'] = adata.obs['labels'].astype(int)

    return adata


def analyse_ALKlung(
        Base_path: str,
        data1_name: str,
        data2_name: str,
        data3_name: str,
        method='louvain'
):
    data1 = read_ALKlung(Base_path=Base_path, data_name=data1_name)
    n_data1 = data1.obs.shape[0]
    data2 = read_ALKlung(Base_path=Base_path, data_name=data2_name)
    n_data2 = data2.obs.shape[0]
    data3 = read_ALKlung(Base_path=Base_path, data_name=data3_name)
    n_data3 = data3.obs.shape[0]

    merge_data1 = anndata.concat([data2, data3], label='batch')
    merge_data1.obs_names_make_unique()

    merge_data = anndata.concat([data1, merge_data1], label='labels')
    merge_data.obs_names_make_unique()
    merge_data.var = data1.var
    merge_data.obs['cell_line'] = [0] * n_data1 + [1] * n_data2 + [2] * n_data3
    merge_data.obs['cell_line'] = merge_data.obs['cell_line'].astype('category')

    del data1, data2, data3, merge_data1
    gc.collect()

    sc.pp.filter_cells(merge_data, min_genes=200)
    sc.pp.filter_genes(merge_data, min_cells=3)
    sc.pp.normalize_total(merge_data, target_sum=1e4)
    sc.pp.log1p(merge_data)

    sc.pp.highly_variable_genes(merge_data, n_top_genes=2000)
    sc.tl.pca(merge_data, svd_solver='arpack')
    sce.pp.harmony_integrate(merge_data, key='cell_line')
    sc.pp.neighbors(merge_data)

    if method == "louvain":
        # louvain
        sc.tl.louvain(merge_data)
        sc.tl.umap(merge_data)
        sc.pl.umap(merge_data, color='cell_line')
        sc.pl.umap(merge_data, color='louvain')

        count_table = pd.pivot_table(merge_data.obs, index='cell_line', columns='louvain', aggfunc='size', fill_value=0)
        proportion_table = count_table.div(count_table.sum(axis=0), axis=1)

        plt.figure(figsize=(10, 6))  
        sns.heatmap(proportion_table, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Cell Counts per Louvain Cluster and Tissue')
        plt.xlabel('Louvain Cluster')
        plt.ylabel('cell_line')
        plt.show()

        plt.figure(figsize=(10, 6))  
        sns.heatmap(count_table, annot=True, fmt='d', cmap='Blues')
        plt.title('Cell Counts per Louvain Cluster and Tissue')
        plt.xlabel('Louvain Cluster')
        plt.ylabel('cell_line')
        plt.show()

        sc.tl.paga(merge_data, groups='louvain')
        sc.pl.paga(merge_data)
        sc.tl.draw_graph(merge_data, init_pos='paga')
        # sc.pl.draw_graph(merge_data, color='louvain', legend_loc='on data')
        sc.pl.draw_graph(merge_data, color='cell_line', legend_loc='on data')

        merge_data.uns['iroot'] = np.flatnonzero(merge_data.obs['louvain'] == '1')[0]
        sc.tl.dpt(merge_data)
        sc.pl.draw_graph(merge_data, color=['louvain', 'dpt_pseudotime'], legend_loc='on data',
                         title=['', 'pseudotime'],
                         frameon=True)
    elif method == "leiden":
        # leiden
        sc.tl.leiden(merge_data)
        sc.tl.umap(merge_data)
        sc.pl.umap(merge_data, color='cell_line')
        sc.pl.umap(merge_data, color='leiden')

        count_table = pd.pivot_table(merge_data.obs, index='cell_line', columns='leiden', aggfunc='size', fill_value=0)
        proportion_table = count_table.div(count_table.sum(axis=0), axis=1)

        plt.figure(figsize=(10, 6)) 
        sns.heatmap(proportion_table, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Cell Counts per leiden Cluster and Tissue')
        plt.xlabel('leiden Cluster')
        plt.ylabel('cell_line')
        plt.show()

        plt.figure(figsize=(10, 6))  
        sns.heatmap(count_table, annot=True, fmt='d', cmap='Blues')
        plt.title('Cell Counts per leiden Cluster and Tissue')
        plt.xlabel('leiden Cluster')
        plt.ylabel('cell_line')
        plt.show()

        sc.tl.paga(merge_data, groups='leiden')
        sc.pl.paga(merge_data)
        sc.tl.draw_graph(merge_data, init_pos='paga')
        # sc.pl.draw_graph(merge_data, color='leiden', legend_loc='on data')
        sc.pl.draw_graph(merge_data, color='cell_line', legend_loc='on data')
        merge_data.uns['iroot'] = np.flatnonzero(merge_data.obs['leiden'] == '1')[0]
        sc.tl.dpt(merge_data)
        sc.pl.draw_graph(merge_data, color=['leiden', 'dpt_pseudotime'], legend_loc='on data', title=['', 'pseudotime'],
                         frameon=True)
    merge_data.write(r"D:\sci_job\casual\causal3.0\DrugR\ALKlung\datasets\preprocessing_data\fa34o.h5ad")
    return merge_data


# ======================================== Beeline.utils ========================================

def gene_TopK_precision(trueGenesDF: pd.DataFrame, predGenesDF: pd.DataFrame):
    from scipy.stats import hypergeom
    precision_matrix = []
    recall_matrix = []
    p_value_matrix = []
    f1_score_matrix = []

    if isinstance(trueGenesDF, pd.DataFrame):
        true_genes = set(trueGenesDF.iloc[:, 0].values)
    if isinstance(trueGenesDF, np.ndarray):
        true_genes = np.unique(trueGenesDF)

    for method in predGenesDF.columns:  # Loop through each method
        precision_col = []
        recall_col = []
        p_value_col = []
        f1_score_col = []

        for K in range(1, 21):  # Loop through K from 1 to 20
            pred_genes = set(predGenesDF[method].iloc[:K])  # Get the top K predicted genes for the current method
            TP = len(pred_genes.intersection(true_genes))
            precision = TP / K
            recall = TP / len(true_genes)
            if precision + recall != 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0  # Define f1_score as 0 when precision + recall is 0
            [k, M, n, N] = [TP, 15000, len(true_genes), K]
            p_value = hypergeom.sf(k, M, n, N)

            precision_col.append(precision)
            recall_col.append(recall)
            p_value_col.append(p_value)
            f1_score_col.append(f1_score)

        precision_matrix.append(precision_col)
        recall_matrix.append(recall_col)
        p_value_matrix.append(p_value_col)
        f1_score_matrix.append(f1_score_col)

    precision_matrix = pd.DataFrame(precision_matrix, columns=range(1, 21), index=predGenesDF.columns)
    recall_matrix = pd.DataFrame(recall_matrix, columns=range(1, 21), index=predGenesDF.columns)
    p_value_matrix = pd.DataFrame(p_value_matrix, columns=range(1, 21), index=predGenesDF.columns)
    f1_score_matrix = pd.DataFrame(f1_score_matrix, columns=range(1, 21), index=predGenesDF.columns)

    result_dict = {'precision': precision_matrix, 'recall': recall_matrix, 'p_value': p_value_matrix,
                   'f1_score': f1_score_matrix}

    return result_dict


def draw_precision_curve_Human(hESC_121_shap_res, Base_path, save_dir, metric='precision', figsize=(24, 6),
                               fontsize=10):
    hESC_all_results = pd.DataFrame(
        {'CEFCON': hESC_121_shap_res.index[:20]})

    input_dir = r"beeline/data/CEFCON-data/GeneSets"
    input_dir = os.path.join(Base_path, input_dir)

    hESC_cell_fate_commitment = pd.read_csv(f'{input_dir}/GO_Human_CELL_FATE_COMMITMENT.txt')  # 278
    hESC_stem_cell_population_maintenace = pd.read_csv(
        f'{input_dir}/GO_Human_STEM_CELL_POPULATION_MAINTENANCE.txt')  # 131
    hESC_endoderm_development = pd.read_csv(f'{input_dir}/GO_Human_ENDODERM_DEVELOPMENT.txt')  # 72
    cell2011 = pd.read_csv(f'{input_dir}/ESC_Cell2011.csv', encoding='latin1')  # 16
    reproduction2008 = pd.read_csv(f'{input_dir}/ESC_Reproduction2008.csv')  # 14
    literature_curated = pd.DataFrame(
        {'literature_curated': pd.concat([cell2011['TFs'], reproduction2008['TFs']]).drop_duplicates()})  # 27

    hESC_cell_fate_commitment_results = gene_TopK_precision(hESC_cell_fate_commitment, hESC_all_results)
    hESC_stem_cell_population_maintenace_results = gene_TopK_precision(hESC_stem_cell_population_maintenace,
                                                                       hESC_all_results)
    hESC_endoderm_development_results = gene_TopK_precision(hESC_endoderm_development, hESC_all_results)
    hESC_literature_curated_results = gene_TopK_precision(literature_curated, hESC_all_results)

    fig, axs = plt.subplots(1, 4, figsize=figsize)

    sns.lineplot(pd.DataFrame(hESC_cell_fate_commitment_results[metric]).T, marker='o', ax=axs[0])
    sns.lineplot(pd.DataFrame(hESC_stem_cell_population_maintenace_results[metric]).T, marker='o', ax=axs[1])
    sns.lineplot(pd.DataFrame(hESC_endoderm_development_results[metric]).T, marker='o', ax=axs[2])
    sns.lineplot(pd.DataFrame(hESC_literature_curated_results[metric]).T, marker='o', ax=axs[3])

    axs[0].set_xlabel('Rank cutoff', fontsize=fontsize)
    axs[0].set_ylabel('GO_Cell fate\ncommitment\nPrecision', fontsize=fontsize)
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks([10, 20])

    axs[1].set_xlabel('Rank cutoff', fontsize=fontsize)
    axs[1].set_ylabel('GO_Stem cell\npopulation maintenace\nPrecision', fontsize=fontsize)
    axs[1].set_ylim(0, 1)
    axs[1].set_xticks([10, 20])

    axs[2].set_xlabel('Rank cutoff', fontsize=fontsize)
    axs[2].set_ylabel('GO_Endoderm\ndevelopment\nPrecision', fontsize=fontsize)
    axs[2].set_ylim(0, 1)
    axs[2].set_xticks([10, 20])

    axs[3].set_xlabel('Rank cutoff', fontsize=fontsize)
    axs[3].set_ylabel('Literature-curated\nkey regulators\nPrecision', fontsize=fontsize)
    axs[3].set_ylim(0, 1)
    axs[3].set_xticks([10, 20])

    plt.savefig(f'{save_dir}/hESC_precision_curve.png')
    plt.show()


def prepare_data_for_R(adata: sc.AnnData,
                       temp_R_dir: str,
                       reducedDim: Optional[str] = None,
                       cluster_label: Optional[str] = None):
    """
    Process the AnnData object and save the necessary data to files.
    These data files are prepared for running the `slingshot_MAST_script.R` or `MAST_script.R` scripts.
    """
    if 'log_transformed' not in adata.layers:
        raise ValueError(
            f'Did not find `log_transformed` in adata.layers.'
        )

    if isinstance(adata.layers['log_transformed'], sparse.csr_matrix):
        exp_normalized = adata.layers['log_transformed'].A
    else:
        exp_normalized = adata.layers['log_transformed']

    # The normalized and log transformed data is used for MAST
    normalized_counts = pd.DataFrame(exp_normalized,
                                     index=adata.obs_names,
                                     columns=adata.var_names)
    normalized_counts.to_csv(temp_R_dir + '/exp_normalized.csv', sep=',')

    # The reduced dimension data is used for Slingshot
    if reducedDim is not None:
        reducedDim_data = pd.DataFrame(adata.obsm[reducedDim], dtype='float32', index=None)
        reducedDim_data.to_csv(temp_R_dir + '/data_reducedDim.csv', index=None)
    else:
        if 'lineages' not in adata.uns:
            raise ValueError(
                f'Did not find `lineages` in adata.uns.'
            )
        else:
            pseudotime_all = pd.DataFrame(index=adata.obs_names)
            for li in adata.uns['lineages']:
                pseudotime_all[li] = adata.obs[li]
            pseudotime_all.to_csv(temp_R_dir + '/pseudotime_lineages.csv', index=True)

    # Cluster Labels (Leiden)
    if cluster_label is not None:
        cluster_labels = pd.DataFrame(adata.obs[cluster_label])
        cluster_labels.to_csv(temp_R_dir + '/clusters.csv')


def process_Slingshot_MAST_R(temp_R_dir: str,
                             split_num: int = 4,
                             start_cluster: int = 0,
                             end_cluster: Optional[list] = None):
    """
    Run the `slingshot_MAST_script.R` to get pseudotim and differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'slingshot_MAST_script.R'
    with res.path('cefcon', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num} {start_cluster}'
    # args = ['Rscript', R_script_path, temp_R_dir, str(split_num), str(start_cluster)]
    if end_cluster is not None:
        args += f' {end_cluster}'
        # args += [str(end_cluster)]
    print('Running Slingshot and MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Trajectory (pseudotime) information: \'pseudotime_lineages.csv\'.')
            lineages = pd.read_csv(temp_R_dir + '/pseudotime_lineages.csv', index_col=0)
            lineages.columns
            print('      Differential expression information: ', end='')
            for l in lineages.columns:
                print(f'\'DEgenes_MAST_sp{split_num}_{l}.csv\' ', end='')
        else:
            print(f'Something error: returncode={p.returncode}.')


def process_MAST_R(temp_R_dir: str, split_num: int = 4):
    """
    Run the `MAST_script.R` to get differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'MAST_script.R'
    with res.path('CauFinder', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num}'
    # args = ['Rscript', R_script_path, temp_R_dir, str(split_num)]
    rscript_exe_path = "D:\\Program Files\\R\\R-4.2.2\\bin\\Rscript.exe"
    args = [rscript_exe_path, str(R_script_path), temp_R_dir, str(split_num)]

    print('Running MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Differential expression information: \'DEgenes_MAST_sp{split_num}_<x>.csv\'')
        else:
            print(f'Something error: returncode={p.returncode}.')
