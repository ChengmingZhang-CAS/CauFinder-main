#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import seaborn as sns
from scipy.stats import hypergeom

import warnings
import sys

sys.path.append(r'E:\Academic_Papers\Causal_Inference\2023_CEFCON_Nat_Commun\CEFCON-main')
sys.path.append(r'E:\Academic_Papers\Causal_Inference\2023_CellOracle_Nature\CellOracle-master')
import cefcon as cf
# import celloracle as co
from CauFinder.caufinder_main import CausalFinder
from CauFinder.utils import set_seed
from CauFinder.benchmark import cumulative_weight_sum_rate
from CauFinder.utils import result_add_direction2
from CauFinder.utils import prepare_data_for_R, process_MAST_R

set_seed(0)

BASE_DIR = r'E:\Project_Research\CauFinder_Project\CauFinder-master'
case_path = os.path.join(BASE_DIR, 'beeline')
data_path = os.path.join(case_path, 'data')
output_path = os.path.join(case_path, 'output')
os.makedirs(output_path, exist_ok=True)


def run_caufinder(data_path, network_path, temp_R_dir, output_path, n_top_genes=1000, seed=0):
    # load data
    expData = pd.read_csv(os.path.join(data_path, 'ExpressionData.csv'), index_col=0).transpose()
    pseudotime = pd.read_csv(os.path.join(data_path, 'PseudoTime.csv'), index_col=0)
    assert expData.index.equals(pseudotime.index)

    # create adata
    adata = sc.AnnData(X=expData, dtype=np.float32)
    adata.obs['all_lineage'] = pseudotime
    adata.layers['log_transformed'] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
    adata = adata[:, adata.var.highly_variable]

    # set labels based on pseudotime
    min_pseudotime = adata.obs['all_lineage'].min()
    max_pseudotime = adata.obs['all_lineage'].max()
    adata.obs['soft_labels'] = (adata.obs['all_lineage'] - min_pseudotime) / (max_pseudotime - min_pseudotime)

    lower_quantile = adata.obs['all_lineage'].quantile(0.25)
    upper_quantile = adata.obs['all_lineage'].quantile(0.75)
    adata.obs['hard_labels'] = adata.obs['all_lineage'].apply(
        lambda x: 0 if x <= lower_quantile else (1 if x >= upper_quantile else 0.5))

    # Obtain differential expression information of genes along each trajectory
    logFC_file_path = os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv')
    if not os.path.exists(logFC_file_path):
        adata.uns['lineages'] = ['all_lineage']
        os.makedirs(temp_R_dir, exist_ok=True)
        prepare_data_for_R(adata, temp_R_dir)
        process_MAST_R(temp_R_dir, split_num=4)
    logFC = pd.read_csv(logFC_file_path, index_col=0)['logFC']
    logFC.index = logFC.index.str.replace('.', '-', regex=False)
    logFC.name = 'all_logFC'
    # logFC = pd.read_csv(os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv'), index_col=0)
    # logFC = pd.read_csv(os.path.join(hESC_data_path, 'DEgenes_MAST_hvg1000_sp4.csv'), index_col=0)
    # logFC.rename(columns={'logFC': 'all_logFC'}, inplace=True)
    adata.var = adata.var.merge(logFC, how='left', left_index=True, right_index=True)
    adata.var['all_logFC'].fillna(value=0, inplace=True)
    adata.obs['labels'] = adata.obs['soft_labels']
    init_weight = np.zeros(adata.X.shape[1])
    init_weight += adata.var['all_logFC'].values * 0
    # run model
    set_seed(seed)
    model = CausalFinder(
        adata=adata,
        # n_controls=n_controls,
        n_latent=10,
        n_causal=2,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        pdp_linear=True,
        # attention=True,
        init_weight=init_weight
    )
    # print(model)
    # prior_probs = np.ones(adata.X.shape[1]) * 0.5
    # init_weight1 = adata.var.copy()
    # min_value = init_weight1['all_logFC'].min()
    # max_value = init_weight1['all_logFC'].max()
    # init_weight1['all_logFC'] = init_weight1['all_logFC'].apply(
    #     lambda x: (x - min_value) / (max_value - min_value) * 0.5 + 0.5 if pd.notnull(x) else x)
    # init_weight1['all_logFC'] = init_weight1['all_logFC'].fillna(0.5)
    # prior_probs = np.array(init_weight1['all_logFC'])
    # model.pretrain_attention(prior_probs=prior_probs)
    model.train(max_epochs=300, stage_training=True, fix_split=False)

    # shap
    # tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="SHAP")
    tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="Grad")
    ESC_121_shap_res = cumulative_weight_sum_rate(tmp)
    # ESC_121_shap_res = result_add_direction2(weights_full=ESC_121_shap_weights_full, result=ESC_121_shap_res,
    #                                          star=np.where(adata.obs['labels'] == 0)[0],
    #                                          end=np.where(adata.obs['labels'] == 1)[0])

    # load prior network
    gene_info = tmp.iloc[:1000]
    # gene_info['weight'] = 1
    network_data = pd.read_csv(network_path, index_col=None, header=0)
    # drivers_df = model.network_master_regulators(network_data, gene_info, driver_union=True)
    drivers_df = model.network_master_regulators(network_data, gene_info, uppercase=True, weight_degree=False, out_lam=0.7, driver_union=True, ILP_lam=1.0,
                                                 execute_in0out0=True, execute_selfloop=True, execute_out1=False,
                                                 execute_in1=False, execute_PIE=True, execute_CORE=False,
                                                 execute_DOME=False)
    # causal_driver = drivers_df[drivers_df['is_causal_driver']]
    # causal_driver = drivers_df[drivers_df['is_MFVS_driver']]
    # causal_driver = drivers_df[drivers_df['is_MDS_driver']]
    causal_driver = drivers_df[drivers_df['is_CauFVS_driver']]
    drivers_info_path = os.path.join(output_path, 'mESC_drivers_info_seed{}.csv'.format(seed))
    causal_driver.to_csv(drivers_info_path)
    return causal_driver


def run_caufinder_wo_cau(data_path, network_path, temp_R_dir, output_path, n_top_genes=1000, seed=0):
    # load data
    expData = pd.read_csv(os.path.join(data_path, 'ExpressionData.csv'), index_col=0).transpose()
    pseudotime = pd.read_csv(os.path.join(data_path, 'PseudoTime.csv'), index_col=0)
    assert expData.index.equals(pseudotime.index)

    # create adata
    adata = sc.AnnData(X=expData, dtype=np.float32)
    adata.obs['all_lineage'] = pseudotime
    adata.layers['log_transformed'] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
    adata = adata[:, adata.var.highly_variable]

    # set labels based on pseudotime
    min_pseudotime = adata.obs['all_lineage'].min()
    max_pseudotime = adata.obs['all_lineage'].max()
    adata.obs['soft_labels'] = (adata.obs['all_lineage'] - min_pseudotime) / (max_pseudotime - min_pseudotime)

    lower_quantile = adata.obs['all_lineage'].quantile(0.25)
    upper_quantile = adata.obs['all_lineage'].quantile(0.75)
    adata.obs['hard_labels'] = adata.obs['all_lineage'].apply(
        lambda x: 0 if x <= lower_quantile else (1 if x >= upper_quantile else 0.5))

    # Obtain differential expression information of genes along each trajectory
    logFC_file_path = os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv')
    if not os.path.exists(logFC_file_path):
        adata.uns['lineages'] = ['all_lineage']
        os.makedirs(temp_R_dir, exist_ok=True)
        prepare_data_for_R(adata, temp_R_dir)
        process_MAST_R(temp_R_dir, split_num=4)
    logFC = pd.read_csv(logFC_file_path, index_col=0)['logFC']
    logFC.index = logFC.index.str.replace('.', '-', regex=False)
    logFC.name = 'all_logFC'
    # logFC = pd.read_csv(os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv'), index_col=0)
    # logFC = pd.read_csv(os.path.join(hESC_data_path, 'DEgenes_MAST_hvg1000_sp4.csv'), index_col=0)
    # logFC.rename(columns={'logFC': 'all_logFC'}, inplace=True)
    adata.var = adata.var.merge(logFC, how='left', left_index=True, right_index=True)
    adata.var['all_logFC'].fillna(value=0, inplace=True)
    adata.obs['labels'] = adata.obs['soft_labels']
    init_weight = np.zeros(adata.X.shape[1])
    init_weight += adata.var['all_logFC'].values * 0
    # run model
    set_seed(seed)
    model = CausalFinder(
        adata=adata,
        # n_controls=n_controls,
        n_latent=10,
        n_causal=2,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        pdp_linear=True,
        # attention=True,
        init_weight=init_weight
    )
    # print(model)
    # prior_probs = np.ones(adata.X.shape[1]) * 0.5
    # init_weight1 = adata.var.copy()
    # min_value = init_weight1['all_logFC'].min()
    # max_value = init_weight1['all_logFC'].max()
    # init_weight1['all_logFC'] = init_weight1['all_logFC'].apply(
    #     lambda x: (x - min_value) / (max_value - min_value) * 0.5 + 0.5 if pd.notnull(x) else x)
    # init_weight1['all_logFC'] = init_weight1['all_logFC'].fillna(0.5)
    # prior_probs = np.array(init_weight1['all_logFC'])
    # model.pretrain_attention(prior_probs=prior_probs)
    model.train(max_epochs=300, stage_training=True, fix_split=False)

    # shap
    # tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="SHAP")
    tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="Grad")
    ESC_121_shap_res = cumulative_weight_sum_rate(tmp)
    # ESC_121_shap_res = result_add_direction2(weights_full=ESC_121_shap_weights_full, result=ESC_121_shap_res,
    #                                          star=np.where(adata.obs['labels'] == 0)[0],
    #                                          end=np.where(adata.obs['labels'] == 1)[0])

    # load prior network
    gene_info = tmp.iloc[:1000]
    gene_info['weight'] = 1
    network_data = pd.read_csv(network_path, index_col=None, header=0)
    # drivers_df = model.network_master_regulators(network_data, gene_info, driver_union=True)
    drivers_df = model.network_master_regulators(network_data, gene_info, uppercase=True, weight_degree=False,
                                                 out_lam=0.7, driver_union=True, ILP_lam=1.0,
                                                 execute_in0out0=True, execute_selfloop=True, execute_out1=False,
                                                 execute_in1=False, execute_PIE=True, execute_CORE=False,
                                                 execute_DOME=False)
    # causal_driver = drivers_df[drivers_df['is_causal_driver']]
    # causal_driver = drivers_df[drivers_df['is_MFVS_driver']]
    # causal_driver = drivers_df[drivers_df['is_MDS_driver']]
    causal_driver = drivers_df[drivers_df['is_CauFVS_driver']]
    # drivers_info_path = os.path.join(output_path, 'mESC_drivers_info_seed{}.csv'.format(seed))
    # causal_driver.to_csv(drivers_info_path)
    return causal_driver


def run_cefcon(data_path, network_path, temp_R_dir, output_path, n_top_genes=1000, seed=0):
    # load data
    expData = pd.read_csv(os.path.join(data_path, 'ExpressionData.csv'), index_col=0).transpose()
    pseudotime = pd.read_csv(os.path.join(data_path, 'PseudoTime.csv'), index_col=0)
    assert expData.index.equals(pseudotime.index)

    # convert to AnnData
    adata = sc.AnnData(X=expData, dtype=np.float32)
    adata.layers['log_transformed'] = adata.X.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
    adata = adata[:, adata.var.highly_variable]

    # add metadata
    adata.obs['all_pseudotime'] = pseudotime['PseudoTime']
    logFC_file_path = os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv')
    logFC = pd.read_csv(logFC_file_path, index_col=0)['logFC']
    logFC.index = logFC.index.str.replace('.', '-', regex=False)
    logFC.name = 'all_logFC'
    # logFC = pd.read_csv(os.path.join(data_path, 'DEgenes_MAST_hvg1000_sp4.csv'), index_col=0)
    # logFC.rename(columns={'logFC': 'all_logFC'}, inplace=True)
    adata.var = pd.merge(adata.var, logFC, left_index=True, right_index=True, how='left')

    # FA plot
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata, resolution=0.7)
    sc.tl.draw_graph(adata)

    # load prior network
    network_data = pd.read_csv(network_path, index_col=None, header=0)
    data = cf.data_preparation(input_expData=adata, input_priorNet=network_data)

    # run CEFCON
    cefcon_results_dict = {}
    for li, data_li in data.items():
        cefcon_GRN_model = cf.NetModel(epochs=350, repeats=1, cuda='0', seed=seed)
        cefcon_GRN_model.run(data_li)

        cefcon_results = cefcon_GRN_model.get_cefcon_results(edge_threshold_avgDegree=8)
        cefcon_results_dict[li] = cefcon_results
    for li, result_li in cefcon_results_dict.items():
        print(f'Lineage - {li}:')
        result_li.gene_influence_score()
        result_li.driver_regulators()
    result_all = cefcon_results_dict['all']
    gene_info_df = result_all.driver_regulator.sort_values(by='influence_score', ascending=False)
    return gene_info_df


def run_caufinder_cefcon(data_path, network_path, temp_R_dir, output_path, n_top_gene=2000, seed=0):
    # load data
    expData = pd.read_csv(os.path.join(data_path, 'ExpressionData.csv'), index_col=0).transpose()
    pseudotime = pd.read_csv(os.path.join(data_path, 'PseudoTime.csv'), index_col=0)
    assert expData.index.equals(pseudotime.index)

    # create adata
    adata = sc.AnnData(X=expData, dtype=np.float32)
    adata.obs['all_lineage'] = pseudotime
    adata.layers['log_transformed'] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_gene, flavor='cell_ranger')
    adata = adata[:, adata.var.highly_variable]

    # set labels based on pseudotime
    min_pseudotime = adata.obs['all_lineage'].min()
    max_pseudotime = adata.obs['all_lineage'].max()
    adata.obs['soft_labels'] = (adata.obs['all_lineage'] - min_pseudotime) / (max_pseudotime - min_pseudotime)

    lower_quantile = adata.obs['all_lineage'].quantile(0.25)
    upper_quantile = adata.obs['all_lineage'].quantile(0.75)
    adata.obs['hard_labels'] = adata.obs['all_lineage'].apply(
        lambda x: 0 if x <= lower_quantile else (1 if x >= upper_quantile else 0.5))

    # Obtain differential expression information of genes along each trajectory
    logFC_file_path = os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv')
    if not os.path.exists(logFC_file_path):
        adata.uns['lineages'] = ['all_lineage']
        os.makedirs(temp_R_dir, exist_ok=True)
        prepare_data_for_R(adata, temp_R_dir)
        process_MAST_R(temp_R_dir, split_num=4)
        del adata.uns['lineages']
    logFC = pd.read_csv(logFC_file_path, index_col=0)['logFC']
    logFC.index = logFC.index.str.replace('.', '-', regex=False)
    logFC.name = 'all_logFC'
    # logFC = pd.read_csv(os.path.join(temp_R_dir, 'DEgenes_MAST_sp4_all_lineage.csv'), index_col=0)
    # logFC = pd.read_csv(os.path.join(hESC_data_path, 'DEgenes_MAST_hvg1000_sp4.csv'), index_col=0)
    # logFC.rename(columns={'logFC': 'all_logFC'}, inplace=True)
    adata.var = adata.var.merge(logFC, how='left', left_index=True, right_index=True)
    adata.var['all_logFC'].fillna(value=0, inplace=True)
    adata.obs['labels'] = adata.obs['soft_labels']
    init_weight = np.zeros(adata.X.shape[1])
    init_weight += adata.var['all_logFC'].values * 0.1
    # run model
    set_seed(seed)
    model = CausalFinder(
        adata=adata,
        # n_controls=n_controls,
        n_latent=25,
        n_causal=5,
        pdp_linear=True,
        attention=False,
        init_weight=init_weight
    )
    # print(model)
    prior_probs = np.ones(adata.X.shape[1]) * 0.5
    # prior_probs[0:5] = 0.9
    # model.pretrain_attention(prior_probs=prior_probs)
    model.train(max_epochs=400, stage_training=True)

    # shap
    # tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="SHAP")
    tmp, ESC_121_shap_weights_full = model.get_feature_weights(sort_by_weight=True, method="Model")
    adata = adata[:, tmp.index[0:1000]]
    # load prior network
    network_data = pd.read_csv(network_path, index_col=None, header=0)
    data = cf.data_preparation(input_expData=adata, input_priorNet=network_data)

    # run CEFCON
    cefcon_results_dict = {}
    for li, data_li in data.items():
        cefcon_GRN_model = cf.NetModel(epochs=350, repeats=1, cuda='0', seed=seed)
        cefcon_GRN_model.run(data_li)

        cefcon_results = cefcon_GRN_model.get_cefcon_results(edge_threshold_avgDegree=8)
        cefcon_results_dict[li] = cefcon_results
    for li, result_li in cefcon_results_dict.items():
        print(f'Lineage - {li}:')
        result_li.gene_influence_score()
        result_li.driver_regulators()
    result_all = cefcon_results_dict['all']
    gene_info_df = result_all.driver_regulator.sort_values(by='influence_score', ascending=False)
    return gene_info_df


def run_celloracle(data_path, network_path, temp_R_dir, output_path,n_top_genes=1000, seed=0):
    data_name = os.path.basename(data_path)
    file_name = f"{data_name}_CellOracle_top20_runs10.csv"
    case_dir = os.path.dirname(os.path.dirname(data_path))
    file_path = os.path.join(case_dir, 'output', file_name)
    df = pd.read_csv(file_path, index_col=0, header=0)
    column_index = seed % len(df.columns)
    gene_list = df.iloc[:, column_index]
    result_df = pd.DataFrame(index=gene_list)

    return result_df


def run_wmdsnet(data_path, network_path, temp_R_dir, output_path, n_top_genes=1000, seed=0):
    data_name = os.path.basename(data_path)
    file_name = f"{data_name}_WMDSnet_top20_runs10.csv"
    case_dir = os.path.dirname(os.path.dirname(data_path))
    file_path = os.path.join(case_dir, 'output', file_name)
    df = pd.read_csv(file_path, index_col=0, header=0)
    column_index = seed % len(df.columns)
    gene_list = df.iloc[:, column_index]
    result_df = pd.DataFrame(index=gene_list)

    return result_df


def run_algorithms(algorithms, run_times, top_k, data_path, network_path, temp_R_dir, output_path, seeds=None,
                   dataset='hESC'):
    if seeds is None:
        seeds = np.arange(run_times)
    algorithm_functions = {
        'CauFinderCEFCON': run_caufinder_cefcon,
        'CauFinder': run_caufinder,
        'CauFinderNC': run_caufinder_wo_cau,
        'CEFCON': run_cefcon,
        'CellOracle': run_celloracle,
        'WMDS.net': run_wmdsnet
    }
    all_runs_results = {}

    for algorithm_name in algorithms:
        # Construct the output file path with the specified naming format
        output_file = os.path.join(output_path, f'{dataset}_{algorithm_name}_top{top_k}_runs{run_times}.csv')
        run_algorithm = algorithm_functions[algorithm_name]
        # Check if the result file already exists
        if os.path.exists(output_file):
            # Read the existing results
            top_k_genes_df = pd.read_csv(output_file, index_col=0)
        else:
            # Initialize DataFrame to store the results
            columns = [f'{algorithm_name}_{i + 1}' for i in range(run_times)]
            top_k_genes_df = pd.DataFrame(index=range(1, top_k + 1), columns=columns)

            for i in range(run_times):
                # Run the algorithm
                gene_info_df = run_algorithm(data_path, network_path, temp_R_dir, output_path, seed=seeds[i])
                # Get the top k genes
                top_k_genes = gene_info_df.index[:top_k]
                top_k_genes_df[f'{algorithm_name}_{i + 1}'] = top_k_genes

            # Save the results to a file
            top_k_genes_df.to_csv(output_file)

        all_runs_results[algorithm_name] = top_k_genes_df

    return all_runs_results


def load_ground_truth(ground_truth_path):
    hESC_ground_truth = {}
    mESC_ground_truth = {}

    # hESC ground truth
    hESC_files = [
        ('cell_fate_commitment', 'GO_CELL_FATE_COMMITMENT.txt'),
        ('stem_cell_population_maintenance', 'GO_STEM_CELL_POPULATION_MAINTENANCE.txt'),
        ('endoderm_development', 'GO_ENDODERM_DEVELOPMENT.txt')
    ]

    for name, file in hESC_files:
        df = pd.read_csv(os.path.join(ground_truth_path, file))
        hESC_ground_truth[name] = set(df.iloc[:, 0])

    # mESC ground truth
    mESC_files = [
        ('cell_fate_commitment', 'GO_mouse_CELL_FATE_COMMITMENT.txt'),
        ('stem_cell_population_maintenance', 'GO_mouse_STEM_CELL_POPULATION_MAINTENANCE.txt'),
        ('endoderm_development', 'GO_mouse_ENDODERM_DEVELOPMENT.txt')
    ]

    for name, file in mESC_files:
        df = pd.read_csv(os.path.join(ground_truth_path, file), sep='\t')
        # mESC_ground_truth[name] = set(df.iloc[:, 0])
        mESC_ground_truth[name] = set(df.iloc[:, 0].str.upper())

    # literature curated key regulators
    cell2011_genes = set(pd.read_csv(os.path.join(ground_truth_path, 'ESC_Cell2011.csv'), encoding='latin1')['TFs'])
    reproduction2008_genes = set(pd.read_csv(os.path.join(ground_truth_path, 'ESC_Reproduction2008.csv'))['TFs'])
    literature_curated = cell2011_genes.union(reproduction2008_genes)

    # add literature curated key regulators to ground truth
    hESC_ground_truth['literature_curated'] = literature_curated
    mESC_ground_truth['literature_curated'] = literature_curated

    # add all genes to ground truth
    hESC_ground_truth['all'] = set.union(*hESC_ground_truth.values())
    mESC_ground_truth['all'] = set.union(*mESC_ground_truth.values())

    return hESC_ground_truth, mESC_ground_truth


def get_top_k_metrics(pred_gene_df, ground_truth, top_k=20):
    """
    Calculate precision, recall, p-value, and F1 score for the top K predicted genes.
    """
    precision_matrix = []
    recall_matrix = []
    p_value_matrix = []
    f1_score_matrix = []

    true_genes = set(ground_truth)
    # Calculate metrics for each method
    for run in pred_gene_df.columns:
        precision_col = []
        recall_col = []
        p_value_col = []
        f1_score_col = []

        for K in range(1, top_k + 1):
            pred_genes = set(pred_gene_df[run].iloc[:K].str.upper())
            TP = len(pred_genes.intersection(true_genes))
            precision = TP / K
            recall = TP / len(true_genes)
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            p_value = hypergeom.sf(TP - 1, 15000, len(true_genes), K)

            precision_col.append(precision)
            recall_col.append(recall)
            p_value_col.append(p_value)
            f1_score_col.append(f1_score)

        precision_matrix.append(precision_col)
        recall_matrix.append(recall_col)
        p_value_matrix.append(p_value_col)
        f1_score_matrix.append(f1_score_col)

    # Convert matrices to DataFrame
    metrics = {
        'precision': pd.DataFrame(precision_matrix, columns=range(1, top_k + 1), index=pred_gene_df.columns),
        'recall': pd.DataFrame(recall_matrix, columns=range(1, top_k + 1), index=pred_gene_df.columns),
        'p_value': pd.DataFrame(p_value_matrix, columns=range(1, top_k + 1), index=pred_gene_df.columns),
        'f1_score': pd.DataFrame(f1_score_matrix, columns=range(1, top_k + 1), index=pred_gene_df.columns)
    }

    return metrics


def metric_eval_plot(algorithm_outputs, ground_truth, metric='precision', dataset='hESC', save_path=None,
                     save_metric=False, gt_keys=None):
    if gt_keys is None:
        gt_keys = ['cell_fate_commitment', 'stem_cell_population_maintenance', 'endoderm_development',
                   'literature_curated', 'all']

    metric_plot = {}
    for i in ground_truth.keys():
        metric_plot[i] = pd.DataFrame()
        for j in algorithm_outputs.keys():
            metric_dict = get_top_k_metrics(algorithm_outputs[j], ground_truth[i])
            metric_plot[i] = pd.concat([metric_plot[i], metric_dict[metric]], axis=0)
            if save_path is not None and save_metric:
                metric_dict[metric].to_csv(os.path.join(save_path, f'{dataset}_{i}_{j}_{metric}.csv'))

    merged_df = pd.DataFrame()
    gt_labels = {
        'cell_fate_commitment': 'cell fate commitment',
        'stem_cell_population_maintenance': 'stem cell population maintenance',
        'endoderm_development': 'endoderm development',
        'literature_curated': 'literature curated',
        'all': 'all'
    }

    for gt_key in gt_keys:
        if gt_key in metric_plot:
            df = metric_plot[gt_key].reset_index().melt(id_vars='index', var_name='timepoint', value_name='value')
            df[['algorithm', 'run']] = df['index'].str.split('_', expand=True)
            df['gt'] = gt_labels.get(gt_key, gt_key)
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    palette = {'CauFinder': '#E41A1C', 'CEFCON': '#377EB8', 'CellOracle': '#4DAF4A', 'WMDS.net': '#984EA3',
               'CauFinderNC': '#1F78B4'}
    g = sns.relplot(data=merged_df, x="timepoint", y="value", hue="algorithm", kind="line", ci=60, marker='o',
                    col='gt', palette=palette)
    (g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
     .set_axis_labels("Rank cutoff", metric.capitalize())
     .set_titles("{col_name}")
     .tight_layout(w_pad=0))

    for ax in g.axes.flat:
        ax.set_xticks([x for x in range(1, 20 + 1) if x % 10 == 0])
    if save_path is not None:
        g.savefig(os.path.join(save_path, f'{dataset}_{metric}.png'))
        g.savefig(os.path.join(save_path, f'{dataset}_{metric}.pdf'))
    # plt.show()

    return metric_plot


def main():
    # ## run the algorithms to get the results
    hESC_data_path = os.path.join(data_path, 'hESC')
    mESC_data_path = os.path.join(data_path, 'mESC')

    # mESC_output_path = os.path.join(output_path, 'mESC_benchmark_FVS_0.7')
    mESC_output_path = os.path.join(output_path, 'mESC_benchmark_CauFVS_0.7_no_degree_all')
    # hESC_output_path = os.path.join(output_path, 'hESC_hard_labels')
    # mESC_output_path = os.path.join(output_path, 'mESC_hard_labels')
    # os.makedirs(hESC_output_path, exist_ok=True)
    os.makedirs(mESC_output_path, exist_ok=True)

    hESC_net_path = os.path.join(BASE_DIR, 'resources', 'network', 'NicheNet_human.csv')
    mESC_net_path = os.path.join(BASE_DIR, 'resources', 'network', 'NicheNet_mouse.csv')

    hESC_temp_R_dir = os.path.join(case_path, 'temp_R_dir', 'hESC')
    mESC_temp_R_dir = os.path.join(case_path, 'temp_R_dir', 'mESC')
    algorithms = ['CauFinder', 'CEFCON', 'CellOracle', 'WMDS.net']
    # algorithms = ['CauFinder', 'CauFinderNC']
    # algorithms = ['CauFinder', 'CEFCON', 'CauFinderNC']
    run_times = 10  # run_times
    top_k = 20  # the number of top genes

    seeds = [0, 16, 23, 35, 43, 49, 58, 87, 88, 92, 99]

    # hESC_algorithm_ret = run_algorithms(algorithms, run_times, top_k, hESC_data_path, hESC_net_path, hESC_temp_R_dir,
    #                                     hESC_output_path, seeds=seeds, dataset='hESC_10seeds')
    mESC_algorithm_ret = run_algorithms(algorithms, run_times, top_k, mESC_data_path, mESC_net_path, mESC_temp_R_dir,
                                        mESC_output_path, seeds=seeds, dataset='mESC_10seeds')

    # ## load the ground truth
    ground_truth_path = os.path.join(BASE_DIR, 'beeline', 'ground_truth')
    hESC_ground_truth, mESC_ground_truth = load_ground_truth(ground_truth_path)

    # ## plot the precision curve
    # metric_eval_plot(hESC_algorithm_ret, hESC_ground_truth, metric='precision', dataset='hESC_10seeds',
    #                  save_path=hESC_output_path)
    metric_eval_plot(mESC_algorithm_ret, mESC_ground_truth, metric='precision', dataset='mESC_10seeds',
                     save_path=mESC_output_path, gt_keys=['all'])


if __name__ == "__main__":
    main()
