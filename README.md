
# CauFinder

Understanding and manipulating cell-state and phenotype transitions is critical for advancing biological research and therapeutic treatments. CauFinder is an advanced framework designed to identify and control causal regulators of these transitions using causal disentanglement modeling and network control, based solely on observed data. By leveraging do-calculus and optimizing information flow, CauFinder distinguishes causal factors from spurious ones, ensuring precise state transitions. It excels in both theoretical and computational settings, revealing key regulatory mechanisms in processes like cell differentiation, cancer transdifferentiation, and drug resistance transitions, providing novel therapeutic strategies.

![CauFinder Overview](docs/CauFinder_overview.png)

**Overview of the CauFinder framework:**
a, Schematic representation of the CauFinder framework for steering state or phenotype transitions by identifying key drivers through causal disentanglement and network control. b, Structural causal model showing the decomposition of original features x={x^c,x^s } into their latent representations z={z^c,z^s}, with y denoting the state or phenotype. c, Causal decoupling model constructed on a variational autoencoder (VAE) framework to identify causal drivers influencing state or phenotype transitions. Features x are processed through a feature selection layer, encoded to create a latent space z, segmented into causal (z^c) and spurious (z^s) components. This latent space is decoded to reconstruct the original features x as x ̂ and to predict the phenotype y. The model strives to maximize the causal information flow, I(z^c→y), from z^c to y, thus delineating the causal pathways from x to y via z^c and identifying the causal drivers for precise transition control. d, Master regulator identification via causality-weighted features and network control. Techniques including SHAP and gradient are used to assign causality weights to features within the causal path defined in c, aiding in the isolation of causal features for integration with prior network insights. Weighed directed feedback vertex set is then employed to pinpoint master regulators critical for directing state or phenotype transitions through counterfactual generation for causal state transition, thereby establishing the foundation for targeted interventions.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ChengmingZhang-CAS/CauFinder-main.git
   ```
2. Create and activate a Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate caufinder_env
   ```
3. Install additional dependencies (if necessary):
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

To quickly get started with CauFinder, you can begin with the following scripts:

1. **Simulated Data**:
   For causal disentanglement analysis without prior network control, run the following script:
   ```bash
   python tutorials/bm_simData_linear.py
   ```

2. **Real Data (LUAS)**:
   For causal analysis with prior network control using real LUAS data, run:
   ```bash
   python tutorials/LUAS_human_analysis.py
   ```

### Using GUROBI for Master Regulator Identification

To effectively identify master regulators using network control, we recommend utilizing GUROBI to solve the optimization problem. GUROBI is a powerful commercial solver that requires a license to run. However, it provides free academic licenses as well as trial licenses for non-academic use. If you have a valid license, install the `gurobipy` package to enable this functionality.

In cases where using GUROBI is not feasible, you can use the non-commercial solver SCIP as an alternative, but successful solutions are not guaranteed with SCIP.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use CauFinder in your research, please cite the following paper:

Chengming Zhang, Zexi Chen, Yuanxiang Miao, Yun Xue, Deyu Cai, Weifeng Guo, Hongbin Ji, Kazuyuki Aihara, Luonan Chen. "Steering cell-state and phenotype transitions by causal disentanglement learning." *bioRxiv* (2024): [https://doi.org/10.1101/2024.08.16.607277](https://doi.org/10.1101/2024.08.16.607277).

## Contact

For questions or issues, please contact Chengming Zhang at zhangchengming@g.ecc.u-tokyo.ac.jp.
