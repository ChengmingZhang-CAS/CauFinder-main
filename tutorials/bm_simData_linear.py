# %% main
import os.path

from CauFinder.benchmark import run_benchmark
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'


# %% Benchmark on high noise simulation data

def benchmark_main(
        noises,
        causal_strengths,
        save_path,
        n_dataset=10,
        method_for_threshold="cumulative_rate",
        is_linear=False
):
    #%%
    evaluate_scores = []
    benchmark_results = []
    for i, noise in enumerate(noises):
        for j, causal_strength in enumerate(causal_strengths):
            print(f"noise:{noise}; causal_strength:{causal_strength}")
            evaluate_res, benchmark_res = run_benchmark(
                # simulation parameters
                n_dataset=n_dataset,
                noise_level=noise,
                causal_strength=causal_strength,
                is_linear=is_linear,
                activation="relu",
                # model parameters
                n_latent=10,
                n_hidden=64,
                n_layers_encoder=0,
                n_layers_decoder=0,
                n_layers_dpd=0,
                dropout_rate_encoder=0.0,
                dropout_rate_decoder=0.0,
                dropout_rate_dpd=0.0,
                use_batch_norm='none',
                use_batch_norm_dpd=True,
                pdp_linear=True,
                # eva parameters
                threshold_method=method_for_threshold,
                save_path=save_path)
            evaluate_scores.append(evaluate_res)
            benchmark_results.append(benchmark_res)
    #%%
    plt.rcParams.update({
        'font.size': 12,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'font.sans-serif': ['Arial'],
        'font.family': 'sans-serif'
    })

    metrics_list = ['AUC', 'ACC', 'MCC', 'Precision', 'Specificity', 'Recall', 'F1_score', 'num_w_cum_rate']
    # metrics_list = ['AUC', 'ACC', 'MCC', 'Specificity', 'Recall', 'F1_score', 'num_w_cum_rate']
    for method in metrics_list:
        fig, axs = plt.subplots(len(noises), len(causal_strengths),
                                figsize=(len(causal_strengths) * 5, len(noises) * 5))

        # Ensure axs is always a 2D array for consistent indexing
        if len(noises) == 1 and len(causal_strengths) == 1:
            axs = np.array([[axs]])
        elif len(noises) == 1:
            axs = axs[np.newaxis, :]  # Add a new axis to make it 2D
        elif len(causal_strengths) == 1:
            axs = axs[:, np.newaxis]  # Add a new axis to make it 2D

        for i, res in enumerate(evaluate_scores):
            x, y = divmod(i, len(causal_strengths))
            ax = axs[x, y] if len(noises) > 1 or len(causal_strengths) > 1 else axs[0]

            sns.boxplot(data=res[f'{method}'].iloc[:, :n_dataset].T, ax=ax, palette='Set2')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f'Noise: {noises[x]} | Causal Strengths: {causal_strengths[y]}', fontsize=14, pad=10)
            if method != 'num_w_cum_rate':
                if method == 'AUC':
                    ax.set_ylim(0, 1.0)
                else:
                    ax.set_ylim(0, 1.0)
            ax.grid(True)
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle(f"Comparison of {method} Values for Different Methods", fontsize=16)
        plt.savefig(os.path.join(save_path, f'{method}_boxplot.png'), format='png')
        plt.savefig(os.path.join(save_path, f'{method}_boxplot.pdf'), format='pdf')
        plt.close()

    # Generating radar chart for the mean values of each metric for different methods
    def create_mean_radar_chart(evaluate_scores, metrics_list, save_path):
        # Retrieve the list of method names
        method_names = evaluate_scores[0][metrics_list[0]].index.tolist()
        mean_values = {method: [] for method in method_names}

        # Calculate mean values for each method
        for metric in metrics_list:
            if metric == "num_w_cum_rate":
                continue  # Skip num_w_cum_rate
            print(f"Processing metric: {metric}")  # Debug: Print current metric
            for method_name in method_names:
                all_values = []
                for res in evaluate_scores:
                    value = res[metric].loc[method_name, 'Mean']
                    all_values.append(value)
                mean_val = np.mean(all_values)
                mean_values[method_name].append(mean_val)

        # Define the categories excluding "num_w_cum_rate"
        categories = [metric for metric in metrics_list if metric != "num_w_cum_rate"]
        num_categories = len(categories)

        # Calculate angles for each category
        angles = [n / float(num_categories) * 2 * np.pi for n in range(num_categories)]
        angles += angles[:1]

        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot and fill for each method
        for method_name in method_names:
            values = mean_values[method_name]
            print(
                f"{method_name} values length: {len(values)}, num_categories: {num_categories}")  # Debug: Print lengths
            if len(values) != num_categories:
                print(f"Skipping {method_name} due to mismatch in number of categories.")
                continue
            values += values[:1]

            # Debug: Print values before plotting
            print(f"Plotting {method_name}: {values}")

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method_name)
            ax.fill(angles, values, alpha=0.25)

        # Remove y-axis labels
        # ax.set_yticklabels([])

        # Add numbers to the circles
        ax.set_rlabel_position(0)
        ax.yaxis.set_tick_params(labelsize=10)

        # Set x-ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

        # Configure legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

        # Save the chart as PDF and PNG
        plt.savefig(os.path.join(save_path, "mean_radar_chart.pdf"), format="pdf", bbox_inches='tight')
        plt.savefig(os.path.join(save_path, "mean_radar_chart.png"), format="png", bbox_inches='tight')
        plt.close()

    create_mean_radar_chart(evaluate_scores, metrics_list, save_path)

    return evaluate_scores, benchmark_results


# %% main
if __name__ == '__main__':
    BASE_DIR = r"E:\Project_Research\CauFinder_Project\CauFinder-master"
    save_path = os.path.join(BASE_DIR, "benchmark", "output", "bm_simData_lin")
    # noises = [0.05, 0.1, 1.0]
    noises = [0.01, 0.1, 1.0]
    # causal_strengths = [3, 5, 8]
    # causal_strengths = [2, 5, 8]
    causal_strengths = [2, 4, 6]
    scores, results = benchmark_main(
                            noises=noises,
                            causal_strengths=causal_strengths,
                            save_path=save_path,
                            n_dataset=10,
                            is_linear=True,
                            method_for_threshold="top_k"
                        )
