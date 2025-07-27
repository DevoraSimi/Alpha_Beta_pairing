import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import mi_with_clustering
import os
import sys
sys.path.append(os.path.abspath(".."))
from data.process_data_new import positive_format_sapir_data
from collections import defaultdict
import re
from scipy.stats import ttest_1samp, mannwhitneyu, gaussian_kde
from scipy.stats import levene, bartlett


font_size = 16


def plot_2d_histograms(alpha_file, beta_file, filename, bins=20):
    # Load data
    alpha_df = pd.read_csv(alpha_file)
    beta_df = pd.read_csv(beta_file)

    features = ['Weight', 'Charge', 'Hydrophobicity']

    # Bin each column into quantile bins (equal number of samples per bin)
    binned_alpha = pd.DataFrame()
    binned_beta = pd.DataFrame()

    for feature in features:
        binned_alpha[feature] = pd.qcut(alpha_df[feature], q=bins, labels=False, duplicates='drop')
        binned_beta[feature] = pd.qcut(beta_df[feature], q=bins, labels=False, duplicates='drop')

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

    for i, beta_feat in enumerate(features):
        for j, alpha_feat in enumerate(features):

            ax = axes[i, j]
            x = binned_alpha[alpha_feat]
            y = binned_beta[beta_feat]

            # Get raw counts without plotting
            counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

            # Apply log transform with offset 0.1
            log_counts = np.log(counts + 0.1)

            # Plot the log_counts using imshow for precise control
            # Need to transpose because imshow expects (rows, cols)
            im = ax.imshow(log_counts.T, origin='lower', cmap="GnBu",
                           vmin=np.log(0.1), vmax=7)  # set vmin and vmax on log scale

            # ax.set_title(f'{alpha_feat} (α) vs {beta_feat} (β)')
            # Show xlabel only on bottom row
            if i == len(features) - 1:
                ax.set_xlabel(f'{alpha_feat} (alpha)', fontsize=25)
            else:
                ax.set_xticklabels([])

            # Show ylabel only on first column
            if j == 0:
                ax.set_ylabel(f'{beta_feat} (beta)', fontsize=25)
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis='both', labelsize=14)

            # fig.colorbar(im, ax=ax)
            if i == 0 and j == len(features) - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    # plt.savefig(f"histograms_{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def with_pep_mhc(column_type, file_exists=True):
    if not file_exists:
        positive_format_sapir_data("all_data_280824.csv", os.path.join("new_data", "sapir_data_with_pep_mhc.csv"),
                                   pep=True)
    df = pd.read_csv(os.path.join("new_data", "sapir_data_with_pep_mhc.csv"))
    if column_type not in ["pep", "hla_seq"]:
        raise ValueError("column_type must be 'pep' or 'hla_seq'")

    # Drop the other column
    column_to_remove = "hla_seq" if column_type == "pep" else "pep"
    df = df.drop(columns=[column_to_remove])

    # Ensure 'pep' column exists
    assert column_type in df.columns, f"CSV must contain a {column_type} column."
    significant = []
    correlation_lists = defaultdict(list)  # keys will be (i,j) for 3x3
    feature_names = ['Weight', 'Charge', 'Hydrophobicity']
    subset_cols = [col for col in df.columns if col != column_type]
    group_keys = list(df.groupby(column_type).groups.keys())
    print(f"Number of groups for '{column_type}': {len(group_keys)}")
    for type_name, group in df.groupby(column_type):
        if column_type == "pep":
            unique_rows = group.drop_duplicates(subset=subset_cols)
            if len(unique_rows) < 10:
                continue
        elif column_type == "hla_seq":
            if str(type_name).strip() == "-1":
                continue  # skip invalid hla_seq

        name = re.sub(r'[\\/:"*?<>|]+', '_', type_name)  # sanitize filename if needed
        print(type_name, name)
        group_wo_pep = group.drop(columns=[column_type])

        # Step 1: calculate bio sequence metrics
        mi_with_clustering.calculate_bio_seq(group_wo_pep, name, file=False)

        # Step 2: compute correlation matrix (3x3)
        corr_matrix = mi_with_clustering.process_sequences_and_calculate_correlations("name", name, "csv",
                                                                                      with_fdr=False)

        # Step 3: collect correlations into lists
        for i in range(3):
            for j in range(3):
                row_name = f"{feature_names[i]} alpha"
                col_name = f"{feature_names[j]} beta"
                correlation_lists[(i, j)].append(corr_matrix.loc[row_name, col_name])

    # Prepare subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 3 rows, 3 columns
    axes = axes.flatten()  # flatten to index easily

    for idx, ((i, j), values) in enumerate(correlation_lists.items()):
        if idx >= 9:
            break  # limit to first 9 plots

        alpha_feat = feature_names[i]
        beta_feat = feature_names[j]

        ax = axes[idx]
        values = [v for v in values if not np.isnan(v)]
        ax.hist(values, bins=30, edgecolor='black')
        ax.set_title(f"Alpha {alpha_feat} vs Beta {beta_feat}")
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Frequency")

        # Perform one-sided t-test against mean = 0
        t_stat, p_val_two_tailed = ttest_1samp(values, popmean=0)

        # Check if mean is significantly > 0
        if t_stat > 0 and (p_val_two_tailed / 2) < 0.05:
            ax.text(0.95, 0.95, "p < 0.05", transform=ax.transAxes,
                    ha='right', va='top', fontsize=12, color='red', fontweight='bold')
            significant.append((alpha_feat, beta_feat))

    # If there are less than 9 plots, remove unused subplots
    for empty_ax in axes[len(correlation_lists):]:
        empty_ax.axis('off')

    if column_type == "hla_seq":
        title = "MHC"
    else:
        title = "peptide"

    plt.tight_layout()
    plt.show()
    plt.close()

    print("All histograms saved in a 3x3 subplot figure.")
    return significant


def pep_distribution(column_type):

    df = pd.read_csv(os.path.join("new_data", "sapir_data_with_pep_mhc.csv"))

    if column_type not in ["pep", "hla_seq"]:
        raise ValueError("column_type must be 'pep' or 'hla_seq'")

    column_to_remove = "hla_seq" if column_type == "pep" else "pep"
    df = df.drop(columns=[column_to_remove])

    assert column_type in df.columns, f"CSV must contain a {column_type} column."

    tcr_distribution = {}  # new
    subset_cols = [col for col in df.columns if col != column_type]
    group_keys = list(df.groupby(column_type).groups.keys())
    print(f"Number of groups for '{column_type}': {len(group_keys)}")
    for type_name, group in df.groupby(column_type):
        if column_type == "pep":
            unique_rows = group.drop_duplicates(subset=subset_cols)
            if len(unique_rows) < 10:
                continue
            tcr_distribution[type_name] = len(unique_rows)  # new line
        elif column_type == "hla_seq":
            if str(type_name).strip() == "-1":
                continue  # skip invalid hla_seq
            unique_rows = group.drop_duplicates(subset=subset_cols)
            tcr_distribution[type_name] = len(unique_rows)  # new line

    # Print or plot the distribution
    print("Distribution of TCRs per HLA:")
    for pep, count in sorted(tcr_distribution.items(), key=lambda x: -x[1]):
        print(f"{pep}: {count}")

    # Optional histogram
    plt.hist(list(tcr_distribution.values()), bins=30)
    # plt.title("Distribution of TCRs per Binding Peptide")
    plt.savefig(f"HLA distribution.pdf", format='pdf', bbox_inches='tight')
    plt.xlabel("Number of unique TCRs")
    plt.ylabel("Number of HLAs")
    plt.tight_layout()
    plt.show()


def sum_weights(alpha_df, beta_df):
    # Step 1: Ensure same length
    assert len(alpha_df) == len(beta_df), "alpha and beta must have same number of rows"

    # Step 2: Compute sum of weights
    original_sum = alpha_df['Weight'].values + beta_df['Weight'].values

    # Step 3: Shuffle beta weights
    shuffled_beta_weights = beta_df['Weight'].sample(frac=1).values
    shuffled_alpha_weights = alpha_df['Weight'].sample(frac=1).values
    shuffled_sum = shuffled_alpha_weights + shuffled_beta_weights
    # ---- F-test (Levene’s test for equal variance) ----
    stat_levene, p_levene = levene(original_sum, shuffled_sum)
    stat_bartlett, p_bartlett = bartlett(original_sum, shuffled_sum)
    print(f"Levene's test:  F = {stat_levene:.4f}, p = {p_levene:.4g}")
    print(f"Bartlett's test: F = {stat_bartlett:.4f}, p = {p_bartlett:.4g}")
    if p_levene < 0.05:
        print("Levene’s test: Variances are significantly different.")
    else:
        print("Levene’s test: No significant difference in variance.")
    # Step 4: Plot
    plt.figure(figsize=(8, 6))
    all_values = np.concatenate([original_sum, shuffled_sum])
    bins = np.histogram_bin_edges(all_values, bins=30)
    plt.hist(original_sum, bins=bins, alpha=0.6, label="Original (alpha + beta)", color='blue', histtype='step',
             linewidth=2)
    plt.hist(shuffled_sum, bins=bins, alpha=0.6, label="Shuffled", color='orange', histtype='step', linewidth=2)
    plt.xlabel("Sum of Weights")
    plt.ylabel("Frequency")
    # plt.title("Comparison of Weight Sums: Original vs Shuffled Beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sum_weights.pdf", format='pdf', bbox_inches='tight')
    plt.show()


name = "VDjdb"
alpha = pd.read_csv(f'alpha_properties_{name}.csv')
beta = pd.read_csv(f'beta_properties_{name}.csv')
sum_weights(alpha, beta)
with_pep_mhc("pep")
pep_distribution("hla_seq")
pep_distribution("pep")
