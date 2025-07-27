import json
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.stats import pointbiserialr, spearmanr
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import Counter
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import new_statistics
import os
import sys
sys.path.append(os.path.abspath(".."))
import v_j
from matplotlib.colors import LinearSegmentedColormap
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from statsmodels.stats.multitest import multipletests


# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')


def compute_mi(alpha_data, beta_data):
    """Compute mutual information between two distributions."""
    if alpha_data.dtype in ['float64', 'int64'] and beta_data.dtype in ['float64', 'int64']:
        return mutual_info_regression(alpha_data.values.reshape(-1, 1), beta_data)[0]  # Reshape for single feature input
    else:
        # Use mutual_info_classif for classification tasks (binary or categorical)
        return mutual_info_classif(alpha_data.values.reshape(-1, 1), beta_data)[0]  # Reshape for single feature input


def load_data(dataset):
    # Load data
    dataset = pd.read_csv(dataset)
    tcra1, tcrb1 = dataset['tcra'], dataset['tcrb']
    va1, ja1 = dataset['va'], dataset['ja']
    vb1, jb1 = dataset['vb'], dataset['jb']
    return tcra1, va1, ja1, tcrb1, vb1, jb1


# Compute amino acid frequencies
def compute_amino_acid_frequency(sequences):
    """Compute the frequency of each amino acid in a sequence."""
    amino_acid_frequencies = sequences.apply(lambda x: {aa: count / len(x) for aa, count in Counter(x).items()})
    return amino_acid_frequencies


# One-hot encoding function
def one_hot_encode(value, name, one_hot_dict):
    """One-hot encode a categorical series."""
    gene_name = "TR" + name[::-1].upper()
    processed_value = v_j.v_j_format(value,  2 if gene_name == "TRBJ" else 1, gene_name)
    # Create a one-hot vector
    one_hot_vector = [0] * len(one_hot_dict)
    # If value exists in dictionary, assign 1 to its index; otherwise use "<UNK>"
    if processed_value in one_hot_dict:
        one_hot_vector[one_hot_dict[processed_value]] = 1
    else:
        one_hot_vector[one_hot_dict["<UNK>"]] = 1

    return one_hot_vector


def process_data(dataset, tcra1, va1, ja1, tcrb1, vb1, jb1):
    # Load one-hot dictionaries
    with open('filtered_counters.json', 'r') as f:
        one_hot_dicts = json.load(f)
    va_2_ix = one_hot_dicts['va_counts']
    vb_2_ix = one_hot_dicts['vb_counts']
    ja_2_ix = one_hot_dicts['ja_counts']
    jb_2_ix = one_hot_dicts['jb_counts']

    data_df = pd.read_csv(dataset)
    # Apply one-hot encoding to all relevant columns
    alpha_v_onehot = pd.DataFrame(va1.apply(lambda x: one_hot_encode(x, 'va', va_2_ix)).tolist(), index=data_df.index)
    beta_v_onehot = pd.DataFrame(vb1.apply(lambda x: one_hot_encode(x, 'vb',  vb_2_ix)).tolist(), index=data_df.index)
    alpha_j_onehot = pd.DataFrame(ja1.apply(lambda x: one_hot_encode(x, 'ja', ja_2_ix)).tolist(), index=data_df.index)
    beta_j_onehot = pd.DataFrame(jb1.apply(lambda x: one_hot_encode(x, 'jb', jb_2_ix)).tolist(), index=data_df.index)

    # Add proper column names for one-hot encoded columns
    alpha_v_onehot.columns = [
        f"Rare V" if key == "<UNK>" else key for key in va_2_ix.keys()
    ]
    beta_v_onehot.columns = [
        f"Rare V" if key == "<UNK>" else key for key in vb_2_ix.keys()
    ]
    alpha_j_onehot.columns = [
        f"Rare J" if key == "<UNK>" else key for key in ja_2_ix.keys()
    ]
    beta_j_onehot.columns = [
        f"Rare J" if key == "<UNK>" else key for key in jb_2_ix.keys()
    ]

    alpha_amino_acid_frequencies = compute_amino_acid_frequency(tcra1)
    beta_amino_acid_frequencies = compute_amino_acid_frequency(tcrb1)

    # Convert frequencies to DataFrames
    alpha_freq_df = pd.DataFrame(list(alpha_amino_acid_frequencies), index=data_df.index).fillna(0)
    beta_freq_df = pd.DataFrame(list(beta_amino_acid_frequencies), index=data_df.index).fillna(0)

    # Align DataFrames by all amino acids
    all_amino_acids = set(alpha_freq_df.columns).union(set(beta_freq_df.columns))
    alpha_freq_df = alpha_freq_df.reindex(columns=all_amino_acids, fill_value=0)
    beta_freq_df = beta_freq_df.reindex(columns=all_amino_acids, fill_value=0)

    # Concatenate all features
    combined_alpha = pd.concat([alpha_freq_df, alpha_v_onehot, alpha_j_onehot], axis=1)
    combined_beta = pd.concat([beta_freq_df, beta_v_onehot, beta_j_onehot], axis=1)

    return combined_alpha, combined_beta


def calc_mi(combined_alpha, combined_beta):
    # Compute MI matrix
    mi_matrix = pd.DataFrame(index=combined_alpha.columns, columns=combined_beta.columns)
    for col_alpha in combined_alpha.columns:
        for col_beta in combined_beta.columns:
            mi_matrix.loc[col_alpha, col_beta] = compute_mi(combined_alpha[col_alpha], combined_beta[col_beta])
    # Display MI matrix
    print("Mutual Information Matrix:\n", mi_matrix)
    return mi_matrix


# Visualize the MI Matrix using a heatmap
def plot_mi(mi_matrix):

    # Plot the heatmap with annotations only for high values
    plt.figure(figsize=(12, 10))
    sns.heatmap(mi_matrix.astype(float), annot=False, cmap='coolwarm', fmt="", linewidths=0.5)
    plt.title("Mutual Information Matrix between Alpha and Beta Chain Features")
    plt.show()


def hierarchical_clustering(matrix, text, color_range, bar, force_row_order=None, force_col_order=None,
                            return_order=False, significant_pairs=None):
    # Perform hierarchical clustering on rows and columns
    matrix = matrix.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors to NaN
    if not text.startswith("charge") and force_row_order is None and force_col_order is None:
        print("in here", text)
        matrix = matrix.dropna(how='all').dropna(axis=1, how='all')
    matrix = matrix.fillna(0)

    max_abs_value = np.nanmax(np.abs(matrix.values))
    # Compute the absolute sum of correlations for rows and columns
    filtered_matrix = matrix.loc[
        ~matrix.index.str.startswith("Rare"),
        ~matrix.columns.str.startswith("Rare")
    ]
    # If this is the first plot (no forced order), select top 20 features
    if force_row_order is None and force_col_order is None:
        # Compute the absolute sum of correlations for rows and columns
        row_scores = filtered_matrix.abs().sum(axis=1)
        column_scores = filtered_matrix.abs().sum(axis=0)

        # Identify the top 20 rows and columns based on the scores
        top_rows = row_scores.nlargest(20).index
        top_columns = column_scores.nlargest(20).index

        # Filter the correlation matrix to keep only the top rows and columns
        filtered_corr_matrix = filtered_matrix.loc[top_rows, top_columns]
    else:
        # Use the forced ordering from the first plot
        # Filter to only include features that exist in current matrix
        available_rows = [row for row in force_row_order if row in filtered_matrix.index]
        available_cols = [col for col in force_col_order if col in filtered_matrix.columns]

        filtered_corr_matrix = filtered_matrix.loc[available_rows, available_cols]
    # Remove rows or columns with zero variance to prevent issues with cosine metric
    if not text.startswith("charge") and force_row_order is None and force_col_order is None:
        filtered_corr_matrix = filtered_corr_matrix.loc[(filtered_corr_matrix != 0).any(axis=1)]
        filtered_corr_matrix = filtered_corr_matrix.loc[:, (filtered_corr_matrix != 0).any(axis=0)]

    # Calculate max absolute value for symmetric color scale
    max_abs_value = np.nanmax(np.abs(filtered_corr_matrix.values))
    # Create a custom colormap that goes from blue (negative) to white (zero) to red (positive)
    colors = [(0, 'red'), (0.5, 'white'), (1, 'blue')]  # This defines the transition
    n_bins = 90  # Number of bins for color interpolation
    cmap_name = 'blue_white_red'  # Custom name for the colormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    # filtered_corr_matrix.columns = [col.split('_')[-1] for col in filtered_corr_matrix.columns]

    # Create clustermap with or without clustering based on forced order
    if force_row_order is not None and force_col_order is not None:
        # Don't cluster - use the provided order
        cluster_map = sns.clustermap(filtered_corr_matrix, method='average', metric='euclidean',
                                     cmap=cmap, figsize=(20, 20),
                                     row_cluster=False, col_cluster=False,  # Disable clustering
                                     xticklabels=filtered_corr_matrix.columns,
                                     yticklabels=filtered_corr_matrix.index,
                                     cbar_pos=(0, .2, .03, .4),
                                     vmin=-color_range,  # Set the range for color bar
                                     vmax=color_range,  # Set the range for color bar
                                     center=0,  # Centering the color map at 0
                                     annot=False)
    else:
        # Perform clustering and potentially return the order
        cluster_map = sns.clustermap(filtered_corr_matrix, method='average', metric='euclidean',
                                     cmap=cmap, figsize=(20, 20),
                                     row_cluster=True, col_cluster=True,
                                     xticklabels=filtered_corr_matrix.columns,
                                     yticklabels=filtered_corr_matrix.index,
                                     cbar_pos=(0, .2, .03, .4),
                                     vmin=-color_range,  # Set the range for color bar
                                     vmax=color_range,  # Set the range for color bar
                                     center=0,  # Centering the color map at 0
                                     annot=False)

        # If return_order is True, extract the clustering order
        if return_order:
            # Get the reordered indices from the clustering
            row_order = [filtered_corr_matrix.index[i] for i in cluster_map.dendrogram_row.reordered_ind]
            col_order = [filtered_corr_matrix.columns[i] for i in cluster_map.dendrogram_col.reordered_ind]

    cluster_map.cax.set_visible(bar)

    # Add "P" annotations for significant pairs
    if significant_pairs is not None:
        for alpha_feature, beta_feature in significant_pairs:
            # Convert feature names to match the matrix format
            alpha_full_name = f"{alpha_feature} alpha"
            beta_full_name = f"{beta_feature} beta"

            # Check if both features are in the current matrix
            if alpha_full_name in filtered_corr_matrix.index and beta_full_name in filtered_corr_matrix.columns:
                if force_row_order is not None and force_col_order is not None:
                    # No clustering - use original positions
                    row_idx = list(filtered_corr_matrix.index).index(alpha_full_name)
                    col_idx = list(filtered_corr_matrix.columns).index(beta_full_name)
                else:
                    # Clustering was performed - find position in reordered matrix
                    original_row_idx = list(filtered_corr_matrix.index).index(alpha_full_name)
                    original_col_idx = list(filtered_corr_matrix.columns).index(beta_full_name)

                    # Find the new position after clustering
                    row_idx = list(cluster_map.dendrogram_row.reordered_ind).index(original_row_idx)
                    col_idx = list(cluster_map.dendrogram_col.reordered_ind).index(original_col_idx)

                # Add "P" annotation at the center of the cell
                cluster_map.ax_heatmap.text(col_idx + 0.5, row_idx + 0.5, 'P',
                                            ha='center', va='center',
                                            color='white', fontsize=50,
                                            fontweight='bold')

    # Apply font properties to x and y tick labels
    font_size = 70
    for label in cluster_map.ax_heatmap.get_xticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(font_size)
        color_label(label)

    for label in cluster_map.ax_heatmap.get_yticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(font_size)
        color_label(label)
    # Customize the color bar tick labels
    for label in cluster_map.ax_cbar.yaxis.get_ticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(font_size)
    # for label in cluster_map.ax_heatmap.get_yticklabels():
    #     label.set_x(1.1)  # Move the labels further left (adjust the value as needed)

    plt.setp(cluster_map.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)  # Horizontal for x-axis
    plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # plt.tight_layout()  # Adjust layout
    plt.savefig(text, bbox_inches="tight")

    plt.show()
    # return cluster_map
    # Return the clustering order if requested (for first plot)
    if return_order and force_row_order is None and force_col_order is None:
        return row_order, col_order

def color_label(label):
    # Color condition for x ticks
    if label.get_text().startswith('Rare V') or label.get_text().startswith('TRAV'):
        label.set_color('red')  # Color 1 for va or TRAV
    elif label.get_text().startswith('Rare V') or label.get_text().startswith('TRBV'):
        label.set_color('red')  # Color 2 for vb or TRBV
    elif label.get_text().startswith('Rare J') or label.get_text().startswith('TRAJ'):
        label.set_color('blue')  # Color 3 for ja or TRAJ
    elif label.get_text().startswith('Rare J') or label.get_text().startswith('TRBJ'):
        label.set_color('blue')  # Color 4 for jb or TRBJ
    elif label.get_text().isalpha():
        label.set_color('purple')  # Color 5 for letters (else condition)


def calc_combined_correlation(combined_alpha, combined_beta, significance_level=0.05, method='fdr_bh'):
    # Initialize results
    correlations = []
    p_values = []
    row_labels = []
    col_labels = []

    for col_alpha in combined_alpha.columns:
        for col_beta in combined_beta.columns:
            row_labels.append(col_alpha)
            col_labels.append(col_beta)

            # If either column is constant
            if combined_alpha[col_alpha].nunique() == 1 or combined_beta[col_beta].nunique() == 1:
                correlations.append(np.nan)
                p_values.append(1.0)
                print(f"Skipping {col_alpha} and {col_beta} due to constant values.")
                continue

            alpha_is_continuous = combined_alpha[col_alpha].nunique() > 2
            beta_is_continuous = combined_beta[col_beta].nunique() > 2

            try:
                if alpha_is_continuous and beta_is_continuous:
                    corr, p_value = spearmanr(combined_alpha[col_alpha], combined_beta[col_beta])
                elif not alpha_is_continuous and not beta_is_continuous:
                    corr, p_value = spearmanr(combined_alpha[col_alpha], combined_beta[col_beta])
                else:
                    if alpha_is_continuous:
                        corr, p_value = pointbiserialr(combined_beta[col_beta], combined_alpha[col_alpha])
                    else:
                        corr, p_value = pointbiserialr(combined_alpha[col_alpha], combined_beta[col_beta])

                correlations.append(corr)
                p_values.append(p_value)
            except Exception as e:
                correlations.append(np.nan)
                p_values.append(1.0)
                print(f"Could not compute correlation between {col_alpha} and {col_beta}: {e}")

    # Apply multiple testing correction
    _, corrected_pvals, _, _ = multipletests(p_values, alpha=significance_level, method=method)

    # Create corrected correlation matrix
    corr_matrix = pd.DataFrame(np.nan, index=combined_alpha.columns, columns=combined_beta.columns)

    for i, (row, col, corr, pval) in enumerate(zip(row_labels, col_labels, correlations, corrected_pvals)):
        if pval < significance_level and not np.isnan(corr):
            corr_matrix.loc[row, col] = corr
        else:
            corr_matrix.loc[row, col] = np.nan

    return corr_matrix

def plot_correlation(corr_matrix):
    print(corr_matrix.shape)
    filtered_corr_matrix = corr_matrix.dropna(how='all').dropna(axis=1, how='all')
    max_abs_value = np.nanmax(np.abs(filtered_corr_matrix.values))
    # Remove rows and columns where the sum is smaller than 0.04
    filtered_corr_matrix = filtered_corr_matrix.loc[
        filtered_corr_matrix.sum(axis=1) >= max_abs_value / 2,  # Filter rows
        filtered_corr_matrix.sum(axis=0) >= max_abs_value / 2  # Filter columns
    ]
    print(filtered_corr_matrix.shape)
    # Plot the heatmap with annotations
    plt.figure(figsize=(15, 15))
    sns.heatmap(filtered_corr_matrix.astype(float), annot=False, cmap='seismic', fmt="", linewidths=0.5)
    plt.title("Correlation Matrix between Alpha and Beta Chain Features")
    plt.show()


def get_properties(sequence):
    """Calculate properties of the given protein sequence."""
    analysis = ProteinAnalysis(sequence)
    weight = analysis.molecular_weight()
    isoelectric_point = analysis.isoelectric_point()
    hydrophobicity = analysis.gravy()  # Grand average of hydrophobicity
    return weight, isoelectric_point, hydrophobicity


def calculate_bio_seq(data, name, file=True):


    # Check if input is CSV or NPZ files
    if file:
        data = pd.read_csv(data)

    # Ensure the required columns exist
    if 'tcra' not in data.columns or 'tcrb' not in data.columns:
        raise ValueError("The input CSV must contain columns named 'tcra' and 'tcrb'.")

    alpha_properties = {'Weight': [], 'Charge': [], 'Hydrophobicity': []}
    beta_properties = {'Weight': [], 'Charge': [], 'Hydrophobicity': []}

    # Process sequences in 'tcra' and 'tcrb'
    for index, row in data.iterrows():
        for chain, properties in zip(['tcra', 'tcrb'], [alpha_properties, beta_properties]):
            sequence = row[chain]
            if pd.isna(sequence):  # Skip missing sequences
                properties['Weight'].append(None)
                properties['Charge'].append(None)
                properties['Hydrophobicity'].append(None)
                continue
            weight, isoelectric_point, hydrophobicity = get_properties(sequence)
            properties['Weight'].append(weight)
            properties['Charge'].append(isoelectric_point)
            properties['Hydrophobicity'].append(hydrophobicity)

        # Convert to DataFrames
    alpha_df = pd.DataFrame(alpha_properties, columns=['Weight', 'Charge', 'Hydrophobicity'])
    beta_df = pd.DataFrame(beta_properties, columns=['Weight', 'Charge', 'Hydrophobicity'])
    alpha_df.to_csv(f"alpha_properties_{name}.csv", index=False)
    beta_df.to_csv(f"beta_properties_{name}.csv", index=False)


def process_sequences_and_calculate_correlations(input_file, name, file_format, with_fdr=True):
    if file_format == "csv":  # Assume input_file_1 is a CSV file
        # calculate_bio_seq(input_file, name)
        alpha_df = pd.read_csv(f"alpha_properties_{name}.csv")
        beta_df = pd.read_csv(f"beta_properties_{name}.csv")
        combined_df = pd.concat([alpha_df, beta_df], axis=1, keys=['alpha', 'beta'])
        # combined_df.dropna(inplace=True)

    else:  # Assume input_file_1 and input_file_2 are NPZ files
        with open(input_file, 'r') as f:

            lines = f.readlines()

        # Extract sequences from file content
        top_alpha_positives_seq = [seq.strip().strip("'") for seq in lines[1].strip()[1:-1].split(',')]
        top_beta_positives_seq = [seq.strip().strip("'") for seq in lines[5].strip()[1:-1].split(',')]

        # Ensure that the number of sequences in alpha and beta match
        if len(top_alpha_positives_seq) != len(top_beta_positives_seq):
            raise ValueError("The number of alpha and beta sequences must match.")

        alpha_properties = {'Weight': [], 'Charge': [], 'Hydrophobicity': []}
        beta_properties = {'Weight': [], 'Charge': [], 'Hydrophobicity': []}

        # Process sequences for 'tcra' and 'tcrb'
        for alpha_sequence, beta_sequence in zip(top_alpha_positives_seq, top_beta_positives_seq):
            if alpha_sequence is None or beta_sequence is None:
                alpha_properties['Weight'].append(None)
                alpha_properties['Charge'].append(None)
                alpha_properties['Hydrophobicity'].append(None)
                beta_properties['Weight'].append(None)
                beta_properties['Charge'].append(None)
                beta_properties['Hydrophobicity'].append(None)
                continue

            # Calculate properties for both alpha and beta sequences
            alpha_weight, alpha_isoelectric_point, alpha_hydrophobicity = get_properties(alpha_sequence)
            beta_weight, beta_isoelectric_point, beta_hydrophobicity = get_properties(beta_sequence)

            # Append the properties
            alpha_properties['Weight'].append(alpha_weight)
            alpha_properties['Charge'].append(alpha_isoelectric_point)
            alpha_properties['Hydrophobicity'].append(alpha_hydrophobicity)

            beta_properties['Weight'].append(beta_weight)
            beta_properties['Charge'].append(beta_isoelectric_point)
            beta_properties['Hydrophobicity'].append(beta_hydrophobicity)

        # Convert properties into DataFrames
        alpha_df = pd.DataFrame(alpha_properties, columns=['Weight', 'Charge', 'Hydrophobicity'])
        beta_df = pd.DataFrame(beta_properties, columns=['Weight', 'Charge', 'Hydrophobicity'])

        # Drop rows with missing values
        combined_df = pd.concat([alpha_df, beta_df], axis=1, keys=['alpha', 'beta'])
        # combined_df.dropna(inplace=True)

    # Calculate the pairwise correlations
    correlations = {}
    p_values = []
    pairs = []
    for alpha_col in ['Weight', 'Charge', 'Hydrophobicity']:
        for beta_col in ['Weight', 'Charge', 'Hydrophobicity']:
            # Get correlation and p-value
            # r, p_value = spearmanr(combined_df['alpha'][alpha_col], combined_df['beta'][beta_col])
            alpha_values = combined_df['alpha'][alpha_col]
            beta_values = combined_df['beta'][beta_col]

            if alpha_values.nunique() <= 1 or beta_values.nunique() <= 1:
                # Constant input — correlation is undefined
                r, p_value = np.nan, 1.0
            elif len(alpha_values) < 3 or len(beta_values) < 3:
                print("here")
                r, p_value = np.nan, 1.0
            else:
                r, p_value = spearmanr(alpha_values, beta_values)
            print(r, p_value)
            pairs.append((alpha_col, beta_col))
            p_values.append(p_value)
            # If not significant, set to 0
            correlations[(alpha_col, beta_col)] = r
                #if p_value <= 0.05 else 0
    if with_fdr:
        # Apply multiple testing correction (e.g. Benjamini-Hochberg FDR)
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        # Update correlations: set to 0 if corrected p-value is NOT significant
        for i, (alpha_col, beta_col) in enumerate(pairs):
            if not reject[i]:
                correlations[(alpha_col, beta_col)] = 0

    # Reshape into a 3x3 DataFrame
    cross_corr_matrix = pd.DataFrame(
        [[correlations[(a, b)] for b in ['Weight', 'Charge', 'Hydrophobicity']]
         for a in ['Weight', 'Charge', 'Hydrophobicity']],
        index=['Weight alpha', 'Charge alpha', 'Hydrophobicity alpha'],
        columns=['Weight beta', 'Charge beta', 'Hydrophobicity beta']
    )

    print(cross_corr_matrix)
    return cross_corr_matrix


def check_duplicate_rows(csv_file):
    """
    Check if there are duplicate rows in a CSV file where all columns are the same.

    Args:
        csv_file (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame of duplicate rows, or an empty DataFrame if no duplicates exist.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Identify duplicate rows
    duplicates = data[data.duplicated(keep=False)]  # keep=False marks all occurrences as duplicates

    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows:")
        print(duplicates)
    else:
        print("No duplicate rows found.")

    return duplicates

def print_duplicate_groups(csv_file):
    """
    Identifies and prints groups of duplicate rows with their row numbers.

    Args:
        csv_file (str): Path to the input CSV file.
    """
    # Load the data
    data = pd.read_csv(csv_file)

    # Create a unique identifier for each duplicate group
    duplicate_groups = data[data.duplicated(keep=False)]  # Keep all duplicates
    duplicate_groups['group_id'] = duplicate_groups.apply(tuple, axis=1).rank(method='dense').astype(int)

    # Group by the unique identifier and print results
    grouped_duplicates = duplicate_groups.groupby('group_id')
    for group_id, group in grouped_duplicates:
        print(f"Duplicate Group {group_id}:")
        print(f"Row numbers: {group.index.tolist()}")
        print(group.drop(columns=['group_id']))  # Drop the helper column for cleaner output
        print("-" * 40)


def process_multiple_datasets(datasets, dataset_names, plot_type="both"):
    """
    Process multiple datasets and generate plots

    Args:
        datasets: list of dataset file paths
        dataset_names: list of names for the datasets (for plot titles)
        plot_type: "correlation", "charge", or "both"
    """
    all_corr_matrices = []
    all_charge_matrices = []

    # Process each dataset
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i + 1}: {dataset_names[i]}")

        # Load and process data
        tcra, va, ja, tcrb, vb, jb = load_data(dataset)
        combined_alpha, combined_beta = process_data(dataset, tcra, va, ja, tcrb, vb, jb)

        # Calculate correlation matrix
        if plot_type in ["correlation", "both"]:
            corr_matrix = calc_combined_correlation(combined_alpha, combined_beta)
            all_corr_matrices.append(corr_matrix)

        # Calculate charge correlation matrix
        if plot_type in ["charge", "both"]:
            charge_matrix = process_sequences_and_calculate_correlations(dataset, dataset_names[i], "csv")
            all_charge_matrices.append(charge_matrix)

    # Generate plots based on plot_type
    if plot_type in ["correlation", "both"]:
        generate_correlation_plots(all_corr_matrices, dataset_names)

    if plot_type in ["charge", "both"]:
        generate_charge_plots(all_charge_matrices, dataset_names)


def generate_correlation_plots(corr_matrices, dataset_names):
    """Generate correlation plots for multiple datasets"""
    names_for_plot = {"VDjdb": "pMHC1", "Minervina": "pMHC2", "Healthy": "All T cells"}
    color_range = 0.3

    # Generate individual plots
    plot_filenames = []
    for i, (matrix, name) in enumerate(zip(corr_matrices, dataset_names)):
        filename = f"clustermap_{name.lower().replace(' ', '_')}.png"
        if i == 0:
            bar = True
            row_order, col_order = hierarchical_clustering(matrix, filename, color_range, bar,
                                                           return_order=True)
        else:
            bar = False
            hierarchical_clustering(matrix, filename, color_range, bar,
                                    force_row_order=row_order, force_col_order=col_order)

        # hierarchical_clustering(matrix, filename, color_range, bar)
        plot_filenames.append(filename)

    # Create combined figure
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(1, n_datasets, figsize=(20 * n_datasets // 2, 10))

    if n_datasets == 1:
        axes = [axes]  # Make it iterable for single plot

    import matplotlib.image as mpimg

    # Load and display the saved heatmaps
    for i, (filename, name) in enumerate(zip(plot_filenames, dataset_names)):
        if os.path.exists(filename):
            img = mpimg.imread(filename)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(names_for_plot[name], fontsize=34)

    # Save the combined figure
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.02)
    plt.savefig("feature_correlations_3_datasets.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def generate_charge_plots(charge_matrices, dataset_names):
    """Generate charge correlation plots for multiple datasets"""
    names_for_plot = {"VDjdb": "pMHC1", "Minervina": "pMHC2", "Healthy": "All T cells"}
    color_range = 0.1

    # Generate individual plots
    plot_filenames = []
    for i, (matrix, name) in enumerate(zip(charge_matrices, dataset_names)):
        filename = f"charge_corr_{name.lower().replace(' ', '_')}.png"
        if i == 0:
            bar = True
            significant_pairs = new_statistics.with_pep_mhc("pep")
            row_order, col_order = hierarchical_clustering(matrix, filename, color_range, bar,
                                                           return_order=True, significant_pairs=significant_pairs)
        else:
            bar = False
            if i == 1:
                hierarchical_clustering(matrix, filename, color_range, bar)
            else:
                hierarchical_clustering(matrix, filename, color_range, bar,
                                        force_row_order=row_order, force_col_order=col_order)
        # hierarchical_clustering(matrix, filename, color_range, bar)
        plot_filenames.append(filename)

    # Create combined figure
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(1, n_datasets, figsize=(20 * n_datasets // 2, 10))

    if n_datasets == 1:
        axes = [axes]  # Make it iterable for single plot

    import matplotlib.image as mpimg

    # Load and display the saved heatmaps
    for i, (filename, name) in enumerate(zip(plot_filenames, dataset_names)):
        if os.path.exists(filename):
            img = mpimg.imread(filename)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(names_for_plot[name], fontsize=34)

    # Save the combined figure
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.02)
    plt.savefig("physicochemical_correlations_3_datasets.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_histograms(df1, df2, alpha_columns, beta_columns):
    """
    Plots histograms for the frequency of alpha and beta triplets from two DataFrames for comparison.

    Args:
        df1 (pd.DataFrame): First DataFrame containing the data.
        df2 (pd.DataFrame): Second DataFrame containing the data.
        alpha_columns (list): List of column names for alpha triplets.
        beta_columns (list): List of column names for beta triplets.
    """
    # Create triplets for alpha and beta from both dataframes
    alpha_triplets_1 = df1[alpha_columns].apply(tuple, axis=1)
    beta_triplets_1 = df1[beta_columns].apply(tuple, axis=1)

    alpha_triplets_2 = df2[alpha_columns].apply(tuple, axis=1)
    beta_triplets_2 = df2[beta_columns].apply(tuple, axis=1)

    # Count frequencies of each unique triplet for both DataFrames
    alpha_counts_1 = alpha_triplets_1.value_counts()
    beta_counts_1 = beta_triplets_1.value_counts()

    alpha_counts_2 = alpha_triplets_2.value_counts()
    beta_counts_2 = beta_triplets_2.value_counts()
    # Get the maximum frequencies for both datasets to ensure same bins
    all_alpha_frequencies = np.concatenate([alpha_counts_1.values, alpha_counts_2.values])
    all_beta_frequencies = np.concatenate([beta_counts_1.values, beta_counts_2.values])
    # Determine common bin edges using log scaling
    alpha_bins = np.logspace(np.log10(1), np.log10(max(all_alpha_frequencies)), 50)
    beta_bins = np.logspace(np.log10(1), np.log10(max(all_beta_frequencies)), 50)

    font_size = 100
    alpha_frequencies_1 = alpha_counts_1.values
    beta_frequencies_1 = beta_counts_1.values

    alpha_frequencies_2 = alpha_counts_2.values
    beta_frequencies_2 = beta_counts_2.values

    # Plot the histograms for both DataFrames
    plt.figure(figsize=(18, 6))

    # Alpha triplets comparison histogram
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(alpha_frequencies_1, bins=alpha_bins,
             color='blue', edgecolor='black',label='pMHC-I', width=0.8, alpha=0.5)
    ax1.hist(alpha_frequencies_2, bins=alpha_bins,
             color='orange', edgecolor='black', label='All T cells', width=0.8, alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency of Alpha', fontsize=font_size, fontproperties=font)
    ax1.set_ylabel('Count (Log Scale)', fontsize=font_size, fontproperties=font)
    legend_font = font.copy()
    legend_font.set_size(20)
    ax1.legend(prop=legend_font)
    # Apply font properties to y-axis labels
    for label in ax1.get_yticklabels() + ax1.get_xticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(font_size)

    # Beta triplets comparison histogram
    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(beta_frequencies_1, bins=beta_bins,
             color='blue', edgecolor='black', label='pMHC-I', width=0.8, alpha=0.5)
    ax2.hist(beta_frequencies_2, bins=beta_bins,
             color='orange', edgecolor='black', label='All T cells', width=0.8, alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Frequency of Beta', fontsize=font_size, fontproperties=font)
    ax2.legend(prop=legend_font)
    # Apply font properties to y-axis labels
    for label in ax2.get_yticklabels() + ax2.get_xticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(font_size)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datasets = [os.path.join("new_data", "sapir_data_positives.csv"),
                os.path.join("new_data", "Minervina_june_positives.csv"),
                os.path.join("new_data", "june_all_positives.csv")
                ]
    dataset_names = ("VDjdb", "Minervina", "Healthy")
    process_multiple_datasets(datasets, dataset_names, plot_type="both")
