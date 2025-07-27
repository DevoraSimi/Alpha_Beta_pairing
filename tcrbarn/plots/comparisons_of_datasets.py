import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(".."))
import v_j
from matplotlib.gridspec import GridSpec
from collections import Counter
from matplotlib.font_manager import FontProperties


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font_size = 16
font.set_size(font_size)


def combine(dataset1, dataset2, dataset3, columns_to_compare, name1, name2, name3, threshold=0.02):

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 0.5], width_ratios=[1, 1])
    ax_alpha_v = fig.add_subplot(gs[0, 0])
    ax_beta_v = fig.add_subplot(gs[0, 1])
    ax_alpha_j = fig.add_subplot(gs[1, 0])
    ax_beta_j = fig.add_subplot(gs[1, 1])
    axes = [ax_alpha_v, ax_beta_v, ax_alpha_j, ax_beta_j]
    compare_one_hot(axes, dataset1, dataset2, dataset3, columns_to_compare, name1, name2, name3, threshold)
    compare_alpha_beta(dataset1, dataset2, dataset3,  name1, name2, name3, fig, gs)
    plt.tight_layout()
    plt.savefig("compare_combined_3_datasets.pdf", format="pdf")
    plt.show()


def compare_one_hot(axes, dataset1, dataset2, dataset3, columns_to_compare, name1, name2, name3, threshold=0.02):

    # Loop over each column and plot the normalized frequencies
    for i, col in enumerate(columns_to_compare):
        # Process va, vb, ja, jb using v_j_format
        gene_name = "TR" + col[::-1].upper()
        processed_column1 = dataset1[col].apply(lambda x: v_j.v_j_format(x, 2 if gene_name == "TRBJ" else 1, gene_name))
        # Calculate value counts and normalize by the dataset size
        freq1 = processed_column1.value_counts(normalize=True).sort_index()
        processed_column2 = dataset2[col].apply(lambda x: v_j.v_j_format(x, 2 if gene_name == "TRBJ" else 1, gene_name))
        freq2 = processed_column2.value_counts(normalize=True).sort_index()
        processed_column3 = dataset3[col].apply(lambda x: v_j.v_j_format(x, 2 if gene_name == "TRBJ" else 1, gene_name))
        freq3 = processed_column3.value_counts(normalize=True).sort_index()
        # Remove small fractions below the threshold first
        freq1 = freq1[freq1 >= threshold]
        freq2 = freq2[freq2 >= threshold]
        freq3 = freq3[freq3 >= threshold]

        # Then align all datasets for comparison by reindexing to common indices
        all_indices = freq1.index.union(freq2.index).union(freq3.index)
        freq1 = freq1.reindex(all_indices, fill_value=0)
        freq2 = freq2.reindex(all_indices, fill_value=0)
        freq3 = freq3.reindex(all_indices, fill_value=0)
        # Define the width of the bars
        bar_width = 0.25  # was 0.35

        # Plot on the corresponding subplot with bars next to each other
        # axes[i].bar(freq1.index, freq1.values, label=name, alpha=0.7, width=bar_width)
        # axes[i].bar(freq2.index, freq2.values, label=name2, alpha=0.7, color='orange', width=bar_width)
        # Add the offset to the x-positions of the second dataset
        positions1 = range(len(freq1))  # x-positions for the first dataset
        positions2 = [p + bar_width for p in positions1]  # Offset x-positions for the second dataset
        positions3 = [p + 2 * bar_width for p in positions1]  # Offset x-positions for the third dataset

        # Plot on the corresponding subplot with bars next to each other
        axes[i].barh(positions1, freq1.values, label=name1, alpha=0.7, color='blue', height=bar_width)
        axes[i].barh(positions2, freq2.values, label=name2, alpha=0.7, color='orange', height=bar_width)
        axes[i].barh(positions3, freq3.values, label=name3, alpha=0.7, color='green', height=bar_width)

        # axes[i].set_title(f'Normalized Comparison of {col.upper()}')
        axes[i].set_yticks([p + bar_width for p in positions1])  # Set x-ticks between the bars
        # axes[i].set_yticks([p + bar_width / 2 for p in positions1])  # Set x-ticks between the bars

        axes[i].set_yticklabels(all_indices)
        letters = {0: "a", 1: "b", 2: "c", 3: "d"}
        axes[i].set_ylabel(f'{col.upper()} Values', fontproperties=font)
        axes[i].set_xlabel(f'Proportion \n ({letters[i]})', fontproperties=font)
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(font)

        for label in axes[i].get_yticklabels():
            label.set_fontproperties(font)
        axes[i].legend(prop=font, loc='lower right')


def compare_alpha_beta(dataset1, dataset2, dataset3, name1, name2, name3, fig, gs):
    # Function to calculate amino acid composition
    def get_aa_composition(sequences):
        aa_counter = Counter()
        total_length = 0
        for seq in sequences.dropna():  # Drop any NaN values
            aa_counter.update(seq)
            total_length += len(seq)
        # Normalize by the total number of amino acids
        return {aa: count / total_length for aa, count in aa_counter.items()}

    # Calculate compositions for alpha and beta CDR3 in each dataset
    alpha_aa_comp1 = get_aa_composition(dataset1['tcra'])
    alpha_aa_comp2 = get_aa_composition(dataset2['tcra'])
    alpha_aa_comp3 = get_aa_composition(dataset3['tcra'])
    beta_aa_comp1 = get_aa_composition(dataset1['tcrb'])
    beta_aa_comp2 = get_aa_composition(dataset2['tcrb'])
    beta_aa_comp3 = get_aa_composition(dataset3['tcrb'])

    # Combine data for plotting
    amino_acids = sorted(set(alpha_aa_comp1.keys()).union(alpha_aa_comp2.keys(), alpha_aa_comp3.keys(),
                                                          beta_aa_comp1.keys(), beta_aa_comp2.keys(),
                         beta_aa_comp3.keys()))
    alpha_freqs1 = [alpha_aa_comp1.get(aa, 0) for aa in amino_acids]
    alpha_freqs2 = [alpha_aa_comp2.get(aa, 0) for aa in amino_acids]
    alpha_freqs3 = [alpha_aa_comp3.get(aa, 0) for aa in amino_acids]
    beta_freqs1 = [beta_aa_comp1.get(aa, 0) for aa in amino_acids]
    beta_freqs2 = [beta_aa_comp2.get(aa, 0) for aa in amino_acids]
    beta_freqs3 = [beta_aa_comp3.get(aa, 0) for aa in amino_acids]

    # Length distributions for alpha and beta chains in each dataset
    alpha_lengths1 = dataset1['tcra'].dropna().apply(len)
    alpha_lengths2 = dataset2['tcra'].dropna().apply(len)
    alpha_lengths3 = dataset3['tcra'].dropna().apply(len)
    beta_lengths1 = dataset1['tcrb'].dropna().apply(len)
    beta_lengths2 = dataset2['tcrb'].dropna().apply(len)
    beta_lengths3 = dataset3['tcrb'].dropna().apply(len)

    # Set a threshold for minimum frequency as a fraction of total counts
    def filter_low_frequencies(lengths, min_fraction=0.01):
        total_count = len(lengths)
        counts = Counter(lengths)
        return {k: v / total_count for k, v in counts.items() if v / total_count >= min_fraction}

    # Filtered length frequencies for each dataset and chain
    alpha_length_freq1 = filter_low_frequencies(alpha_lengths1)
    alpha_length_freq2 = filter_low_frequencies(alpha_lengths2)
    alpha_length_freq3 = filter_low_frequencies(alpha_lengths3)
    beta_length_freq1 = filter_low_frequencies(beta_lengths1)
    beta_length_freq2 = filter_low_frequencies(beta_lengths2)
    beta_length_freq3 = filter_low_frequencies(beta_lengths3)

    # Prepare lengths and frequencies for plotting
    all_lengths = sorted(set(alpha_length_freq1.keys()).union(alpha_length_freq2.keys(), alpha_length_freq3.keys(),
                                                              beta_length_freq1.keys(), beta_length_freq2.keys(),
                                                              beta_length_freq3.keys()))
    alpha_freqs1_filtered = [alpha_length_freq1.get(length, 0) for length in all_lengths]
    alpha_freqs2_filtered = [alpha_length_freq2.get(length, 0) for length in all_lengths]
    alpha_freqs3_filtered = [alpha_length_freq3.get(length, 0) for length in all_lengths]
    beta_freqs1_filtered = [beta_length_freq1.get(length, 0) for length in all_lengths]
    beta_freqs2_filtered = [beta_length_freq2.get(length, 0) for length in all_lengths]
    beta_freqs3_filtered = [beta_length_freq3.get(length, 0) for length in all_lengths]

    # Amino acid composition comparison for Alpha CDR3
    ax_alpha = fig.add_subplot(gs[2, 0])
    width = 0.25
    x = range(len(amino_acids))
    ax_alpha.barh(x, alpha_freqs1, height=width, label=name1, color='blue', alpha=0.7)
    ax_alpha.barh([p + width for p in x], alpha_freqs2, height=width, label=name2, color='orange',
                 alpha=0.7)
    ax_alpha.barh([p + 2 * width for p in x], alpha_freqs3, height=width, label=name3, color='green', alpha=0.7)
    ax_alpha.set_yticks([p + width for p in x])
    # ax_alpha.set_yticks([p + width / 2 for p in x])

    ax_alpha.set_yticklabels(amino_acids)
    ax_alpha.set_ylabel('Amino Acids - Alpha CDR3', fontproperties=font)
    ax_alpha.set_xlabel('Normalized Frequency \n (e)', fontproperties=font)
    # ax_alpha.set_title('Amino Acid Composition - Alpha CDR3')
    for label in ax_alpha.get_xticklabels():
        label.set_fontproperties(font)

    for label in ax_alpha.get_yticklabels():
        label.set_fontproperties(font)
    ax_alpha.legend(prop=font, loc='upper right')

    # Amino acid composition comparison for Beta CDR3
    ax_beta = fig.add_subplot(gs[2, 1]) # add
    ax_beta.barh(x, beta_freqs1, height=width, label=name1, color='blue', alpha=0.7)
    ax_beta.barh([p + width for p in x], beta_freqs2, height=width, label=name2, color='orange',
                alpha=0.7)
    ax_beta.barh([p + 2 * width for p in x], beta_freqs3, height=width, label=name3, color='green', alpha=0.7)
    ax_beta.set_yticks([p + width for p in x])
    # ax_beta.set_yticks([p + width / 2 for p in x])
    ax_beta.set_yticklabels(amino_acids)
    ax_beta.set_ylabel('Amino Acids - Beta CDR3', fontproperties=font)
    ax_beta.set_xlabel('Normalized Frequency \n (f)', fontproperties=font)
    # ax_beta.set_title('Amino Acid Composition - Beta CDR3')
    for label in ax_beta.get_xticklabels():
        label.set_fontproperties(font)

    for label in ax_beta.get_yticklabels():
        label.set_fontproperties(font)
    ax_beta.legend(prop=font, loc='upper right')

    # CDR3 Length Distribution Comparison (centered below Alpha and Beta composition plots)
    # Two subplots for Alpha and Beta
    ax_alpha = fig.add_subplot(gs[3, 0])
    ax_beta = fig.add_subplot(gs[3, 1])

    width = 0.2
    x = np.arange(len(all_lengths))

    # ---- Alpha Plot ----
    ax_alpha.bar(x - width, alpha_freqs1_filtered, width=width, color='blue', alpha=0.6, label=name1)
    ax_alpha.bar(x, alpha_freqs2_filtered, width=width, color='orange', alpha=0.6, label=name2)
    ax_alpha.bar(x + width, alpha_freqs3_filtered, width=width, color='green', alpha=0.6, label=name3)
    ax_alpha.set_xticks(x)
    ax_alpha.set_xticklabels(all_lengths)
    ax_alpha.set_xlabel('CDR3 Length - Alpha \n (g)', fontproperties=font)
    ax_alpha.set_ylabel('Fraction of Total', fontproperties=font)
    ax_alpha.legend(prop=font, loc='upper right')
    for label in ax_alpha.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax_alpha.get_yticklabels():
        label.set_fontproperties(font)

    # ---- Beta Plot ----
    ax_beta.bar(x - width, beta_freqs1_filtered, width=width, color='blue', alpha=0.6, label=name1)
    ax_beta.bar(x, beta_freqs2_filtered, width=width, color='orange', alpha=0.6, label=name2)
    ax_beta.bar(x + width, beta_freqs3_filtered, width=width, color='green', alpha=0.6, label=name3)
    ax_beta.set_xticks(x)
    ax_beta.set_xticklabels(all_lengths)
    ax_beta.set_xlabel('CDR3 Length - Beta \n (h)', fontproperties=font)
    ax_beta.legend(prop=font, loc='upper right')
    for label in ax_beta.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax_beta.get_yticklabels():
        label.set_fontproperties(font)


def compare_datasets_combined(dataset1, dataset2, columns_to_compare, name1, name2, threshold=0.02):
    # Helper function to calculate amino acid composition
    def get_aa_composition(sequences):
        aa_counter = Counter()
        total_length = 0
        for seq in sequences.dropna():  # Drop any NaN values
            aa_counter.update(seq)
            total_length += len(seq)
        return {aa: count / total_length for aa, count in aa_counter.items()}

    # Amino acid composition for alpha and beta chains
    alpha_aa_comp1 = get_aa_composition(dataset1['tcra'])
    alpha_aa_comp2 = get_aa_composition(dataset2['tcra'])
    beta_aa_comp1 = get_aa_composition(dataset1['tcrb'])
    beta_aa_comp2 = get_aa_composition(dataset2['tcrb'])

    # Combine data for amino acids
    amino_acids = sorted(set(alpha_aa_comp1.keys()).union(alpha_aa_comp2.keys(),
                                                          beta_aa_comp1.keys(), beta_aa_comp2.keys()))
    alpha_freqs1 = [alpha_aa_comp1.get(aa, 0) for aa in amino_acids]
    alpha_freqs2 = [alpha_aa_comp2.get(aa, 0) for aa in amino_acids]
    beta_freqs1 = [beta_aa_comp1.get(aa, 0) for aa in amino_acids]
    beta_freqs2 = [beta_aa_comp2.get(aa, 0) for aa in amino_acids]

    # Length distributions for alpha and beta chains
    alpha_lengths1 = dataset1['tcra'].dropna().apply(len)
    alpha_lengths2 = dataset2['tcra'].dropna().apply(len)
    beta_lengths1 = dataset1['tcrb'].dropna().apply(len)
    beta_lengths2 = dataset2['tcrb'].dropna().apply(len)

    def filter_low_frequencies(lengths, min_fraction=0.01):
        total_count = len(lengths)
        counts = Counter(lengths)
        return {k: v / total_count for k, v in counts.items() if v / total_count >= min_fraction}

    alpha_length_freq1 = filter_low_frequencies(alpha_lengths1)
    alpha_length_freq2 = filter_low_frequencies(alpha_lengths2)
    beta_length_freq1 = filter_low_frequencies(beta_lengths1)
    beta_length_freq2 = filter_low_frequencies(beta_lengths2)

    all_lengths = sorted(set(alpha_length_freq1.keys()).union(alpha_length_freq2.keys(),
                                                              beta_length_freq1.keys(), beta_length_freq2.keys()))
    alpha_freqs1_filtered = [alpha_length_freq1.get(length, 0) for length in all_lengths]
    alpha_freqs2_filtered = [alpha_length_freq2.get(length, 0) for length in all_lengths]
    beta_freqs1_filtered = [beta_length_freq1.get(length, 0) for length in all_lengths]
    beta_freqs2_filtered = [beta_length_freq2.get(length, 0) for length in all_lengths]

    # Initialize a unified plot
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

    # Subplots for amino acid compositions
    ax_alpha_aa = fig.add_subplot(gs[0, 0])
    width = 0.35
    x = range(len(amino_acids))
    ax_alpha_aa.bar(x, alpha_freqs1, width=width, label=name1 + ' - Alpha CDR3', color='blue', alpha=0.7)
    ax_alpha_aa.bar([p + width for p in x], alpha_freqs2, width=width, label=name2 + ' - Alpha CDR3', color='orange', alpha=0.7)
    ax_alpha_aa.set_xticks([p + width / 2 for p in x])
    ax_alpha_aa.set_xticklabels(amino_acids)
    ax_alpha_aa.set_title('Amino Acid Composition - Alpha CDR3')
    ax_alpha_aa.legend()

    ax_beta_aa = fig.add_subplot(gs[0, 1])
    ax_beta_aa.bar(x, beta_freqs1, width=width, label=name1 + ' - Beta CDR3', color='blue', alpha=0.7)
    ax_beta_aa.bar([p + width for p in x], beta_freqs2, width=width, label=name2 + ' - Beta CDR3', color='orange', alpha=0.7)
    ax_beta_aa.set_xticks([p + width / 2 for p in x])
    ax_beta_aa.set_xticklabels(amino_acids)
    ax_beta_aa.set_title('Amino Acid Composition - Beta CDR3')
    ax_beta_aa.legend()

    # Subplot for length distributions
    ax_length = fig.add_subplot(gs[2, :])
    x = np.arange(len(all_lengths))
    ax_length.bar(x - width, alpha_freqs1_filtered, width=width, color='blue', alpha=0.6, label=name1 + ' - Alpha')
    ax_length.bar(x, alpha_freqs2_filtered, width=width, color='green', alpha=0.6, label=name2 + ' - Alpha')
    ax_length.bar(x + width, beta_freqs1_filtered, width=width, color='skyblue', alpha=0.6, label=name1 + ' - Beta')
    ax_length.bar(x + 2 * width, beta_freqs2_filtered, width=width, color='lightgreen', alpha=0.6, label=name2 + ' - Beta')
    ax_length.set_xticks(x)
    ax_length.set_xticklabels(all_lengths)
    ax_length.set_title('CDR3 Length Distribution')
    ax_length.legend()

    # One-hot frequency plots
    for i, col in enumerate(columns_to_compare):
        ax_onehot = fig.add_subplot(gs[1, i])
        gene_name = "TR" + col[::-1].upper()
        processed_column1 = dataset1[col].apply(lambda x: v_j.v_j_format(x, 2 if gene_name == "TRBJ" else 1, gene_name))
        freq1 = processed_column1.value_counts(normalize=True).sort_index()
        processed_column2 = dataset2[col].apply(lambda x: v_j.v_j_format(x, 2 if gene_name == "TRBJ" else 1, gene_name))
        freq2 = processed_column2.value_counts(normalize=True).sort_index()

        all_indices = freq1.index.union(freq2.index)
        freq1 = freq1.reindex(all_indices, fill_value=0)
        freq2 = freq2.reindex(all_indices, fill_value=0)
        freq1 = freq1[freq1 >= threshold]
        freq2 = freq2[freq2 >= threshold]

        positions1 = range(len(freq1))
        positions2 = [p + width for p in positions1]

        ax_onehot.bar(positions1, freq1.values, label=name1, alpha=0.7, width=width)
        ax_onehot.bar(positions2, freq2.values, label=name2, alpha=0.7, color='orange', width=width)
        ax_onehot.set_xticks([p + width / 2 for p in positions1])
        ax_onehot.set_xticklabels(all_indices)
        ax_onehot.set_title(f'Normalized {col.upper()} Comparison')
        ax_onehot.legend()

    plt.tight_layout()
    # plt.savefig("compare_datasets_combined.png")
    plt.show()


if __name__ == "__main__":

    datasets = [os.path.join("new_data", "sapir_data_positives.csv"),
                os.path.join("new_data", "Minervina_june_positives.csv"),
                os.path.join("new_data", "june_all_positives.csv")
                ]

    dataset_names = ("pMHC1", "pMHC2", "All T cells")

    dataset1, name1 = pd.read_csv(datasets[0]), dataset_names[0]
    dataset2, name2 = pd.read_csv(datasets[1]), dataset_names[1]
    dataset3, name3 = pd.read_csv(datasets[2]), dataset_names[2]

    # Columns to compare
    columns_to_compare = ['va', 'ja', 'vb', 'jb']
    combine(dataset1, dataset2, dataset3, columns_to_compare, name1, name2, name3, threshold=0.02)