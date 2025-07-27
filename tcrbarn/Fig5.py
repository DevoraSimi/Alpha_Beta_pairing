import main
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font_size = 16
font.set_size(font_size)

def run_on_test_set(data_lists):
    """
    Run the best models on the test set and plot the combined results.
    """
    param_file = "hyperparameters.json"
    label_prob_pairs = []
    for dataset, mode, name in data_lists:
        if mode == "pMHC":
            save_paths = {
                'model': "models/combined_5fold/best_model.pth",
                'alpha_encoder': "models/combined_5fold/best_alpha_encoder.pth",
                'beta_encoder': "models/combined_5fold/best_beta_encoder.pth"
            }
        elif mode == "pMHC1":
            save_paths = {
                'model': "models/pMHC1_5fold/best_model.pth",
                'alpha_encoder': "models/pMHC1_5fold/best_alpha_encoder.pth",
                'beta_encoder': "models/pMHC1_5fold/best_beta_encoder.pth"
            }
        elif mode == "pMHC2":
            save_paths = {
                'model': "models/pMHC2_5fold/best_model.pth",
                'alpha_encoder': "models/pMHC2_5fold/best_alpha_encoder.pth",
                'beta_encoder': "models/pMHC2_5fold/best_beta_encoder.pth"
            }
        elif mode == "steps":
            save_paths = {
                'model': "models/train_in_steps/best_model.pth",
                'alpha_encoder': "models/train_in_steps/best_alpha_encoder.pth",
                'beta_encoder': "models/train_in_steps/best_beta_encoder.pth"
            }
        else:
            save_paths = {
                'model': "models/non-paired/best_model.pth",
                'alpha_encoder': "models/non-paired/best_alpha_encoder.pth",
                'beta_encoder': "models/non-paired/best_beta_encoder.pth"
            }
        labels, predicted_probs = main.load_and_evaluate_model(
            dataset, param_file, save_paths)
        # mean_auc, std_auc = bootstrap_auc(labels_minervina_m_minervina_d, predicted_minervina_m_minervina_d)
        # mean_auc, std_auc = bootstrap_auc(labels_vdjdb_m_vdjdb_d, predicted_vdjdb_m_vdjdb_d)
        # mean_auc, std_auc = bootstrap_auc(labels_combined_m_combined_d, predicted_combined_m_combined_d)
        # print(f"Bootstrap AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        if name is not None:
            label_prob_pairs.append((labels, predicted_probs, name))
    return label_prob_pairs

def results_on_all():
    main.filter_test('data/pMHC1_with_negatives_train.csv', 'data/pMHC1_with_negatives_test.csv', 'data/pMHC1_with_negatives_test_filtered.csv')
    d_pMHC1_m_pMHC1 = ('data/pMHC1_with_negatives_test_filtered.csv', 'pMHC1', "M pMHC1 - D pMHC1")
    d_pMHC2_m_pMHC1 = ('data/pMHC2_with_negatives.csv', 'pMHC1', "M pMHC1 - D pMHC2")
    d_all_tcells_m_pMHC1 = ('data/All T cells with_negatives.csv', 'pMHC1', None)
    d_pMHC1_m_pMHC2 = ('data/pMHC1_with_negatives.csv', 'pMHC2', "M pMHC2 - D pMHC1")
    d_pMHC2_m_pMHC2 = ('data/pMHC2_with_negatives_test.csv', 'pMHC2', "M pMHC2 - D pMHC2")
    d_all_tcells_m_pMHC2 = ('data/All T cells with_negatives.csv', 'pMHC2', None)
    d_pMHC1_m_combined = ('data/pMHC1_with_negatives_test_filtered.csv', 'pMHC', None)
    d_pMHC2_m_combined = ('data/pMHC2_with_negatives_test.csv', 'pMHC', None)
    d_all_tcells_m_combined = ('data/All T cells with_negatives.csv', 'pMHC', None)
    d_all_tcells_m_all_tcells = ('data/All T cells with_negatives.csv', 'all_T_cells', None)
    read_merge_shuffle_save('data/pMHC2_with_negatives_test.csv', 'data/pMHC1_with_negatives_test_filtered.csv', 'data/merged_test.csv')

    d_combined_m_combined = ('data/merged_test.csv', 'pMHC', "M combined - D combined")

    data_lists = [
        d_pMHC1_m_pMHC1,
        d_pMHC2_m_pMHC1,
        d_pMHC2_m_pMHC2,
        d_pMHC1_m_pMHC2,
        d_combined_m_combined,
        d_pMHC1_m_combined,
        d_pMHC2_m_combined,
        d_all_tcells_m_combined,
        d_all_tcells_m_pMHC1,
        d_all_tcells_m_pMHC2,
        d_all_tcells_m_all_tcells
    ]
    return data_lists

def plot_auc_multiple(label_prob_pairs, ax=None):
    """
    Plot ROC curve for multiple sets of labels/predictions.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for (labels, probs, label_text), color in zip(label_prob_pairs, ['orange', 'green', 'blue', 'red', 'skyblue', 'brown']):
        if labels is not None:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)
            ax.plot(fpr, tpr, lw=2, label=f'{label_text} (AUC = {auc:.2f})', color=color)

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties=font, fontsize=font_size + 5)
    ax.set_ylabel('True Positive Rate', fontproperties=font, fontsize=font_size + 5)
    ax.legend(loc="lower right", prop=font)
    ax.grid()

    ax.set_xticklabels(np.round(ax.get_xticks(), 1), fontproperties=font, fontsize=font_size + 5)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), fontproperties=font, fontsize=font_size + 5)

def plot_precision_recall_multiple(label_prob_pairs, ax=None):
    """
    Plot Precision-Recall curves for multiple sets of labels/predictions.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for (labels, probs, label_text), color in zip(label_prob_pairs, ['orange', 'green', 'blue', 'red', 'skyblue', 'brown']):
        if labels is not None:
            precision, recall, _ = precision_recall_curve(labels, probs)
            avg_precision = average_precision_score(labels, probs)

            precision = np.insert(precision, 0, 0.0)
            recall = np.insert(recall, 0, 1.0)

            ax.plot(recall, precision, lw=2, label=f'{label_text} (AP = {avg_precision:.2f})', color=color)

    ax.set_xlabel('Recall', fontproperties=font, fontsize=font_size + 5)
    ax.set_ylabel('Precision', fontproperties=font, fontsize=font_size + 5)
    ax.legend(loc="upper right", prop=font)
    ax.grid()

    ax.set_xticklabels(np.round(ax.get_xticks(), 1), fontproperties=font, fontsize=font_size + 5)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), fontproperties=font, fontsize=font_size + 5)


def plot_combined(label_prob_pairs, axes):
    """
    Plot combined ROC and Precision-Recall curves for multiple datasets and models.
    Args:
        label_prob_pairs (list of tuples): Each tuple is (true_labels, predicted_probs, label_text)
        axes: Axes to plot on.
    """

    plot_auc_multiple(label_prob_pairs, ax=axes[0])
    plot_precision_recall_multiple(label_prob_pairs, ax=axes[1])


def bootstrap_auc(y_true, y_scores, n_bootstrap=1000, random_state=None):
    rng = np.random.RandomState(random_state)
    aucs = []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip iteration if only one class is present

        auc = roc_auc_score(y_true[indices], y_scores[indices])
        aucs.append(auc)

    print(np.mean(aucs), np.std(aucs), "boots")

    return np.mean(aucs), np.std(aucs)


def read_merge_shuffle_save(csv1, csv2, output_path, sep=","):
    # Read both files with header
    df1 = pd.read_csv(csv1, sep=sep)
    df2 = pd.read_csv(csv2, sep=sep)
    # Concatenate (excluding header duplication)
    merged = pd.concat([df1, df2], ignore_index=True)
    # Shuffle rows
    shuffled = merged.sample(frac=1).reset_index(drop=True)
    # Save with header
    shuffled.to_csv(output_path, index=False, header=True, sep=sep)
    return shuffled


def pu_negatives(input_file, output_file):
    # positive unknown
    df = pd.read_csv(input_file)
    alpha_columns = ['tcra', 'va', 'ja']
    beta_columns = ['tcrb', 'vb', 'jb']
    df = df[alpha_columns + beta_columns]
    # Drop rows with "nan" in any of the selected columns
    df = df.dropna()
    # Add sign column
    df_true = df.assign(sign=1)
    # Initialize a list to store the shuffled DataFrames
    shuffled_dfs = []
    # Split into two groups: A (first 3 columns) and B (last 3 columns)
    a_columns = df_true[alpha_columns].values
    b_columns = df_true[beta_columns].values
    # Perform the shuffle and append process 5 times
    for _ in range(5):
        # Shuffle the rows while keeping the columns in each group together
        shuffled_indices_a = np.random.permutation(df_true.index)
        shuffled_a = a_columns[shuffled_indices_a]
        shuffled_indices_b = np.random.permutation(df_true.index)
        shuffled_b = b_columns[shuffled_indices_b]
        # Combine the shuffled groups back into a DataFrame
        shuffled_df = pd.DataFrame(np.hstack((shuffled_a, shuffled_b)), columns=alpha_columns+beta_columns)
        # Set 'sign' to 0 for the shuffled DataFrame
        shuffled_df['sign'] = 0
        # Add the shuffled DataFrame to the list
        shuffled_dfs.append(shuffled_df)
    # Concatenate the original DataFrame with the 5 shuffled copies
    result_df = pd.concat([df_true] + shuffled_dfs, ignore_index=True)
    # Shuffle the entire DataFrame (shuffle rows together)
    result_df = result_df.sample(frac=1).reset_index(drop=True)
    # Save the final DataFrame to a CSV file
    result_df.to_csv(output_file, index=False)


def auc_per_peptide(ax, full_data, test_set):

    # Load data
    full_df = pd.read_csv(full_data)  # Contains tcra, va, ja, tcrb, vb, jb, pep, hla_seq
    test_df = pd.read_csv(test_set)  # Contains tcra, va, ja, tcrb, vb, jb
    # Merge to get peptides in test set
    merged = test_df.merge(full_df, on=['tcra', 'va', 'ja', 'tcrb', 'vb', 'jb'], how='left')
    # Count peptides
    pep_counts = merged['pep'].value_counts()
    valid_peptides = pep_counts[pep_counts > 20].index

    param_file_ireceptor = "hyperparameters.json"
    save_paths = {
        'model': "models/combined_5fold/best_model.pth",
        'alpha_encoder': "models/combined_5fold/best_alpha_encoder.pth",
        'beta_encoder': "models/combined_5fold/best_beta_encoder.pth"
    }
    # Filepath to use temporarily
    temp_file = "temp_peptide_test.csv"

    # Dictionary to store AUCs
    aucs = {}
    samples = {}

    for pep in valid_peptides:
        print(pep)
        pep_rows = merged[merged['pep'] == pep][['tcra', 'va', 'ja', 'tcrb', 'vb', 'jb']]
        num_samples = len(pep_rows)
        if num_samples == 0:
            continue
        # Save to temporary file
        pep_rows.to_csv(temp_file, index=False)
        # Create augmented PU dataset
        pu_negatives(temp_file, temp_file)
        # Evaluate model
        labels, predicted = main.load_and_evaluate_model(
            temp_file, param_file_ireceptor, "ireceptor", save_paths
        )

        # Skip peptide if not enough class variation
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            print(f"Skipping peptide {pep}: only one class ({unique_labels}) in y_true")
            continue

        # Compute AUC
        auc = roc_auc_score(labels, predicted)
        aucs[pep] = auc
        samples[pep] = num_samples

    # Print results
    for pep, auc in aucs.items():
        print(f"{pep}: AUC = {auc:.4f}, num samples = {samples[pep]}")

    # Optional: remove the temporary file afterward
    os.remove(temp_file)
    # Prepare labels with sample count
    labels = [f"{pep} (n={samples[pep]})" for pep in aucs.keys()]
    values = [aucs[pep] for pep in aucs.keys()]

    # Plot
    # plt.figure(figsize=(8, 5))
    bars = ax.barh(labels, values, color='skyblue')
    ax.set_xlabel("AUC", fontproperties=font, fontsize=font_size + 5)
    ax.set_xlim(0.5, 1.0)

    # Add AUC values on the bars
    for bar, auc in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{auc:.2f}", va='center', fontproperties=font, fontsize=font_size + 5)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontproperties=font, fontsize=font_size + 5)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font, fontsize=font_size + 5)



def train_in_steps(train_file, json_param_file, output_file, step=5000, start=1000):
    # Load full train data
    full_train_df = pd.read_csv(train_file)
    max_rows = len(full_train_df)

    # Prepare output file
    results = []

    # Temp file path
    temp_train_file = "temp_train_subset.csv"

    for size in range(start, max_rows + 1, step):
        print(f"Training with {size} samples...")

        # Take subset of training data
        subset = full_train_df.iloc[:size]
        subset.to_csv(temp_train_file, index=False)

        # Train model
        main.find_best_model(temp_train_file, json_param_file)

        # Evaluate
        labels, predicted_probes, _ = run_on_test_set(('data/merged_test.csv', "steps", "steps"))
        auc = roc_auc_score(labels, predicted_probes)

        print(f"AUC with {size} samples: {auc:.4f}")
        results.append({'num_samples': size, 'auc': auc})

    # Save results
    pd.DataFrame(results).to_csv(output_file, index=False)

    # Clean up
    os.remove(temp_train_file)


def train_steps_plot(ax, filename):
    # Read data from CSV file
    df = pd.read_csv(filename)

    # Scatter plot
    ax.scatter(df['num_samples'], df['auc'], color='blue', label='Data Points')

    # Linear regression
    slope, intercept = np.polyfit(df['num_samples'], df['auc'], 1)
    regression_line = slope * df['num_samples'] + intercept
    ax.plot(df['num_samples'], regression_line, color='red', label='Regression Line')

    ax.set_xlabel('Number of samples in training set', fontproperties=font, fontsize=font_size + 5)
    ax.set_ylabel('AUC', fontproperties=font, fontsize=font_size + 5)
    ax.grid(True)
    ticks = ax.get_xticks()
    ticks_int = [int(tick) if tick.is_integer() else tick for tick in ticks]
    ax.set_xticklabels(ticks_int, fontproperties=font, fontsize=font_size + 5)
    ax.set_yticklabels(np.round(ax.get_yticks(), 3), fontproperties=font, fontsize=font_size + 5)


def plot_all_2x2(label_prob_pairs, full_data, test_data, steps_file, output_file=None):
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    axs = axs.flatten()
    plot_combined(label_prob_pairs, axs[:2])
    auc_per_peptide(axs[2], full_data, test_data)
    train_steps_plot(axs[3], steps_file)

    plt.tight_layout()
    pos = axs[2].get_position()
    axs[2].set_position([
        0.2,
        pos.y0,
        pos.width * 0.75,  # make plot itself narrower
        pos.height
    ])
    pos = axs[0].get_position()
    print(pos)
    axs[0].set_position([
        0.1,
        pos.y0,
        pos.width,
        pos.height
    ])
    if output_file:
        plt.savefig(output_file + ".pdf", format='pdf')
    plt.show()


if __name__ == "__main__":
    read_merge_shuffle_save('data/pMHC2_with_negatives_train.csv', 'data/pMHC1_with_negatives_train.csv', 'data/merged_train.csv')
    train_in_steps('data/merged_train.csv', "hyperparameters.json",
                   "train_in_steps/auc_steps.txt")
    run_on = results_on_all()
    label_probs = run_on_test_set(run_on)
    plot_all_2x2(label_probs, 'data/pMHC1_with_pep_mhc.csv', 'data/pMHC1_with_negatives_test_filtered.csv',
                 "train_in_steps/auc_steps.txt", output_file="all_auc_results")
