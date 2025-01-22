import torch
import Models
import Loader
from torch.utils.data import DataLoader
import Trainer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import optuna
import pandas as pd
import os
import json
from matplotlib.font_manager import FontProperties

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
font = FontProperties()
font.set_family('serif')
font_size = 16
font.set_size(font_size)


def read_dictionaries_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['va_counts'], data['vb_counts'], data['ja_counts'], data['jb_counts']


def process_data_with_k_fold(file_path, batch_size, k=5):
    pairs = Loader.read_data(file_path)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)

    # Initialize the KFold class
    kf = KFold(n_splits=k, shuffle=True)
    # Lists to store the datasets and data loaders for each fold
    fold_data = []

    for train_index, test_index in kf.split(pairs):
        train_data = [pairs[i] for i in train_index]
        test_data = [pairs[i] for i in test_index]

        # Create the dataset and DataLoader for the training set
        train_dataset = Loader.ChainClassificationDataset(train_data, vj_data)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)

        # Create the dataset and DataLoader for the testing set
        test_dataset = Loader.ChainClassificationDataset(test_data, vj_data)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)

        # Store the datasets and data loaders for this fold
        fold_data.append(
            (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data, len_one_hot))

    return fold_data


def train_models(vocab_size, hyperparameters, batch_size, acid_2_ix, train_data_loader,
                 len_one_hot, model_of, test_data_loader=None):

    (embed_size, hidden_size, num_layers, latent_size, weight_decay_cl, weight_decay_encoder, dropout_prob, layer_norm,
     nhead, dim_feedforward) = hyperparameters
    # Initialize the models
    alpha_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                       num_layers).to(DEVICE)
    beta_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                      num_layers).to(DEVICE)
    alpha_decoder = Models.DecoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                       num_layers).to(DEVICE)
    beta_decoder = Models.DecoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                      num_layers).to(DEVICE)

    if model_of == "irec":
        model = Models.FFNN(latent_size * 2 * 2 + len_one_hot, dropout_prob, layer_norm).to(DEVICE)
    else:
        model = Models.FFNN(latent_size * 2 * 2 + len_one_hot + 1, dropout_prob, layer_norm).to(DEVICE)

    Trainer.merge_train_model(model, "LSTM", (alpha_encoder, alpha_decoder),
                              (beta_encoder, beta_decoder), train_data_loader, test_data_loader, DEVICE,
                              base_folder, batch_size, acid_2_ix, weight_decay_encoder, weight_decay_cl)
    return model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder)


def evaluate_model(model, encoders, data_loader):
    (encoder_alpha, encoder_beta) = encoders
    correct = 0
    total = 0
    model.eval()
    encoder_alpha.eval()
    encoder_beta.eval()
    all_labels = []
    all_predicted_probs = []
    all_alpha = []
    all_beta = []
    with torch.no_grad():
        for alpha, beta, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, labels, stage1 in data_loader:
            labels = labels.to(DEVICE)
            va_one_hot = va_one_hot.to(DEVICE)
            vb_one_hot = vb_one_hot.to(DEVICE)
            ja_one_hot = ja_one_hot.to(DEVICE)
            jb_one_hot = jb_one_hot.to(DEVICE)

            alpha = alpha.to(DEVICE)
            beta = beta.to(DEVICE)
            _, alpha_vector, _ = encoder_alpha(alpha)
            _, beta_vector, _ = encoder_beta(beta)
            if alpha_vector.shape[0] >= 2:
                # If the tensor has at least two layers, return the second one with an added dimension
                alpha_vector = alpha_vector[1].unsqueeze(0)
            if beta_vector.shape[0] >= 2:
                # If the tensor has at least two layers, return the second one with an added dimension
                beta_vector = beta_vector[1].unsqueeze(0)
            concatenated_a_b = torch.cat((alpha_vector, beta_vector), dim=2)
            if stage1 is not None and any(o is not None for o in stage1):
                stage1 = stage1.to(DEVICE)
                stage1 = stage1.view(1, 64, 1)
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0), stage1), dim=2)
            else:
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0)), dim=2)
            outputs = model(concatenated_inputs)
            predicted_probabilities = torch.sigmoid(outputs)
            predicted = (predicted_probabilities >= 0.5).squeeze().int()
            add = (predicted == labels).sum().item()
            total += len(predicted)
            correct += add
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            if predicted_probabilities.is_cuda:
                predicted_probabilities = predicted_probabilities.cpu().numpy()
            else:
                predicted_probabilities.numpy()

            # Append the true labels and predicted probabilities to the lists
            all_alpha.extend(alpha.cpu().numpy() if alpha.is_cuda else alpha.numpy())
            all_beta.extend(beta.cpu().numpy() if beta.is_cuda else beta.numpy())
            all_labels.extend(labels)
            all_predicted_probs.extend(predicted_probabilities.squeeze())
    auc = roc_auc_score(all_labels, all_predicted_probs)
    acc = correct / total
    print(f'AUC: {auc}')
    print(f'Accuracy: {acc}')

    # Identify top positive and bottom negative cases
    top_positive_indices = np.argsort([prob for prob, label in zip(all_predicted_probs, all_labels) if label == 1])[
                           -len(all_labels) // 20:]
    bottom_negative_indices = np.argsort([prob for prob, label in zip(all_predicted_probs, all_labels) if label == 0])[
                              :len(all_labels) // 20]

    # Gather the corresponding alpha and beta values based on sorted indices
    top_alpha_positives = [all_alpha[i] for i in top_positive_indices]
    top_beta_positives = [all_beta[i] for i in top_positive_indices]

    # Gather the corresponding alpha and beta values based on sorted indices
    bottom_alpha_negatives = [all_alpha[i] for i in bottom_negative_indices]
    bottom_beta_negatives = [all_beta[i] for i in bottom_negative_indices]

    return (auc, all_labels, all_predicted_probs, all_alpha, all_beta, top_alpha_positives, top_beta_positives,
            bottom_alpha_negatives, bottom_beta_negatives)


def plot_auc(all_labels, all_predicted_probs, all_labels2=None, all_predicted_probs2=None, ax=None, text=None):
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_predicted_probs)
    auc = roc_auc_score(all_labels, all_predicted_probs)

    # Plot ROC curve
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if all_labels2 is not None and all_predicted_probs2 is not None:
        fpr2, tpr2, _ = roc_curve(all_labels2, all_predicted_probs2)
        auc2 = roc_auc_score(all_labels2, all_predicted_probs2)
        ax.plot(fpr2, tpr2, color='green', lw=2, label=f'pMHC-I binding (AUC = {auc2:.2f})')
    ax.plot(fpr, tpr, color='orange', lw=2, label=f'All T cells (AUC = {auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties=font)
    ax.set_ylabel('True Positive Rate', fontproperties=font)
    ax.legend(loc="lower right", prop=font)
    ax.grid()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Round the tick values to one decimal place
    rounded_xticks = np.round(xticks, 1)
    rounded_yticks = np.round(yticks, 1)
    # Set the rounded tick labels with font properties
    ax.set_xticklabels(rounded_xticks, fontproperties=font, fontsize=font_size)
    ax.set_yticklabels(rounded_yticks, fontproperties=font, fontsize=font_size)

    if ax is None:
        if text is not None:
            plt.savefig(os.path.join(base_folder, "Roc curve " + text + ".png"))
        else:
            Trainer.save_plot_with_incremental_filename(base_folder, "Roc curve.png")


def plot_precision_recall(all_labels, all_predicted_probs, all_labels2=None, all_predicted_probs2=None, ax=None,
                          text=None):
    # Compute Precision-Recall curve and Average Precision (AP) for the first dataset
    precision, recall, _ = precision_recall_curve(all_labels, all_predicted_probs)
    avg_precision = average_precision_score(all_labels, all_predicted_probs)
    precision = np.insert(precision, 0, 0.0)
    recall = np.insert(recall, 0, 1.0)

    # Plot Precision-Recall curve for the first dataset
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if all_labels2 is not None and all_predicted_probs2 is not None:
        precision2, recall2, _ = precision_recall_curve(all_labels2, all_predicted_probs2)
        avg_precision2 = average_precision_score(all_labels2, all_predicted_probs2)
        precision2 = np.insert(precision2, 0, 0.0)
        recall2 = np.insert(recall2, 0, 1.0)

        ax.plot(recall2, precision2, color='green', lw=2, label=f'pMHC-I binding (AP = {avg_precision2:.2f})')
    ax.plot(recall, precision, color='orange', lw=2, label=f'All T cells (AP = {avg_precision:.2f})')

    ax.plot([1, 1], [0, 0], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('Recall', fontproperties=font)
    ax.set_ylabel('Precision', fontproperties=font)
    ax.legend(loc="lower left", prop=font)
    ax.grid()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Round the tick values to one decimal place
    rounded_xticks = np.round(xticks, 1)
    rounded_yticks = np.round(yticks, 1)
    # Set the rounded tick labels with font properties
    ax.set_xticklabels(rounded_xticks, fontproperties=font, fontsize=font_size)
    ax.set_yticklabels(rounded_yticks, fontproperties=font, fontsize=font_size)

    if ax is None:
        # Save the plot
        if text is not None:
            plt.savefig(os.path.join(base_folder, "Precision-Recall curve " + text + ".png"))
        else:
            plt.savefig(os.path.join(base_folder, "Precision-Recall curve.png"))


def plot_combined(all_labels, all_predicted_probs, all_labels2=None, all_predicted_probs2=None, text=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot ROC curve on the first subplot
    plot_auc(all_labels, all_predicted_probs, all_labels2, all_predicted_probs2, ax=axes[0])

    # Plot Precision-Recall curve on the second subplot
    plot_precision_recall(all_labels, all_predicted_probs, all_labels2, all_predicted_probs2, ax=axes[1])
    # Save the combined figure
    if text is not None:
        plt.savefig(os.path.join(base_folder, f"Combined Curve {text}.pdf"), format="pdf")
    else:
        plt.savefig(os.path.join(base_folder, "Combined Curve.pdf"), format="pdf")


def objective(trial, file_path):

    # Define the search space for hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 1024])
    weight_decay_cl = trial.suggest_float('weight_decay_cl', 1e-5, 1e-1, log=True)
    weight_decay_encoder = trial.suggest_float('weight_decay_encoder', 1e-5, 1e-1, log=True)
    latent_size = trial.suggest_categorical('latent_size', [256, 512, 1024])
    num_layers = trial.suggest_categorical('num_layers', [1, 2])
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)
    layer_norm = trial.suggest_categorical('layer_norm', [True, False])

    encoder_type = "LSTM"
    embed_size = trial.suggest_categorical('embed_size', [128, 256, 512])
    if encoder_type == "LSTM" or encoder_type == "BiLSTM":
        nhead = None
        dim_feedforward = None
    else:
        nhead = trial.suggest_categorical('nhead', [4, 8, 2])
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [2048, 1024])
        # Ensure that embed_size is divisible by nhead
        if embed_size % nhead != 0:
            raise ValueError("embed_size must be divisible by nhead")

    hyperparameter = (embed_size, hidden_size, num_layers, latent_size, weight_decay_cl,
                      weight_decay_encoder, dropout_prob, layer_norm, nhead, dim_feedforward)

    batch_size = 64

    # K-fold cross-validation loop
    k = 2
    fold_data = process_data_with_k_fold(file_path, batch_size, k)
    fold_aucs = []
    vocab_size = len(fold_data[0][0].vocab)

    for fold in range(k):
        print(f"Training on fold {fold + 1}/{k}")
        (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data,
         len_one_hot) = fold_data[fold]
        # Train models with current hyperparameters
        model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
            vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
            len_one_hot, test_data_loader
        )
        # Evaluate model and calculate AUC
        auc_test, _, _, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), test_data_loader
        )
        auc_train, _, _, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), train_data_loader
        )
        print("auc_train", auc_train)
        fold_aucs.append(auc_test)
        with open(os.path.join(base_folder, "auc_results.txt"), 'a') as f:
            f.write(f"Train AUC: {auc_train}, Test AUC: {auc_test}\n")

    return np.mean(fold_aucs)


def save_best_params(study, trial, file_path='best_hyperparameters_vdjdb_19.1.json'):
    """Save the best hyperparameters to a file if the current trial's score is better."""
    current_best_value = study.best_value
    print("in_saving")
    try:
        # Check the current saved best score
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
            saved_best_value = saved_data.get("AUC", float('-inf'))
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is invalid, treat as no prior best
        saved_best_value = float('-inf')

    # Save only if the current trial's score is better
    if current_best_value > saved_best_value:
        best_params = study.best_params
        best_params["AUC"] = current_best_value  # Add the best score
        with open(file_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Updated best hyperparameters saved to {file_path}")


def run_optuna(input_file, hyperparameters_file_path):
    study = optuna.create_study(direction='maximize')

    def wrapped_objective(trial):
        return objective(trial, input_file)

    study.optimize(wrapped_objective, n_trials=100,
                   callbacks=[lambda study, trial:save_best_params(study, trial, hyperparameters_file_path)])

    # Print best hyperparameters and their result
    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Save best hyperparameters to a file
    with open(hyperparameters_file_path) as f:
        json.dump(study.best_params, f)
    print("Best hyperparameters saved")


def save_20_to_file(ix_to_acid, all_alpha, all_beta, top_alpha_positives, top_beta_positives, bottom_alpha_negatives,
                    bottom_beta_negatives, text):
    top_alpha_positives_seq = arrays_to_sequences(top_alpha_positives, ix_to_acid)
    top_beta_positives_seq = arrays_to_sequences(top_beta_positives, ix_to_acid)
    bottom_alpha_negatives_seq = arrays_to_sequences(bottom_alpha_negatives, ix_to_acid)
    bottom_beta_negatives_seq = arrays_to_sequences(bottom_beta_negatives, ix_to_acid)
    all_alpha_seq = arrays_to_sequences(all_alpha, ix_to_acid)
    all_beta_seq = arrays_to_sequences(all_beta, ix_to_acid)

    with open(os.path.join(base_folder, f'alpha_beta_vj_{text}.txt'), 'w') as f:
        f.write(f"alpha\n")
        f.write(f"{top_alpha_positives_seq}\n")
        f.write(f"{bottom_alpha_negatives_seq}\n")
        f.write(f"{all_alpha_seq}\n")
        f.write("beta\n")
        f.write(f"{top_beta_positives_seq}\n")
        f.write(f"{bottom_beta_negatives_seq}\n")
        f.write(f"{all_beta_seq}\n")


def find_best_model(model_of, file_path, best_params_file):
    # Load the best hyperparameters from the JSON file
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params.get('embed_size', None)
    hidden_size = best_params.get('hidden_size', None)
    num_layers = best_params.get('num_layers', None)
    latent_size = best_params.get('latent_size', None)
    weight_decay_cl = best_params['weight_decay_cl']
    weight_decay_encoder = best_params.get('weight_decay_encoder', None)
    dropout_prob = best_params['dropout_prob']
    layer_norm = best_params['layer_norm']

    # Extract parameters specific to non-LSTM models if available
    nhead = best_params.get('nhead', None)
    dim_feedforward = best_params.get('dim_feedforward', None)

    # Prepare the hyperparameter tuple
    hyperparameter = (embed_size, hidden_size, num_layers, latent_size, weight_decay_cl, weight_decay_encoder,
                      dropout_prob, layer_norm, nhead, dim_feedforward)

    # Other fixed hyperparameters
    batch_size = 64
    k = 5

    # Process data
    fold_data = process_data_with_k_fold(file_path, batch_size, k)
    fold_aucs_test = []
    fold_aucs_train = []
    vocab_size = len(fold_data[0][0].vocab)

    best_auc = 0
    best_model_state_dict = None
    best_alpha_encoder = None
    best_beta_encoder = None
    print(base_folder)
    best_model_path = os.path.join(base_folder, 'best_model.pth')
    best_alpha_encoder_path = os.path.join(base_folder,
                                           'best_alpha_encoder.pth')
    best_beta_encoder_path = os.path.join(base_folder,
                                          'best_beta_encoder.pth')

    for fold in range(k):
        print(f"Training on fold {fold + 1}/{k}")
        (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data,
         len_one_hot) = fold_data[fold]

        # Train models with current hyperparameters
        model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
            vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
            len_one_hot, model_of, test_data_loader
        )

        # Evaluate model and calculate AUC
        auc_train, all_labels_train, all_predicted_probs_train, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), train_data_loader
        )
        plot_auc(all_labels_train, all_predicted_probs_train)
        auc_test, all_labels_test, all_predicted_probs_test, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), test_data_loader
        )
        fold_aucs_test.append(auc_test)
        fold_aucs_train.append(auc_train)

        # Save the model if the test AUC is the best so far
        if auc_test > best_auc:
            best_auc = auc_test
            best_model_state_dict = model.state_dict()
            best_alpha_encoder = alpha_encoder.state_dict()
            best_beta_encoder = beta_encoder.state_dict()
            print(f'New best AUC: {best_auc} found on fold {fold + 1}')

    # Save only the best model after all folds have been processed
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, best_model_path)
        torch.save(best_alpha_encoder, best_alpha_encoder_path)
        torch.save(best_beta_encoder, best_beta_encoder_path)
        print(f'Best model saved to {best_model_path} with AUC: {best_auc}')
        print(f'Best alpha encoder saved to {best_alpha_encoder_path}')
        print(f'Best beta encoder saved to {best_beta_encoder_path}')
    # Return the mean AUC across the folds
    return model, alpha_encoder, beta_encoder, test_data_loader


def arrays_to_sequences(arrays, ix_2_acid):
    sequences = []
    for array in arrays:
        sequence = ''.join(
            [ix_2_acid[ix] for ix in array if ix_2_acid[ix] not in ["<PAD>", "<EOS>"]])  # Skip PAD and EOS tokens
        sequences.append(sequence)
    return sequences


def load_models(model_save_paths, len_one_hot, vocab_size, best_params_file, model_of):

    # Load the best hyperparameters from the JSON file
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params['embed_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    latent_size = best_params['latent_size']
    dropout_prob = best_params['dropout_prob']
    layer_norm = best_params['layer_norm']
    # Load the main model
    if model_of == "vdjdb":
        main_model = Models.FFNN(latent_size * 2 * 2 + len_one_hot + 1, dropout_prob, layer_norm).to(DEVICE)
    else:
        main_model = Models.FFNN(latent_size * 2 * 2 + len_one_hot, dropout_prob, layer_norm).to(DEVICE)

    main_model.load_state_dict(torch.load(model_save_paths['model']))
    main_model.eval()  # Set the model to evaluation mode

    # Load the alpha encoder
    alpha_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                       num_layers).to(DEVICE)
    alpha_encoder.load_state_dict(torch.load(model_save_paths['alpha_encoder']))
    alpha_encoder.eval()

    # Load the beta encoder
    beta_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                      num_layers).to(DEVICE)
    beta_encoder.load_state_dict(torch.load(model_save_paths['beta_encoder']))
    beta_encoder.eval()

    return main_model, alpha_encoder, beta_encoder


def load_data(file_path):
    pairs = Loader.read_data(file_path)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    batch_size = 64
    # return train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data, len_one_hot
    full_dataset = Loader.ChainClassificationDataset(pairs, vj_data)
    full_data_loader = DataLoader(full_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)
    return full_dataset, full_data_loader, pairs, len_one_hot


def load_and_evaluate_model(file_path, param_file, model_of='vdjdb'):
    if model_of == "vdjdb":
        model_save_paths = {
            'model': "5fold_vdjdb/best_model.pth",
            'alpha_encoder': "5fold_vdjdb/best_alpha_encoder.pth",
            'beta_encoder': "5fold_vdjdb/best_beta_encoder.pth"
        }
    else:
        model_save_paths = {
            'model': "5fold_irec/best_model.pth",
            'alpha_encoder': "5fold_irec/best_alpha_encoder.pth",
            'beta_encoder': "5fold_irec/best_beta_encoder.pth"
        }
    # Load your dataset
    full_dataset, full_data_loader, full_data, len_one_hot = load_data(file_path)
    vocab_size = len(full_dataset.vocab)
    # Load the best models
    model, alpha_encoder, beta_encoder = load_models(model_save_paths, len_one_hot, vocab_size, param_file, model_of)
    # Evaluate the model on the test set
    (auc_test, all_labels_test, all_predicted_probs_test, all_alpha_te, all_beta_te, top_alpha_positives_te,
     top_beta_positives_te, bottom_alpha_negatives_te, bottom_beta_negatives_te) = evaluate_model(
        model, (alpha_encoder, beta_encoder), full_data_loader
    )

    # Plot AUC
    plot_auc(all_labels_test, all_predicted_probs_test, text="test_evaluation_auc")
    save_20_to_file(full_dataset.ix_2_acid, all_alpha_te, all_beta_te, top_alpha_positives_te, top_beta_positives_te,
                    bottom_alpha_negatives_te, bottom_beta_negatives_te, f"loaded_model_{model_of}")
    print(f'Test AUC: {auc_test}')
    return all_labels_test, all_predicted_probs_test


def ireceptor_model_on_dataset(param_file, file_to_update, new_file_name):
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    data = Loader.read_data(file_to_update)

    # Create the dataset and DataLoader for the training set
    dataset = Loader.ChainClassificationDataset(data, vj_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, collate_fn=Loader.collate_fn)
    vocab_size = len(dataset.vocab)

    model_save_paths = {'model': "models/model_irec.pth", 'alpha_encoder': "models/alpha_encoder_irec.pth",
                        'beta_encoder': "models/beta_encoder_irec.pth"}
    main_model, alpha_encoder, beta_encoder = load_models(model_save_paths, len_one_hot, vocab_size, param_file,
                                                          "ireceptor")
    (auc, all_labels, all_predicted_probs, all_alpha, all_beta, top_alpha_positives, top_beta_positives,
     bottom_alpha_negatives, bottom_beta_negatives) = evaluate_model(main_model,
                                                                     (alpha_encoder, beta_encoder), data_loader)
    # Load the file into a DataFrame
    df = pd.read_csv(file_to_update)
    # Ensure the length of `all_predicted_probs` matches the number of rows in the file
    if len(all_predicted_probs) != len(df):
        raise ValueError("The length of `all_predicted_probs` does not match the number of rows in the file.")
    # Add the new column
    df["output"] = all_predicted_probs
    # Save the updated DataFrame back to the file
    df.to_csv(new_file_name, index=False)


def run_on_all_save_models(param_file, dataset_file, paths_for_best_models):
    # Load the best hyperparameters from the JSON file
    with open(param_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params.get('embed_size', None)
    hidden_size = best_params.get('hidden_size', None)
    num_layers = best_params.get('num_layers', None)
    latent_size = best_params.get('latent_size', None)
    weight_decay_cl = best_params['weight_decay_cl']
    weight_decay_encoder = best_params.get('weight_decay_encoder', None)
    dropout_prob = best_params['dropout_prob']
    layer_norm = best_params['layer_norm']

    # Extract parameters specific to non-LSTM models if available
    nhead = best_params.get('nhead', None)
    dim_feedforward = best_params.get('dim_feedforward', None)

    # Prepare the hyperparameter tuple
    hyperparameter = (embed_size, hidden_size, num_layers, latent_size,
                      weight_decay_cl, weight_decay_encoder, dropout_prob,
                      layer_norm, nhead, dim_feedforward)

    batch_size = 64
    pairs = Loader.read_data(dataset_file)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    train_data = pairs

    # Create the dataset and DataLoader for the training set
    train_dataset = Loader.ChainClassificationDataset(train_data, vj_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)
    vocab_size = len(train_dataset.vocab)

    best_model_path = os.path.join(base_folder, paths_for_best_models['model'])
    best_alpha_encoder_path = os.path.join(base_folder, paths_for_best_models['alpha_encoder'])
    best_beta_encoder_path = os.path.join(base_folder, paths_for_best_models['beta_encoder'])

    # Train models with current hyperparameters
    model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
        vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
        len_one_hot, "vdjdb"
    )

    best_model_state_dict = model.state_dict()
    best_alpha_encoder = alpha_encoder.state_dict()
    best_beta_encoder = beta_encoder.state_dict()

    # Save only the best model after all folds have been processed
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, best_model_path)
        torch.save(best_alpha_encoder, best_alpha_encoder_path)
        torch.save(best_beta_encoder, best_beta_encoder_path)
        print(f'Best model saved to {best_model_path}')
        print(f'Best alpha encoder saved to {best_alpha_encoder_path}')
        print(f'Best beta encoder saved to {best_beta_encoder_path}')


def run_on_test_set():
    param_file_vdjdb = "best_hyperparameters_vdjdb.json"
    param_file_ireceptor = "best_hyperparameters_ireceptor.json"
    all_labels_test_vdjdb, all_predicted_probs_test_vdjdb = load_and_evaluate_model(
        'vdjdb_validation.csv', param_file_vdjdb, "vdjdb")
    # # param_file = "best_hyperparameters_stage2.json"
    all_labels_test_irec, all_predicted_probs_test_irec = load_and_evaluate_model(
        'ireceptor_validation.csv', param_file_ireceptor, "ireceptor")

    plot_combined(all_labels_test_irec, all_predicted_probs_test_irec, all_labels_test_vdjdb,
                  all_predicted_probs_test_vdjdb)


if __name__ == "__main__":

    base_folder = "/home/eng/devorasimi/Alpha_Beta_pairing/correct_models"
    os.makedirs(base_folder, exist_ok=True)





