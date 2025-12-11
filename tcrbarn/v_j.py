import pandas as pd
from collections import Counter
import json

def v_j_format(gene, levels, gene_name):
    """
    Format V and J gene names according to specific rules.
    Args:
        gene (str): The gene name to format.
        levels (int): The number of levels to include in the formatted name.
        gene_name (str): The base name of the gene (e.g., "TRBV", "TRBJ").
    Returns:
        str: The formatted gene name.
    """
    # Handle special cases where the gene is missing or invalid
    if pd.isna(gene) or gene in ["~", "nan", "", "NA"]:
        return "~"

    if gene_name == "TRBV":
        if "TCRBV" in gene:
            gene = gene.replace("TCRBV", "TRBV")
        if "TRBV" not in gene:
            return "~"
    if gene_name == "TRBJ":
        if not gene.startswith("TRBJ"):
            return "~"
    if "TRDAV" in gene:
        gene = gene.replace("TRDAV", "TRAV")
    if "TRA21" in gene:
        gene = gene.replace("TRA21", "TRAV21")
    if "TRAJF" in gene:
        return "~"
    if "TRDJ" in gene:
        return "~"

    gene = gene.split("/")[0]
    gene_list = gene.replace(" ", "").replace("*", "-").replace(":", "-").split("-")

    # add zero to numbers with one digit
    gene_value = gene_list[0].replace(gene_name, "")
    if len(gene_value) == 1:
        gene_value = "0" + gene_value
    gene_list[0] = gene_name + gene_value

    if levels == 1:
        return gene_list[0]
    if levels == 2:
        if len(gene_list) == 1:
            return gene_list[0]
        return gene_list[0] + "-" + str(int(gene_list[1]))


def get_vj_dict(file_paths, filtering_number, filename, va_c='va', vb_c='vb', ja_c='ja', jb_c='jb'):
    # Initialize counters
    va_counter = Counter()
    vb_counter = Counter()
    ja_counter = Counter()
    jb_counter = Counter()

    for file_path in file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Extract chains and label from each row
        for _, row in df.iterrows():
            va = v_j_format(row[va_c], 1, "TRAV")
            vb = v_j_format(row[vb_c], 1, "TRBV")
            ja = v_j_format(row[ja_c], 1, "TRAJ")
            jb = v_j_format(row[jb_c], 2, "TRBJ")

            # Update counters
            va_counter[va] += 1
            vb_counter[vb] += 1
            ja_counter[ja] += 1
            jb_counter[jb] += 1

    # Create dictionaries from the counters
    va_d = dict(va_counter)
    vb_d = dict(vb_counter)
    ja_d = dict(ja_counter)
    jb_d = dict(jb_counter)
    va_d.pop("~", None)
    vb_d.pop("~", None)
    ja_d.pop("~", None)
    jb_d.pop("~", None)

    # Filter the dictionaries
    va_d_filtered = {key: count for key, count in va_d.items() if count > filtering_number}
    vb_d_filtered = {key: count for key, count in vb_d.items() if count > filtering_number}
    ja_d_filtered = {key: count for key, count in ja_d.items() if count > filtering_number}
    jb_d_filtered = {key: count for key, count in jb_d.items() if count > filtering_number}

    va_2_ix = {key: i for i, key in enumerate(va_d_filtered)}
    vb_2_ix = {key: i for i, key in enumerate(vb_d_filtered)}
    ja_2_ix = {key: i for i, key in enumerate(ja_d_filtered)}
    jb_2_ix = {key: i for i, key in enumerate(jb_d_filtered)}

    va_2_ix["<UNK>"] = len(va_2_ix)
    vb_2_ix["<UNK>"] = len(vb_2_ix)
    ja_2_ix["<UNK>"] = len(ja_2_ix)
    jb_2_ix["<UNK>"] = len(jb_2_ix)

    # Save the filtered dictionaries to a JSON file
    with open(filename, 'w') as f:
        json.dump({
            'va_counts': va_2_ix,
            'vb_counts': vb_2_ix,
            'ja_counts': ja_2_ix,
            'jb_counts': jb_2_ix
        }, f, indent=4)

if __name__ == "__main__":
    get_vj_dict(["data/sapir_data_positives_vj.csv"], 20, "filtered_counters_20.json")