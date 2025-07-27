import csv

import pandas as pd
import numpy as np
import scipy.io


def get_positives_format_irec(input_file, output_file):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Filter rows where locus is TRA or TRB
    df_filtered = df[df['locus'].isin(['TRA', 'TRB'])]

    # Group by 'cell_id'
    new_data = []
    for cell_id, group in df_filtered.groupby('cell_id'):
        tra = group[group['locus'] == 'TRA'].iloc[:1]  # Get the first TRA
        trb = group[group['locus'] == 'TRB'].iloc[:1]  # Get the first TRB

        if not tra.empty and not trb.empty:
            row1, row2 = tra.iloc[0], trb.iloc[0]

            # Ensure required columns are non-empty
            if all(row1[col] != '' for col in ['junction_aa', 'v_call', 'j_call']) and \
                    all(row2[col] != '' for col in ['junction_aa', 'v_call', 'j_call']):
                new_data.append({
                    'tcra': row1['junction_aa'],
                    'va': row1['v_call'],
                    'ja': row1['j_call'],
                    'tcrb': row2['junction_aa'],
                    'vb': row2['v_call'],
                    'jb': row2['j_call']
                })

    # Create a new DataFrame from collected data
    new_df = pd.DataFrame(new_data)
    print(len(df)
          )
    print(len(new_df))
    # return new_df

    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_file, index=False)


def tsv_to_csv(tsv_file_path, csv_file_path):
    # Save the TSV data to a file (tsv file)
    df = pd.read_csv(tsv_file_path, sep='\t', on_bad_lines='skip', engine='python')
    # Convert and save the file as CSV
    df.to_csv(csv_file_path, index=False)


def get_positives_format(input_file, output_file):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Group the rows by barcode and process each group
    new_data = []

    # Iterate over each group of rows with the same barcode
    for barcode, group in df.groupby('barcode'):
        # Filter out rows where raw_consensus_id is 'None'
        valid_rows = group[
            group['raw_consensus_id'].notna() &
            (group['raw_consensus_id'] != "None") &
            (group['raw_consensus_id'] != "") &
            (group['productive'] == True)
            ]
        # Check if there are exactly one TRA row and one TRB row
        if len(valid_rows) >= 2:
            # Separate TRA and TRB rows
            tra_row = valid_rows[valid_rows['chain'] == 'TRA']
            trb_row = valid_rows[valid_rows['chain'] == 'TRB']

            # Ensure both TRA and TRB rows exist
            if not tra_row.empty and not trb_row.empty:
                # Extract values from the TRA row (first row)
                row_tra = tra_row.iloc[0]
                # Extract values from the TRB row (first row)
                row_trb = trb_row.iloc[0]

                # Assuming that `tcra` is represented by `cdr3`, `va` by `v_gene`, `ja` by `j_gene`,
                # `tcrb` by `cdr3`, `vb` by `v_gene`, and `jb` by `j_gene`.
                new_data.append({
                    'tcra': row_tra['cdr3'],  # or use the correct column for tcra
                    'va': row_tra['v_gene'],
                    'ja': row_tra['j_gene'],
                    'tcrb': row_trb['cdr3'],  # or use the correct column for tcrb
                    'vb': row_trb['v_gene'],
                    'jb': row_trb['j_gene']
                })
        #     else:
        #         print(f"Missing TRA or TRB for barcode {barcode}")
        # else:
        #     print(f"Invalid number of valid rows for barcode {barcode}: {len(valid_rows)}")

    # Create a new DataFrame from the collected data
    new_df = pd.DataFrame(new_data)
    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_file, index=False)


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


def filter_humans(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    # Filter to keep rows where 'mhc' starts with "HLA"
    df_hla = df[df['mhc'].str.startswith("HLA", na=False)]
    # Save the result back to a CSV file (optional)
    df_hla.to_csv(output_file, index=False)


def delete_rows_exceeding_threshold(input_file, output_file, column_name, threshold=24):
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_file)
    print(len(df))
    # Filter the rows where the length of the specified column is less than or equal to the threshold
    df_filtered = df[df[column_name].apply(len) <= threshold]
    print(len(df_filtered))
    # # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_file, index=False)


def remove_duplicates(csv_file, output_file):
    """
    Remove duplicate rows from a CSV file and save the cleaned file.

    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
    """
    # Load the data
    data = pd.read_csv(csv_file)
    print(len(data), "1")
    # Remove duplicates, keeping only the first occurrence
    cleaned_data = data.drop_duplicates()
    print(len(cleaned_data), "2")
    # Save the cleaned data to a new CSV file
    cleaned_data.to_csv(output_file, index=False)


def positive_format_sapir_data(input_file, output_file, pep=False):
    # Columns in the desired order for the new file
    # columns_in_order = ["tcra", "va", "ja", "tcrb", "vb", "jb", "dataset_name"]
    if pep:
        columns_in_order = ["tcra", "va", "ja", "tcrb", "vb", "jb", "pep", "hla_seq"]
    else:
        columns_in_order = ["tcra", "va", "ja", "tcrb", "vb", "jb"]
    # Load the CSV into a DataFrame
    # Specify custom missing value markers
    missing_values = ["~", "NA"]
    # Load the CSV into a DataFrame, treating custom markers as NaN
    df = pd.read_csv(input_file, na_values=missing_values)
    # Filter rows with no missing values in the relevant columns
    if pep:
        df_filtered = df.dropna(subset=["cdr3a", "va", "ja", "cdr3b", "vb", "jb", "pep", "hla_seq"])
    else:
        df_filtered = df.dropna(subset=["cdr3a", "va", "ja", "cdr3b", "vb", "jb"])
    # Rename columns to match the desired output format
    df_filtered = df_filtered.rename(columns={
        "cdr3a": "tcra",
        "cdr3b": "tcrb"
    })

    # Reorder columns
    df_filtered = df_filtered[columns_in_order]
    df_filtered.to_csv(output_file, index=False)
    # remove_duplicates(output_file, output_file)
    delete_rows_exceeding_threshold(output_file, output_file, "tcra")
    delete_rows_exceeding_threshold(output_file, output_file, "tcrb")


def all_stages(base_file_name):
    tsv_to_csv(base_file_name + '.tsv', base_file_name + '.csv')

    # filter_humans(base_file_name + '_positives.csv', base_file_name + '_positives.csv')

    get_positives_format_irec(base_file_name + '.csv', base_file_name + '_positives.csv')
    delete_rows_exceeding_threshold(base_file_name + '_positives.csv', base_file_name + '_positives.csv', "tcra")
    delete_rows_exceeding_threshold(base_file_name + '_positives.csv', base_file_name + '_positives.csv', "tcrb")
    remove_duplicates(base_file_name + '_positives.csv', base_file_name + '_positives.csv')
    pu_negatives(base_file_name + '_positives.csv', base_file_name + '_with_negatives.csv')

def mtx_to_csv(mtx_file, csv_file):
    # Load the matrix from the MTX file
    matrix = scipy.io.mmread(mtx_file).toarray()

    # Write the matrix to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each row of the matrix to the CSV
        for row in matrix:
            writer.writerow(row)


def merge_donors():
    # all_stages('vdj_v1_hs_aggregated_donor4_all_contig_annotations')
    for i in range(5):
        print(i)
        # all_stages('donor' + str(i+1))
        all_stages('june' + str(i+1))

    # List of files to merge
    # files = ["donor1_positives.csv", "donor2_positives.csv", "donor3_positives.csv", "donor4_positives.csv"]
    files = ["june1_positives.csv", "june2_positives.csv", "june3_positives.csv", "june4_positives.csv",
             "june5_positives.csv"]

    # Read and concatenate all files
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Shuffle the rows
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop_duplicates()
    # Save to a new file

    df.to_csv("june_all_positives.csv", index=False)

    pu_negatives("june_all_positives.csv", 'june_all_with_negatives.csv')



if __name__ == "__main__":
    positive_format_sapir_data("../all_data_280824.csv", "sapir_data_positives.csv")
    pu_negatives("sapir_data_positives.csv", "sapir_data_with_negatives.csv")
    df = pd.read_csv("Minervina_june_positives.csv")
    all_stages("Minervina_june")
    merge_donors()