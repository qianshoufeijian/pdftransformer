import scipy.io
import pandas as pd
import numpy as np
import os

# Define the directory where the .mat files are located
mat_files_dir = "D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset – Vibration and Acoustic Faults under Constant and Variable Speed Conditions (UOEMD-VAFCVS)/测试集"

# Initialize empty lists to store the features
acc_features = []
sound_features = []
temp_features = []

# Loop over each .mat file in the directory
for file_name in os.listdir(mat_files_dir):
    if file_name.endswith(".mat"):  # Check if it's a .mat file
        file_path = os.path.join(mat_files_dir, file_name)

        # Load the .mat file
        mat = scipy.io.loadmat(file_path)

        # Extract the 'data' variable
        data = mat['data']

        # Convert to a DataFrame
        df = pd.DataFrame(data)

        # Sample 200 rows from each of the specified columns (1st, 2nd, and 5th: index 0, 1, 4)
        sample_col_1 = df.iloc[:, 0].sample(n=200, random_state=42).reset_index(drop=True)
        sample_col_2 = df.iloc[:, 1].sample(n=200, random_state=42).reset_index(drop=True)
        sample_col_5 = df.iloc[:, 4].sample(n=200, random_state=42).reset_index(drop=True)

        # Append the sampled data to the respective lists
        acc_features.append(sample_col_1)
        sound_features.append(sample_col_2)
        temp_features.append(sample_col_5)

# Concatenate all the collected features into single DataFrames (vertically)
acc_features_df = pd.concat(acc_features, ignore_index=True)
sound_features_df = pd.concat(sound_features, ignore_index=True)
temp_features_df = pd.concat(temp_features, ignore_index=True)

# Define the output file paths
output_file_1 = os.path.join(mat_files_dir, "acc_features.csv")
output_file_2 = os.path.join(mat_files_dir, "sound_features.csv")
output_file_5 = os.path.join(mat_files_dir, "temp_features.csv")

# Save the concatenated data to CSV files
acc_features_df.to_csv(output_file_1, index=False, header=['acc_feature'])
sound_features_df.to_csv(output_file_2, index=False, header=['sound_features'])
temp_features_df.to_csv(output_file_5, index=False, header=['temp_features'])
