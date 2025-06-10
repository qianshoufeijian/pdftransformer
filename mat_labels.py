import pandas as pd
import os

# Define the directory where the CSV files will be saved
output_dir = "D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset – Vibration and Acoustic Faults under Constant and Variable Speed Conditions (UOEMD-VAFCVS)/训练集"

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a list of labels: 200 of each label from 0 to 9
labels = [i for i in range(10) for _ in range(200)]  # Each number 0-9 repeated 200 times

# Convert to DataFrame
acc_labels_df = pd.DataFrame(labels, columns=["acc_labels"])
sound_labels_df = pd.DataFrame(labels, columns=["sound_labels"])
temp_labels_df = pd.DataFrame(labels, columns=["temp_labels"])

# Define the output file paths
output_file_1 = os.path.join(output_dir, "acc_labels.csv")
output_file_2 = os.path.join(output_dir, "sound_labels.csv")
output_file_5 = os.path.join(output_dir, "temp_labels.csv")

# Save the data to CSV files
acc_labels_df.to_csv(output_file_1, index=False)
sound_labels_df.to_csv(output_file_2, index=False)
temp_labels_df.to_csv(output_file_5, index=False)
