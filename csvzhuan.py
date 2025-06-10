import numpy as np
import pandas as pd

# 定义输入的 CSV 文件路径和输出的 NPY 文件路径
csv_file_path = "D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset/acc/val/acc_labels.csv"  # 输入的 CSV 文件路径
npy_file_path = "D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset/acc/val/acc_labels.npy"  # 输出的 NPY 文件路径

# 使用 pandas 读取 CSV 文件
# 如果你的 CSV 文件没有列名，可以设置 header=None
data = pd.read_csv(csv_file_path, header=None).values  # 读取 CSV 文件并转换为 NumPy 数组

# 使用 numpy 保存为 .npy 文件
np.save(npy_file_path, data)

print(f"CSV 文件已成功转换为 NPY 文件：{npy_file_path}")