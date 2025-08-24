import os
import numpy as np
import csv

landmarks_dir = "landmarks"
labels_csv = "labels.csv"
output_file = "sequence_dataset.npz"
window_size = 5  # 1サンプルあたりのフレーム数

# ラベル辞書を作成
labels_dict = {}
with open(labels_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels_dict[row['filename']] = row['label']

# ファイルリストをソート
file_list = sorted([f for f in os.listdir(landmarks_dir) if f.endswith('.npy')])

X_seq = []
y_seq = []

for i in range(len(file_list) - window_size + 1):
    window = file_list[i:i+window_size]
    # すべてのフレームにラベルがある場合のみ
    if all(fname in labels_dict for fname in window):
        X_window = [np.load(os.path.join(landmarks_dir, fname)) for fname in window]
        # 中央フレームのラベルを代表ラベルに
        label = labels_dict[window[window_size//2]]
        X_seq.append(np.stack(X_window))
        y_seq.append(label)

X_seq = np.array(X_seq)  # [サンプル数, フレーム数, 特徴量数]
y_seq = np.array(y_seq)

np.savez(output_file, X=X_seq, y=y_seq)
print(f"時系列データセットを {output_file} に保存しました")
print(f"X shape: {X_seq.shape}, y shape: {y_seq.shape}")