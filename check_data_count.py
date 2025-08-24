import numpy as np
from collections import Counter
import os

def check_data_counts():
    print("=== 現在のデータ数確認 ===\n")
    
    # 1. sequence_dataset.npz の確認
    if os.path.exists("sequence_dataset.npz"):
        print("1. sequence_dataset.npz (LSTM用)")
        data = np.load("sequence_dataset.npz")
        y = data['y']
        counts = Counter(y)
        print(f"   総サンプル数: {len(y)}")
        print(f"   ラベル数: {len(counts)}")
        print("   各ラベルのデータ数:")
        for label, count in sorted(counts.items()):
            print(f"     - {label}: {count}サンプル")
        print()
    else:
        print("1. sequence_dataset.npz が見つかりません")
        print()
    
    # 2. 3d_cnn_dataset.npz の確認
    if os.path.exists("3d_cnn_dataset.npz"):
        print("2. 3d_cnn_dataset.npz (3D-CNN用)")
        data = np.load("3d_cnn_dataset.npz")
        y = data['y']
        counts = Counter(y)
        print(f"   総サンプル数: {len(y)}")
        print(f"   ラベル数: {len(counts)}")
        print("   各ラベルのデータ数:")
        for label, count in sorted(counts.items()):
            print(f"     - {label}: {count}サンプル")
        print()
    else:
        print("2. 3d_cnn_dataset.npz が見つかりません")
        print()
    
    # 3. labels.csv の確認
    if os.path.exists("labels.csv"):
        print("3. labels.csv (元データ)")
        import csv
        labels = []
        with open("labels.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row['label'])
        
        counts = Counter(labels)
        print(f"   総サンプル数: {len(labels)}")
        print(f"   ラベル数: {len(counts)}")
        print("   各ラベルのデータ数:")
        for label, count in sorted(counts.items()):
            print(f"     - {label}: {count}サンプル")
        print()
    else:
        print("3. labels.csv が見つかりません")
        print()
    
    # 4. 推奨データ数
    print("4. 推奨データ数")
    print("   各ラベル最低10サンプル以上を推奨")
    print("   理想は各ラベル20-50サンプル")
    print()

if __name__ == "__main__":
    check_data_counts()
