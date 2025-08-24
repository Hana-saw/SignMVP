import numpy as np
import csv
import os

def make_sequence_dataset_clean(window_size=5):
    """
    クリーンなデータセットから時系列データセットを作成
    """
    landmarks_dir = "landmarks"
    labels_csv = "labels_clean.csv"
    
    if not os.path.exists(labels_csv):
        print(f"エラー: {labels_csv} が見つかりません")
        return
    
    # ラベル辞書を作成
    labels_dict = {}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['label']
    
    print(f"ラベル数: {len(labels_dict)}")
    
    # ファイル名をソート
    filenames = sorted(labels_dict.keys())
    
    # 時系列データを作成
    X_seq, y_seq = [], []
    
    for i in range(len(filenames) - window_size + 1):
        # window_size分のファイルを取得
        window_files = filenames[i:i+window_size]
        
        # ランドマークデータを読み込み
        X_window = []
        for filename in window_files:
            filepath = os.path.join(landmarks_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                X_window.append(data)
            else:
                print(f"警告: {filepath} が見つかりません")
                break
        else:
            # すべてのファイルが存在する場合
            if len(X_window) == window_size:
                # 中央のフレームのラベルを使用
                center_idx = window_size // 2
                label = labels_dict[window_files[center_idx]]
                
                X_seq.append(np.stack(X_window))
                y_seq.append(label)
    
    # NumPy配列に変換
    X = np.array(X_seq)
    y = np.array(y_seq)
    
    # 保存
    np.savez("sequence_dataset_clean.npz", X=X, y=y)
    
    print(f"時系列データセットを sequence_dataset_clean.npz に保存しました")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # ラベル別のサンプル数を表示
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\nラベル別サンプル数:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}サンプル")

if __name__ == "__main__":
    make_sequence_dataset_clean(window_size=5)
