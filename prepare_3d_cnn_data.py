import numpy as np
import cv2
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def prepare_3d_data_from_landmarks(landmarks_dir, labels_csv, target_frames=16):
    """
    ランドマークデータから3D-CNN用のデータを準備
    """
    # ラベル読み込み
    labels_dict = {}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['label']
    
    # ファイルリスト
    file_list = sorted([f for f in os.listdir(landmarks_dir) if f.endswith('.npy')])
    
    X_3d = []
    y_labels = []
    
    # ウィンドウサイズ（例：5フレーム）
    window_size = 5
    
    for i in range(len(file_list) - window_size + 1):
        window = file_list[i:i+window_size]
        if all(fname in labels_dict for fname in window):
            # ランドマークデータを読み込み
            landmarks_window = []
            for fname in window:
                landmarks = np.load(os.path.join(landmarks_dir, fname))
                landmarks_window.append(landmarks)
            
            # 3Dテンソルに変換（[frames, features] → [frames, height, width, channels]）
            landmarks_3d = np.stack(landmarks_window)
            
            # 特徴量を2D画像風にリシェイプ（例：sqrt(features) × sqrt(features)）
            feat_dim = landmarks_3d.shape[1]
            side_length = int(np.sqrt(feat_dim))
            if side_length * side_length == feat_dim:
                # 完全平方数の場合
                landmarks_2d = landmarks_3d.reshape(window_size, side_length, side_length, 1)
            else:
                # 不完全な場合はパディング
                next_square = int(np.ceil(np.sqrt(feat_dim))) ** 2
                padded = np.zeros((window_size, next_square))
                padded[:, :feat_dim] = landmarks_3d
                side_length = int(np.sqrt(next_square))
                landmarks_2d = padded.reshape(window_size, side_length, side_length, 1)
            
            # フレーム数を調整
            if landmarks_2d.shape[0] < target_frames:
                # パディング
                padding = np.zeros((target_frames - landmarks_2d.shape[0], side_length, side_length, 1))
                landmarks_2d = np.concatenate([landmarks_2d, padding], axis=0)
            elif landmarks_2d.shape[0] > target_frames:
                # サンプリング
                indices = np.linspace(0, landmarks_2d.shape[0]-1, target_frames, dtype=int)
                landmarks_2d = landmarks_2d[indices]
            
            X_3d.append(landmarks_2d)
            y_labels.append(labels_dict[window[window_size//2]])
    
    return np.array(X_3d), np.array(y_labels)

if __name__ == "__main__":
    # データ準備
    landmarks_dir = "landmarks"
    labels_csv = "labels.csv"
    
    if not os.path.exists(landmarks_dir):
        print(f"エラー: {landmarks_dir} フォルダが見つかりません")
        exit(1)
    
    if not os.path.exists(labels_csv):
        print(f"エラー: {labels_csv} ファイルが見つかりません")
        exit(1)
    
    print("3D-CNN用データを準備中...")
    X_3d, y_labels = prepare_3d_data_from_landmarks(landmarks_dir, labels_csv, target_frames=16)
    
    print(f"3D-CNN用データ形状: {X_3d.shape}")
    print(f"ラベル数: {len(set(y_labels))}")
    print(f"ラベル一覧: {sorted(set(y_labels))}")
    
    # データ保存
    np.savez('3d_cnn_dataset.npz', X=X_3d, y=y_labels)
    print("3D-CNN用データセットを 3d_cnn_dataset.npz に保存しました")
