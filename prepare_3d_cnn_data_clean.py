import numpy as np
import csv
import os

def prepare_3d_data_from_landmarks_clean(landmarks_dir, labels_csv, target_frames=16):
    """
    クリーンなランドマークデータから3D-CNN用データを準備
    """
    if not os.path.exists(labels_csv):
        print(f"エラー: {labels_csv} が見つかりません")
        return None, None
    
    # ラベル辞書を作成
    labels_dict = {}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['label']
    
    print(f"ラベル数: {len(labels_dict)}")
    
    # ファイル名をソート
    filenames = sorted(labels_dict.keys())
    
    # 3D-CNN用データを作成
    X_3d, y_labels = [], []
    
    for i in range(len(filenames) - target_frames + 1):
        # target_frames分のファイルを取得
        window_files = filenames[i:i+target_frames]
        
        # ランドマークデータを読み込み
        landmarks_window = []
        for filename in window_files:
            filepath = os.path.join(landmarks_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                # 2D "image-like" 形式に変換 (例: 40x40)
                # 1629次元を40x40+29に変換
                if data.shape[0] == 1629:
                    # 1600次元を40x40に変換、残り29次元を追加
                    main_data = data[:1600].reshape(40, 40)
                    extra_data = data[1600:].reshape(29, 1)
                    # 29x1を40x40の最後の行に追加
                    padded_extra = np.zeros((40, 40))
                    padded_extra[:29, 0] = extra_data.flatten()
                    combined = np.vstack([main_data, padded_extra])
                    # チャンネル次元を追加 (height, width, 1)
                    combined = combined[:, :, np.newaxis]
                    landmarks_window.append(combined)
                else:
                    print(f"警告: 予期しない次元 {data.shape[0]} in {filename}")
                    break
            else:
                print(f"警告: {filepath} が見つかりません")
                break
        else:
            # すべてのファイルが存在する場合
            if len(landmarks_window) == target_frames:
                # 中央のフレームのラベルを使用
                center_idx = target_frames // 2
                label = labels_dict[window_files[center_idx]]
                
                # 3Dテンソルに変換 (frames, height, width, channels)
                landmarks_3d = np.stack(landmarks_window)
                X_3d.append(landmarks_3d)
                y_labels.append(label)
    
    # NumPy配列に変換
    X = np.array(X_3d)
    y = np.array(y_labels)
    
    return X, y

def main():
    print("3D-CNN用データを準備中...")
    
    landmarks_dir = "landmarks"
    labels_csv = "labels_clean.csv"
    
    X_3d, y_labels = prepare_3d_data_from_landmarks_clean(landmarks_dir, labels_csv, target_frames=16)
    
    if X_3d is not None:
        # 保存
        np.savez("3d_cnn_dataset_clean.npz", X=X_3d, y=y_labels)
        
        print(f"3D-CNN用データセットを 3d_cnn_dataset_clean.npz に保存しました")
        print(f"X shape: {X_3d.shape}, y shape: {y_labels.shape}")
        
        # ラベル別のサンプル数を表示
        unique_labels, counts = np.unique(y_labels, return_counts=True)
        print("\nラベル別サンプル数:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}サンプル")
    else:
        print("データの準備に失敗しました")

if __name__ == "__main__":
    main()
