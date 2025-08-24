import numpy as np
import os

def check_and_fix_data_shape():
    """データの形状を確認し、3D-CNNに適した形式に変換"""
    
    # 3D-CNNデータセットを読み込み
    data_file = "3d_cnn_dataset_clean.npz"
    if not os.path.exists(data_file):
        print(f"エラー: {data_file} が見つかりません")
        return
    
    data = np.load(data_file)
    X = data['X']
    y = data['y']
    
    print(f"現在のデータ形状: X={X.shape}, y={y.shape}")
    
    # 3D-CNNは (batch, frames, height, width) の4次元を期待
    # 現在は (batch, frames, height, width) の4次元になっているはず
    if len(X.shape) == 4:
        print("データ形状は正しい4次元です")
        print(f"各次元の意味:")
        print(f"  - バッチサイズ: {X.shape[0]}")
        print(f"  - フレーム数: {X.shape[1]}")
        print(f"  - 高さ: {X.shape[2]}")
        print(f"  - 幅: {X.shape[3]}")
        
        # データの統計情報
        print(f"\nデータの統計情報:")
        print(f"  - 最小値: {X.min():.4f}")
        print(f"  - 最大値: {X.max():.4f}")
        print(f"  - 平均値: {X.mean():.4f}")
        print(f"  - 標準偏差: {X.std():.4f}")
        
        # サンプルデータの可視化
        print(f"\nサンプルデータ (最初のフレーム):")
        sample_frame = X[0, 0]  # 最初のサンプルの最初のフレーム
        print(f"  形状: {sample_frame.shape}")
        print(f"  値の範囲: {sample_frame.min():.4f} ~ {sample_frame.max():.4f}")
        
        return True
    else:
        print(f"エラー: 予期しないデータ形状 {X.shape}")
        return False

def create_simple_3d_cnn_data():
    """シンプルな3D-CNN用データを作成"""
    print("\n=== シンプルな3D-CNN用データ作成 ===")
    
    # 元のランドマークデータを読み込み
    landmarks_dir = "landmarks"
    labels_csv = "labels_clean.csv"
    
    if not os.path.exists(labels_csv):
        print(f"エラー: {labels_csv} が見つかりません")
        return
    
    import csv
    labels_dict = {}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['label']
    
    filenames = sorted(labels_dict.keys())
    
    # シンプルな3D-CNN用データを作成
    X_3d, y_labels = [], []
    target_frames = 16
    
    for i in range(len(filenames) - target_frames + 1):
        window_files = filenames[i:i+target_frames]
        
        landmarks_window = []
        for filename in window_files:
            filepath = os.path.join(landmarks_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                if data.shape[0] == 1629:
                    # 1629次元を40x40に変換（余りは切り捨て）
                    reshaped = data[:1600].reshape(40, 40)
                    landmarks_window.append(reshaped)
                else:
                    break
            else:
                break
        else:
            if len(landmarks_window) == target_frames:
                center_idx = target_frames // 2
                label = labels_dict[window_files[center_idx]]
                
                landmarks_3d = np.stack(landmarks_window)
                X_3d.append(landmarks_3d)
                y_labels.append(label)
    
    X = np.array(X_3d)
    y = np.array(y_labels)
    
    print(f"シンプルな3D-CNNデータ形状: X={X.shape}, y={y.shape}")
    
    # 保存
    np.savez("3d_cnn_dataset_simple.npz", X=X, y=y)
    print("シンプルな3D-CNNデータセットを 3d_cnn_dataset_simple.npz に保存しました")
    
    return X, y

if __name__ == "__main__":
    print("=== データ形状チェック ===")
    
    # 現在のデータをチェック
    if check_and_fix_data_shape():
        print("\n現在のデータは3D-CNNに適しています")
    else:
        print("\nデータ形状に問題があります")
        # シンプルなデータを作成
        create_simple_3d_cnn_data()
