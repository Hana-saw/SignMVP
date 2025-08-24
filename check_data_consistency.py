import numpy as np
import os
import csv
from collections import Counter

def check_landmark_dimensions():
    """ランドマークファイルの次元をチェック"""
    landmarks_dir = "landmarks"
    if not os.path.exists(landmarks_dir):
        print("landmarksフォルダが存在しません")
        return
    
    files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
    print(f"ランドマークファイル数: {len(files)}")
    
    dimensions = {}
    for file in files:
        filepath = os.path.join(landmarks_dir, file)
        try:
            data = np.load(filepath)
            dim = data.shape[0]
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(file)
        except Exception as e:
            print(f"エラー: {file} - {e}")
    
    print("\n=== 次元別ファイル数 ===")
    for dim, file_list in dimensions.items():
        print(f"次元 {dim}: {len(file_list)}ファイル")
        if len(file_list) <= 5:
            print(f"  ファイル: {file_list}")
        else:
            print(f"  ファイル: {file_list[:3]} ... {file_list[-2:]}")
    
    return dimensions

def check_labels_csv():
    """labels.csvの内容をチェック"""
    if not os.path.exists("labels.csv"):
        print("labels.csvが存在しません")
        return
    
    labels = []
    with open("labels.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['label'])
    
    counter = Counter(labels)
    print("\n=== ラベル別データ数 ===")
    for label, count in counter.most_common():
        print(f"{label}: {count}サンプル")
    
    return counter

def fix_data_consistency():
    """データの整合性を修正"""
    print("\n=== データ整合性修正 ===")
    
    # 現在の状況を確認
    dimensions = check_landmark_dimensions()
    label_counts = check_labels_csv()
    
    if not dimensions:
        return
    
    # 最も多い次元を標準とする
    most_common_dim = max(dimensions.keys(), key=lambda k: len(dimensions[k]))
    print(f"\n標準次元: {most_common_dim}")
    
    # 異なる次元のファイルを削除
    files_to_remove = []
    for dim, file_list in dimensions.items():
        if dim != most_common_dim:
            files_to_remove.extend(file_list)
    
    if files_to_remove:
        print(f"\n削除対象ファイル数: {len(files_to_remove)}")
        print("削除するファイル:")
        for file in files_to_remove:
            print(f"  {file}")
        
        # 確認
        response = input("\nこれらのファイルを削除しますか？ (y/n): ")
        if response.lower() == 'y':
            for file in files_to_remove:
                os.remove(os.path.join("landmarks", file))
            print("ファイルを削除しました")
            
            # labels.csvからも対応する行を削除
            remove_labels_from_csv(files_to_remove)
        else:
            print("削除をキャンセルしました")
    else:
        print("すべてのファイルが同じ次元です")

def remove_labels_from_csv(files_to_remove):
    """labels.csvから指定されたファイルの行を削除"""
    if not os.path.exists("labels.csv"):
        return
    
    # 削除対象のファイル名（拡張子なし）
    files_to_remove_no_ext = [f.replace('.npy', '') for f in files_to_remove]
    
    # 既存データを読み込み
    existing_data = []
    with open("labels.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['filename'].replace('.npy', '') not in files_to_remove_no_ext:
                existing_data.append(row)
    
    # 保存
    with open("labels.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(existing_data)
    
    print(f"labels.csvから{len(files_to_remove)}行を削除しました")

def create_clean_dataset():
    """クリーンなデータセットを作成"""
    print("\n=== クリーンなデータセット作成 ===")
    
    # 現在の状況を確認
    dimensions = check_landmark_dimensions()
    if not dimensions:
        return
    
    # 最も多い次元のファイルのみを使用
    most_common_dim = max(dimensions.keys(), key=lambda k: len(dimensions[k]))
    valid_files = dimensions[most_common_dim]
    
    print(f"有効なファイル数: {len(valid_files)}")
    
    # 新しいlabels.csvを作成
    new_labels = []
    for file in valid_files:
        # 既存のlabels.csvからラベルを取得
        label = get_label_from_csv(file)
        if label:
            new_labels.append({'filename': file, 'label': label})
    
    # 保存
    with open("labels_clean.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(new_labels)
    
    print(f"クリーンなlabels_clean.csvを作成しました ({len(new_labels)}サンプル)")

def get_label_from_csv(filename):
    """labels.csvからファイル名に対応するラベルを取得"""
    if not os.path.exists("labels.csv"):
        return None
    
    with open("labels.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['filename'] == filename:
                return row['label']
    return None

if __name__ == "__main__":
    print("=== データ整合性チェック ===")
    
    # 1. 現在の状況を確認
    check_landmark_dimensions()
    check_labels_csv()
    
    # 2. 修正オプションを提示
    print("\n=== 修正オプション ===")
    print("1. データの整合性を自動修正")
    print("2. クリーンなデータセットを作成")
    print("3. 現在の状況のみ確認")
    
    choice = input("\n選択してください (1-3): ")
    
    if choice == '1':
        fix_data_consistency()
    elif choice == '2':
        create_clean_dataset()
    elif choice == '3':
        print("確認のみ完了")
    else:
        print("無効な選択です")
