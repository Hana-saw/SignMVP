#!/usr/bin/env python3
"""
Webから収集されたデータを処理するスクリプト
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import glob

def process_web_data():
    """
    web_dataフォルダ内のJSONファイルを処理して、
    既存のデータセットと統合する
    """
    print("=== Webデータ処理開始 ===")
    
    # web_dataフォルダの確認
    web_data_dir = "web_data"
    if not os.path.exists(web_data_dir):
        print("web_dataフォルダが存在しません")
        return
    
    # JSONファイルの検索
    json_files = glob.glob(os.path.join(web_data_dir, "*.json"))
    if not json_files:
        print("処理するJSONファイルが見つかりません")
        return
    
    print(f"処理対象ファイル数: {len(json_files)}")
    
    # 既存のlabels_clean.csvを読み込み
    existing_labels = {}
    if os.path.exists("labels_clean.csv"):
        df = pd.read_csv("labels_clean.csv")
        for _, row in df.iterrows():
            existing_labels[row['filename']] = row['label']
    
    # 新しいデータを処理
    new_landmarks = []
    new_labels = []
    
    for json_file in json_files:
        print(f"処理中: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データの検証
            if not isinstance(data, list):
                print(f"警告: {json_file} は配列形式ではありません")
                continue
            
            for sample in data:
                # 必須フィールドの確認
                if 'signWord' not in sample or 'features' not in sample:
                    print(f"警告: 必須フィールドが不足しています")
                    continue
                
                # 特徴量の次元チェック
                features = sample['features']
                if len(features) != 1629:
                    print(f"警告: 特徴量の次元が不正です: {len(features)}")
                    continue
                
                # ファイル名の生成
                timestamp = sample.get('timestamp', int(datetime.now().timestamp() * 1000))
                filename = f"web_frame_{timestamp:013d}.npy"
                
                # ランドマークファイルの保存
                landmark_path = os.path.join("landmarks", filename)
                np.save(landmark_path, np.array(features))
                
                # ラベル情報の保存
                new_landmarks.append(filename)
                new_labels.append(sample['signWord'])
                
        except Exception as e:
            print(f"エラー: {json_file} の処理に失敗しました - {e}")
            continue
    
    print(f"処理完了: {len(new_landmarks)} サンプル")
    
    # 新しいラベルを既存のCSVに追加
    if new_landmarks:
        new_data = []
        for filename, label in zip(new_landmarks, new_labels):
            new_data.append({
                'filename': filename,
                'label': label
            })
        
        # 既存データと統合
        if os.path.exists("labels_clean.csv"):
            existing_df = pd.read_csv("labels_clean.csv")
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = pd.DataFrame(new_data)
        
        # 保存
        combined_df.to_csv("labels_clean.csv", index=False)
        print(f"labels_clean.csv を更新しました (総サンプル数: {len(combined_df)})")
        
        # 統計情報の表示
        label_counts = combined_df['label'].value_counts()
        print("\nラベル別サンプル数:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}サンプル")
    
    # 処理済みファイルの移動
    processed_dir = os.path.join(web_data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    for json_file in json_files:
        try:
            filename = os.path.basename(json_file)
            new_path = os.path.join(processed_dir, filename)
            os.rename(json_file, new_path)
            print(f"移動: {filename} -> processed/")
        except Exception as e:
            print(f"警告: {filename} の移動に失敗しました - {e}")

def validate_data_quality():
    """
    データの品質を検証する
    """
    print("\n=== データ品質検証 ===")
    
    if not os.path.exists("labels_clean.csv"):
        print("labels_clean.csv が存在しません")
        return
    
    df = pd.read_csv("labels_clean.csv")
    
    # 基本統計
    print(f"総サンプル数: {len(df)}")
    print(f"ユニークラベル数: {df['label'].nunique()}")
    
    # ラベル別統計
    label_counts = df['label'].value_counts()
    print(f"\nラベル別サンプル数:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}サンプル")
    
    # データの不均衡チェック
    min_samples = label_counts.min()
    max_samples = label_counts.max()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    print(f"\nデータ不均衡比: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print("警告: データの不均衡が大きいです")
    
    # ランドマークファイルの存在確認
    missing_files = []
    for filename in df['filename']:
        landmark_path = os.path.join("landmarks", filename)
        if not os.path.exists(landmark_path):
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n警告: {len(missing_files)} 個のランドマークファイルが見つかりません")
        for filename in missing_files[:5]:  # 最初の5個のみ表示
            print(f"  {filename}")
    else:
        print("\n✓ すべてのランドマークファイルが存在します")

if __name__ == "__main__":
    process_web_data()
    validate_data_quality()
    print("\n=== Webデータ処理完了 ===")
