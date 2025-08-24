#!/usr/bin/env python3
"""
学習済みモデルをTensorFlow.js形式に変換するスクリプト
"""

import os
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np

def convert_model_to_tfjs():
    """
    学習済みの3D-CNNモデルをTensorFlow.js形式に変換
    """
    print("=== モデル変換開始 ===")
    
    # モデルファイルの確認
    model_path = "3d_cnn_model_clean.h5"
    if not os.path.exists(model_path):
        print(f"エラー: {model_path} が見つかりません")
        return
    
    # 出力ディレクトリの作成
    output_dir = "web_app/models"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # モデルの読み込み
        print("モデルを読み込み中...")
        model = tf.keras.models.load_model(model_path)
        
        # モデル情報の表示
        print(f"モデル入力形状: {model.input_shape}")
        print(f"モデル出力形状: {model.output_shape}")
        print(f"モデルパラメータ数: {model.count_params():,}")
        
        # TensorFlow.js形式に変換
        print("TensorFlow.js形式に変換中...")
        tfjs.converters.save_keras_model(model, output_dir)
        
        print(f"✓ モデルを {output_dir} に保存しました")
        
        # モデルファイルの確認
        model_files = os.listdir(output_dir)
        print(f"生成されたファイル:")
        for file in model_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file} ({file_size:,} bytes)")
        
        # ラベルエンコーダーの変換
        convert_label_encoder()
        
    except Exception as e:
        print(f"エラー: モデル変換に失敗しました - {e}")
        return

def convert_label_encoder():
    """
    ラベルエンコーダーをJSON形式に変換
    """
    print("\n=== ラベルエンコーダー変換 ===")
    
    label_encoder_path = "label_encoder_clean.npy"
    if not os.path.exists(label_encoder_path):
        print(f"警告: {label_encoder_path} が見つかりません")
        return
    
    try:
        # ラベルエンコーダーの読み込み
        label_classes = np.load(label_encoder_path, allow_pickle=True)
        
        # JSON形式に変換
        label_data = {
            "classes": label_classes.tolist(),
            "num_classes": len(label_classes)
        }
        
        # 保存
        output_path = "web_app/models/labels.json"
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ ラベルエンコーダーを {output_path} に保存しました")
        print(f"クラス数: {len(label_classes)}")
        print(f"クラス: {label_classes.tolist()}")
        
    except Exception as e:
        print(f"エラー: ラベルエンコーダー変換に失敗しました - {e}")

def create_model_info():
    """
    モデル情報ファイルを作成
    """
    print("\n=== モデル情報ファイル作成 ===")
    
    model_info = {
        "name": "SignMVP 3D-CNN Model",
        "version": "1.0.0",
        "description": "手話認識用3D-CNNモデル",
        "input_shape": [16, 80, 40, 1],
        "output_shape": [17],
        "model_type": "3D-CNN",
        "framework": "TensorFlow.js",
        "created_date": "2025-01-01",
        "accuracy": "9.09%",
        "classes": [
            "?", "ha!", "hai", "ichi", "namae", "nihon", "no", "ok?", 
            "other", "syuwa", "tugi", "wakaranai", "wakarimasitaka?", 
            "wakaru", "waktta", "watashi", "zaki"
        ]
    }
    
    output_path = "web_app/models/model_info.json"
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"✓ モデル情報を {output_path} に保存しました")

def validate_conversion():
    """
    変換結果を検証
    """
    print("\n=== 変換結果検証 ===")
    
    output_dir = "web_app/models"
    if not os.path.exists(output_dir):
        print("エラー: 出力ディレクトリが存在しません")
        return
    
    required_files = [
        "model.json",
        "labels.json",
        "model_info.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✓ {file} ({file_size:,} bytes)")
        else:
            print(f"✗ {file} が見つかりません")
    
    # 重みファイルの確認
    weight_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
    if weight_files:
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in weight_files)
        print(f"✓ 重みファイル: {len(weight_files)}個 (合計: {total_size:,} bytes)")
    else:
        print("✗ 重みファイルが見つかりません")

if __name__ == "__main__":
    convert_model_to_tfjs()
    create_model_info()
    validate_conversion()
    print("\n=== モデル変換完了 ===")
