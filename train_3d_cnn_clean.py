import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

def create_3d_cnn_model(input_shape, num_classes):
    """
    3D-CNNモデルを作成
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # 1層目: 3D畳み込み層
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(0.25),
        
        # 2層目: 3D畳み込み層
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(0.25),
        
        # 3層目: 3D畳み込み層
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(0.25),
        
        # 全結合層
        layers.GlobalAveragePooling3D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_3d_cnn_clean():
    """
    クリーンなデータセットで3D-CNNを学習
    """
    # データ読み込み
    data_file = "3d_cnn_dataset_clean.npz"
    if not os.path.exists(data_file):
        print(f"エラー: {data_file} が見つかりません")
        return
    
    data = np.load(data_file)
    X = data['X']
    y = data['y']
    
    print(f"データ形状: X={X.shape}, y={y.shape}")
    
    # ラベルエンコーディング
    le = LabelEncoder()
    y_cat = le.fit_transform(y)
    
    print(f"クラス数: {len(le.classes_)}")
    print(f"クラス: {le.classes_}")
    
    # データ分割（stratifyを削除してエラーを回避）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    
    # モデル作成
    input_shape = X.shape[1:]  # (frames, height, width)
    num_classes = len(le.classes_)
    
    print(f"入力形状: {input_shape}")
    print(f"クラス数: {num_classes}")
    
    model = create_3d_cnn_model(input_shape, num_classes)
    
    # モデルコンパイル
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # モデル概要
    model.summary()
    
    # コールバック
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # 学習
    print("\n学習開始...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # 評価
    print("\n=== 評価結果 ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"テスト精度: {test_accuracy:.4f}")
    
    # 予測
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 分類レポート
    print("\n=== 分類レポート ===")
    # テストセットに存在するクラスのみを対象とする
    test_classes = np.unique(y_test)
    target_names = [le.inverse_transform([cls])[0] for cls in test_classes]
    
    print(classification_report(
        y_test, y_pred_classes, 
        target_names=target_names,
        labels=test_classes
    ))
    
    # モデル保存
    model.save("3d_cnn_model_clean.h5")
    print("\nモデルを 3d_cnn_model_clean.h5 に保存しました")
    
    # ラベルエンコーダーも保存
    np.save("label_encoder_clean.npy", le.classes_)
    print("ラベルエンコーダーを label_encoder_clean.npy に保存しました")

if __name__ == "__main__":
    train_3d_cnn_clean()
