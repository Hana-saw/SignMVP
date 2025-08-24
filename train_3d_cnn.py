import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import os

def create_3d_cnn_model(num_classes, input_shape=(16, 64, 64, 1)):
    """
    L=3層の3D-CNNモデル（ReLU活性化関数使用）
    """
    model = Sequential([
        # 1st 3D Conv Block (L=1)
        Conv3D(32, kernel_size=(3, 3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.25),
        
        # 2nd 3D Conv Block (L=2)
        Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.25),
        
        # 3rd 3D Conv Block (L=3)
        Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.25),
        
        # Classification Head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    # データ読み込み
    data_file = "3d_cnn_dataset.npz"
    if not os.path.exists(data_file):
        print(f"エラー: {data_file} が見つかりません。先にprepare_3d_cnn_data.pyを実行してください。")
        exit(1)
    
    data = np.load(data_file)
    X_3d = data['X']
    y_labels = data['y']
    
    print(f"データ形状: {X_3d.shape}")
    print(f"ラベル数: {len(set(y_labels))}")
    
    # ラベルエンコーディング
    le = LabelEncoder()
    y_num = le.fit_transform(y_labels)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_num, num_classes=num_classes)
    
    print(f"ラベル一覧: {list(le.classes_)}")
    
    # データ分割（stratifyを外して分割）
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_3d, y_cat, test_size=0.2, random_state=42, stratify=y_cat
        )
    except ValueError:
        print("警告: stratifyが使えないため、通常の分割を使用します")
        X_train, X_test, y_train, y_test = train_test_split(
            X_3d, y_cat, test_size=0.2, random_state=42
        )
    
    # モデル作成
    input_shape = X_3d.shape[1:]  # (frames, height, width, channels)
    model = create_3d_cnn_model(num_classes, input_shape)
    
    # モデル概要表示
    model.summary()
    
    # コンパイル
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # コールバック
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    print("学習開始...")
    # 学習
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,  # 3D-CNNはメモリを大量消費するため小さく
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 評価
    print("\n--- 評価結果 ---")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"テスト精度: {test_accuracy:.4f}")
    
    # 詳細な分類レポート
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n--- 詳細分類レポート ---")
    # 実際にテストデータに含まれるクラスのみを対象とする
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [le.classes_[i] for i in unique_classes]
    print(classification_report(y_true, y_pred, target_names=target_names, labels=unique_classes))
    
    # モデル保存
    model.save('3d_cnn_sign_model.h5')
    print("\nモデルを 3d_cnn_sign_model.h5 に保存しました")
    
    # ラベルエンコーダーも保存
    np.save('3d_cnn_label_encoder.npy', le.classes_)
    print("ラベルエンコーダーを 3d_cnn_label_encoder.npy に保存しました")
