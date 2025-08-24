import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# データ読み込み
data = np.load('sequence_dataset_all_other.npz')
X = data['X']  # [サンプル数, フレーム数, 特徴量数]
y = data['y']  # [サンプル数]（ラベル名）

# ラベルを数値に変換
le = LabelEncoder()
y_num = le.fit_transform(y)
num_classes = len(le.classes_)

# one-hot化
y_cat = to_categorical(y_num, num_classes=num_classes)

# 訓練・テスト分割
# 変更前
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
# 変更後
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# モデル構築
model = Sequential([
    Masking(mask_value=0., input_shape=(X.shape[1], X.shape[2])),
    LSTM(128, return_sequences=False),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# 推論・評価
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print('--- 精度評価 ---')
print('Accuracy:', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=le.classes_))

# モデル保存（必要なら）
model.save('lstm_sign_model.h5')
print('モデルを lstm_sign_model.h5 に保存しました')

# 使い方：新しいデータで推論する場合
# X_new = ... # shape: [サンプル数, フレーム数, 特徴量数]
# y_pred_new = model.predict(X_new)
# pred_labels = le.inverse_transform(np.argmax(y_pred_new, axis=1))

# どのラベルが1つしかないか確認する
print(Counter(y))