import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
import os

# --- 設定 ---
LABELS = ["matane", "arigatou", "sumimasenn","onegai","iiyo"]  # "またね", "ありがとう", "すみません","お願い","いいよ"
DATA_FILE = "sign_data.npy"


# --- 音声合成エンジン ---
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- MediaPipe初期化 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- データ保存・読み込み ---
def save_data(X, y):
    np.save(DATA_FILE, {"X": X, "y": y})
def load_data():
    if os.path.exists(DATA_FILE):
        d = np.load(DATA_FILE, allow_pickle=True).item()
        return d["X"], d["y"]
    return [], []

# --- 特徴量抽出（手のランドマーク座標を1次元配列化） ---
def extract_features(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# --- メイン ---
def main():
    mode = input("[1] データ収集  [2] 推論  を選択してください: ")
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    X, y = load_data()
    clf = None
    if mode == "2" and len(X) > 0:
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
    last_label = None
    label_guide = " / ".join([f"{i+1}={l}" for i, l in enumerate(LABELS)])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        label = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                feat = extract_features(hand_landmarks)
                if mode == "1":
                    # データ収集
                    for i, l in enumerate(LABELS):
                        if cv2.waitKey(1) & 0xFF == ord(str(i+1)):
                            X.append(feat)
                            y.append(i)
                            print(f"{l} のデータを追加")
                elif mode == "2" and clf:
                    # 推論
                    pred = clf.predict([feat])[0]
                    label = LABELS[pred]
                    cv2.putText(frame, f"{label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                    if label != last_label:
                        speak(label)
                        last_label = label
        if mode == "1":
            txt = f"データ収集: {label_guide}  (qで保存終了)"
        else:
            txt = "推論モード (qで終了)"
        cv2.putText(frame, txt, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Sign Recognition PoC", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if mode == "1":
        save_data(X, y)
        print("データ保存しました")

if __name__ == "__main__":
    main() 