import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
import mediapipe as mp
import os
import pyttsx3

# データ読み込み
DATA_FILE = "sequence_dataset.npz"
if not os.path.exists(DATA_FILE):
    print(f"{DATA_FILE} が見つかりません。先にデータセットを作成してください。")
    exit(1)
data = np.load(DATA_FILE)
X = data['X']
y = data['y']

# ラベルリストを自動生成し、エンコーダーを使う
le = LabelEncoder()
y_num = le.fit_transform(y)
LABELS = list(le.classes_)
print("推論対象ラベル:", LABELS)

# 分類器学習
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X.reshape(X.shape[0], -1), y_num)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

print("カメラに手話・表情・動きを見せてください。ESCで終了。")
last_label = None
N = 3  # 連続判定回数
label_history = []
window_size = X.shape[1]  # データセットから自動取得
feat_buffer = []
expected_feat_dim = X.shape[2]  # 1フレームあたりの特徴量数

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推定
    hand_results = hands.process(rgb)
    face_results = face.process(rgb)
    pose_results = pose.process(rgb)

    def flatten_landmarks(landmarks, n_points):
        if landmarks is None:
            return np.zeros(n_points*3)
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    hand_feats = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_feats.append(flatten_landmarks(hand_landmarks.landmark, 21))
    while len(hand_feats) < 2:
        hand_feats.append(np.zeros(21*3))
    face_feat = np.zeros(468*3)
    if face_results.multi_face_landmarks:
        face_feat = flatten_landmarks(face_results.multi_face_landmarks[0].landmark, 468)
    pose_feat = np.zeros(33*3)
    if pose_results.pose_landmarks:
        pose_feat = flatten_landmarks(pose_results.pose_landmarks.landmark, 33)
    feat = np.concatenate(hand_feats + [face_feat, pose_feat])

    if feat.shape[0] != expected_feat_dim:
        print(f"警告: 特徴量のshapeが異なります。スキップします。shape={feat.shape}")
        continue
    feat_buffer.append(feat)
    if len(feat_buffer) > window_size:
        feat_buffer.pop(0)
    if len(feat_buffer) == window_size:
        X_input = np.stack(feat_buffer).reshape(1, -1)  # [1, window_size * feature_dim]
        pred = clf.predict(X_input)[0]
        label = le.inverse_transform([pred])[0]
        label_history.append(label)
        if len(label_history) > N:
            label_history.pop(0)
        # N回連続で同じラベルなら音声・字幕を出す（other以外のみ）
        if len(label_history) == N and all(l == label for l in label_history):
            if label != last_label and label != "other":
                speak(label)
                last_label = label
        # 字幕もother以外のみ表示
        if label != "other":
            cv2.putText(frame, f"推論: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.putText(frame, "ESCで終了", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Landmark Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() 