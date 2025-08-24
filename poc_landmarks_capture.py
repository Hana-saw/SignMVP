import cv2
import mediapipe as mp
import numpy as np
import os

LABELS = ["matane", "arigatou", "sumimasenn", "onegai", "iiyo"]  # 必要に応じて追加
DATA_FILE = "multimodal_sign_data.npz"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# データ保存用
X, y = [], []
label_guide = " / ".join([f"{i+1}={l}" for i, l in enumerate(LABELS)])

print("キー番号でラベルを指定し、データを保存します。qで終了。")

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

    # 描画
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )

    # 特徴量抽出
    def flatten_landmarks(landmarks, n_points):
        if landmarks is None:
            return np.zeros(n_points*3)
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    # 両手
    hand_feats = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_feats.append(flatten_landmarks(hand_landmarks.landmark, 21))
    while len(hand_feats) < 2:
        hand_feats.append(np.zeros(21*3))
    # 顔
    face_feat = np.zeros(468*3)
    if face_results.multi_face_landmarks:
        face_feat = flatten_landmarks(face_results.multi_face_landmarks[0].landmark, 468)
    # 肩・全身
    pose_feat = np.zeros(33*3)
    if pose_results.pose_landmarks:
        pose_feat = flatten_landmarks(pose_results.pose_landmarks.landmark, 33)
    # すべて連結
    feat = np.concatenate(hand_feats + [face_feat, pose_feat])

    # キー入力でデータ保存
    key = cv2.waitKey(1) & 0xFF
    for i, l in enumerate(LABELS):
        if key == ord(str(i+1)):
            X.append(feat)
            y.append(i)
            print(f"{l} のデータを追加（合計 {len(X)} 件）")
    if key == ord('q'):
        break

    txt = f"データ収集: {label_guide}  (qで保存終了)"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Landmark Capture", frame)

cap.release()
cv2.destroyAllWindows()

if len(X) > 0:
    np.savez(DATA_FILE, X=np.array(X), y=np.array(y))
    print(f"{len(X)}件のデータを{DATA_FILE}に保存しました")
else:
    print("データは保存されませんでした") 