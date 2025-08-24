import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

frames_dir = "frames"
output_dir = "landmarks"
os.makedirs(output_dir, exist_ok=True)

hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh(max_num_faces=1)
pose = mp_pose.Pose()

def flatten_landmarks(landmarks, n_points):
    if landmarks is None:
        return np.zeros(n_points*3)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

for fname in sorted(os.listdir(frames_dir)):
    if not fname.endswith(".jpg"):
        continue
    img = cv2.imread(os.path.join(frames_dir, fname))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face.process(rgb)
    pose_results = pose.process(rgb)

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

    np.save(os.path.join(output_dir, fname.replace('.jpg', '.npy')), feat)
print("ランドマーク抽出が完了しました")