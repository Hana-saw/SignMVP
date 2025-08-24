import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import pyttsx3

def inference_3d_cnn():
    # モデル読み込み
    model_file = "3d_cnn_sign_model.h5"
    encoder_file = "3d_cnn_label_encoder.npy"
    
    if not os.path.exists(model_file):
        print(f"エラー: {model_file} が見つかりません")
        return
    
    if not os.path.exists(encoder_file):
        print(f"エラー: {encoder_file} が見つかりません")
        return
    
    model = load_model(model_file)
    le_classes = np.load(encoder_file, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = le_classes
    
    print(f"推論対象ラベル: {list(le.classes_)}")
    
    # 音声合成エンジン
    engine = pyttsx3.init()
    def speak(text):
        engine.say(text)
        engine.runAndWait()
    
    # MediaPipe初期化
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    
    # 3D-CNN用のバッファ
    frame_buffer = []
    target_frames = 16  # モデルの入力フレーム数
    window_size = 5     # ランドマーク抽出時のウィンドウサイズ
    
    # 学習時の特徴量次元数を確認（3D-CNNデータセットから取得）
    expected_feat_dim = None
    if os.path.exists("3d_cnn_dataset.npz"):
        data = np.load("3d_cnn_dataset.npz")
        expected_feat_dim = data['X'].shape[2] * data['X'].shape[3] * data['X'].shape[4]  # height * width * channels
        print(f"期待される特徴量次元数: {expected_feat_dim}")
    
    # 連続判定用
    N = 3
    label_history = []
    last_label = None
    
    print("カメラに手話・表情・動きを見せてください。ESCで終了。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe推定
        hand_results = hands.process(rgb)
        face_results = face.process(rgb)
        pose_results = pose.process(rgb)
        
        # 特徴量抽出
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
        
        # 特徴量のshapeチェック（デバッグ情報付き）
        if expected_feat_dim is not None:
            print(f"現在の特徴量次元: {feat.shape[0]}, 期待される次元: {expected_feat_dim}")
            if feat.shape[0] != expected_feat_dim:
                print(f"警告: 特徴量のshapeが異なります。スキップします。")
                print(f"  - 現在: {feat.shape[0]}次元")
                print(f"  - 期待: {expected_feat_dim}次元")
                print(f"  - 手の特徴量: {len(hand_feats)}個")
                print(f"  - 顔の特徴量: {face_feat.shape[0]}次元")
                print(f"  - 体の特徴量: {pose_feat.shape[0]}次元")
                continue
        
        # 3D-CNN用データ準備
        frame_buffer.append(feat)
        if len(frame_buffer) > window_size:
            frame_buffer.pop(0)
        
        if len(frame_buffer) == window_size:
            print(f"推論実行中... バッファサイズ: {len(frame_buffer)}")
            # 3Dテンソルに変換
            landmarks_3d = np.stack(frame_buffer)
            
            # 特徴量を2D画像風にリシェイプ
            feat_dim = landmarks_3d.shape[1]
            side_length = int(np.sqrt(feat_dim))
            if side_length * side_length == feat_dim:
                landmarks_2d = landmarks_3d.reshape(window_size, side_length, side_length, 1)
            else:
                next_square = int(np.ceil(np.sqrt(feat_dim))) ** 2
                padded = np.zeros((window_size, next_square))
                padded[:, :feat_dim] = landmarks_3d
                side_length = int(np.sqrt(next_square))
                landmarks_2d = padded.reshape(window_size, side_length, side_length, 1)
            
            # フレーム数を調整
            if landmarks_2d.shape[0] < target_frames:
                padding = np.zeros((target_frames - landmarks_2d.shape[0], side_length, side_length, 1))
                landmarks_2d = np.concatenate([landmarks_2d, padding], axis=0)
            elif landmarks_2d.shape[0] > target_frames:
                indices = np.linspace(0, landmarks_2d.shape[0]-1, target_frames, dtype=int)
                landmarks_2d = landmarks_2d[indices]
            
            # 推論
            X_input = landmarks_2d.reshape(1, *landmarks_2d.shape)
            pred_prob = model.predict(X_input, verbose=0)
            pred_idx = np.argmax(pred_prob[0])
            label = le.inverse_transform([pred_idx])[0]
            
            # 連続判定
            label_history.append(label)
            if len(label_history) > N:
                label_history.pop(0)
            
            if len(label_history) == N and all(l == label for l in label_history):
                if label != last_label and label != "other":
                    speak(label)
                    last_label = label
            
            # 結果表示
            if label != "other":
                cv2.putText(frame, f"3D-CNN: {label}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        
        cv2.putText(frame, "ESCで終了", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("3D-CNN Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_3d_cnn()
