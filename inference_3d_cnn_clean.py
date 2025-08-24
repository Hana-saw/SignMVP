import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyttsx3
import time

def inference_3d_cnn_clean():
    """
    クリーンな3D-CNNモデルを使用したリアルタイム推論
    """
    # モデルとラベルエンコーダーの読み込み
    model = keras.models.load_model("3d_cnn_model_clean.h5")
    label_classes = np.load("label_encoder_clean.npy", allow_pickle=True)
    
    print(f"読み込まれたクラス: {label_classes}")
    
    # MediaPipe初期化
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    
    # 音声エンジン初期化
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    # フレームバッファ（16フレーム）
    frame_buffer = []
    window_size = 16
    
    # N-consecutive-frames logic
    N = 3
    consecutive_predictions = []
    last_spoken_label = None
    last_speech_time = 0
    speech_cooldown = 2.0  # 2秒間のクールダウン
    
    print("推論開始...")
    print("ESCキーで終了")
    
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
        
        # 特徴量の次元チェック
        expected_feat_dim = 1629
        if feat.shape[0] != expected_feat_dim:
            print(f"スキップ: 特徴量次元が不一致 {feat.shape[0]} != {expected_feat_dim}")
            continue
        
        # 2D "image-like" 形式に変換
        main_data = feat[:1600].reshape(40, 40)
        extra_data = feat[1600:].reshape(29, 1)
        padded_extra = np.zeros((40, 40))
        padded_extra[:29, 0] = extra_data.flatten()
        combined = np.vstack([main_data, padded_extra])
        combined = combined[:, :, np.newaxis]  # チャンネル次元を追加
        
        # フレームバッファに追加
        frame_buffer.append(combined)
        
        # バッファが満杯になったら推論
        if len(frame_buffer) >= window_size:
            # 古いフレームを削除
            frame_buffer = frame_buffer[-window_size:]
            
            # 3D-CNN入力形式に変換
            input_data = np.stack(frame_buffer)
            input_data = np.expand_dims(input_data, axis=0)  # バッチ次元を追加
            
            # 推論
            prediction = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            predicted_label = label_classes[predicted_class]
            
            # N-consecutive-frames logic
            consecutive_predictions.append(predicted_label)
            if len(consecutive_predictions) > N:
                consecutive_predictions.pop(0)
            
            # 音声出力（other以外）
            current_time = time.time()
            if (len(consecutive_predictions) == N and 
                len(set(consecutive_predictions)) == 1 and 
                consecutive_predictions[0] != "other" and
                consecutive_predictions[0] != last_spoken_label and
                current_time - last_speech_time > speech_cooldown):
                
                print(f"認識: {predicted_label} (信頼度: {confidence:.3f})")
                engine.say(predicted_label)
                engine.runAndWait()
                last_spoken_label = predicted_label
                last_speech_time = current_time
            
            # 画面表示
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{window_size}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("3D-CNN Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキー
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_3d_cnn_clean()
