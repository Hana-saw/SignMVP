import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from collections import Counter

def load_current_data():
    """現在のデータ数を確認"""
    if os.path.exists("labels_clean.csv"):
        labels = []
        with open("labels_clean.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row['label'])
        return Counter(labels)
    return Counter()

def focused_data_collection_phase2(target_labels, samples_per_label=20):
    """
    第2段階：より多くのデータを収集
    """
    # 現在のデータ数を確認
    current_counts = load_current_data()
    
    print("=== 第2段階：集中データ収集 ===")
    print(f"対象ラベル: {target_labels}")
    print(f"目標サンプル数: 各ラベル{samples_per_label}サンプル")
    print()
    
    # 現在の状況を表示
    print("現在のデータ数:")
    for label in target_labels:
        current = current_counts.get(label, 0)
        needed = max(0, samples_per_label - current)
        print(f"  {label}: {current}サンプル → あと{needed}サンプル必要")
    print()
    
    # MediaPipe初期化
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    
    # landmarksフォルダの作成
    os.makedirs("landmarks", exist_ok=True)
    
    # 既存のファイル数を確認
    existing_files = len([f for f in os.listdir("landmarks") if f.endswith('.npy')])
    current_file_index = existing_files
    
    # 各ラベルの収集状況
    collected_counts = {label: 0 for label in target_labels}
    
    print("データ収集開始...")
    print("キー操作:")
    for i, label in enumerate(target_labels[:9]):
        print(f"  {i+1}: {label}")
    print(f"  0: {target_labels[9]}")
    print(f"  a: {target_labels[10]}")
    print("  q: 終了")
    print()
    
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
        
        # キー入力でデータ保存
        key = cv2.waitKey(1) & 0xFF
        
        # ラベル指定のキー入力（1-9）
        for i, label in enumerate(target_labels[:9]):
            if key == ord(str(i+1)):
                if collected_counts[label] < samples_per_label:
                    # ランドマークファイルを保存
                    filename = f"frame_{current_file_index:05d}.npy"
                    np.save(os.path.join("landmarks", filename), feat)
                    
                    # labels_clean.csvに追加
                    with open("labels_clean.csv", 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, label])
                    
                    collected_counts[label] += 1
                    current_file_index += 1
                    print(f"{label} のデータを追加 ({collected_counts[label]}/{samples_per_label}) - {filename}")
                else:
                    print(f"{label} は目標数に達しています")
        
        # 10番目のラベル
        if key == ord('0'):
            label = target_labels[9]
            if collected_counts[label] < samples_per_label:
                filename = f"frame_{current_file_index:05d}.npy"
                np.save(os.path.join("landmarks", filename), feat)
                
                with open("labels_clean.csv", 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, label])
                
                collected_counts[label] += 1
                current_file_index += 1
                print(f"{label} のデータを追加 ({collected_counts[label]}/{samples_per_label}) - {filename}")
            else:
                print(f"{label} は目標数に達しています")
        
        # 11番目のラベル
        if key == ord('a'):  # 'a'キーで11番目
            label = target_labels[10]
            if collected_counts[label] < samples_per_label:
                filename = f"frame_{current_file_index:05d}.npy"
                np.save(os.path.join("landmarks", filename), feat)
                
                with open("labels_clean.csv", 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, label])
                
                collected_counts[label] += 1
                current_file_index += 1
                print(f"{label} のデータを追加 ({collected_counts[label]}/{samples_per_label}) - {filename}")
            else:
                print(f"{label} は目標数に達しています")
        
        if key == ord('q'):
            break
        
        # 進捗表示
        progress_text = "進捗: "
        for label in target_labels:
            progress_text += f"{label}({collected_counts[label]}/{samples_per_label}) "
        
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "ESCで終了", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Phase 2 Data Collection", frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 結果表示
    print("\n=== 収集結果 ===")
    for label in target_labels:
        print(f"{label}: {collected_counts[label]}サンプル収集")
    
    total_collected = sum(collected_counts.values())
    print(f"\n合計: {total_collected}サンプル収集完了")
    print("labels_clean.csv と landmarks/ フォルダの両方に保存されました")

if __name__ == "__main__":
    # 第2段階：より多くのデータを収集
    target_labels = ["ichi", "nihon", "no", "ok?", "syuwa", "waktta", "watashi", "namae", "wakaranai", "wakaru", "zaki"]
    samples_per_label = 20
    
    print("第2段階: より多くのデータ収集")
    print(f"対象: {target_labels}")
    print(f"目標: 各ラベル{samples_per_label}サンプル")
    print()
    
    input("Enterキーを押して開始...")
    focused_data_collection_phase2(target_labels, samples_per_label)
