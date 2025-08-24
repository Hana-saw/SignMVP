import cv2
import mediapipe as mp

# MediaPipe初期化
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# VideoCapture
cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 各種ランドマーク推定
    hand_results = hands.process(rgb)
    face_results = face.process(rgb)
    pose_results = pose.process(rgb)

    # 両手
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 顔
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

    # 肩・全身
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )

    cv2.putText(frame, "ESCで終了", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Multi-modal Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESCキー
        break

cap.release()
cv2.destroyAllWindows() 