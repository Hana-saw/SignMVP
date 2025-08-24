import cv2
import os

video_path = "shuwa1.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps)  # 1秒ごとに1枚

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % interval == 0:
        cv2.imwrite(f"{output_dir}/frame_{count//interval:05d}.jpg", frame)
    count += 1
cap.release()
print("フレーム画像の抽出が完了しました")