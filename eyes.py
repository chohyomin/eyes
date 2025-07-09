# === test change for git ===
import cv2
import mediapipe as mp
import numpy as np
import os

INPUT_FOLDER = 'clips'
OUTPUT_FOLDER = 'pose_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.mp4'):
        continue

    video_path = os.path.join(INPUT_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)

    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            all_landmarks.append(landmarks)

    cap.release()

    output_file = os.path.splitext(filename)[0] + '.npy'
    np.save(os.path.join(OUTPUT_FOLDER, output_file), np.array(all_landmarks))
    print(f' 저장 완료: {output_file}')

pose.close()

