import os
import cv2
import numpy as np
import mediapipe as mp

VIDEO_ROOT = "clips"
OUTPUT_ROOT = "pose_data"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for action in os.listdir(VIDEO_ROOT):
    action_dir = os.path.join(VIDEO_ROOT, action)
    if not os.path.isdir(action_dir):
        continue

    output_action_dir = os.path.join(OUTPUT_ROOT, action)
    os.makedirs(output_action_dir, exist_ok=True)

    for filename in os.listdir(action_dir):
        if not filename.endswith(".mp4"):
            continue

        file_path = os.path.join(action_dir, filename)
        cap = cv2.VideoCapture(file_path)

        pose_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in result.pose_landmarks.landmark
                ]
                pose_sequence.append(landmarks)

        cap.release()

        pose_sequence = np.array(pose_sequence)
        output_path = os.path.join(output_action_dir, filename.replace(".mp4", ".npy"))
        np.save(output_path, pose_sequence)

        print(f"✅ 포즈 데이터 저장 완료: {output_path}")
