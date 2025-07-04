import cv2
import numpy as np
import mediapipe as mp
import pickle

# ✅ 예측할 영상 경로
VIDEO_PATH = 'test_videos/test_pass1.mp4'  # 여기를 원하는 파일명으로 바꾸세요

# ✅ 모델 불러오기
with open('pose_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(VIDEO_PATH)

keypoints_all = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        for lm in landmarks:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        keypoints_all.append(keypoints)

cap.release()

# 평균 포즈 벡터 사용
if keypoints_all:
    avg_keypoints = np.mean(keypoints_all, axis=0).reshape(1, -1)
    prediction = clf.predict(avg_keypoints)
    print(f'🎯 예측된 행동: {prediction[0]}')
else:
    print("❌ 영상에서 포즈를 추출하지 못했습니다.")
