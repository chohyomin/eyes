# convert_pose_to_dataset.py
import os
import numpy as np
import pickle

DATA_PATH = 'pose_data'
OUTPUT_FILE = 'pose_dataset.pkl'

X = []
y = []

for action in os.listdir(DATA_PATH):
    action_dir = os.path.join(DATA_PATH, action)
    for filename in os.listdir(action_dir):
        filepath = os.path.join(action_dir, filename)
        data = np.load(filepath)

        # (frames, 33, 4) → 평균 내기 → (33, 4) → 평탄화 → (132,)
        avg_pose = data.mean(axis=0).flatten()
        X.append(avg_pose)
        y.append(action)

X = np.array(X)
y = np.array(y)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump((X, y), f)

print(f"✅ pose_dataset.pkl 저장 완료 ({len(X)} samples)")
