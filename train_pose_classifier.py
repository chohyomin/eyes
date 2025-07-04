# train_pose_classifier.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 불러오기
with open("pose_dataset.pkl", "rb") as f:
    X, y = pickle.load(f)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 모델 저장
with open("pose_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# 정확도 출력
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 행동 분류 정확도: {acc:.2%}")
