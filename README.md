# 👁️ eyes

MediaPipe를 사용해 축구 영상을 분석하고, 포즈 데이터를 추출하여  
MMAction2로 슛, 패스 등 다양한 축구 행동을 인식하는 모델을 학습하는 프로젝트입니다.

## 🎯 목표

- 축구 훈련 및 경기 영상에서 플레이어 행동(예: 슛, 패스)을 자동 인식
- MediaPipe로 포즈 좌표를 추출하고 `.npy` 형태로 저장
- 추출된 데이터를 `.pkl`로 변환하여 MMAction2로 행동 분류 학습
- 아마추어 선수들에게 분석 리포트를 제공하는 앱 개발을 장기 목표로 함

### 📊 출력
- 행동 인식 결과
- 프레임별 분류 시각화
- 향후 PDF 리포트 생성 예정

#### ⚙️ 사용 기술 스택

- **Python 3.10**
- [MediaPipe](https://github.com/google/mediapipe) – 포즈 추출
- [MMAction2](https://github.com/open-mmlab/mmaction2) – 행동 분류 모델 학습
- OpenCV – 영상 프레임 추출
- NumPy, Pickle – 데이터 저장 및 변환
- Git + VSCode – 개발 및 협업

- ##### 🚀 향후 개발 계획

- 다양한 경기 데이터 확보 및 전처리 자동화
- 모델 정확도 평가 및 개선
- 분석 결과 PDF 리포트 생성 기능 추가
- 웹 기반 인터페이스 구현 (React or Streamlit)
- 앱 버전 개발 및 사용자 피드백 적용

 - 팀 구성
  chohyomin
  kheejae 

 
 * 협업 방식

- GitHub 기반 버전 관리
- 각 기능 단위 커밋 및 PR
- VSCode + Git 연동으로 공동 작업
