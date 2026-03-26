
# Digital Twin-based Light Tunnel Inspection System
> 3D 렌더링 기반 합성 데이터와 온보드 추론 환경을 결합해 자동차 외관 결함을 검사하는 머신 비전 시스템
> 최종 결과에 관한 자세한 내용은 results/docs/presentation/머신 비전을 활용한 라이트 터널 검사 시스템 에서 확인 가능합니다
## Overview
본 프로젝트는 자동차 제조 공정의 외관 검사 단계인 **Light Tunnel Inspection**을 머신 비전 기반으로 구현한 프로젝트입니다.  
실제 불량 데이터 수집이 어렵고 비용이 크다는 문제를 해결하기 위해, **Blender 기반 디지털 트윈 환경에서 결함 데이터를 생성**하고 이를 활용해 detection 모델을 학습한 뒤, **온보드 환경에서 실제 검사 시스템 형태로 동작**하도록 구성했습니다. :contentReference[oaicite:1]{index=1}

검사 대상 결함은 다음 4가지입니다.

- dent (찍힘)
- smash (우그러짐)
- stain (오염)
- scratch (스크래치) :contentReference[oaicite:2]{index=2}

---

## Why This Project
실제 외관 불량 데이터는 수집 비용이 높고 확보가 어렵기 때문에, 본 프로젝트는 먼저 **3D Render를 통한 데이터 생성 및 학습의 실현 가능성**을 검증하는 데 초점을 두었습니다.  
또한 최종적으로는 **적은 투자금으로 확장 가능한 On-board 검사 시스템**을 구현해, 유지보수 비용을 줄이고 현장 적용 가능성을 확인하고자 했습니다. 

---

## Project Goals
- 디지털 트윈 환경을 활용한 결함 데이터 생성
- 실제 라이트 터널과 유사한 검사 환경 구현
- 합성 데이터 기반 detection 모델 학습
- Jetson Nano 기반 온보드 추론 환경 구성
- 검사 결과 수신 및 시각화를 위한 UI 연동 구현 

---

## System Pipeline
본 프로젝트는 크게 **Data → Model → Visualization** 흐름으로 구성되었습니다.

1. **Data**
   - Blender 기반 3D 렌더링
   - 랜덤 결함 생성
   - 합성 데이터 약 2000장 제작

2. **Model**
   - SSD MobileNet V2 기반 detection 모델 학습
   - 실제 데이터 일부를 test에 활용

3. **Visualization**
   - 온보드 환경에서 추론 결과 전송
   - UI에서 이미지 및 결함 결과 수신 후 시각화 

---

## Key Features

### 1. Digital Twin-based Data Generation
실제 라이트 터널과 유사한 환경을 Blender로 재구성하고, 실제 데이터와 비슷한 크기와 구성을 갖도록 가상 검사 환경을 제작했습니다.  
이를 통해 실제 현장에서 수집하기 어려운 불량 데이터를 가상으로 생성하고, 모델 학습에 활용했습니다. 

### 2. Automatic Defect Annotation
결함 생성 시 annotation도 함께 자동 생성되도록 구성했습니다.

- dent / scratch: 변형된 좌표 기반 min/max 계산 후 bbox 생성
- stain: 중심점과 반경 기반 샘플링 후 bbox 생성
- smash: 전체 영역을 bbox로 생성 :contentReference[oaicite:7]{index=7}

### 3. Embedded & On-board Inspection Flow
임베디드 단에서는 다음 기능을 구성했습니다.

- start 스위치
- 스텝모터 제어
- LED 제어
- 카메라 트리거 신호 생성

또한 테스트 환경에서는 가림막 설치로 외부 환경 변수를 줄이고, 레일 및 이동 경로를 구성해 경로를 표준화했으며, 모터와 도르레를 설치해 검사 대상 이동을 구현했습니다. 

### 4. Inspection Scenario Implementation
검사 공정은 다음과 같은 시나리오로 구성했습니다.

- 진입
- 검사
- 출하

검사 단계에서는 좌/중/우 조명 환경을 반영해 표면 결함이 드러나도록 하고, 이를 모델 입력 데이터로 활용했습니다. 

---

## Data

### Real-world Data Environment
실증 환경은 다음과 같이 구성했습니다.

- 자동차 대체물: 틴케이스 (10 x 7 x 2 cm)
- 라이트 터널: 박스 (24 x 20 x 15.5 cm), LED, 웹캠 :contentReference[oaicite:10]{index=10}

### Synthetic Data Environment
가상 데이터 환경은 실제 데이터와 유사한 환경과 같은 크기로 제작했습니다.  
실제 데이터와 가상 데이터를 비교한 결과, 조명 반사와 표면 형상이 유사하게 표현되어 합성 데이터 기반 학습의 가능성을 확인할 수 있었습니다. 

### Defect Types
랜덤으로 생성한 결함 유형은 다음과 같습니다.

- stain
- dent
- smash
- scratch :contentReference[oaicite:12]{index=12}

---

## Model

### Detection Model
본 프로젝트에서는 **SSD MobileNet V2 (DetectNet)** 구조를 사용했습니다.  
MobileNet V2 기반 경량 CNN과 SSD(one-stage detector)를 적용했고, Jetson Nano에서 실시간 추론이 가능하도록 ONNX 변환과 TensorRT 최적화를 수행했습니다. :contentReference[oaicite:13]{index=13}

### Training / Deployment Setting
- epoch: 50
- batch size: 8
- image size: 320
- output: `best.pth`, `detectnet.onnx`
- environment:
  - Windows: training
  - Linux + Docker: deployment
  - JetPack 4.6
  - TensorRT 8.0.1
  - opset 13 :contentReference[oaicite:14]{index=14}

---

## Project Flow
프로젝트는 아래 순서로 진행되었습니다.

1. 사전 기획 및 역할 분담
2. 가상 데이터 제작 / 실제 데이터 제작
3. SSD MobileNet V2 및 YOLO 11n 검토
4. 라이트 터널 환경 제작
5. 웹캠 연결 및 모델 추론 기능 구현
6. 기능 통합
7. UI 제작 및 기능 연동 

---

## Results
본 프로젝트를 통해 짧은 기간 내에 다음 전 과정을 연결해 구현했습니다.

- 디지털 트윈 환경 구성
- 합성 데이터 생성 및 annotation 자동화
- 임베디드 제어 환경 제작
- 온보드 추론 환경 구성
- 웹 기반 시각화 연동 :contentReference[oaicite:16]{index=16}

또한  
가상 환경에서 제작한 데이터가 예상보다 실물과 유사하게 표현되었고, 데이터가 부족한 현장 상황에서 디지털 트윈 기반 데이터 생성 접근이 충분히 의미 있다는 점을 확인했습니다. 다만 이후에는 **실제 데이터와 가상 데이터를 균형 있게 혼합한 학습**까지 확장해볼 필요가 있다고 정리했습니다. :contentReference[oaicite:17]{index=17}

---

## What I Learned
- 실제 데이터 수집이 어려운 문제를 디지털 트윈 기반 합성 데이터로 보완할 수 있음을 확인했습니다.
- 단순 모델 학습에 그치지 않고, 임베디드 제어와 온보드 추론, UI 시각화까지 연결해야 실제 시스템 형태가 된다는 점을 체감했습니다.
- 검사 시스템은 데이터 품질뿐 아니라 조명 환경, 이동 경로, 외부 환경 변수 차단 같은 하드웨어적 요소도 성능에 큰 영향을 준다는 점을 배웠습니다. 

---

## Tech Stack
- **3D / Digital Twin**
  - Blender

- **AI / Vision**
  - SSD MobileNet V2
  - YOLO 11n
  - ONNX
  - TensorRT

- **On-board / Embedded**
  - stm32f411
  - Jetson Nano
  - Webcam
  - Step Motor
  - LED
  - Camera Trigger

- **UI / Visualization**
  - React
  - Web-based Visualization 

---
