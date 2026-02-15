# DeepCredit - QHedge 시스템 상세 분석

## 📌 프로젝트 개요

### 프로젝트명
**DeepCredit - 저축은행 신용평가를 위한 인공지능기반의 신용예측 시스템**

### 주요 목적
저축은행의 신용평가를 위한 AI 기반 신용예측 시스템으로, 여러 머신러닝/딥러닝 모델을 앙상블하여 정확한 신용 리스크 예측을 수행합니다.

### 기술 스택
- **언어**: Python 3.x
- **프레임워크**: Django 3.2, Django REST Framework
- **ML/DL 라이브러리**: 
  - XGBoost
  - TensorFlow
  - PyTorch
  - scikit-learn
  - TabNet
- **데이터 처리**: pandas, numpy, dask
- **데이터베이스**: MariaDB
- **분산 컴퓨팅**: Dask Distributed
- **GPU**: NVIDIA TITAN x 4 (수냉식)

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
DeepCredit/
├── QHedge/                    # 메인 신용평가 시스템
│   ├── Data_preprocessing/    # 데이터 전처리
│   ├── Learning/              # 모델 학습
│   ├── Prediction/            # 예측 수행
│   ├── Ensemble/              # 앙상블 처리
│   ├── Optimization/          # 하이퍼파라미터 최적화
│   ├── ENV/                   # 환경설정 및 DB 연결
│   ├── ResfulApi/             # Django REST API
│   └── module/                # 공통 모듈
├── base_model/                # 기본 모델 및 DB 유틸리티
├── data/                      # 데이터셋
├── data_profiling/            # 데이터 프로파일링
├── data_imbalance/            # 불균형 데이터 처리
├── docs/                      # 문서
└── main/                      # 메인 실행 스크립트
```

---

## 🔧 주요 컴포넌트 상세 분석

### 1. **launcher.py - 배치 자동 실행 시스템**

#### 역할
- 데이터베이스에서 대기 중인 배치 작업을 자동으로 감지하고 실행
- 배치 작업의 전체 생명주기 관리 (I → U → C/E)

#### 주요 기능
```python
class Launcher:
    - __init__(): 서버 ID 설정, 로깅 초기화
    - read_dc_batch(): dc_batch 테이블에서 작업 조회 및 잠금
    - set_parameters(): 배치 파라미터 설정
    - execute_main(): 메인 프로그램 실행
    - finish_batch(): 작업 완료 처리
```

#### 배치 상태 관리
- **I (Initial)**: 대기 중
- **U (Under Processing)**: 처리 중
- **C (Completed)**: 완료
- **E (Error)**: 오류 발생

#### 핵심 로직
1. `state='I'`인 배치를 10개 조회
2. 랜덤하게 하나 선택 (성능 최적화)
3. `FOR UPDATE`로 행 잠금
4. `state='U'`로 변경하여 다른 프로세스 차단
5. 작업 실행
6. 완료 시 `state='C'`, 실패 시 `state='E'`

---

### 2. **interface.py - 메인 워크플로우 제어**

#### 역할
전체 시스템의 진입점으로 모드에 따라 적절한 처리를 수행

#### 지원 모드
1. **train**: 모델 학습
2. **predict**: 학습된 모델로 예측
3. **ensemble**: 앙상블 수행
4. **real**: 실제 예측 (구현 예정)

#### 워크플로우
```python
def DeepCredit_main(batch_info, batch_param):
    if mode == "train":
        데이터 읽기 → 전처리 → 학습
    
    elif mode == "predict":
        데이터 읽기 → 전처리 → 모델 로드 → 예측 → 결과 저장
    
    elif mode == "ensemble":
        앙상블 수행 (데이터 읽기 생략)
```

---

### 3. **Data_preprocessing - 데이터 전처리**

#### 주요 파일
- **Read_data.py**: 데이터베이스에서 데이터 조회
- **preprocessing.py**: 전처리 파이프라인 실행
- **transformer.py**: 데이터 변환 로직

#### 전처리 단계
```python
def preprocessing(data, mode, batch_info, batch_param):
    1. X, Y 분리 (features, target)
    2. train_test_split (비율: batch_param["testSize"])
    3. Transformer 적용:
       - 카테고리 인코딩 (OneHotEncoder)
       - 수치형 스케일링 (RobustScaler)
       - 결측치 처리
    4. train/test 모드에 따라 적절한 데이터 반환
```

#### 주요 기능
- **drop_column()**: 모든 값이 동일한 컬럼 제거
- **preprocessing_beta()**: 카테고리/수치형 데이터 분리 및 인코딩
- **data_scaler()**: RobustScaler를 이용한 스케일링

---

### 4. **Learning - 모델 학습**

#### 4.1 Training.py

분산 컴퓨팅을 활용한 모델 학습 수행

```python
def training(x_train, y_train, batch_param, batch_info):
    1. Dask Distributed Client 연결
    2. 모델 리스트 조회 (dc_model_list)
    3. 병렬 처리를 위한 파라미터 설정
    4. 각 모델에 대해:
       - 데이터 불균형 처리 (Resampling)
       - 모델 학습
       - 모델 저장 (SFTP)
       - 결과 DB 저장
```

#### 4.2 models.py

지원 모델 종류:
1. **XGBoost (XGB)**
   - n_estimators: 400
   - learning_rate: 0.1
   - max_depth: 3

2. **Deep Neural Network (DNN)**
   ```python
   - Layer 1: Dense(128) + BatchNormalization + ReLU
   - Layer 2: Dense(1) + Sigmoid
   - Loss: binary_crossentropy
   - Optimizer: Adam
   ```

3. **Random Forest (RF)**
   - n_estimators: 100

#### 모델 저장/로드
- **ML 모델**: joblib로 `.pkl` 저장
- **DL 모델**: TensorFlow `.h5` 저장
- **저장 위치**: SFTP 서버 (`165.246.34.142`)

---

### 5. **Prediction - 예측 수행**

#### Prediction.py

```python
def predict(model, x_test, y_test):
    1. 모델로 예측 수행
    2. DNN의 경우 확률값을 0/1로 변환 (threshold=0.5)
    3. Confusion Matrix 계산
    4. 결과 반환 (TN, FP, FN, TP)
```

#### Save_result.py
- 예측 결과를 `dc_batch_result` 테이블에 저장
- Confusion Matrix 기반 성능 지표 저장

---

### 6. **Ensemble - 앙상블 시스템**

#### 핵심 아이디어
여러 베이스 모델의 예측을 결합하여 더 정확한 예측 수행

#### 앙상블 프로세스

```python
def ensemble(batch_info, batch_param):
    1. 데이터 로드 및 전처리
    
    2. 베이스 모델 조회 (candidate_id)
       - dc_candidate_model_list 테이블 참조
    
    3. 모델 로딩 (약 374개)
       - TABNET: .pt 파일
       - DNN: .pt 파일
       - XGB/RF: .pkl 파일
    
    4. 베이스 모델로 예측 수행 (모든 모델)
    
    5. 앙상블 수행
       - 100만 번의 앙상블 반복
       - 각 반복마다 랜덤하게 30개 모델 선택
       - 20번 반복하여 평균
    
    6. 보팅 (Voting)
       - 연체 기준: 11회 이상
       - 다수결 투표
    
    7. 결과 저장
       - dc_batch_result_t4
       - dc_batch_detail_t4
```

#### 병렬 처리
- 100만 건의 앙상블을 1000개 범위로 분할
- Dask Distributed로 병렬 처리

---

### 7. **Optimization - 하이퍼파라미터 최적화**

#### optimizer.py

```python
class Optimizer:
    - Bayesian Optimization 지원
    - GPyOpt 라이브러리 활용
    - 모델별 최적 파라미터 탐색
```

---

### 8. **ENV - 환경 설정 및 DB 연결**

#### DB_Handler.py

```python
class DBHandler:
    - SQLAlchemy 기반 DB 연결
    - MariaDB 연결 (165.246.34.133:3306)
    - Connection Pool 관리
    
    주요 메서드:
    - get_connection(): 엔진 반환
    - retrive_stmt(): SELECT 쿼리 실행
    - execute_stmt(): INSERT/UPDATE 실행
```

#### config.py
- 데이터베이스 연결 정보
- 서버 설정 정보
- 환경 변수 관리

---

### 9. **ResfulApi - Django REST API**

#### 구조
```
ResfulApi/
├── manage.py              # Django 관리 스크립트
├── restfulAPI/            # 프로젝트 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── resful_main/           # 메인 앱
    ├── models.py          # DB 모델
    ├── views.py           # API 뷰
    ├── serializers.py     # 시리얼라이저
    └── urls.py            # URL 라우팅
```

#### 주요 기능
- 배치 작업 조회/등록
- 예측 결과 조회
- 모델 목록 관리
- RESTful API 제공

---

### 10. **distributed_computing - 분산 컴퓨팅**

#### 아키텍처
```
server_config.ini → 서버 설정 로드
    ↓
Dask Scheduler 연결
    ↓
Worker 노드들에 작업 분산
    ↓
병렬 처리 수행
```

#### 주요 파일
- **cluster_start.py**: 클러스터 시작
- **cluster_stop.py**: 클러스터 중지
- **cluster_run.py**: 작업 실행
- **common.py**: 공통 유틸리티

#### 설정 예시
```ini
[server1]
host=165.246.34.142
scheduler_port=8786
```

---

## 📊 데이터 흐름

### 학습 모드 (train)
```
DB (dc_dataset) 
    → Read_data
    → preprocessing (train_test_split)
    → 불균형 데이터 처리 (SMOTE/Undersampling)
    → 모델 학습 (병렬)
    → 모델 저장 (SFTP)
    → 결과 저장 (dc_batch_result)
```

### 예측 모드 (predict)
```
DB (dc_dataset)
    → Read_data
    → preprocessing (test 데이터)
    → 모델 로드 (SFTP)
    → 예측 수행 (병렬)
    → 결과 저장 (dc_batch_result)
```

### 앙상블 모드 (ensemble)
```
로컬 CSV 파일
    → 전처리
    → 베이스 모델 예측 (374개)
    → 앙상블 (1,000,000회 * 30개 모델)
    → 보팅
    → 결과 저장 (dc_batch_result_t4)
```

---

## 🗄️ 데이터베이스 스키마

### 주요 테이블

#### dc_batch
배치 작업 정보 테이블
```
- batch_id: 배치 ID (PK)
- batch_desc: 배치 설명
- batch_memo: 메모
- batch_param: JSON 파라미터
- script_no: 스크립트 번호
- mode: train/predict/ensemble/real
- train_batch_id: 학습 배치 ID (예측 시 참조)
- dataset_group: 데이터셋 그룹
- dataset_version: 데이터셋 버전
- model_group: 모델 그룹
- state: I/U/C/E
- serve_server_id: 서버 ID
- work_date: 작업 시작 시각
- work_end_date: 작업 종료 시각
```

#### dc_model_list
모델 목록 테이블
```
- model_no: 모델 번호 (PK)
- model_id: 모델 식별자 (예: ML-XGB, DL-DNN)
- model_group_id: 모델 그룹 ID
- model_param: JSON 파라미터
- use_yn: 사용 여부
```

#### dc_batch_result
배치 결과 테이블
```
- batch_id: 배치 ID
- model_no: 모델 번호
- model_sub_no: 모델 서브 번호
- confusion_matrix: TP, TN, FP, FN
- accuracy, precision, recall, f1_score 등
```

#### dc_candidate_model_list
앙상블 후보 모델 목록
```
- candidate_id: 후보 ID
- batch_id: 배치 ID
- model_no: 모델 번호
- model_sub_no: 모델 서브 번호
```

---

## ⚙️ 설정 및 실행

### 환경 설정

#### GPU 서버
```bash
# Docker 컨테이너 실행
docker run --name deepcredit --hostname DEEPCREDIT \
  --restart always --gpus all --ipc host --privileged \
  -p 17022:22 -p 17888:8888 \
  -it djyoon0223/base:full
```

#### Python 라이브러리 설치
```bash
pip install calculate numpy import_ipynb tqdm sklearn pandas dask
pip install sqlalchemy pymysql imblearn xgboost category_encoders
pip install tensorflow shap bayesian-optimization pytorch-tabnet
pip install wget matplotlib torchvision paramiko asyncssh distributed
pip install GPyOpt scipy==1.9.0 django==3.2 djangorestframework
```

### 실행 방법

#### 1. Django 서버 실행
```bash
cd ResfulApi
python manage.py runserver
# http://127.0.0.1:8000/
```

#### 2. 배치 런처 실행
```bash
# SERVER_ID를 인자로 전달
python launcher.py SERVER_001
```

#### 3. 단일 배치 실행
```bash
python main.py
```

---

## 🔍 주요 알고리즘

### 1. 데이터 불균형 처리

```python
from Learning.Imbalance import imbalance_data

# SMOTE, RandomUnderSampler 등을 활용한 리샘플링
x_resampled, y_resampled = imbalance_data(x_train, y_train, model_info)
```

### 2. 앙상블 알고리즘

```python
# 30개 모델을 랜덤 선택하여 20번 반복
for i in range(20):
    random_models = random.sample(all_models, 30)
    predictions = [model.predict(X) for model in random_models]
    ensemble_result = voting(predictions)

# 최종 보팅: 11회 이상 연체로 분류된 경우 1로 판정
```

### 3. 베이지안 최적화

```python
from Optimization.optimizer import Optimizer

opt = Optimizer("Bayesian")
best_params = opt.run(objective_function, search_space)
```

---

## 📈 성능 및 확장성

### 분산 처리 성능
- **베이스 모델 로딩**: 약 20초 (374개 모델)
- **베이스 모델 예측**: 약 35초 (서버), 53초 (로컬)
- **앙상블 처리**: 병렬화로 대폭 단축

### GPU 활용
- NVIDIA TITAN x 4 (수냉식)
- 딥러닝 모델 학습 가속
- `--gpus all` 옵션으로 모든 GPU 활용

### 확장 가능성
- Dask Distributed로 워커 노드 추가 가능
- 모델 개수 무제한 확장 가능
- 데이터셋 크기 제한 없음

---

## 🛡️ 안정성 및 에러 처리

### 로깅 시스템
```python
# launcher.py
- 로그 파일: ./logs/{server_id}.log
- 5MB 이상 시 자동 백업
- 상세한 Exception 추적
```

### 재시도 메커니즘
```python
# state='E' 배치는 20분 후 재시도
# state='U' 배치는 1시간 후 재시도
```

### 트랜잭션 관리
```python
with self.dbHandler.engine.begin() as transaction:
    # FOR UPDATE로 행 잠금
    # 롤백 자동 처리
```

---

## 🔐 보안 고려사항

### 데이터베이스 인증
- 하드코딩된 비밀번호 → 환경 변수로 이동 권장
- SSH 키 기반 인증 고려

### SFTP 연결
```python
# models.py
SERVER_IP = "165.246.34.142"
USER_ID = "deep"
USER_PASS = "credit!0721"  # 환경 변수로 이동 권장
```

---

## 📚 주요 의존성

### 핵심 라이브러리
| 라이브러리 | 버전 | 용도 |
|----------|------|------|
| Django | 3.2 | REST API 서버 |
| TensorFlow | latest | DNN 모델 |
| PyTorch | latest | TabNet, DNN |
| XGBoost | latest | Gradient Boosting |
| scikit-learn | latest | ML 모델, 전처리 |
| Dask | latest | 분산 컴퓨팅 |
| scipy | 1.9.0 | 통계 및 최적화 |
| pandas | latest | 데이터 처리 |
| SQLAlchemy | latest | ORM |

---

## 🎯 향후 개선 방향

### 1. 코드 품질
- [ ] 하드코딩된 설정을 환경 변수로 이동
- [ ] 중복 코드 제거 및 리팩토링
- [ ] 타입 힌트 추가
- [ ] 유닛 테스트 작성

### 2. 기능 개선
- [ ] 실시간 예측 모드 구현
- [ ] 모델 버전 관리 시스템
- [ ] A/B 테스트 프레임워크
- [ ] 모니터링 대시보드

### 3. 성능 최적화
- [ ] 모델 캐싱 메커니즘
- [ ] 데이터베이스 쿼리 최적화
- [ ] 비동기 처리 확대
- [ ] GPU 메모리 최적화

### 4. 보안 강화
- [ ] 비밀번호 암호화
- [ ] API 인증/인가 강화
- [ ] 민감 정보 마스킹
- [ ] 감사 로그 추가

---

## 💡 사용 예시

### 1. 새로운 배치 작업 등록

```sql
INSERT INTO dc_batch (
    batch_desc, batch_memo, batch_param, mode,
    dataset_group, dataset_version, model_group, state
) VALUES (
    '신용평가 모델 학습',
    'XGBoost 모델 그룹 학습',
    '{"testSize": "0.3", "validationSize": "0.2", "randomState": "42"}',
    'train',
    'DATA001', 1, 'M100', 'I'
);
```

### 2. 학습된 모델로 예측

```sql
INSERT INTO dc_batch (
    batch_desc, train_batch_id, mode,
    dataset_group, dataset_version, state
) VALUES (
    '신용평가 예측',
    123,  -- 학습 배치 ID
    'predict',
    'DATA001', 1, 'I'
);
```

### 3. 앙상블 수행

```sql
INSERT INTO dc_batch (
    batch_desc, candidate_id, mode, state
) VALUES (
    '앙상블 신용평가',
    'C001',  -- 후보 모델 그룹
    'ensemble',
    'I'
);
```

---

## 📞 시스템 운영 정보

### 서버 정보
- **MariaDB 서버**: 165.246.34.133:3306
- **GPU 서버**: 165.246.34.142
  - SSH: 17022
  - Jupyter: 17888
- **ID**: deepcredit

### 디렉토리 구조
```
/opt/data/DeepCredit/
├── models/              # 학습된 모델 저장
│   ├── T{batch_id}/     # 배치별 폴더
│   │   └── {model_no}/{model_sub_no}.pkl/.h5
├── logs/                # 로그 파일
└── data/                # 데이터셋
```

---

## 🏁 결론

DeepCredit QHedge 시스템은 저축은행의 신용평가를 위한 종합 AI 플랫폼입니다.

**주요 강점:**
1. 다양한 ML/DL 모델 지원
2. 대규모 앙상블 처리 능력
3. 분산 컴퓨팅 기반 확장성
4. 자동화된 배치 처리 시스템
5. RESTful API 제공

**핵심 워크플로우:**
```
데이터 전처리 → 모델 학습(분산) → 예측 → 앙상블 → 결과 저장
```

이 시스템은 수백 개의 모델을 효율적으로 관리하고, 앙상블을 통해 높은 정확도의 신용 리스크 예측을 제공합니다.
