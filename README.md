# DeepCredit
### 저축은행 신용평가를 위한 인공지능기반의 DeepCredit 신용예측 시스템 개발
![20220412091343](https://user-images.githubusercontent.com/6267896/162853362-fcbb1856-ad78-422b-bb22-765d1a30d1c1.png)

### 소스 버젼 관리(GitHub)
  + 폴더 설명
    + data : 데이터 파일
    + pre_processing : 오상민
    + data_imbalance : 김영훈
    + docs : 관련 문서 파일

#### ※ 소스는 하루에 한번 이상 커밋 해주세요.

### 데이터베이스(Maria DB)
  + SERVER IP : 165.246.34.133
  + SERVER PORT : 3306
  + ID : deepcredit

### GPU 서버(수냉식 - TITAN * 4)
   + SERVER IP : 165.246.34.142
   + DOCKER POST : 17022(ssh), 17888(notebook)
   + ID : deepcredit
   + docker run --name deepcredit --hostname DEEPCREDIT --restart always --gpus all --ipc host --privileged -p 17022:22 -p 17888:8888 -it djyoon0223/base:full

#### ※ GPU를 Full로 사용하거나 장기간 실행 할때에는 메시지 주세요.(재부팅 시킬수 있어서..)

### DJango 서버 구동
+ python manage.py runserver
+ http://127.0.0.1:8000/

### 파이썬 라이브러리
#### scipy 버전 맞지 않으면 모델 로딩시 오류 발생

+ pip install calculate
+ pip install numpy
+ pip install import_ipynb
+ pip install tqdm
+ pip install sklearn
+ pip install pandas
+ pip install dask
+ pip install sqlalchemy
+ pip install pymysql
+ pip install imblearn
+ pip install xgboost
+ pip install category_encoders
+ pip install tensorflow
+ pip install shap
+ pip install bayesian-optimization
+ pip install pytorch-tabnet
+ pip install wget
+ pip install matplotlib
+ pip install pytorch_tabnet
+ pip install torchvision
+ pip install paramiko
+ pip install asyncssh
+ pip install distributed
+ pip install GPyOpt
+ pip install scipy==1.9.0
+ 
+ pip install django==3.2
+ pip install djangorestframework
