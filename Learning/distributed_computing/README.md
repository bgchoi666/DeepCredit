# 분산 처리를 위한 예제 코드

# 1. Usage
```bash
## 1. Server 설정(3.1 참고)
server_config.ini 생성


## 2. Cluster 생성
$ python cluster_start.py


## 3. 작업 수행
$ python example.py
$ python example_connection_error_handled.py  # client.gather() error is handled


## 4. Cluster 해제
$ python cluster_stop.py 
```

# 2. Example python script code
- [example.py](example.py)
- [example_connection_error_handled.py](example_connection_error_handled.py)
- [example_connection_error_handled.ipynb](example_connection_error_handled.ipynb)


# 3. File structure
```
distributed_computing
├── README.md
├── cluster_run.py
├── cluster_start.py
├── cluster_stop.py
├── common.py
├── example.py
├── example_connection_error_handled.py
├── example_connection_error_handled.ipynb
└── server_config.ini (private)
```

### 3.1 `server_config.ini`
Cluster를 구성하는 server 정보가 저장된 파일
- `ip`, `port`, `username` 등을 저장
- 비공개 file 혹은 DB로 관리하여 server의 정보가 online 상에 upload 되지 않도록 주의 (`.gitignore`에 추가)

e.g. 
```
[server1]
host=123.456.78.910
ssh_port=10022
username=root
scheduler_port=18786
gpus=4


[server2]
host=123.456.78.911
ssh_port=10022
username=root
scheduler_port=18786
gpus=4

...
```

### 3.2 `cluster_run.py --cmd <command> [--ini <server config path>]`
모든 server들에 병렬적으로 명령을 실행시키는 python script
- Arguments
    1. `--cmd` : 수행할 명령어 (복수의 명령을 실행시킬 경우, `;`를 구분자로 사용)
    2. `--ini` : server 정보가 저장된 ini 파일 경로 (default: `server_config.ini`)

### 3.3 `cluster_start.py [--scheduler <id of scheduler server>] [--worker <ids of worker server>] [--ini <server config path>]`
Cluster를 연결하는 python script
- Arguments
    1. `--scheduler` : scheduler server id (default: `server1`)
    2. `--worker` : worker server ids (복수의 server를 사용할 경우, ` `를 구분자로 사용) (default: `server1 server2 server3 server4`)
    3. `--ini` : server 정보가 저장된 ini 파일 경로 (default: `server_config.ini`)

### 3.4 `cluster_stop.py [--scheduler <id of scheduler server>] [--ini <server config path>]`
생성된 cluster를 구성하는 server들 간의 연결을 끊는 python script
- Arguments
    1. `--scheduler` : scheduler server id (default: `server1`)
    2. `--ini` : server 정보가 저장된 ini 파일 경로 (default: `server_config.ini`)

### 3.5 `common.py`
Utility function 등이 있는 공용 module

### 3.6 `example.py`
분산 처리 예제 python script

### 3.7 `example_connection_error_handled.py`, `example_connection_error_handled.ipynb`
분산 처리 예제 python script (connection error가 발생하는 경우 사용)
- worker port 접근이 제한되어 있는 경우, `client.gather()`이 사용불가하기 때문에 파일 전송 등을 통해 worker에서 실행된 결과를 받아와야 한다.