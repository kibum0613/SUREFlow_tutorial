# SUREFlow 서버 실습 및 재현 기록

작성일: 2026-07-04  
작성자: kbkim  
실습 주제: SUREFlow를 이용한 LIBERO Spatial 학습 및 평가 재현

---

## 1. 실습 목적

본 실습의 목적은 SUREFlow 논문 및 GitHub 코드 기반으로 LIBERO Spatial 벤치마크에서 로봇 조작 정책을 학습하고 평가하는 것이다. 로컬 WSL2 환경에서는 GPU 메모리 및 시스템 메모리 한계로 장시간 학습과 전체 평가가 어렵기 때문에, 연구실 공용 GPU 서버에 새 계정을 생성한 뒤 서버 환경에서 SUREFlow 학습 및 평가를 진행하였다.

본 실습에서는 다음 과정을 수행하였다.

- 공용 서버 접속 및 개인 계정 환경 구성
- Miniconda 및 conda 가상환경 구성
- PyTorch, CUDA, Mamba, causal-conv1d, robosuite, robomimic, LIBERO 관련 의존성 설치
- SUREFlow GitHub 코드 클론 및 실행 환경 구성
- LIBERO 데이터셋 다운로드 및 경로 설정
- robosuite, MuJoCo, LIBERO import 문제 해결
- tmux를 이용한 장시간 학습 실행
- 학습 완료 checkpoint 기반 LIBERO Spatial 평가
- 논문 결과와 재현 결과 차이 분석

---

## 2. 서버 환경

실습은 연구실 공용 서버에서 진행하였다.

| 항목 | 내용 |
|---|---|
| 사용자 계정 | `kbkim` |
| 작업 경로 | `/home/kbkim/projects/SUREFlow` |
| Conda 환경 | `sureflow` |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU 메모리 | 약 24GB |
| 시스템 메모리 | 약 256GB |
| CUDA Toolkit | `/usr/local/cuda-12.4` |
| PyTorch | `2.4.1+cu121` |

GPU 확인 명령어는 다음과 같다.

```bash
nvidia-smi
```

실습 당시 서버에는 RTX 3090 Ti GPU가 3장 존재하였으며, GPU 0에는 다른 사용자의 Python 프로세스가 일부 존재하였다. 따라서 학습과 평가는 주로 GPU 2를 지정하여 실행하였다.

```bash
CUDA_VISIBLE_DEVICES=2
```

---

## 3. Conda 환경 구성

개인 계정에서 Miniconda를 설치한 뒤 `sureflow` 가상환경을 생성하였다.

```bash
conda create -n sureflow python=3.10
conda activate sureflow
```

PyTorch는 CUDA 12.1 빌드를 사용하였다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

확인 결과는 다음과 같았다.

```text
torch: 2.4.1+cu121
cuda: True
gpu: NVIDIA GeForce RTX 3090 Ti
```

---

## 4. SUREFlow 코드 클론

SUREFlow 코드는 다음 저장소를 기준으로 클론하였다.

```bash
cd /home/kbkim/projects
git clone https://github.com/tanvirnwu/SUREFlow_IROS_2026.git SUREFlow
cd /home/kbkim/projects/SUREFlow
```

실제 실습은 `/home/kbkim/projects/SUREFlow`에서 진행하였다.

---

## 5. 주요 의존성 설치

Mamba 계열 모듈과 SUREFlow 실행에 필요한 패키지를 설치하였다.

```bash
pip install ftfy regex tqdm colorama h5py einops wandb omegaconf huggingface_hub
pip install matplotlib pandas scipy scikit-learn opencv-python
```

Mamba 관련 패키지는 PyTorch 및 CUDA 버전과 호환되도록 설치하였다.

```bash
pip install causal-conv1d
pip install mamba-ssm
```

설치 확인은 다음과 같이 수행하였다.

```bash
python - <<'PY'
import causal_conv1d
import mamba_ssm
print('causal_conv1d OK')
print('mamba_ssm OK')
PY
```

---

## 6. LIBERO 및 robosuite 관련 설치

SUREFlow 실행에는 LIBERO, robosuite, robomimic 등이 필요하였다. LIBERO-PRO requirements 전체를 그대로 설치하면 `transformers` 등 일부 패키지가 과도하게 다운그레이드되어 Mamba import가 깨질 수 있어, 필요한 패키지를 선별적으로 설치하였다.

```bash
pip install hydra-core==1.2.0 easydict==1.9 robomimic==0.2.0 thop==0.1.1-2209072238
pip install robosuite==1.4.0 bddl==1.0.1 future==0.18.2 cloudpickle==2.1.0 gym==0.25.2
```

설치 후 다음 모듈들의 import를 확인하였다.

```bash
python - <<'PY'
import robosuite
import robomimic
import bddl
import gym
print('robosuite OK')
print('robomimic OK')
print('bddl OK')
print('gym OK')
PY
```

---

## 7. robosuite 로그 권한 문제 해결

공용 서버에서 robosuite import 시 다음과 같은 문제가 발생하였다.

```text
PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'
```

원인은 `/tmp/robosuite.log`가 다른 사용자 소유 파일로 생성되어 있었기 때문이다. 공용 서버이므로 `sudo`, `chmod`, `chown` 등 시스템 전체에 영향을 주는 방식은 사용하지 않았다.

대신 개인 conda 환경 안의 robosuite 로그 파일 경로를 `/home/kbkim/tmp/robosuite.log`로 변경하였다.

```bash
mkdir -p /home/kbkim/tmp
```

Python site-packages 내 `robosuite/utils/log_utils.py`에서 `/tmp/robosuite.log`를 개인 경로로 패치하였다.

패치 후 robosuite import는 정상 동작하였다. 단, 다음 경고는 계속 출력되었다.

```text
[robosuite WARNING] No private macro file found!
```

이는 치명적인 오류가 아니라 robosuite 설정 파일 관련 권고 경고로 판단하였다.

---

## 8. LIBERO 설정

LIBERO import 시 `~/.libero/config.yaml`이 없으면 import 과정에서 dataset path를 입력하라는 prompt가 뜨며, 비대화형 Python 실행에서는 `EOFError`가 발생하였다.

이를 해결하기 위해 LIBERO config 파일을 직접 생성하였다.

```bash
mkdir -p /home/kbkim/.libero
mkdir -p /home/kbkim/datasets/robot

cat > /home/kbkim/.libero/config.yaml <<'YAML'
benchmark_root: /home/kbkim/projects/LIBERO/libero/libero
bddl_files: /home/kbkim/projects/LIBERO/libero/libero/bddl_files
init_states: /home/kbkim/projects/LIBERO/libero/libero/init_files
datasets: /home/kbkim/datasets/robot
assets: /home/kbkim/projects/LIBERO/libero/libero/assets
YAML
```

다만 평가 과정에서는 SUREFlow 내부의 `LIBERO-PRO` 경로를 우선 사용해야 했기 때문에 `PYTHONPATH`를 별도로 정리해야 했다.

---

## 9. PYTHONPATH 문제와 해결

평가 중 다음 오류가 반복적으로 발생하였다.

```text
ModuleNotFoundError: No module named 'libero.libero'
```

원인은 `PYTHONPATH`에 다음 경로가 섞여 있었기 때문이다.

```bash
/home/kbkim/projects/LIBERO/libero
```

이 경로를 넣으면 Python이 내부 `libero` 폴더를 최상위 package로 잡아버려 `libero.libero` 구조를 찾지 못한다.

따라서 평가 및 재실행 전에는 항상 다음과 같이 `PYTHONPATH`를 초기화하였다.

```bash
unset PYTHONPATH
export PYTHONPATH=/home/kbkim/projects/SUREFlow/LIBERO-PRO:/home/kbkim/projects/SUREFlow
```

정상 확인 명령은 다음과 같다.

```bash
python - <<'PY'
import importlib

m1 = importlib.import_module('libero.libero')
print('libero.libero OK:', m1.__file__)

m2 = importlib.import_module('libero.libero.benchmark')
print('benchmark OK:', m2.__file__)
PY
```

정상적으로는 SUREFlow 내부의 `LIBERO-PRO` 경로가 출력되어야 한다.

---

## 10. MuJoCo 버전 문제 해결

평가 시 다음 오류가 발생하였다.

```text
TypeError: mj_fullM(): incompatible function arguments
```

이는 `robosuite==1.4.0`과 설치된 최신 `mujoco` 패키지의 API 호환성 문제로 판단하였다. 해결을 위해 mujoco 버전을 낮추었다.

```bash
pip uninstall -y mujoco
pip install mujoco==2.3.7
```

서버에서 offscreen rendering을 위해 다음 환경변수를 사용하였다.

```bash
export MUJOCO_GL=egl
```

---

## 11. tmux 사용

장시간 학습 중 SSH 연결이 끊겨도 학습이 유지되도록 `tmux`를 사용하였다.

새 세션 생성:

```bash
tmux new -s sureflow100
```

세션에서 빠져나오기:

```text
Ctrl + B → D
```

다시 접속:

```bash
tmux attach -t sureflow100
```

세션 목록 확인:

```bash
tmux ls
```

학습 또는 평가가 종료된 뒤 세션을 정리할 때는 다음을 사용하였다.

```bash
exit
```

또는 필요한 경우:

```bash
tmux kill-session -t sureflow100
```

---

## 12. 데이터셋 구성

LIBERO Spatial 데이터셋은 다음 경로에 저장하였다.

```bash
/home/kbkim/datasets/robot/libero_spatial
```

데이터셋 다운로드 예시는 다음과 같다.

```bash
mkdir -p /home/kbkim/datasets/robot

huggingface-cli download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --include "libero_spatial/*" \
  --local-dir /home/kbkim/datasets/robot
```

config의 `DATA_ROOT`는 다음과 같이 맞추었다.

```python
DATA_ROOT = "/home/kbkim/datasets/robot/"
```

---

## 13. 학습 설정

초기에는 논문 조건과 최대한 동일하게 재현하기 위해 다음 설정을 확인하였다.

| 항목 | 논문 조건 | 실습에서 맞춘 값 |
|---|---:|---:|
| Epoch | 200 | 가장 최근 실험은 200 epoch 수행 |
| Batch size | 256 | 로그 폴더 기준 B256 설정 사용 |
| Inference sampling steps | 50 | 50 |
| Rollouts per task | 50 | 최종 논문 조건 기준 50, 중간 확인은 10 사용 |
| Max step per episode | 350 | 350으로 맞춤 |
| Demos per task | 70 | 70 |

실제로 학습한 checkpoint 경로는 다음과 같다.

```text
/home/kbkim/projects/SUREFlow/logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/20260703_192803/checkpoints/final_model.pth
```

폴더명은 `LO_E400_B256_TS100k`로 생성되었으나, 사용자가 최종적으로 보여준 평가 결과는 가장 최근에 수행한 `epoch 200` 학습 결과에 대한 평가이다. 따라서 본 보고서에서는 최종 평가 결과를 `epoch 200 학습 후 평가 결과`로 기록한다. 다만 폴더명이 `E400`으로 남아 있어, 추후 재현성을 높이기 위해서는 실행 시점의 config snapshot과 checkpoint 내부 metadata를 함께 저장하는 것이 필요하다.

checkpoint 목록 확인 명령어는 다음과 같다.

```bash
find /home/kbkim/projects/SUREFlow/logs -type f \( -name "*.pth" -o -name "*.pt" \) -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort
```

---

## 14. 학습 실행 명령어

최종적으로 사용한 기본 학습 실행 형태는 다음과 같다.

```bash
cd /home/kbkim/projects/SUREFlow
conda activate sureflow

unset PYTHONPATH
export PYTHONPATH=/home/kbkim/projects/SUREFlow/LIBERO-PRO:/home/kbkim/projects/SUREFlow

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python run.py --train_suite libero_spatial
```

---

## 15. 평가 실행 명령어

학습 완료 후 checkpoint를 지정하여 평가를 수행하였다.

```bash
cd /home/kbkim/projects/SUREFlow
conda activate sureflow

unset PYTHONPATH
export PYTHONPATH=/home/kbkim/projects/SUREFlow/LIBERO-PRO:/home/kbkim/projects/SUREFlow

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

CKPT=/home/kbkim/projects/SUREFlow/logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/20260703_192803/checkpoints/final_model.pth

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path "$CKPT"
```

평가 영상 저장 경로는 다음과 같이 출력되었다.

```text
/home/kbkim/projects/SUREFlow/logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/20260703_192803/checkpoints/eval_videos
```

---

## 16. 최종 평가 결과

최종 평가 결과는 다음과 같이 출력되었다.

```text
EVALUATION RESULTS

Overall Average Success Rate: 0.302

Per-Task Success Rates:
Task 0: 0.900
Task 1: 0.020
Task 2: 0.720
Task 3: 0.660
Task 4: 0.080
Task 5: 0.580
Task 6: 0.060
Task 7: 0.100
Task 8: 0.500
Task 9: 0.000
```

전체 평균 성공률은 30.2%였다.

과제별 편차가 매우 컸다. Task 0, 2, 3은 비교적 높은 성공률을 보였지만 Task 1, 4, 6, 7, 9는 낮은 성공률을 보였다. 특히 Task 9는 0.0으로 평가되었다.

---

## 17. 논문 결과와의 차이

논문에서는 LIBERO Spatial 성공률을 94.0%로 보고하였다. 반면 본 실습에서는 LIBERO Spatial 평가에서 평균 성공률 30.2%를 얻었다.

가능한 원인은 다음과 같다.

1. 논문과 완전히 동일한 training/evaluation protocol이 아닐 수 있음
   - 최종 실험은 epoch 200으로 수행하였다.
   - 그러나 폴더명이 `E400`으로 남아 있고, 중간에 config를 여러 번 수정했기 때문에 정확한 재현을 위해서는 실행 당시 config snapshot 확인이 필요하다.
   - 특히 batch size, rollout 수, max step, sampling step, URFlow 관련 설정이 논문 조건과 완전히 동일했는지 확인해야 한다.

2. checkpoint 선택 문제가 있었을 가능성
   - `find ... | tail -1` 방식으로 checkpoint를 자동 선택하면 원하는 epoch의 checkpoint가 아닐 수 있다.
   - 이후에는 checkpoint 경로를 직접 지정하는 방식으로 평가하였다.

3. PYTHONPATH와 LIBERO package 충돌
   - 외부 `/home/kbkim/projects/LIBERO`와 SUREFlow 내부 `LIBERO-PRO`가 섞이면서 import 오류가 발생하였다.
   - 이후 `unset PYTHONPATH` 후 SUREFlow 내부 `LIBERO-PRO`만 사용하도록 정리하였다.

4. MuJoCo / robosuite 버전 호환성
   - `mj_fullM()` API 오류가 발생하여 mujoco를 2.3.7로 낮추었다.
   - 이 과정에서 평가 환경이 논문 환경과 완전히 동일하지 않을 수 있다.

5. 평가 rollout 수 차이
   - 중간 평가에서는 `rollouts=10`, 즉 총 100 episodes로 평가하였다.
   - 논문 비교를 위해서는 task당 50 rollouts, 총 500 episodes 기준 평가가 필요하다.

6. action rollout의 누적 오차
   - 일부 task에서는 초반 동작은 유사하지만 장기 rollout 과정에서 작은 action 오차가 누적되어 실패할 가능성이 있다.

---

## 18. 현재 결론

SUREFlow 학습 및 평가 환경을 서버에서 구축하고, LIBERO Spatial 평가까지 실행하는 데 성공하였다. 다만 최종 성공률은 논문 보고 수치와 큰 차이를 보였다.

이번 실습의 주요 성과는 다음과 같다.

- 공용 GPU 서버에서 SUREFlow 실행 환경 구축 완료
- CUDA, PyTorch, Mamba, robosuite, LIBERO, MuJoCo 관련 의존성 해결
- tmux 기반 장시간 학습 실행
- 가장 최근 실험에서 epoch 200 학습 수행
- checkpoint 기반 평가 실행 성공
- LIBERO Spatial task별 success rate 산출
- 논문 재현 시 발생 가능한 환경 및 protocol 차이 확인

현재 결과는 `epoch 200 학습 후 평가 결과`이지만, 논문 성능과 차이가 크므로 완전 재현 성공이라기보다는 논문 코드 기반 서버 재현 실습 및 troubleshooting 기록으로 보는 것이 적절하다.

---

## 19. 향후 개선 방향

다음 단계에서는 논문 조건과 실습 조건을 더 엄밀하게 맞춘 뒤 재평가할 필요가 있다.

1. 새 실험 폴더에서 config를 완전히 초기화한 뒤 재학습
   - `EPOCHS = 200`
   - `TRAIN_BATCH_SIZE = 256`
   - `VAL_BATCH_SIZE = 256`
   - `NUM_SAMPLING_STEPS = 50`
   - `rollouts = 50`
   - `max_step_per_episode = 350`

2. checkpoint 자동 선택 금지
   - `tail -1` 방식 대신 정확한 checkpoint 경로를 직접 지정한다.

3. 학습 로그 저장
   - epoch별 training loss
   - validation loss
   - checkpoint 저장 시점
   - GPU memory 사용량

4. task별 실패 영상 분석
   - Task 1, 4, 6, 7, 9의 실패 원인을 영상으로 확인한다.
   - grasp 실패, object 접근 실패, placement 실패, 장기 rollout drift 등을 분류한다.

5. 논문 protocol 기반 full evaluation 수행
   - 총 500 episodes로 평가한다.
   - 결과를 논문 Table I의 LIBERO Spatial 94.0%와 비교한다.

6. 재현 실험 결과 표준화
   - 실험 날짜
   - commit hash
   - config snapshot
   - checkpoint path
   - evaluation command
   - task별 success rate
   - 영상 저장 경로를 함께 기록한다.

---

## 20. 실습 요약

이번 실습은 단순히 모델을 실행한 것이 아니라, 실제 연구 코드 재현 과정에서 자주 발생하는 문제를 단계적으로 해결한 과정이었다.

특히 다음 문제가 중요했다.

- 공용 서버에서는 `/tmp` 권한 문제가 발생할 수 있음
- robosuite, mujoco, LIBERO는 버전 호환성이 민감함
- `PYTHONPATH`가 조금만 꼬여도 같은 `libero` 이름의 package가 충돌함
- checkpoint 자동 선택은 실험 비교에서 위험함
- 논문 정확도와 비교하려면 학습 epoch, batch size, rollout 수, max step, sampling step을 모두 맞춰야 함

최종적으로 서버에서 SUREFlow 학습과 평가를 실행하는 전체 pipeline을 구축하였고, 가장 최근 epoch 200 학습 checkpoint를 평가하여 LIBERO Spatial 기준 평균 성공률 0.302의 결과를 얻었다.
