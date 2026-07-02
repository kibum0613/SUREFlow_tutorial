# SUREFlow + LIBERO-spatial 실습 가이드

## 1. 실습 개요

이번 실습은 SUREFlow를 WSL2 Ubuntu 환경에서 실행하고, LIBERO-spatial 데이터셋을 이용해 Vision-Language-Action 계열 모델의 학습과 평가 과정을 확인하는 것이 목적이다.

핵심 흐름은 다음과 같다.

```text
데이터셋 준비
→ 의존성 설치
→ 코드 호환성 패치
→ mini subset smoke test
→ 전체 데이터 학습
→ checkpoint 저장
→ checkpoint 기반 평가
→ rollout video 확인
```

---

## 2. 실습 환경

```text
Windows + WSL2 Ubuntu 22.04
Conda environment: libero
Project directory: ~/projects/SUREFlow
Dataset directory: ~/datasets/robot/libero_spatial
Training suite: libero_spatial
```

실행 전 기본 위치:

```bash
cd ~/projects/SUREFlow
conda activate libero
```

---

## 3. Mamba 관련 패키지 설치

초기 실행 시 `causal-conv1d`, `mamba-ssm` 설치 문제가 발생했다. CUDA Toolkit과 `nvcc` 인식 문제를 해결한 뒤 아래 명령으로 설치했다.

```bash
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

설치 확인:

```bash
python - <<'PY'
import causal_conv1d
import mamba_ssm
print("causal_conv1d OK")
print("mamba_ssm OK")
PY
```

정상 출력:

```text
causal_conv1d OK
mamba_ssm OK
```

---

## 4. Python 3.8 타입 힌트 호환성 패치

SUREFlow 코드 일부가 Python 3.10 스타일 타입 힌트를 사용한다. 현재 환경은 Python 3.8이므로 다음 구문을 전체 `.py` 파일에 추가했다.

```python
from __future__ import annotations
```

일괄 적용 스크립트:

```bash
cd ~/projects/SUREFlow

python - <<'PY'
from pathlib import Path

for p in Path(".").rglob("*.py"):
    parts = set(p.parts)
    if "__pycache__" in parts or ".git" in parts:
        continue

    text = p.read_text(errors="ignore")
    if "from __future__ import annotations" in text:
        continue

    lines = text.splitlines()
    if not lines:
        continue

    insert_idx = 0
    if lines[0].startswith("#!"):
        insert_idx = 1
    if insert_idx < len(lines) and "coding" in lines[insert_idx]:
        insert_idx += 1

    lines.insert(insert_idx, "from __future__ import annotations")
    p.write_text("\n".join(lines) + "\n")

print("added future annotations to all python files")
PY
```

---

## 5. 누락 패키지 설치

실행 중 `ftfy`, `colorama` 등이 없어서 아래 패키지를 설치했다.

```bash
pip install ftfy regex tqdm
pip install colorama
```

---

## 6. LIBERO-spatial 데이터셋 확인

학습 로그에서 데이터셋이 정상 로딩되는지 확인한다.

정상 로그 예시:

```text
INFO:root:The dataset is loading from /home/ubuntu/datasets/robot/libero_spatial
INFO:dataloader.libero_dataset:Loading demo: pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5
INFO:dataloader.libero_dataset:Loading demo: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
...
INFO:SUREFlow.utils.scaler:Training dataset size: target (62250, 7)
INFO:SUREFlow.main:Number of training samples: 57750
```

전체 학습 sample 수는 57,750개이다.

---

## 7. Mini subset smoke test

처음부터 전체 데이터를 학습하면 시간이 오래 걸리고 OOM 위험이 있다. 그래서 먼저 subset으로 smoke test를 수행했다.

`SUREFlow/main.py`의 `_setup_data_loaders()`에서 DataLoader에만 subset을 적용한다.

주의할 점은 `self.trainset` 자체를 `Subset`으로 바꾸면 안 된다는 것이다. `self.trainset.get_all_actions()`가 scaler 설정에 사용되기 때문이다.

예시:

```python
from torch.utils.data import Subset

max_train_samples = 1000

trainset_for_loader = self.trainset
if len(self.trainset) > max_train_samples:
    trainset_for_loader = Subset(self.trainset, range(max_train_samples))
    logging.info(f"Using mini train subset: {max_train_samples} samples")

self.train_dataloader = DataLoader(
    trainset_for_loader,
    batch_size=self.train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    prefetch_factor=None
)
```

Mini subset 학습 결과 예시:

```text
Epoch 0: Mean train loss is 0.8127
Epoch 1: Mean train loss is 0.4664
Epoch 2: Mean train loss is 0.3665
```

loss는 감소했지만 evaluation 성공률은 대부분 0이었다.

---

## 8. 평가 설정 축소

기본 evaluation은 task와 episode 수가 많아 RAM 사용량이 크다. WSL2 환경에서는 다음처럼 평가를 축소했다.

`configs/config.py`:

```python
rollouts = 1   # 또는 영상 확인용으로 3
save_video = True
render_image = False
use_multiprocessing = False
```

`dataloader/libero_sim.py`:

```python
if self.benchmark_type == "libero_90":
    num_tasks = 1
else:
    num_tasks = 1
```

평가 로그 예시:

```text
Task Suite: 1 tasks
Episodes per Task: 3
Total Evaluations: 3
Multiprocessing: Disabled
Real-time Rendering: Disabled
```

---

## 9. 영상 저장

영상 저장이 켜져 있으면 실행 초기에 다음 로그가 출력된다.

```text
Evaluation videos will be saved under /home/ubuntu/projects/SUREFlow/logs/.../checkpoints/eval_videos
```

저장된 영상 확인:

```bash
find ~/projects/SUREFlow/logs -type f -name "*.mp4" | tail
```

저장 경로 예시:

```text
logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/20260702_135954/checkpoints/eval_videos/libero_spatial/<task_name>/episode_0.mp4
```

---

## 10. 전체 데이터 학습 전환

Mini subset만으로는 성공률이 잘 나오지 않아 전체 데이터 학습으로 전환했다.

`SUREFlow/main.py`에서 subset 제한을 제거하고 전체 trainset을 사용한다.

```python
def _setup_data_loaders(self):
    """Setup training and validation data loaders."""

    trainset_for_loader = self.trainset

    self.train_dataloader = DataLoader(
        trainset_for_loader,
        batch_size=self.train_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )
```

더 안정적으로 돌리고 싶으면 다음처럼 보수적으로 설정한다.

```python
self.train_dataloader = DataLoader(
    trainset_for_loader,
    batch_size=self.train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    prefetch_factor=None
)
```

---

## 11. 전체 데이터 학습 config

`configs/config.py`:

```python
EPOCHS = 1
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
```

전체 1 epoch 기준 batch 수:

```text
57750 / 2 = 28875 batches
```

현재 확장 실험에서는 다음과 같이 3 epoch로 늘렸다.

```python
EPOCHS = 3
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
```

---

## 12. 학습 후 자동 evaluation 생략

전체 데이터 학습 후 바로 evaluation으로 들어가면 RAM 사용량이 커질 수 있다. 따라서 학습만 수행하고 종료하도록 `run.py`를 수정했다.

수정 위치:

```python
else:
    # Train the model if no checkpoint provided
    trainer.main(model)
```

수정 후:

```python
else:
    # Train the model if no checkpoint provided
    trainer.main(model)

    log.info("Training-only mode: skipping simulation evaluation.")
    log.info("state_dict saved in {}".format(model.working_dir))

    if run is not None:
        wandb.finish()

    return
```

이렇게 하면 다음 명령은 학습만 수행한다.

```bash
WANDB_MODE=disabled python run.py --train_suite libero_spatial
```

반면 checkpoint 평가만 하려면 다음처럼 실행한다.

```bash
WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path /home/ubuntu/projects/SUREFlow/logs/.../checkpoints/final_model.pth
```

---

## 13. 학습 실행

전체 데이터 학습:

```bash
cd ~/projects/SUREFlow
conda activate libero
WANDB_MODE=disabled python run.py --train_suite libero_spatial
```

정상 시작 로그:

```text
INFO:SUREFlow.main:Number of training samples: 57750
Epochs: 0/1
Batches: 0/28875
```

3 epoch인 경우:

```text
Epochs: 0/3
Batches: 0/28875
```

---

## 14. checkpoint 확인

학습 완료 후 checkpoint를 찾는다.

```bash
find ~/projects/SUREFlow/logs -type f -name "*.pth" | tail
```

또는 checkpoint 폴더 확인:

```bash
find ~/projects/SUREFlow/logs -type d -name "checkpoints" | tail
```

---

## 15. checkpoint 평가

```bash
WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path /home/ubuntu/projects/SUREFlow/logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/<run_id>/checkpoints/final_model.pth
```

평가 결과 예시:

```text
Task 0, Episode 0: SUCCESS after ... steps
Overall Average Success Rate: ...
```

전체 데이터 1 epoch 학습 후 evaluation에서 1회 성공을 확인했다.

---

## 16. RAM/GPU 모니터링

학습/평가 전 상태 확인:

```bash
free -h
nvidia-smi
ps aux | grep python
```

학습 중 모니터링:

```bash
watch -n 1 free -h
watch -n 1 nvidia-smi
```

WSL2를 완전히 초기화하고 싶으면 Windows PowerShell에서:

```powershell
wsl --shutdown
```

---

## 17. 현재 실험 상태

- mini subset 100 samples 학습 성공
- mini subset 1000 samples, 3 epoch 학습 성공
- 전체 데이터 1 epoch 학습 성공
- 전체 데이터 1 epoch checkpoint 평가에서 1회 성공 확인
- 현재 전체 데이터 3 epoch 학습 진행 중

---

## 18. 다음 실험 추천

1. 3 epoch checkpoint로 `1 task × 3 episodes` 평가
2. 성공이 나오면 `1 task × 5 episodes` 평가
3. 메모리가 안정적이면 `3 tasks × 3 episodes` 평가
4. 영상 저장 결과를 README에 추가

---

## 19. 실습 결과 해석

초기 mini subset 학습에서는 loss가 감소했지만 evaluation success rate는 0이었다. 전체 데이터 1 epoch 학습 후에는 closed-loop rollout에서 1회 성공이 확인되었다. 이는 단순한 training loss 감소뿐 아니라, 전체 데이터 학습이 실제 LIBERO simulation 행동 성능 개선으로 이어질 수 있음을 보여준다.
