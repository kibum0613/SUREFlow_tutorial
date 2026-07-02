# 코드 수정 패치 정리

이 폴더는 SUREFlow 실습 과정에서 수정한 핵심 코드 패치를 정리한다.

## 1. Python 3.8 타입 힌트 패치

### 문제

Python 3.8 환경에서 Python 3.10 스타일 타입 힌트가 에러를 일으켰다.

예:

```python
str | None
list[str]
tuple[str, ...]
```

### 해결

모든 `.py` 파일 상단에 다음 구문을 추가했다.

```python
from __future__ import annotations
```

적용 스크립트:

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

## 2. DataLoader mini subset 패치

### 문제

전체 데이터 학습이 무겁기 때문에 처음에는 subset만 사용해야 했다. 하지만 `self.trainset` 자체를 `Subset`으로 바꾸면 scaler에서 `get_all_actions()`를 호출할 때 에러가 발생했다.

### 잘못된 방식

```python
self.trainset = Subset(self.trainset, range(max_train_samples))
```

이 방식은 `self.trainset.get_all_actions()`를 사용할 수 없게 만든다.

### 올바른 방식

`self.trainset`은 그대로 두고, DataLoader에 들어가는 local 변수만 subset으로 바꾼다.

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

---

## 3. 전체 데이터 학습용 DataLoader 패치

전체 데이터 학습으로 전환할 때는 subset 코드를 제거한다.

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

메모리가 불안정하면 다음 설정으로 낮춘다.

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

## 4. 학습 후 자동 evaluation 생략 패치

### 문제

기본 `run.py`는 학습이 끝난 뒤 바로 simulation evaluation을 실행한다. 전체 데이터 학습 후 바로 evaluation으로 넘어가면 WSL2 RAM 부족 문제가 발생할 수 있다.

### 수정 위치

`run.py`에서 다음 부분을 찾는다.

```python
else:
    # Train the model if no checkpoint provided
    trainer.main(model)
```

### 수정 후

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

checkpoint 평가만 할 때는 다음처럼 실행한다.

```bash
WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path /home/ubuntu/projects/SUREFlow/logs/.../checkpoints/final_model.pth
```

---

## 5. Evaluation 축소 패치

`dataloader/libero_sim.py`에서 evaluation task 수를 1개로 제한했다.

```python
if self.benchmark_type == "libero_90":
    num_tasks = 1
else:
    num_tasks = 1
```

`configs/config.py`에서 rollout 수를 줄였다.

```python
rollouts = 1  # 또는 3
```

평가 episode 수는 다음과 같다.

```text
Total Evaluations = num_tasks × rollouts
```

---

## 6. 영상 저장 설정

`configs/config.py`:

```python
save_video = True
render_image = False
use_multiprocessing = False
```

저장 파일 확인:

```bash
find ~/projects/SUREFlow/logs -type f -name "*.mp4" | tail
```
