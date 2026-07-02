# 실험 로그 정리

## 1. Mini subset smoke test

### 설정

```text
Dataset: LIBERO-spatial
Subset size: 100 samples
Epochs: 1
Evaluation: 1 task × 1 episode
```

### 결과

```text
Mean train loss: 약 1.2883
Checkpoint 저장 성공
Evaluation success rate: 0.000
```

### 해석

100개 sample은 파이프라인 확인용으로는 충분했지만, 실제 robot manipulation 성공을 기대하기에는 부족했다.

---

## 2. Mini subset 1000 samples, 3 epochs

### 설정

```text
Dataset: LIBERO-spatial
Subset size: 1000 samples
Epochs: 3
Evaluation: 1 task × 3 episodes
```

### 학습 결과

```text
Epoch 0: Mean train loss is 0.8127
Epoch 1: Mean train loss is 0.4664
Epoch 2: Mean train loss is 0.3665
```

### 평가 결과

```text
Task 0, Episode 0: FAILED after 300 steps
Task 0, Episode 1: FAILED after 300 steps
Task 0, Episode 2: FAILED after 300 steps
Overall Average Success Rate: 0.000
```

### 해석

loss는 안정적으로 감소했지만, closed-loop evaluation에서는 성공하지 못했다. 작은 action 오차가 rollout 중 누적되면서 task 실패로 이어진 것으로 보인다.

---

## 3. 전체 데이터 1 epoch

### 설정

```text
Dataset: LIBERO-spatial
Training samples: 57,750
Epochs: 1
Train batch size: 2
Total batches per epoch: 28,875
Evaluation: 1 task 기준
```

### 정상 시작 로그

```text
INFO:SUREFlow.main:Number of training samples: 57750
Epochs: 0/1
Batches: 0/28875
```

### 결과

```text
Checkpoint 저장 성공
Evaluation에서 1회 성공 확인
```

### 해석

Mini subset과 달리 전체 데이터 1 epoch 학습 후에는 실제 LIBERO closed-loop rollout에서 성공 사례가 나왔다. 이는 전체 데이터 학습이 단순 loss 감소뿐 아니라 실제 policy 성능 개선으로 이어질 수 있음을 보여준다.

---

## 4. 전체 데이터 3 epochs

### 설정

```text
Dataset: LIBERO-spatial
Training samples: 57,750
Epochs: 3
Train batch size: 2
Total batches per epoch: 28,875
Total batches: 86,625
```

### 현재 상태

```text
학습 진행 중
```

### 완료 후 평가 계획

1. checkpoint 확인

```bash
find ~/projects/SUREFlow/logs -type f -name "*.pth" | tail
```

2. checkpoint evaluation 실행

```bash
WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path /home/ubuntu/projects/SUREFlow/logs/SUREFlow_demo70_FiLM/LO_E400_B256_TS100k/<run_id>/checkpoints/final_model.pth
```

3. 평가 설정

```text
1 task × 3 episodes
save_video=True
render_image=False
use_multiprocessing=False
```

4. 확인 항목

```text
Success rate
Episode length
Rollout video
Object grasp 여부
Plate 위 배치 성공 여부
```

---

## 5. 메모리 관련 기록

### 발생했던 문제

Evaluation 중 WSL2 RAM 부족으로 `Killed`가 발생했다.

확인 로그:

```text
Out of memory: Killed process ... pt_main_thread ...
```

### 해결 방향

```text
1. evaluation task 수 축소
2. rollouts 축소
3. multiprocessing 비활성화
4. render_image 비활성화
5. 학습과 평가 분리
6. WSL2 memory / swap 설정 조정
```

### 권장 실행 전 확인

```bash
free -h
nvidia-smi
ps aux | grep python
```

필요하면 Windows PowerShell에서:

```powershell
wsl --shutdown
```

---

## 6. 현재까지 결론

```text
Mini subset은 파이프라인 확인에 적합하다.
실제 성공률 확인에는 전체 데이터 학습이 필요하다.
전체 데이터 1 epoch만으로도 성공 사례가 확인되었다.
평가는 학습보다 simulation RAM 사용량이 커서 별도로 작게 실행하는 것이 안전하다.
```
