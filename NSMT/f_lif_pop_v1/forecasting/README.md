# f-LIF population: first forecasting experiment

`model_v1/forecasting`의 파일 역할과 함수 이름을 따르는 작은 실험이다.
확정된 기억 사용 의미는 **증거 보강**이며, 실제 모델은 직접 막전위 검색을 수행한다.
분수 미분 방정식의 정확한 f-LIF 구현은 아니다.

## 코드를 읽는 순서

1. `config.py`: `parse_arguments`, `Config.set_args/save_arg/print_info`.
2. `model.py`: `LOAD_MODEL['myModel']` → `ours.py`의 `myModel`.
3. `ours.py`: patch 분할 → `Embedding` → population spikes → forecasting head.
4. `layers.py`: 현재 입력 충전 → 과거 검색 → 증거 가산 → 발화/reset → 저장.
5. `data_provider/`: 기존 ETT-hour 12/4/4개월 분할과 train-only StandardScaler.
6. `train.py`: `train_one_epoch`, `val_one_epoch`, `train` 및 validation checkpoint 선택.
7. `test.py`: best checkpoint의 전체 test MSE/MAE 및 기억 개입 평가.
8. `utils.py`: neorecall_v1의 `EpochLog`, `EarlyStopping`를 필요한 import만으로 옮겼다.

전체 저장소 소스/config/script 314개(98,854줄, 고유 내용 202개)의 구조 조사는
`results/source_review.json`에 경로·hash·정의·import와 함께 기록했다. 모든 소스를
행별로 수동 검증했다는 뜻은 아니다. model_v1/neorecall의 모델·설정·학습·평가·로그와
ETT loader를 상세히 읽고 이 실험에 필요한 부분만 구성했다. 기존의 미사용 import,
관련 없는 auxiliary loss, 시간축 BN, 시뮬레이션 반복, cross-window memory는 가져오지 않았다.

## 첫 실험의 고정 설계

- `x [B,336,C] → patch [42,BC,8] → Linear(8,32) × 2 → PopulationLIF [42,BC,32,4]`.
- 시간축은 실제 8개 관측으로 구성된 patch. 각 window마다 상태/기억을 초기화한다.
- 채널별 파라미터 공유, 상태는 별개. Gaussian input coding이나 별도 temporal attention 없음.
- Input은 train-only 표준화. Window norm/BN/LN 없음: 입력 크기와 patch 인과성을 유지한다.
- `tau=[2,4,8,16]`, `beta=exp(-1/tau)` 고정. 균일 대조군은 이 beta의 평균을 K번 반복.
- 입력 전류는 같은 logical neuron의 K 구성원에 공유. 학습 projection에는 bias가 있다.
- 현재 pre-retrieval membrane의 Q와 과거 post-reset membrane의 K를 학습된 공유 K×K
  행렬로 변환하고 cosine/temperature(.25)의 softmax로 조회. 값은 원래 막전위다.
- Q/K identity 초기화, gate=Linear(K,1)의 sigmoid(초기 .5), gamma=.05.
  `v=u_bar+gamma*gate*memory`. Gate는 logical neuron별 값이며 파라미터는 공유한다.
- threshold=1, subtractive reset `u=v-s.detach()`, Sigmoid surrogate(alpha4), history는
  detach하지 않아 full BPTT. 첫 시점 기억 기여 0. 주입 계수에 `max(beta)+gamma<1`을
  강제한다. 이는 bounded input/convex memory/nonexpansive reset의 보수적 크기 제어이며,
  full training gradient stability나 원래 f-SNN robustness theorem의 증명이 아니다.
- K spikes를 head까지 유지. `Linear(32*4,32) → flatten(42*32) → Linear(...,96)`.
  보조 last head는 마지막 patch의 32차원만 동일한 horizon으로 예측한다.
- 같은 head/seed의 4조건은 trainable parameter 초기값/개수 동일. Retrieval off의
  Q/K/gate는 미사용 파라미터다. Nominal capacity는 같아도 effective capacity는 다르다.
- AdamW lr .001/wd .01, MSE, gradient clip1, batch128, 최대10epoch/early-stop3,
  ReduceLROnPlateau(valMSE factor.5/patience1), 최소 valMSE checkpoint만 test 평가.
- 주 실험: ETTh1/ETTh2 × homogeneous/heterogeneous × retrieval on/off × seed7/13/21 =24.
  Last head: 같은 8조건의 seed7만 추가. 보조 결과는 단일 seed 탐색 결과로 해석한다.
- 모든 test window의 MSE/MAE, persistence/window-mean 기준값. Retrieval 모델은 같은
  checkpoint에서 off/uniform/recent memory 개입을 전체 test에 평가한다. 인과적 변수
  중요도나 재학습 대조와 동일하지 않다. 내부 진단은 첫 test8window만 측정한다.

## 실행과 저장

NSMT 디렉터리에서:

```bash
bash f_lif_pop_v1/forecasting/scripts/run_ett.sh --suite ett-first-20260914
# 단일 실행 예: 새로운 suite 이름을 사용한다.
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/train.py --suite my-new-run --data ETTh1 --seed 7 --heterogeneous --retrieval
```

`log/<suite>/<dataset>/<YYMMDD>/<date+config>/seed<seed>_<head>_<variant>/` 아래:

```text
logargs.txt
log/best_log_0.csv
log/final+result.csv
log/train_0/                 # TensorBoard (local)
log/val_0/                   # TensorBoard (local)
log/history.json
log/provenance.json
log/horizon_metrics.csv
log/forecast_example.json
model_state/config.pt        # local
model_state/best+model.pt     # local, raw state_dict
```

CSV/TensorBoard epoch는 기존 template처럼 0-based. CSV는 6자리, results JSON에는
full precision/정확한 command/환경/source hash/초기 parameter hash/데이터 hash를 보존한다.
Checkpoint/raw events/stdout은 로컬, text 결과/설정/코드는 Git에 보존한다.
원래처럼 발생하지 않은 energy/ops를 만들어 기록하지 않는다.
Canonical session 기록은 저장소 루트 `docs/PROJECT_LOG.md`이며 NSMT/docs의 연결을 유지한다.
