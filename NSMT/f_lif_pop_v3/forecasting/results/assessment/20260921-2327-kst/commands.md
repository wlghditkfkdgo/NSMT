# 감사 재현 명령 — 2026-09-21

Cwd는 `NSMT/`. 모델 소스는 감사 시작 시 `/tmp/nsmt_assessment_20260921`에 복사하여 고정했다. `inventory.json`의 hash와 실제 소스를 대조한다. 해당 임시 사본은 로컬 산출물이며 Git에 복제하지 않는다. 원본은 commit `df3a3407b9ae653b4b1031320c4e8240b4c943a0`으로 식별한다.

## 수치 반례와 생성기

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3.py /tmp/nsmt_assessment_20260921 > f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/probes.json
```

Source snapshot, seed7, synthetic data RNG20260921, 조건별300 sequences 및 전류 보정256 sequences. Optimizer/GPU 실행 없음. 후속 실행은 새 폴더에 저장한다.

## 기존 gate 재실행

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python /tmp/nsmt_assessment_20260921/f_lif_pop_v3/forecasting/check_model.py --phase all > /tmp/nsmt_assessment_20260921/gates.txt
```

Exit0, 17 pass/0 fail/1 not run. 구조화한 결과는 `gates.json`; raw stdout은 로컬 `forecasting/log/assessment/20260921-2327-kst/gates.stdout`으로 복사 보존했다.

## CPU torch2 환경 가능 여부

```bash
/home/yschoi/.conda/envs/snn_jelly/bin/python -c 'import torch; x=torch.tensor([1.,2.],requires_grad=True); x.square().sum().backward(); print(torch.__version__, x.grad.tolist(), hasattr(torch,"compile"))'
```

관측: `2.11.0+cu130 [2.0, 4.0] True`, exit0. CUDA driver warning은 발생했다. `torch.compile` 자체 실행 및 spikeDE golden 실행은 하지 않았다.

## 실제 ETTh1 loader

다음 inline Python으로 `ett_loader.json`을 작성했다. 모든 split은 읽기 전용이며 원본 데이터는 변경하지 않았다.

```bash
LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib OMP_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0,'/tmp/nsmt_assessment_20260921/f_lif_pop_v3/forecasting')
from data_provider.data_loader import Dataset_ETT_hour
root=Path('forecasting/dataset/ETT-small').resolve();path=root/'ETTh1.csv'
raw=pd.read_csv(path).drop(columns=['date']).to_numpy(dtype=np.float64)
rows=[]
for horizon in (96,720):
 args=SimpleNamespace(seq_len=336,pred_len=horizon,root_path=str(root),data_path='ETTh1.csv')
 for split in ('train','val','test'):
  ds=Dataset_ETT_hour(args,split); start=ds.start+336; end=ds.end
  mean_error=np.abs(ds.scaler.mean_-raw[:8640].mean(0)).max()
  expected=((raw[start:start+horizon]-ds.scaler.mean_)/ds.scaler.scale_).astype('float32')
  rows.append({'horizon':horizon,'split':split,'count':len(ds),'target_first':start,'target_last_exclusive':end,'mean_max_error':float(mean_error),'first_target_exact':bool(np.array_equal(ds[0][1].numpy(),expected))})
out={'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':rows,'training':'not run'}
print(json.dumps(out,indent=2));Path('f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/ett_loader.json').write_text(json.dumps(out,indent=2)+'\n')
PY
```

## 문서의 구조화된 전체 표 검사

`reorganization_manifest.json`은 모든 move source/destination의 중복, 이동 파일 수 합, copy별 SHA256을 검사했다. `split_results`의 모든 원래 행 인덱스에 현재 raw.txt의 해당 행을 대응시켜 정렬·결합한 bytes의 SHA256이 `original_results_tune_sha256`과 같은지 검사했다. 결과는 `manifest_review.json`. 현재 불일치 copy는 `config.py`, `__pycache__/config.cpython-310.pyc` 두 개다. 과거 이동 당시 동일성 기록과 현재 변경 여부를 구분한다.

Canonical PROJECT_LOG의 Markdown table73개, table 행864개를 전체 파싱했다. 32/32/24/24/72/72/72/72개의 긴 개별 run 표에서 MSE/MAE finite, best epoch 범위를 확인하고, v2 Run ID의 horizon/variant별6행(dataset2×seed3) MSE를 재평균했다. 결과는 `historical_table_review.json`. 반올림된 문서 표의 재계산이며 과거 full precision checkpoint audit를 새로 실행한 것이 아니다. 추출 원본 table JSON은 임시 snapshot 폴더에 로컬 보존했다.

## 기록 규칙

모델/생성기/기존 calibration 수정, 학습, package 설치, branch/HEAD/index 변경, commit/tag/push는 하지 않았다. 이 폴더의 JSON은 텍스트 감사 결과다. Dataset/checkpoint/raw console log는 Git에 추가하지 않는다.
