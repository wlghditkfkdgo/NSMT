# nsmt_base — NSMT 논문 기준 구현 (뉴런 교체 실험의 base)

논문 *Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer* 의 공개 코드(`origin/main`, commit `191f366c6`)를
이 저장소의 작업 규칙(`<model>/<task>/{코드, scripts/, log/, results/}`)에 맞춰 옮긴 폴더다. 방법 정리는 `NSMT/papers/NSMT_method_summary.md`,
코드·옵션 대응과 그 근거는 `NSMT/papers/NSMT_code_mapping.md` 를 본다.

## 출처
| 경로 | 출처 | 변경 |
|---|---|---|
| `forecasting/{train,test,ours,layers,config,utils}.py`, `data_provider/` | `origin/main:forecasting/` | 없음 (`model.py`만 아래) |
| `forecasting/model.py` | `origin/main:forecasting/model.py` | 논문 모델 외의 ablation/baseline import 를 선택적으로 바꿈 (파일 머리 주석) |
| `forecasting/{analysis,positional,noise_injector}.py` | 저장소 루트 `forecasting/` (공개 저장소에 빠져 있으나 `test.py`가 import) | 없음 |
| `forecasting/scripts/upstream_*.sh` | `origin/main:forecasting/scripts/` 원본 (sweep 잔재 포함, 참고용) | 없음 |
| `forecasting/scripts/run_ett.sh` | 새로 작성: 논문 Table 13 을 재현한 저장 실행의 옵션 | — |
| `anomaly_detection/`, `classification/` | `origin/main` 그대로 (루트의 `anomaly_detection/`는 공개본과 다름) | 없음, 미검증 |
| `*/dataset` | 공유 데이터셋 symlink (`NSMT/forecasting/dataset`, `NSMT/anomaly_detection/dataset`) | — |

## 실행 (NSMT/ 에서)
```bash
bash nsmt_base/forecasting/scripts/run_ett.sh ETTh1 96 42 0        # <DS> <PL> <seed> <gpu>
bash scripts/gpu_pool2.sh nsmt_base/forecasting/scripts/queues/<queue>.txt 0,1,2,3
```
환경: `snn_recall` (Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend). 출력은 `forecasting/log/<suite>/`(stdout, 실행별 `log/`·`model_state/`), `forecasting/results/<suite>/raw.txt`.
