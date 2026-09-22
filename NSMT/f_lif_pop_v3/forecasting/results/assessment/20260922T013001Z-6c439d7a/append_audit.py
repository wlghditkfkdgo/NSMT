from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
run='20260922T013001Z-6c439d7a';stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
out=Path('f_lif_pop_v3/forecasting/results/assessment')/run
body=f'''

## 추적 감사 10 — {stamp} (예약 `{run}`)

### 관찰 범위와 고정 증거

기억·감사09·사전등록·canonical 기록을 복구하고 trigger와 실제 파일을 대조했다. **10:30:36 KST**에111개 파일을 `/tmp/nsmt_assessment_{run}`에 snapshot/hash한 뒤 검사했다. 관찰 HEAD **73599f02a6cec05e485ea078669a9cb1723a83e7** 위 미커밋 수정, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 캡처된 trigger 대상 hash는 일치했다. 검사 중 추가된 사전등록 **§2F D-AC/AD** 및 canonical10:31 기록도 읽고 별도 postscript로 보존했다.

Source SHA256: train `146ab71656368df19556007eed1c4384a9c24414849ca6c8e4beebe995de200a`, test `c4558bfda69fd39ee433f9765d6b739548ed8761e97c26e0ec600158024eb1d0`, model `d5ba914759d6906f2d5293601237f409005bd8aed7382cc6d4311f7d92d6bcef`, layers `71c5e17e8b32fac4619f000356d80e1cd34c6198127833d0959f21fd0a92fc7d`, calibrate `f6e19b754fa08ca63382d448155e8eba3b2c9b4dffe4c12fd7604fa7d5701c95`。 전체 목록은 [inventory](../f_lif_pop_v3/forecasting/results/assessment/{run}/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/{run}/changes.diff), [후속 문서](../f_lif_pop_v3/forecasting/results/assessment/{run}/document_postscript_inventory.json), [재검사](../f_lif_pop_v3/forecasting/results/assessment/{run}/diagnostic_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/{run}/validation.json)에 있다.

감사 도중 config/layers/ours 및 analog checkpoint가 다시 변경됐다. **이번 판단은 캡처 버전에 한정**하며 진행 중 analog 출력을 확정 오류로 세지 않는다. 종료 전 관찰 HEAD d54caa1d07bd3a4f4de464852fac0bd8d2262421의 새 변경과 추가 ETT/no-test/Full/oracle 결과는 다음 주기 대상이다. 감사 문서 자체의 append는 연구 변화로 세지 않는다.

### (a) 구현: 진단·거부 경로는 수정 확인, 정규화 통계 복원에는 결함

**A02-DIAG / D-AC, VERIFIED(주요 지표 집계):** 현재 API로 기존 pilot-eta와 새 diag3 checkpoint를 각각 로드해 모든256 sequence/8085 recall query를 계산했다. `kind>0`, query→sequence 평균 및 전 embedding unit hit 계산을 소스와 수치로 확인했다. Batch128과16에서 M_eff/hit/kernel_mass 차이가 모두0이었다. 기존 pilot M_eff **0.13309303159470623**은 이전 독립값0.1330930309와 약7e-10 차이로 일치한다. 기존 파일의0.231467을 덮어쓰지 않는다. Hit는 정확히0이 아니라 **3.72888448e-5**여서 문서의0.0000은 반올림값이다. 기본 diagnostics는 여전히 첫4batch 표본이고, support/eta 등의 보조 지표는 배치별 평균이므로 전체 데이터·불균등 배치 집계로 확대 해석하지 않는다.

**A05-BRANCH, VERIFIED(실패 판정 배선):** 이전과 같은 sparse branch [0,1,1,1] 및 [.01,1,1,10] 주입은 이제 picked=null/exit1이고 진단 row가 남는다. Healthy는 승인/exit0, nonfinite·bound초과·발화율 미달은 거부/exit1도 재확인했다. 실제 새 보정을 수행한 것은 아니다. [branch 주입](../f_lif_pop_v3/forecasting/results/assessment/{run}/branch_paths.json), [기존 실패 경로](../f_lif_pop_v3/forecasting/results/assessment/{run}/failure_paths.json).

**A10-CAL / D-AD, 부분 VERIFIED:** 실제 loader가 기존 호환 파일을 반환하고 tau40/80/160/320 및 cue 불일치를 ValueError로 거부한다. require_calibration 기본True와 상한 없는 경우의 guard를 해당 AST 블록만 실행해 확인했다(optimizer/학습 미실행). 하지만 key_norm=frozen으로 변경해도 기존 none 보정이 승인됐으며, 입력 분포·정책별 적용 계약 전체를 검증한 것은 아니다. 최신 mtime 하나를 선택하는 방식은 artifact를 다시 복사하면 선택이 달라질 수 있다. 보정 ID/hash와 호환 schema를 명시하고 의도적으로 공유하는 축과 재확인이 필요한 축을 구분할 것. 정식 실행 전체가 검증됐다고 닫지 않는다.

**A10-KEYNORM-CKPT: legacy 정상 복원 VERIFIED / A10-RESTORE-STATS 신규 OPEN(P1).** 새 인자 기본값과 누락 key_mean/key_std의 항등 복원으로 옛 pilot config/checkpoint가 실제 로드되며 기존 MSE가 재현됐다. 그러나 OPTIONAL_BUFFERS에 **embedding.norm_mean/std까지 무조건 포함**되어 있다. 정상 diag3 checkpoint의 두 통계만 제거한 임시 사본을 input_norm=frozen으로 읽으면 로드가 승인되고 경고만 출력되며 **동일 입력2개에서 예측 최대차0.8706527948**이 발생한다. 일반 weight 제거는 거부됐다. 이는 원본 파일 손상을 주장하는 것이 아니라 **누락을 안전하게 거부하지 못하는 확정된 복원 결함**이다. Schema/version과 당시 config를 근거로 migration을 제한하고, frozen으로 학습한 통계가 없으면 실패하도록 할 것. eta_value도 항등 통계가 아니므로 누락 허용을 별도 계약으로 다룰 것.

**A12-ETT, VERIFIED(빈 truth 진단 경로):** CPU ETT 형태의 임의 입력2개/빈(B,0) truth에서 예외 없이 finite 상태 진단을 반환하고 M_eff/hit/kernel_mass=None, sequences/queries=0이었다. 실제 ETTh1 전체 재평가·학습은 **not run**; 담당10:31 문서의1epoch 완료 주장은 아직 이번 독립 검사 범위가 아니다.

**A12-TEST-SPLIT 및 A10-HASH, VERIFIED(학습 이후 배선):** 고정 train 소스에서 학습 이후 AST 블록만 실행했다. test와 data_provider를 호출 시 실패하는 sentinel로 대체했는데 --no-test 경로는 접근하지 않고 test=null/test_skipped=true로 완료했다. Last 모델을 메모리에서만 변경해 best와 hash가 다른 조건에서 last/evaluated/checkpoint SHA256이 각각 맞았다. 새 학습을 통한 end-to-end 검증은 not run이다. no-test에서도 parameter_hash_evaluated라는 이름을 쓰므로 실제 평가 여부는 test_skipped와 함께 읽어야 한다.

### (b) 검증과 새 실행 결과: 성능 주장은 보류

캡처된 diag-102225 및 errchk-102436 JSON에는 train/provenance가 없어 중간 저장물로 분류한다. 동일 metric을 독립 성공 반복으로 세지 않는다. errchk2는1epoch 기능 결과, diag3 sparse는12epoch 완료 결과다. 저장 소스 hash가 실행 종료 시 live 파일에서 계산되므로 새 test hash가 있어도 실제 저장 진단은 옛 집계인 경우가 있다. **실행 시작 시 source snapshot/hash와 완료 상태를 별도로 남겨야 한다(A10-PROVENANCE OPEN).**

| checkpoint 재평가 | 전체 MSE | recall MSE | 정정 M_eff | kernel_mass |
|---|---:|---:|---:|---:|
| 기존 pilot-eta,15epoch | 0.239489035682 | 0.269893671218 | 0.133093031595 | 0.130328893616 |
| 새 diag3 sparse,12epoch | 0.241255409829 | 0.272620149439 | 0.132756536374 | 0.130328893616 |

두 checkpoint의 MSE는 기존 기록과 최대2.8e-17 차이로 재현됐다. Seed7/data_seed20260921, r2/k3, train/val/test2048/256/256, frozen norm/scale8이다. Theta는 기존1.0 대 새5.561343350061557로 다르고 epoch 예산도 달라 **θ 효과의 통제 비교가 아니다**. 새 diag3의 옛 보고 M_eff0.231213은 새 집계에서0.132757로 내려가며 정답 추가 질량은 약0.002428, hit0이다. 새 성능 우위·선택 검색 성공의 근거가 되지 않는다. 독립8seed/CI·정식 O7 판정은 **not run**이다.

**A09-GRAD OPEN / canonical10:31 표현 정정 필요:** 새 support_size/score_std와 pre-clipping Q/K/eta_hat gradient norm은 유용하다. score.std(unbiased=False)는 history1에서도 유한값을 만들며 이번 forward/diagnostic JSON도 NaN 없이 저장됐다. 하지만 gradient는 g11_every 표본의 **L2 norm 평균**이며 사전등록 max|grad|가 아니다. 최종 JSON은 마지막 epoch의 표본 평균이고, 수집한 *_nonfinite 수도 payload 필드에서 빠진다. diag3 마지막 epoch의 singleton_frac0.06798, Q/K norm0.004664/0.013687, eta_hat norm0.003896과 errchk2의0.03826/0.005332/0.013558/0.000352는 **관측한 표본의 값**이다. 이를 근거로 ‘singleton 가설 기각’, ‘폭주도 소멸도 없다’, ‘eta 포화가 유력 원인’이라고 인과 결론을 내릴 수 없다. 전형적 singleton 점유가 낮았다는 제한된 관찰로 수정하고, query·시점·unit별 분포/최대값·nonfinite·실제 업데이트 크기 및 고정η 통제 결과로 확인할 것.

### (c) 개선 우선순위와 문헌 근거

1. **복원 신뢰성:** frozen 통계 누락 거부 및 버전별 migration을 먼저 해결한다. 보정 artifact ID/hash·완료 상태·실행 시작 소스·실제 평가 checkpoint identity를 함께 기록한다. 기존 파일은 보존하고 정정 진단을 별도 artifact로 연결한다.
2. **원인 분리:** D-Y의 고정η 격자와 spike/analog·Full/oracle 개입을 같은 데이터/예산에서 비교하고, gradient/선택 질량/회상 오차가 함께 바뀌는지 validation에서 살핀다. 이미 본 test에 맞춰 threshold나 유리한 조건을 고르지 않는다. 아직 완료되지 않은 analog 결과로 병목을 확정하지 않는다.
3. **진단의 범위:** 기존에 원문 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단·여러 seed 비교 방향, [Pascanu et al.(2013)](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간축 gradient/norm 분석을 유지한다. 한 번의 norm이나 support 비율을 인과 증거로 취급하지 않는다. 새 문헌 사실은 추가하지 않았다.

A10-LOG CSV 미호출, A12-ORACLE fallback 계약, A07-REGEN scaffold 및 미검증 통계/대조군 요구는 그대로 OPEN이다. 신규 학습·공식 gate 전체 재실행·GPU 검사는 not run이다.

### 실행·보존

CPU snn_recall Python3.10/torch1.12.0+cu113/2threads/seed7, 기존 checkpoint 평가·작은 forward·임시 파일 fault injection·학습 이후 AST/guard 검사만 수행했다. 모델/학습 소스 수정·train 함수/optimizer 실행·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송 없음. 사본111개 hash 일치, 평가한 spike checkpoint 보존 및 기존3문서 prefix를 확인했다. 진행 중 analog checkpoint의 외부 변경은 validation에 따로 남겼다.

Exact commands(cwd NSMT; 각 명령 앞 환경은 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib`, Python은 `/home/yschoi/.conda/envs/snn_recall/bin/python`):

```text
python scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_{run} f_lif_pop_v3/forecasting/results/assessment/{run}/failure_paths.json
python f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_probe.py /tmp/nsmt_assessment_{run} f_lif_pop_v3/forecasting/results/assessment/{run}/branch_paths.json
python f_lif_pop_v3/forecasting/results/assessment/{run}/diagnostic_probe.py /tmp/nsmt_assessment_{run} f_lif_pop_v3/forecasting/results/assessment/{run}/diagnostic_probes.json
```

<!-- assessment-watch:{run} -->
'''
mem=f'''

## 추적 갱신 — {stamp} (예약 감사10)

- Snapshot10:30:36/HEAD73599f0,111개 파일. 후속 사전등록 §2F D-AC/AD·canonical10:31도 별도 보존/검토. A02 recall-only query→sequence 및 모든unit hit 배선 VERIFIED: pilot M_eff.133093031595, batch128/16 차0. 기본 첫4batch 표본 범위 주의.
- A05 sparse dead/imbalance 거부 VERIFIED. A10-CAL tau/cue 거부·require guard 부분확인, key_norm 변경은 여전히승인. A12 빈truth 및 no-test tail, A10 best/last/checkpoint hash 배선 scoped VERIFIED; 새학습not run.
- Legacy pilot 정상복원/MSE재현. 신규 **A10-RESTORE-STATS OPEN(P1)**: frozen norm_mean/std 제거 임시checkpoint가 승인되고 예측최대차.870653. 버전없는무조건optional허용 금지. 일반weight누락은거부.
- 새 diag3 sparse12epoch/seed7 recall.272620149439,M_eff.132756536374,kernel.130328893616,hit0. 8seed/CI·성능우위미확인. 중간 JSON을 완료로 세지 않음; sourcehash는종료시live라실행시점증거부족.
- A09: 표본 L2 norm 평균은 max|grad|가 아니며 canonical10:31의singleton기각/폭주소멸없음/eta원인단정은과도. 문헌기반통제진단권고유지. 추가 analog/ETT/Full/oracle 및 config/layers/ours의 감사중변경은다음주기. ASSESMENT 감사10과 results/assessment/{run}/ 참조.
'''
log=f'''

## {stamp} — v3 예약 추적 감사10: 진단 집계·실행 계약 재검사

- 목적: A02/A05/A09/A10/A12 수정과 새diag3 결과 감사. Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7, 관찰HEAD73599f02a6cec05e485ea078669a9cb1723a83e7+미커밋수정. 감사자commit/tag 없음. Snapshot /tmp/nsmt_assessment_{run}(10:30:36 KST,111파일) 및 sourcehash inventory 고정.
- 모델/학습변경 없음. CPU Python3.10/torch1.12/2threads,기존seed7/data_seed20260921/r2k3/frozen-scale8 checkpoint 재평가. Pilot15epoch(theta1) recall.269893671218/M_eff.133093031595;diag3 12epoch(theta5.56134335) recall.272620149439/M_eff.132756536374. 각각2048/256/256 split, MSE차<=2.8e-17. epoch/θ달라통제비교아님. batch16/128 primary차0.
- A02 집계/A05 branch 실패배선 VERIFIED. Legacy정상복원,빈truth진단/no-test 학습후AST/hash분리 scoped VERIFIED. Train함수·optimizer미실행. Calibration tau/cue거부·기본guard확인,전체계약OPEN.
- **A10-RESTORE-STATS OPEN(P1)**: norm_mean/std누락임시checkpoint승인,예측최대차.8706527948. 원본손상주장아님. 일반weight누락거부. Schema/version 기반 migration 필요.
- **10:31 canonical항목 정정 권고:** singleton.038 및 gradient표본 norm만으로 singleton원인기각/폭주소멸없음/eta원인단정불가. 이번diag3 singleton.06798,gradQ.004664/K.013687/eta.003896은마지막epoch표본평균. 사전등록max|grad|·nonfinite기록과통제조건이필요. 성능·8seed/CI not run.
- Exact commands/세부수치/문헌링크는 NSMT/docs/ASSESMENT.md 감사10, artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/. 실행script는동폴더diagnostic_probe.py, 기존scripts/audit_calibration_failure_paths.py 및감사06 branch_failure_probe.py, 모두frozenroot 인자와CPU환경으로실행. source/model변경·학습·GPU·설치·프로세스중단·git변이없음. 사본hash/평가checkpoint/기존문서prefix보존. 다른세션진행중변경은다음주기.
'''
for name,text in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 with Path(name).open('a') as f:f.write(text)
print(stamp,'appended',run)
