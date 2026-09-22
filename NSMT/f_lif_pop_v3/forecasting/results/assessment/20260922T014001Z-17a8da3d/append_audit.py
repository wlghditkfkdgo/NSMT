from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
run='20260922T014001Z-17a8da3d';out=Path('f_lif_pop_v3/forecasting/results/assessment')/run;stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');inv=json.loads((out/'inventory.json').read_text());hashes={Path(x['path']).name:x['sha256'] for x in inv['files'] if x['path'].startswith('f_lif_pop_v3/forecasting/')}
body=f'''

## 추적 감사 11 — {stamp} (예약 `{run}`)

### 관찰·고정 범위

기억·감사10·canonical PROJECT_LOG·사전등록 최신 **§2F D-AC/AD**를 읽었다. 사전등록 변경 감지는 이미 감사10에서 읽은 개정으로 새 규정으로 중복 집계하지 않는다. **10:40:39 KST**, HEAD **639d2293dcf12e3db3738641ff774a30e9f7439a**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7에서143개 실제 파일을 `/tmp/nsmt_assessment_{run}`에 복사·hash했다. Trigger와 대상 파일 hash는 전부 일치하고 HEAD만 감지 이후 변경됐다. ETT 원본도 재평가 전에 별도 사본을 고정했다(SHA256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066).

Source SHA256: ours `{hashes['ours.py']}`, layers `{hashes['layers.py']}`, train `{hashes['train.py']}`, test `{hashes['test.py']}`, config `{hashes['config.py']}`. Model/calibrate/utils는 감사10과 동일하므로 A10-RESTORE-STATS 등 해당 열린 이슈는 재검사 없이 유지한다.

증거: [inventory](../f_lif_pop_v3/forecasting/results/assessment/{run}/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/{run}/changes.diff), [ETT 사본 hash](../f_lif_pop_v3/forecasting/results/assessment/{run}/data_snapshot.json), [독립 재평가/진단](../f_lif_pop_v3/forecasting/results/assessment/{run}/readout_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/{run}/validation.json). Console은 forecasting/log/assessment/{run}/readout_probe.log에 저장했다.

### (a) 구현: oracle 수정 확인, drive는 추가 탐색 경로

**A12-ORACLE, VERIFIED(현재 공개 모델 경로):** truth_to_oracle_p에 kind를 전달해 copy 사건은 uniform, recall만 정답 집합에 균등 질량을 주도록 바뀌었다. Train/evaluate/diagnostics의 전달 배선도 소스로 확인했다. 동일256 sequence/8085 recall query에서 copy 비균등 사건이 **옛 규칙1243 → 현재0**, recall 정답 질량 오차0이며 공개 모델은 oracle에서 kind 누락을 ValueError로 거부했다. 이전610건은 다른 확인 표본의 수치로, 이번 전체256 sequence 수치와 혼동하지 않는다. 새 csvchk의 실제 checkpoint 평가도 현재 규칙에서 재현됐다. 감사자는 학습 함수를 실행하지 않았다.

**A08-DRIVE, VERIFIED(배선·인과성·gradient의 제한된 검사):** drive는 리셋 후 soma voltage인 analog와 달리 **soma 이전 학습된 가지 가중합**을 읽는다. 입력2개에서 aux live drive와 `(state * soma.weight).sum(-1)`의 차0, gradient 연결을 확인했다. 입력/soma weight/Q gradient norm은0.604835/1.133588/0.056158로 유한했다.20번째 patch 이후 입력만 바꿨을 때 앞20개 예측 차0이었다. 새 학습/optimizer step 없이 고정 checkpoint의 단일 역전파만 수행했다.

Drive와 analog는 학습 목적·gradient 경로까지 달라지는 **end-to-end 통제 조건**이다. Drive 개선만으로 ‘정보 손실이 soma에만 있다’거나 ‘가지가 정답을 충분히 저장한다’고 단정할 수 없다. Analog도 reset 이후 상태를 읽으므로 spike 비선형 하나만 분리한 실험은 아니다. 사전등록 §2F까지 drive 정의/판정 계획의 추가 개정은 없으므로 현 결과를 탐색적으로 취급하고, 정식 비교 전에 readout과 oracle 정책 버전을 고정할 것.

**A10-LOG, 부분 VERIFIED:** train이 logging/verbose 대신 EpochLog.write를 호출한다. csvchk-103953의 실제 best_log_0.csv에 epoch0/1의2행, train/val 및 selector 진단 열이 존재한다. CSV min val_loss0.371975는 JSON0.37197544992758175를6자리로 반올림한 값이다. 과거 run에 CSV가 생긴 것으로 소급하지 않는다. `log/final+result.csv`는 여전히 없고, 동적 *_nonfinite 열을 나중에 추가하면 고정된 CSV 열 목록에서 누락될 수 있으므로 프로젝트 표준 전체 준수는 OPEN이다.

### (b) 검증: 재현되는 결과와 정책 변경의 영향을 분리

**12epoch 탐색 readout 비교:** seed7/data_seed20260921, r2/k3, train/val/test2048/256/256, batch64, 제한 batch0, scale8/theta5.561343350061557, frozen input norm. 아래 **비-oracle** checkpoint들은 현재 고정 소스에서도 기록한 MSE와 차0이었다. Kernel_mass는 약0.130328894로 동일하다.

| readout | Full recall MSE | 학습η sparse recall MSE | sparse M_eff |
|---|---:|---:|---:|
| spike | 0.276338636158 | 0.272620149439 | 0.132756536374 |
| analog(리셋 후) | 0.283003154904 | 0.261619795824 | 0.133927164587 |
| drive(soma 전) | 0.274684306208 | 0.212159097751 | 0.134012595802 |

Drive sparse의 오차 감소는 **이 seed/예산의 관찰**이다. M_eff는 kernel보다 약0.003684만 높다. 낮아진 오차와 선택 검색 성공을 동일시하지 않는다. 여러 seed/CI·정식 O7·용량 통제 GRU 비교는 **not run**이다. 기존 Full 재실행은 같은 seed의 반복이며 독립 seed 표본으로 세지 않는다.

**A12-ORACLE-VERSION / A10-PROVENANCE OPEN:** 아래 세12epoch oracle checkpoint는 **옛 정책**으로 평가하면 저장 MSE가 차0으로 재현되지만, 현재 정책으로 평가하면 달라진다. 재평가에서 kind를 무시하는 옛 helper를 메모리에서만 대체했으며 원본 소스를 수정하지 않았다.

| checkpoint | 저장/옛 규칙 recall MSE | 같은 checkpoint·새 규칙 recall MSE |
|---|---:|---:|
| diag3 spike oracleη1 | 0.029680017366 | 0.035348084881 |
| diag3 analog oracleη1 | 0.042595001411 | 0.071973921201 |
| drive oracleη1 | 0.031637966001 | 0.045209064883 |

이는 모델 실패나 파일 손상이 아니라 **평가 정책 변경**이다. 새 규칙으로 재학습한12epoch 대조군은 이번 사본에 없다. Copy 정책 변경이 recurrent 상태를 통해 이후 recall에도 영향을 주므로 과거 headroom/G 지표와 새 정책 지표를 섞지 않는다. 특히 drive oracle JSON은 새 ours/test 소스 hash를 담고도 **옛 정책에서만 저장 수치가 재현**된다. 종료 시 live 파일 hash만 읽는 현재 방식으로는 실행 중 로드된 코드를 식별하지 못한다. 시작 시 코드 사본/정책 version을 고정하고 결과에 연결할 것. 새 규칙으로 실행된 csvchk는2epoch/256·64·64 기능 점검이며 recall0.377701122733을 재현했다. 이를12epoch oracle의 대체 성능값으로 쓰지 않는다.

**A10-HASH 실제 artifact 확인:** 새 parameter_hash_evaluated와 checkpoint_sha256이 있는 run은 모두 실제 로드한 값과 일치했다. 특히 drive oracle은 last≠best 조건에서도 evaluated hash가 맞았다. 기존 diag3 spike sparse의 옛 parameter_hash 불일치는 이미 알려진 과거 결함이며 보존한다. 한 가지 개선이 과거 artifact를 자동 정정하지 않는다.

**A12-ETT, VERIFIED(제한된 실제 평가):** ETTh1 checkpoint를 별도 데이터 사본으로 읽어 **test 첫3batch=192개 창/129024요소**에서 저장 MSE **0.911763752704**를 차0으로 재현했다. 전체 test 창은2785개다. train/val도 max_batches=3인1epoch 기능 실행이므로 ‘ETTh1 전체 검증 완료’로 기록하지 않는다. 같은 표본의 persistence1.664264396996, window_mean0.856615248951이며 모델이 window_mean보다 좋지 않다. 기존 loader는 scaler를 train 첫8640행에만 맞추고, target 경계는 train0–8640/val8640–11520/test11520–14400으로 분리한다. 전체 데이터 성능/반복통계는 not run이다. Selection diagnostics는 기본4batch를 읽어 평가의3batch와 범위가 다를 수 있으므로 표본 수/범위를 명시할 것.

**A12-TEST-SPLIT:** 실제 notest-103025 결과에서 test=null/test_skipped=true를 확인했고 checkpoint identity도 일치했다. 이 run의 test를 감사자가 새로 평가하지 않았다. 감사10의 호출 차단 검사와 합쳐 현재 배선 상태를 유지하며, 이미 본 다른 run의 test가 다시 미관측 데이터가 되는 것은 아니다.

### (c) 개선 방향과 남은 조건

- 새 oracle 계약과 drive를 **탐색 개정**으로 기록하고 같은 정책 버전에서 Full/sparse/고정η/oracle 비교를 맞춘다. 진행 중 결과와 과거 정책 결과를 분리한다. 최종 규칙은 validation에서 정하고 이미 반복 관찰한 test로 threshold·조건을 선택하지 않는다.
- Drive의 낮은 MSE가 선택 기능과 관련되는지 동일 예산·여러 seed에서 확인한다. 선택 질량/정답 hit/회상 유형별 오차를 함께 보고, 학습된 같은 표현에 대한 정책 개입과 별도 학습 readout 비교의 질문을 구분한다. 기존에 원문 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단 방향과 [Zoology/MQAR](https://arxiv.org/abs/2312.04927)의 recall 평가 맥락을 유지한다. 이번에 새 문헌 사실은 추가하지 않았다.
- A10-RESTORE-STATS의 frozen 통계 누락 거부, 실행 시작 source/정책 고정, 보정 호환 schema를 우선 해결한다. A09 표본 norm의 인과 해석 제한/epoch max|grad|·nonfinite 기록, A07-REGEN scaffold, 다중 seed·통계 요구는 계속 OPEN이다. 동일 결함 재검사는 not run.

### 실행·보존

CPU Python3.10/torch1.12.0+cu113/2threads/seed7. 사본 checkpoint 평가와 drive의 입력2개 gradient/인과성 검사, oracle helper 검사 및 CSV/소스 읽기만 수행했다. 새 학습·optimizer·GPU·환경 설치·모델/학습 소스 수정·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람/전송 없음.143개 사본과 모든 캡처 checkpoint의 현재 hash 일치, 기존 문서 prefix를 확인했다. 감사자 파일 외 연구 내용은 수정하지 않았다.

Exact command(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \\
/home/yschoi/.conda/envs/snn_recall/bin/python \\
f_lif_pop_v3/forecasting/results/assessment/{run}/readout_probe.py \\
/tmp/nsmt_assessment_{run} \\
f_lif_pop_v3/forecasting/results/assessment/{run}/readout_probes.json \\
> f_lif_pop_v3/forecasting/log/assessment/{run}/readout_probe.log 2>&1
```

<!-- assessment-watch:{run} -->
'''
mem=f'''

## 추적 갱신 — {stamp} (예약 감사11)

- Snapshot10:40:39/HEAD639d2293/143파일 + ETT CSV 고정. Prereg §2F 동일. A12-ORACLE 현재 kind 배선 VERIFIED(copy비균등1243→0/전체256sequence,recall질량오차0). 새csvchk2epoch 현재정책MSE재현.
- 과거12epoch oracle(spike/analog/drive)은 옛정책에서만 저장MSE차0. 새정책으로 같은checkpoint 평가시 recall .035348/.071974/.045209. 새정책재학습결과로세지않음. Driveoracle은새sourcehash기록에도옛정책만재현하므로 시작source/정책version고정필수(A10-PROVENANCE).
- Drive=소마전가지혼합,실제aux차0/gradient유한/causalprefix차0. Sparse recall spike.272620/analog.261620/drive.212159; drive Meff.134013 대kernel.130329로검색성공미입증. 1seed탐색;O7/8seedCI not run.
- A10-LOG 실제2행CSV확인,final+result.csv없음/동적nonfinite열누락위험잔여. Actualevaluated/checkpointhash일치,driveoracle last≠best확인. ETT첫3batch192창 MSE.911764재현(전체2785창아님),window_mean.856615. No-test결과test_skipped확인,감사자는해당test평가안함.
- A10-RESTORE-STATS/model동일로OPEN유지,A09과잉인과해석/A07재생성등도유지. ASSESMENT 감사11 및 results/assessment/{run}/ 참조.
'''
log=f'''

## {stamp} — v3 예약 감사11: readout·oracle 버전·실제 기능 결과 검토

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD639d2293dcf12e3db3738641ff774a30e9f7439a. Snapshot /tmp/nsmt_assessment_{run}(10:40:39,143파일),ETT CSV SHA f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066. 모델수정/새학습없음,감사commit/tag없음.
- CPU Python3.10/torch1.12/2threads/seed7,data_seed20260921,r2k3,2048/256/256,batch64,12epoch 기존checkpoint평가. Sparse recall spike.272620149439/analog.261619795824/drive.212159097751;각MSE차0. DriveMeff.134012595802/kernel.130328894. 1seed탐색이며성능우위/검색성공일반화보류;8seedCI not run.
- A12-ORACLE copyuniform수정 VERIFIED,현재정책copy비균등0/옛1243. 과거12epochoracle은옛정책에서만저장MSE재현. 새정책재평가 recall spike.035348084881/analog.071973921201/drive.045209064883은재학습아님. Driveoracle JSON의새sourcehash와실행정책불일치가 A10-PROVENANCE 재확인근거.
- Drive혼합값차0/유한gradient/앞20patch인과성차0. 실제CSV2행확인(A10-LOG부분VERIFIED),final+result.csv미생성. Hash분리새artifact모두일치/driveoracle last≠best. ETTmax_train/eval_batches3 기능실행test192/2785창MSE.911763752704재현,window_mean.856615248951;전체성능주장금지. No-testartifact확인,해당test미접근.
- Exact command와근거는 NSMT/docs/ASSESMENT.md 감사11, NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/readout_probe.py 및readout_probes.json/inventory/data_snapshot/validation; rawstdout forecasting/log/assessment/{run}/readout_probe.log. OMP/MKL2,LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib,snn_recall python으로snapshot인자실행. 새학습/optimizer/GPU/설치/git변이/프로세스중단없음,원본checkpoint/문서prefix보존. A10-RESTORE-STATS 등미수정이슈유지.
'''
for name,text in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 with Path(name).open('a') as f:f.write(text)
print(stamp,'appended')
