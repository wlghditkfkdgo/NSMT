from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T080001Z-b5162cb6';a=Path('f_lif_pop_v3/forecasting/results/assessment')/run;iv=json.loads((a/'inventory.json').read_text());s=Path(iv['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');v={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':{}}
for p,h in iv['files'].items():
 if hashlib.sha256((s/p).read_bytes()).hexdigest()!=h:v['snapshot_changed'].append(p)
 live=hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
 if live!=h:v['live_changed'][p]=live
assert not v['snapshot_changed'];(a/'validation.json').write_text(json.dumps(v,indent=2))
entry='''

## 추적 감사 24 — NOW (예약 20260922T080001Z-b5162cb6)

### 범위·관찰 버전

**Absmax/관찰 횟수의 계산 수정과 post-clip norm 추가는 직접 검사로 확인했다. 다만 이 snapshot에서는 새 정수 관찰 횟수가 기존 logger를 깨뜨려 정상 CSV 저장에 도달하지 못한다.** 이는 모델 수치 실패가 아닌 기록 경로의 타입 결함이다.

HEAD `adb43e4ae47848d682a06b7ba5b067ed3d8015ab`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사23·문서 기억·사전등록§2H·canonical PROJECT_LOG를 읽고 **17:00:50 KST** 실제207파일을 `/tmp/nsmt_assessment_RUN`에 snapshot·hash했다. 이전 감사 이후 연구 변경은 train.py이며 이전 감사자 append는 연구 변화로 세지 않았다.

- Trigger train SHA256: `97f737bf6f63aba74897a1ad46d997040f06a4147b476855d09729b20d71ddc4`.
- **검사 snapshot train SHA256: `d5a6faa339fc918052c96ccfaec40b7c63e564068775155b0c7f9429abcac160`**.
- 감지 이후 수정되어 manifest 불일치1개였다. 감사 도중 live train은 다시 변경됐다(종료 hash는 validation.json). 본 판정은 위 고정 snapshot에 한정하며 뒤의 수정은 다음 주기 대상이다. `absmax2-170000` 디렉터리도 새로 관찰됐지만 이번 snapshot/trigger 밖의 실행 결과를 미완료나 실패로 분류하지 않았다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/RUN/source.diff), [실제 AST 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probe.py), [수치·logger 결과](../f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probes.json), [보존/추가 변화](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현 정확성 — 계산은 부분 VERIFIED, 기록 경로 잔여 OPEN

`train_one_epoch` 전체를 실행하지 않고 실제 AST의 reducer·clipping 블록·최종 train JSON 투영식만 추출해 CPU 합성값으로 검사했다.

| 검사 | 실제 결과 | 판정 범위 |
|---|---:|---|
| grad_absmax_all / WQ [1,9] | **9** | 기존 평균5 결함 수정 VERIFIED |
| WK absmax [2,10] | **10** | 관찰값 최대 계산 VERIFIED |
| grad_absmax_all_observations | **2** | 기존 1의 평균 대신 finite 관찰 수 VERIFIED |
| WQ norm [1,9], η [.1,.3] | **5**, **.2** | 평균 의미 유지 |
| absmax [1,NaN,9] | max9, observations2, nonfinite1 | finite count의 정의 확인 |
| 전 batch pre [.5,5,.5,5] | mean2.75, max5, count4 | 전 batch 통계 유지 |
| post [.5,1,.5,1], clip [0,1,0,1] | post max1, clip rate.5 | reducer 확인 |

**A09-ABSMAX의 관찰값 최대 reducer는 VERIFIED.** 다만 absmax 수집 자체는 여전히 watch batch에 한정된다. `grad_absmax_all`의 all은 해당 관찰 시점의 전체 파라미터를 뜻하며 epoch의 모든 batch를 조사했다는 의미가 아니다. 전체 epoch 최대를 요구하는 해석·운영은 여전히 잔여다.

**A09-OBSERVATIONS 계산 및 global count JSON 투영은 VERIFIED.** 기존 `grad_observations`는 제거되고 `grad_absmax_all_observations` 등 필드별 finite count가 생겼다. 실제 train JSON 투영식에서 global count2와 post max1이 보존됐다. WQ/WK 등 나머지 `*_observations`는 이 JSON whitelist에 아직 포함되지 않는다. 모두 NaN인 필드는 `_nonfinite`만 생기고 observations=0이 명시되지는 않는다. “실제 유효 gradient가 존재한 횟수”가 아니라 “코드가 기록한 finite 수치의 수”이며 gradient=None도 기존 로직상0으로 기록된다.

**A09-POSTCLIP 계산은 VERIFIED.** 작은 파라미터에 gradient [3,4]를 직접 주입해 실제 clipping→post 측정 블록을 실행하면 pre5→post **.9999998211860657**, clip1이다. [.3,.4]는 pre/post **.5**, clip0이다. Backward나 optimizer를 호출하지 않았다. Post는 norm의 epoch 최대이며 mean은 추가되지 않았다.

**A10-LOG-COUNT-TYPE OPEN — 이 snapshot의 출력 타입 불일치.** reducer가 `len(finite)`를 Python int로 반환하지만 기존 `EpochLog._verbose`는 float가 아닌 값을 `v.mean()`으로 처리한다. 실제 `EpochLog.write`에 위 reducer 출력을 전달하면 다음 예외가 발생한다.

`AttributeError: 'int' object has no attribute 'mean'`

실제 write→verbose 경로를 실행했고 TensorBoard 전송 부분만 no-op으로 대체했다. 예외는 CSV 쓰기 이전에 발생해 파일이 생성되지 않았다. **감사 입력만 float로 변환한 대조에서는 같은 logger가 CSV 쓰기에 성공**했다. 그 대조는 원본 소스 수정이나 실제 학습 성공이 아니다. Scalar 타입 처리를 logger에서 일관되게 하거나 count 전달 타입을 맞추고, 서로 다른 두 관찰값을 가진 reducer→logger→JSON 경로를 다시 확인해야 한다. 현재 진행 중 수정이 있을 수 있으므로 고정 snapshot의 확정 오류와 이후 버전의 상태를 구분한다.

### (b) 검증 과정·재현성 판정

이전1관찰 실행에서 드러나지 않던 mean/max·count 결함을 이번2관찰 반례가 구분했고, post-clip도 직접 계산했다. 그러나 계산 단위 검사의 성공만으로 완료 artifact 저장을 VERIFIED로 닫을 수는 없다. 현재 snapshot의 타입 오류가 그 반례다. 합성 `synthetic_logger.csv`는 진단용이며 연구 결과가 아니다.

새 완료 학습 결과·checkpoint·데이터 분리/통계 결과는 이번 감사 범위에서 확인하지 않았다. 새 성능 판단 근거 없음. **학습·완료 run CSV/JSON 보존 검증·validation η 통제 비교·독립8seed paired CI는 not run.** 기존 데이터 분리·대조군·사전등록 정의는 변경되지 않았다. A02-STAGE-RANK, A05-ETA-INTERVENTION-BOUND, A08 개입 해석, A10 provenance 등 감사23의 열린 조건도 재검사 없이 닫지 않는다.

### (c) 다음 개선 방향

우선 기록 경로의 타입을 맞추고 **관찰값 최대/finite count/post-clip→logger→최종 JSON**을 학습 없이 연결 검사할 것. sampled absmax와 all-batch norm의 관찰 범위를 표시하고 필요 시 전 batch absmax로 확장해야 epoch 최대라는 표현이 성립한다. 기존 완료 artifact의 과거 평균값을 소급해 최대값으로 재해석하지 않는다.

그 다음 감사23의 **동일 validation checkpoint에서 η0/학습η/격자, 고정 궤적 대조와 전체 forward 개입, G11 동시 보고** 우선순위를 유지한다. η1 개입의 상태 상한 초과를 무시한 채 M_eff만으로 채택하지 않는다. 새 logging 수정은 QK·entmax·Gram·Delta 후보의 효능 순위를 바꿀 근거가 아니다. 안정성 및 clipping 지표를 구분하는 근거는 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)을 재사용한다. 신규 문헌 주장·성능 주장은 추가하지 않았다.

### 수행·보존

CPU torch1.12.0+cu113,2threads,합성 진단으로 dataset/seed/학습 hyperparameter 해당 없음. 모델 학습/backward/optimizer/GPU·설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 접근/전송 없음. 감사 artifact와3문서 append만 작성하고 기존 문서 prefix를 보존했다. 최초 감사 스크립트에는 `.4.` 오타로 SyntaxError가 있었고 감사 코드만 수정해 재실행했다. 최초 raw log도 보존했으며 이 도구 실행 오류를 모델 실패로 분류하지 않았다. 재실행 exit0 및 의도한 logger 예외의 포착 결과는 artifact에 있다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probes.json > f_lif_pop_v3/forecasting/log/assessment/RUN/reducer_probe_retry.log 2>&1
```

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
mem=f'''\n\n## 추적 갱신 — {now} (예약 감사24)\n\nHEADadb43e4/snapshot17:00:50/207파일;train감지이후변경으로manifest불일치1,검사SHA d5a6faa339fc918052c96ccfaec40b7c63e564068775155b0c7f9429abcac160. 실제AST absmax[1,9]→9/WK[2,10]→10,finite관찰수2/globalcount JSON전달VERIFIED범위. Postclip주입[3,4]norm5→.999999821/[.3,.4] .5유지VERIFIED. 단watch표본absmax≠전batch최대. 신규A10-LOG-COUNT-TYPE OPEN:len(finite)int→EpochLog._verbose v.mean() AttributeError,CSV쓰기전실패;감사입력float변환대조만성공. 실제완료artifact검증not run. 감사중live재수정/새absmax2결과는다음주기. 기존A02/A05/개입해석OPEN및validation우선순위유지,새학습/성능판정/8seedCI not run. ASSESMENT감사24/results/assessment/{run}/.\n'''
log=f'''\n\n## {now} — 예약 추적 감사24: 집계 수정 부분 검증·logger 정수 count 호환 결함\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADadb43e4ae47848d682a06b7ba5b067ed3d8015ab. Snapshot17:00:50/207파일,train SHA d5a6faa339fc918052c96ccfaec40b7c63e564068775155b0c7f9429abcac160;trigger hash와다름/감사중추가변경은다음주기. 감사commit/tag없음.\n- CPUtorch1.12/2threads,실제AST만합성검사:absmax[1,9]→9/WK[2,10]→10/count2·globalcount JSON투영VERIFIED. Postclip gradient주입[3,4] norm5→.999999821/.3,.4는.5유지. Watch표본최대와전batch최대구분잔여.\n- A10-LOG-COUNT-TYPE OPEN:새int observations가실제EpochLog.write→_verbose에서'int has no mean'으로CSV쓰기전실패. No-op TensorBoard외실제경로,입력만float변환한진단대조성공. 소스수정없음;완료실행저장검증not run. 신규absmax2결과이번snapshot외라판정안함.\n- 우선reducer→logger→JSON타입/의미연결검사후감사23 validationη대조/G11순위유지. 새로운성능근거없음/학습·8seedCI not run. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,source.diff,reducer_probe.py,reducer_probes.json,synthetic_logger.csv,validation,append_validation);rawlogs forecasting/log/assessment/동일run/. Exactcommand·문헌은ASSESMENT감사24. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음,감사3문서append.\n'''
checks={}
for p,t in [('docs/ASSESMENT.md',entry),('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log)]:
 p=Path(p);b=p.read_bytes()
 with p.open('ab') as f:f.write(t.encode())
 assert p.read_bytes()[:len(b)]==b;checks[str(p)]={'prefix_bytes':len(b),'prefix_sha256':hashlib.sha256(b).hexdigest(),'preserved':True}
(a/'append_validation.json').write_text(json.dumps(checks,indent=2));assert Path('docs/ASSESMENT.md').read_text().rstrip().endswith('<!-- assessment-watch:'+run+' -->');print(now,v)
