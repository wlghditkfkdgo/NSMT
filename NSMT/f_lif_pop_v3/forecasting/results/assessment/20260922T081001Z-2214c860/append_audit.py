from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T081001Z-2214c860';a=Path('f_lif_pop_v3/forecasting/results/assessment')/run;iv=json.loads((a/'inventory.json').read_text());s=Path(iv['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');v={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':{}}
for p,h in iv['files'].items():
 if hashlib.sha256((s/p).read_bytes()).hexdigest()!=h:v['snapshot_changed'].append(p)
 h2=hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
 if h2!=h:v['live_changed'][p]=h2
assert not v['snapshot_changed'];(a/'validation.json').write_text(json.dumps(v,indent=2))
entry='''

## 추적 감사 25 — NOW (예약 20260922T081001Z-2214c860)

### 관찰 범위·핵심 판정

**Logger 타입 오류의 수정과 실제 완료 CSV/JSON 보존을 확인했다. Validation의 궤적 고정/전체 forward 분리도 재현됐으며, η=.2는 같은 모델의 원 학습η 대비 recall MSE가 약1.86% 낮았다. 단일 seed 탐색 결과로 채택 확증은 보류한다.**

HEAD `5db6e315fec55de2962b843d9348deb1acfc18ed`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사24·기억·사전등록§2H·canonical17:03 기록을 복구하고 **17:10:50 KST**, 실제230파일을 `/tmp/nsmt_assessment_RUN`로 snapshot·SHA256 기록했다. Manifest 불일치0. 감사자 append는 연구 결과로 세지 않았다.

- train.py SHA256 `b2ab437e5cc9bf3ccdff11bf3153a27e84e003edcd534f39bbba90384e7c0a8f`.
- stage_decomposition.py SHA256 `afae7c9b6cbbf68d41c31103ec6a961146626c0843adcedc38fa10151da28124`.
- eta_intervention_split.py SHA256 `7e013ab47c716893a0be9d47b309f2f133f5e057003261674e6ec5a5c6532afb`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/RUN/source.diff), [reducer/logger probe](../f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probes.json), [분리 분석 probe](../f_lif_pop_v3/forecasting/results/assessment/RUN/split_probe.py), [정밀 결과](../f_lif_pop_v3/forecasting/results/assessment/RUN/split_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현 정확성 — 실제 재검사로 닫은 범위

**A10-LOG-COUNT-TYPE VERIFIED.** 새 reducer는 observations와 nonfinite count를 float로 반환한다. 실제 AST reducer의 absmax [1,9]→9/count2.0 결과를 **변환 없이** 실제 EpochLog.write→verbose→CSV 경로에 전달해 성공했다(TensorBoard 전송만 no-op). 감사24의 정수 타입 예외는 발생하지 않는다. 동일 probe에서 clipping 전후 계산·JSON 투영도 통과했다. Logger 자체가 모든 int를 지원하게 된 것은 아니며 현재 train 전달 타입을 고친 범위로 판정한다.

**A09-ABSMAX/OBSERVATIONS/POSTCLIP의 완료 artifact 저장까지 VERIFIED(해당 run).** `absmax4-170100`의 실제 CSV/JSON을 대조했다.

| 항목 | 완료 JSON 값 | CSV |
|---|---:|---:|
| grad_absmax_all | **1.4416555166244507** | 1.441656 |
| grad_absmax_all_observations | **8.0** | 8.000000 |
| grad_WQ_absmax | .004537541419267654 | .004538 |
| grad_WK_absmax | .012127026915550232 | .012127 |
| grad_total_norm_pre_max | 2.9906373023986816 | 2.990637 |
| grad_total_norm_post_max | **.999999669963376** | 1.000000 |
| clip_rate_all_batches / batches_seen | **1 / 8** | 1 / 8 |

Seed7/data_seed20260921,1epoch,train/val/test512/64/64,batch64,**g11_every1**이므로 이번 실행의8batch는 모두 관찰됐다. 이 run의 absmax를 전체8batch의 최대라고 읽는 것은 타당하다. 일반 설정에서 watch 간격이 커지면 여전히 표본 최대이며, 다른 run의 값까지 전체 batch 최대라고 소급하지 않는다.

이전 `absmax2-170000`은 평균으로 기록된 .8510724902153015 및 count 없음이 남아 있다. 두 run의 checkpoint SHA는 모두 `34c6965ca09cf823cff6cd3d9f3ea1725976888e5090d60b19f513cf539b6ab1`이고 각 실제 파일 hash와 일치한다. Absmax4의 기록 train source hash는 이번 snapshot과 같다. 같은 모델 결과에서 집계 방식만 달라졌다는 해석을 지지하되, **gradient 자체를 backward로 재생성한 것은 아니다**. Absmax2의 과거 평균 값을 수정하거나 최대값으로 재명명하지 않는다. Absmax3는 완료 JSON이 없는 것을 확인한 범위이며 종료 원인/실패 판정은 하지 않았다. WQ/WK 등 개별 관찰 수는 CSV에 있지만 최종 JSON은 global count만 보존한다는 잔여 제한도 유지한다.

**A02-STAGE-RANK VERIFIED.** 현재 실제 decompose 함수의 같은 test256 표본에서 chance **.21481773387565073**, QK rank **.19589813019628083**이 재현됐고 출력 설명도 무작위0.5 오기를 고쳤다. 이는 기존 test 표의 기준이며 validation에 수치를 그대로 이식하지 않는다.

**A08-ETA-INTERVENTION 분리 구현 확인.** 궤적 고정 함수는 학습η의 상태에서 p,b를 얻어 계수만 재계산하고, 전체 forward 함수는 η buffer를 바꾼 상태로 순환 모델을 다시 실행한다. 각 호출 후 buffer 복원, 전체 검사 후 학습 파라미터 hash 불변을 확인했다. 현재 sparse/cap=True/QK 학습η checkpoint에 한정한 확인이다. 함수가 다른 mode/no-cap 설정까지 일반적으로 맞는다고 판정하지 않는다.

### (b) 새 validation 결과·해석 제한

`qk2-140541` learnedη QK checkpoint를 CPU로 재평가했다. Seed7/data_seed20260921,train/val/test2048/256/256,batch64,θ=.38146987702788376,soft QK ε=.01,학습η=.026292985305190086. 실제 val256 전체(4batch)에서 원8행의 수치가 표시 정밀도까지 재현됐다. 다음은 추가한 원 학습η 기준선까지 포함한 전체 forward다.

| 평가η | val recall MSE | post-cap M_eff | max abs state |
|---|---:|---:|---:|
| 원 학습η | **.259146662958** | .136868366013 | 22.07177544 |
| 0 | .263483596918 | .133246449823 | 22.01334572 |
| .2 | **.254324974062** | .155864393992 | 22.48150253 |
| .5 | .260108542602 | .181384548465 | 43.09806061 |
| 1 | .293643652987 | .210006701282 | **656.12957764** |

η.2는 η0 대비 .009158622857(**3.4760%**), 원 학습η 대비 .004821688896(**1.8606%**) recall MSE가 낮다. 기존 “약3.5%”는 η0 대조를 뜻하며 원 학습 모델보다3.5% 좋아졌다는 뜻은 아니다. η1은 frozen G11 bound **305.0375175476074**를 초과해 불합격이며 **A05-ETA-INTERVENTION-BOUND는 수정/해결로 닫지 않는다**. Canonical이 해당 조건을 사후 상한 확대 없이 불합격으로 보고한 것은 수용한다. 이 val 결과에 test oracle 분모를 섞어 G를 만들지 않았다.

**궤적 고정 분석은 유용하지만 두 표기의 보완이 필요하다.**

1. 궤적 고정 표의 max|u|=22.0718은 모든 η에서 **원 학습η 궤적의 상태**다. 바뀐 계수를 재귀에 다시 넣지 않았으므로 G11 ‘OK’는 변경η 시스템의 안정성 판정이 아니다. `원 궤적 peak`와 `변경η G11: not run/N/A`를 구분해야 한다. 동일η1의 실제 전체 forward는656.13으로 실패한다. 계수만 계산하는 조건에서 MSE를 보고하지 않은 것은 적절하다.
2. Canonical17:03의 “p mass에 남은 차이는 cap”은 틀리다. 고정 궤적 η1에서 **p .296238626751 → 커널 가중 w .270998621964 (=pre) → post .274052337754**다. Cap은 오히려 정답 질량 비율을 약.00305 올렸으며, p와의 차이에는 `w=bp/Σbp` 변환이 있다. η0→1의 계수 변화는 혼합과 cap의 결합 효과다. **순수 pre-cap 희석**은 `M_pre=(1−η)M_kernel+ηM_w`로 별도 보고해야 한다. 고정 궤적의 pre는 .13324645/.16079688/.20212253/.27099862로 식과 일치한다.

전체 forward와 고정 궤적의 η1 post .21000670 vs .27405234 차이는 현재 checkpoint에서 상태 피드백을 포함한 개입의 차이를 뒷받침한다. 다만 이를 보편적인 분해 비율이나 새 학습의 성능 예측으로 일반화하지 않는다. 이전 서로 다른 궤적의 score/c hit 혼용을 철회하고 QK η.2 score 조건을 정정한 canonical 기록은 수용한다.

Validation 활용·동일 checkpoint η0 대조·실행 가능한 분석 코드의 추가는 검증 설계를 개선했다. **하지만 validation은 이미 학습 checkpoint 선택에 사용됐고, 후보η를 test에서 관찰한 뒤 다시 validation에서 비교했다.** 독립 확증 데이터는 아니다. 원 학습η보다 η.2가 낫다는 것은 이 한 seed의 탐색 결과이며 독립8seed paired CI/미사용 평가자료의 확증·새 학습은 **not run**이다. 두 분석 txt는4자리 표이며 원 실행 명령/정밀 결과/시작 source provenance는 여전히 부족하다. 현재 감사의 code·hash·정밀 JSON 재현 근거와 원 run provenance는 구분한다.

### (c) 채택 검증 우선순위

**① 구조/η 진단의 다음 단계로 η.2 후보를 유지하되, 채택은 사전등록한 선택 규칙과 독립 검증 뒤로 둔다.** 원 학습η·η0를 항상 포함하고, 후보 수/η 선택 기준/G11 탈락 규칙을 고정한 뒤 별도 미사용 평가 자료·독립seed로 비교할 것. 기록에는 score→p→w→pre→post와 전체 forward의 실제 상태 상한을 함께 남긴다. 현재 validation 개선은 후보 유지의 근거이며 새 확증 완료가 아니다.

그 다음 **②QK 조건 검증 → ③causal key → ④entmax → ⑤별도 Gram/ridge → ⑥별도 Delta**의 기존 순위를 유지한다. 이번 분해는 혼합·커널 가중·상태 피드백을 주요 점검 대상으로 지지하며 entmax 자동 승격이나 residual 즉시 삭제를 지지하지 않는다. State-derived key 개선은 조건부 후속 후보로 남기되 새로운 효능을 주장하지 않는다.

문헌은 이미 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 커널 가중·정규화/scale 구분과 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 순환 안정성 분석을 재사용했다. 이번 권고는 직접 수치 검토에 기반하며 새로운 논문 사실을 추가하지 않았다.

### 수행·보존

CPU torch1.12.0+cu113,2threads,seed7. AST reducer/logger/clip 합성 검사와 snapshot checkpoint forward만 수행했다. 학습/backward/optimizer/GPU·설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송 없음. 기존 문서 prefix를 보존하고 감사3문서 append 및 진단 artifacts만 작성했다. 실제 새 gradient 수치 재생성·absmax 실행의 성능 재평가·독립seed 통계는 not run이다. 감사 중 추가 변경은 validation.json에 남기고 다음 주기에 검토한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/reducer_probes.json > f_lif_pop_v3/forecasting/log/assessment/RUN/reducer_probe.log 2>&1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/RUN/split_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/split_probes.json > f_lif_pop_v3/forecasting/log/assessment/RUN/split_probe.log 2>&1
```

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
mem=f'''\n\n## 추적 갱신 — {now} (예약 감사25)\n\nHEAD5db6e315/snapshot17:10:50/230파일/manifest불일치0. A10-LOG-COUNT-TYPE float전달실제reducer→loggerCSV성공VERIFIED. Absmax4 1epoch512/64/64,g11_every1:8batch모두관찰,JSON/CSV max1.4416555166/count8/postnorm.99999966996확인;absmax2 .85107249는과거평균. 두checkpoint SHA34c6965c동일,gradient재생성not run. A02-STAGE-RANK실제함수.214817733876 VERIFIED. Val분리8행재현+원ηbaseline .259146662958,η0 .263483596918/η.2 .254324974062/η.5 .260108542602/η1 .293643652987. η.2원η대비1.8606%,η0대비3.4760%개선(단seed1). η1peak656.12958>305.03752 OPEN불합격유지. 고정궤적G11 OK는원궤적만/개입시스템안전아님;η1 p.296239→w/pre.270999→post.274052로남은차이cap단독설명정정필요. η.2후보유지/사전등록·독립검증필요;8seedCI·새학습not run. ASSESMENT감사25/results/assessment/{run}/.\n'''
log=f'''\n\n## {now} — 예약 추적 감사25: logger/완료 기록 검증 및 validation η 분리 재현\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD5db6e315fec55de2962b843d9348deb1acfc18ed,snapshot17:10:50/230파일,manifest불일치0. 감사commit/tag없음.\n- A10-LOG-COUNT-TYPE:현재float관찰수의실제reducer→logger→CSV재검사성공VERIFIED. Absmax4실제JSON/CSV max1.4416555166/count8/postnorm.99999966996보존확인;1epoch512/64/64,batch64,g11_every1로전8batch관찰. Absmax2평균.85107249소급재명명금지,두checkpoint SHA34c6965c동일. A02 rankchance.214817733876 실제함수VERIFIED.\n- CPUtorch1.12/2threads,seed7/data20260921,val256/4batch,QKε.01/θ.381469877,동일learnedηcheckpoint의고정궤적/전체forward8행재현. 추가원ηvalrecall .259146662958;η0 .263483596918/.2 .254324974062/.5 .260108542602/1 .293643652987. .2의원η대비1.8606%,η0대비3.4760%탐색개선. η1max656.129578>305.037518로안정성불합격유지.\n- 17:03보완:고정궤적max22.07/G11 OK는원궤적기준,변경η시스템G11은not run/N/A. 고정η1 p.296239→w/pre.270999→post.274052로cap은비율을올림;남은차이cap단독주장기각. 원학습η대조·정밀필드·실행시작provenance와독립seed검증필요. Candidate .2유지,기존③key/④entmax조건부순위유지.\n- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,source.diff,reducer_probe.py,reducer_probes.json,split_probe.py,split_probes.json,validation,append_validation);rawlogs forecasting/log/assessment/동일run/. Exactcommands·문헌은ASSESMENT감사25. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음. 실제gradient재생성·absmax성능재평가·독립8seedCI·새학습not run,감사3문서append.\n'''
checks={}
for p,t in [('docs/ASSESMENT.md',entry),('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log)]:
 p=Path(p);b=p.read_bytes()
 with p.open('ab') as f:f.write(t.encode())
 assert p.read_bytes()[:len(b)]==b;checks[str(p)]={'prefix_bytes':len(b),'prefix_sha256':hashlib.sha256(b).hexdigest(),'preserved':True}
(a/'append_validation.json').write_text(json.dumps(checks,indent=2));assert Path('docs/ASSESMENT.md').read_text().rstrip().endswith('<!-- assessment-watch:'+run+' -->');print(now,v)
