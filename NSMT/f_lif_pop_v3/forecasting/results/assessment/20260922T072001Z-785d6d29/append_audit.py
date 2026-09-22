from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T072001Z-785d6d29'; art=Path('f_lif_pop_v3/forecasting/results/assessment')/run
inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
validation={'head_at_end':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':[]}
for p,h in inv['files'].items():
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=h:validation['snapshot_changed'].append(p)
 if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:validation['live_changed'].append(p)
assert not validation['snapshot_changed']
(art/'validation.json').write_text(json.dumps(validation,indent=2))
text='''

## 추적 감사 23 — NOW (예약 20260922T072001Z-785d6d29)

### 관찰 범위·핵심 판정

**단계별 분해6행과 η 개입8행의 수치를 재현했다. 다만 무작위 순위 기준·개입의 인과 해석을 정정해야 하며, QK 학습η checkpoint에 η=1을 적용하면 G11 상태 상한을 초과한다.** HEAD `adb43e4ae47848d682a06b7ba5b067ed3d8015ab`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 16:20:44 KST에 실제231파일(분석·소스·config/checkpoint·문서)을 `/tmp/nsmt_assessment_20260922T072001Z-785d6d29`로 snapshot하고 SHA256을 기록했다. Trigger hash 불일치0. 최신 사전등록은 §2H이며 forecasting Python 소스는 감사22 snapshot과 동일하다. 감사자의 이전 append는 새 연구 성과로 세지 않았다.

`stage_decomposition.py` SHA256 `4c384e3d846e9e2d6fca309a86f5d925c874b71b4f938df79791e0d8b23ebe6c`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/RUN/stage_probe.py), [전체 정밀 수치](../f_lif_pop_v3/forecasting/results/assessment/RUN/stage_probes.json), [추가 비교](../f_lif_pop_v3/forecasting/results/assessment/RUN/supplement_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현·진단 정확성

**A02 단계 분해의 실제 계수 재구성은 확인됐다.** 현재 API에서 state[n]=u(n+1), query는 갱신 전 u(n), history는 각 시점의 갱신 전 상태와 current다. 분석 함수의 인덱싱·kind>0 mask·D 평균→query 평균→sequence 평균이 이에 맞는다. 6개 checkpoint의 같은 test256sequence/8085recall query에서 원문 표의 모든 지표가 표시 정밀도4자리까지 재현됐다. 재구성 c와 실제 forward가 남긴 c의 차이는0이었다. 추가로 QK 고정η.2 조건을 검사했다. 원래 스크립트의 batches=4는 이 설정의 batch64/test256 전체를 덮지만 큰 데이터에 자동으로 일반화되지는 않는다.

**A02-STAGE-RANK OPEN — “0.5=무작위” 표기 오류.** 지표는 m개의 정답 중 최상위 정답의 0-based 순위를 n−1로 나눈 값이다. 무작위 permutation의 기대값은

`E[best rank/(n−1)] = (n−m)/((m+1)(n−1))`, n>1.

0.5는 m=1에 한정된다. 실제 query→sequence 집계 기준은 **.214817733876**이다. n=2..8의 모든 정답 부분집합35조건을 열거해 식과의 차이 ≤5.56e−17을 확인했다([검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/rank_formula_check.json)). 따라서 학습η 비QK rank .258861은 이 순위 기준에서 무작위보다 나쁘고, 학습η QK .195898은 더 좋다. Top1 hit의 별도 기준 .156583749693은 그대로다. 두 지표가 서로 다른 양상을 보이는 것은 모순이 아니다. 향후 각 query의 정답 수에 맞춘 순위 기준을 함께 기록할 것.

**A02 단계 사이에 커널 가중 정책을 추가할 것.** 현재 수식의 정확한 분해는

`w_j = b_j p_j / Σ b p`, `M_pre = (1−η) M_kernel + η M_w`.

| 학습η checkpoint | p mass | w mass | pre-cap | post-cap |
|---|---:|---:|---:|---:|
| 비QK | .2443073973 | .2245154195 | .1327565364 | .1327565364 |
| QK ε.01 | .2872716602 | .2622897784 | .1337985402 | .1337985402 |

혼합식은 float32 오차 범위에서 직접 확인됐다(각 query/unit 최대 오차의 집계 <5e−8). 작은 η가 커널 기준선과의 차이를 축소한다는 설명은 지지된다. 그러나 **p→w 커널 재가중과 w→pre 혼합은 별개**이며 모든 질량 감소가 η에서 생긴다고 쓰지 않는다. η=1 비QK 학습 모델도 p .137611→w/pre .121563→post .136899로 중간 변화가 있으므로 양 끝의 근접함만으로 “손실이 거의 없다”고 일반화하지 않는다. Score와 p의 top1 일치는 이 표의 순위 보존 확인이지 전체 선택·상태 동역학의 검증을 대체하지 않는다.

**A09-ABSMAX/OBSERVATIONS는 OPEN 유지.** train.py가 감사22와 같은 hash여서 실제 reducer/count/JSON 전달 결함에 수정 근거가 없다. 이번에는 같은 반례를 다시 실행하지 않았다. 다른 기존 OPEN도 새 근거 없이 닫지 않았다.

### (b) 새 개입 결과·검증 적절성

Snapshot의 learned 및 learned+QK 두 checkpoint에서 η buffer만 메모리 내 변경해 학습값/.2/.5/1의8조건을 CPU forward로 재검사했다. 모든 조건에서 trainable parameter bytes는 그대로였고 저장 checkpoint는 변경하지 않았다. 표의 recall MSE/M_eff/hit8행은4자리까지 일치한다. 설정은 seed7/data_seed20260921, train/val/test2048/256/256,batch64,최대12epoch,평가 batch 제한0이다. Split 재생성 hash는 서로 달랐다(추가 artifact에 수록). 이는 split 생성의 구분 확인이며 반복 test 선택 문제를 해소하지 않는다.

| QK 학습η checkpoint의 평가η | recall MSE | M_eff | c hit | 현재 궤적 score top1 | 최대 상태 |
|---|---:|---:|---:|---:|---:|
| .0262929853 | .273289752622 | .133798540174 | 0 | .340716419226 | 22.05676270 |
| .2 | **.268067504245** | .151894157073 | .145839501694 | .279855566571 | 22.29709053 |
| .5 | .273393590377 | .176371790477 | .262352495659 | .244263590342 | 47.29192734 |
| 1 | .306745081982 | .202267859521 | .341512628376 | **.189319316592** | **664.14135742** |

**A05-ETA-INTERVENTION-BOUND OPEN: η1의 실제 상태 초과를 확인했다.** 기존 config G11 bound는 **305.0375175476074**이며 재평가 max **664.141357421875**가 이를 넘는다(`within_bound=false`, 유한값). 이는 현재 test 고정 checkpoint 개입의 상태 기준 위반이며 학습 프로세스 중단을 관찰했다는 뜻은 아니다. M_eff .2023도 O7 .5에 미달한다. 기존 qk2 고정η1 학습 중단·g11chk 사건과 서로 다른 실험이다. 사후 상한 확대 없이 이 개입 조건의 안정성 불합격을 성능과 함께 보고해야 한다.

**A08-ETA-INTERVENTION 해석 정정:** 가중치 고정은 score/상태 고정과 다르다. η가 fractional recurrence에 들어가 u·query·history를 바꾸므로, QK score top1은 .340716에서 η1의 **.189319**로 달라졌다. η1의 최종 c hit .341513을 원래 궤적 score .340716과 “일치”한다고 해석하면 다른 상태의 통계를 혼합한다. b 재가중과 cap도 c의 순위에 영향을 준다. 현재 결과는 **η 변경의 전체 순환 모델 개입 효과**다. p/score 고정 하의 혼합·cap 직접 효과를 주장하려면 저장된 동일 궤적의 p,b에서 계수만 다시 계산하는 별도 대조가 필요하다. 상태 변화·총질량·cap·spike readout이 함께 바뀌므로 “dense 이력이 유용한 정보를 나르기 때문에 MSE가 나빠졌다”는 특정 매개 원인은 아직 미확인이다.

**Canonical16:13 §3의 조건 혼용:** QK 고정η.2 학습 recall .306119에 대응하는 score top1은 **.150596957017**, rank **.336628487234**다. 인용한 .1214는 비QK η.2 모델의 값이다. QK의 작은η 학습 모델이 더 나은 점수를 보였다는 관찰 방향은 유지되지만 정확한 조건을 짝지어야 한다. 또 QK/nonQK 두 학습 checkpoint의 차이는 θ·정규화·학습 궤적을 포함한 조건 비교이며 QK 변환만의 단독 인과 효과는 아니다.

η.2 개입의 full 대비 산술 G는 **.03426702117**, 상대 recall 감소는 **2.9931%**로 재현된다(기존 full .276338636158, oracle1 .034965688339). 그러나 full은 별도로 학습한 모델이다. 같은 checkpoint의 η0 개입 대조는 원8행에 없으므로 이 값만으로 그 모델 내부 선택의 기여를 고립하지 못한다. “선택이 도움이 된 첫 관측”도 과거 학습η의 양의 G(.0154056 등)가 이미 있어 부정확하다. 이번 결과는 기존보다 큰 **단일 seed 탐색 개선 관측**으로 기록한다.

Test가 반복 관찰되고 testη 후보가 비교됐다는 한계를 명시한 점은 적절하다. 현재 사전등록에 없는 학습/추론η 불일치 후보이며 확증 채택은 보류한다. V2의 MDE1.2%를 다른 과제·분산·설계의 v3 단일 seed에 옮겨 통계 근거로 삼지 않는다. **Validation 통제 비교·독립8seed paired CI·새 학습·효능 확증은 not run.** eta_intervention.txt에는 원 실행 코드/정확 명령/정밀 JSON이 없어 A10-PROVENANCE 잔여다. 이번 감사의 재구성 코드·정밀 결과·checkpoint hash를 제공하지만 당시 실행 provenance를 소급 보장하지 않는다.

### (c) 채택 검증 우선순위

**AUDIT-PRIORITY-01 보완: 다음 우선 작업은 같은 validation checkpoint에서 η 개입의 효과·안정성을 분리하는 것이다.** η=.2는 탐색 후보로 유지하되, 원 학습η·η0·고정η 격자를 같은 checkpoint/validation 표본에서 비교하고 score→p→w→pre→post 및 G11을 함께 보고할 것. 궤적 고정 계수 재계산과 모델 전체 forward 개입을 구분하면 실제 희석과 상태 피드백을 분리할 수 있다. Test에서 발견한 후보를 validation에서 다시 본다고 이미 관찰한 test가 새로운 확증 자료가 되는 것은 아니다. 후보와 η 선택 규칙을 사전등록 개정으로 고정한 뒤, 별도 미사용 평가 자료·독립seed 검증을 설계해야 한다.

기존 순서 **①η/혼합 구조 진단 → ②QK 조건 검증 → ③causal key → ④entmax → ⑤별도 Gram/ridge → ⑥별도 Delta**는 유지한다. 이번 표는 주된 손실이 p→w→혼합에서 나타나며 score top1→p top1은 보존된다. 따라서 **entmax를 자동 승격할 근거는 없다**. Top1 보존만으로 entmax가 불필요하다고 결론내리지도 않는다. 동일 validation에서 score는 충분한데 p의 질량 배분이 병목이라는 대조가 확보될 때만③/④의 조건부 순위를 바꾼다. η1의 큰 M_eff를 이유로 residual 제거 또는 η1 채택을 권하지 않는다.

문헌 근거는 앞서 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 커널·정규화/scale 구분, [Adaptive Sparse Transformers §3](https://aclanthology.org/D19-1223.pdf)의 entmax 변환 및 Jacobian, [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 순환 상태/gradient 안정성 분석을 재사용한다. 이번 우선순위 보완은 직접 수치 검사에 기반한 연구 판단이며 해당 논문이 이 모델의 효능을 보장한다는 주장은 아니다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7. 고정 checkpoint forward와 작은 조합론 검사를 수행했다. 학습/backward/optimizer/GPU·환경 설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송은 하지 않았다. 진단 코드/텍스트 결과와 감사3문서 append만 작성했다. 중간 inventory 비교 명령은 과거 schema의 list를 dict로 가정해 AttributeError가 났으며, 실제 snapshot bytes 비교로 바로잡았다. 이는 감사 보조 명령 오류로 모델 실패가 아니다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/RUN/stage_probe.py \
/tmp/nsmt_assessment_RUN \
f_lif_pop_v3/forecasting/results/assessment/RUN/stage_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/RUN/stage_probe.log 2>&1
```

추가 QK 고정η.2 비교는 같은 환경에서 `supplement_probe.py`에 같은 snapshot과 `supplement_probes.json` 경로를 인자로 주어 수행했으며 raw log는 동일 task의 `log/assessment/RUN/supplement_probe.log`다. Snapshot/현재 변화 대조는 validation.json에 기록했으며 감사 중 새 변화는 다음 주기가 처리한다.

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
mem=f'''\n\n## 추적 갱신 — {now} (예약 감사23)\n\nHEADadb43e4/snapshot16:20:44/231파일/manifest불일치0,forecasting소스동일. Stage6행·eta개입8행 CPUtest256/8085query 재현. 신규A02-STAGE-RANK:최상위복수정답무작위rank .214817734(0.5아님),조합열거검증. p→w=bp/Σbp→η혼합→cap분리:QK학습η .287272→.262290→.133799→.133799. η변경은가중치고정이나상태/score는변경(QKscore .340716→η1 .189319);c hit .341513과원score근접을동일신호증명으로못씀. QKη.2개입recall .268067504245/G .03426702117 탐색재현,η1peak664.141357>G11bound305.037518 신규A05-ETA-INTERVENTION-BOUND OPEN. QKη.2학습score .150597이며canonical .1214는비QK혼용. A09absmax/count잔여유지. 우선같은validation η0/학습η/격자+고정궤적대조/G11;entmax자동승격근거없음. Validation/8seedCI/새학습not run. ASSESMENT감사23 및results/assessment/{run}/.\n'''
log=f'''\n\n## {now} — 예약 추적 감사23: 단계·η 개입 재현, 순위 기준 정정 및 η1 상태 초과\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADadb43e4ae47848d682a06b7ba5b067ed3d8015ab,snapshot16:20:44/231파일,manifest불일치0. 감사commit/tag없음. Forecasting소스변경없음.\n- CPUtorch1.12/2threads,seed7/data20260921,test256/8085recallquery:stage6행및2checkpoint×4η개입8행표재현;QK고정η.2의추가stage검사. 가중치bytes보존. QKη.2개입 recall .26806750424450215/G .0342670는탐색개선. QKη1개입maxstate664.141357421875>305.0375175476074로상태기준위반,학습중단사건과구별(A05-ETA-INTERVENTION-BOUND OPEN).\n- 16:13 해석정정:복수정답best-rank무작위기준 .214817733876(0.5아님);p→bp/Σbp→η혼합분리. η변경은순환상태와score변경:QKscore .340716→η1 .189319이므로η1 c hit .341513과원score의근접을신호보존증명으로못씀. QKη.2학습score .150596957(.1214는비QK). 첫양의G주장/V2 MDE이식/이력유용성원인단정제한.\n- 동일validation checkpoint에서η0/원η/격자+고정궤적계수대조/G11우선. η.2후보보류검증,η1안정성실패병기;entmax자동승격없음. η개입원실행코드/명령/정밀기록미비잔여. A09absmax/count등OPEN유지. Validation/독립8seedCI/새학습not run.\n- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,stage_probe.py,stage_probes.json,supplement_probe.py,supplement_probes.json,rank_formula_check,validation,append_validation);rawlog forecasting/log/assessment/동일run/. Exact command·문헌링크는ASSESMENT감사23. Snapshot source만CPUforward,학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근·전송없음. 문서prefix보존·감사3문서append.\n'''
checks={}
for path,content in [('docs/ASSESMENT.md',text),('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log)]:
 p=Path(path);before=p.read_bytes()
 with p.open('ab') as f:f.write(content.encode())
 after=p.read_bytes();assert after[:len(before)]==before
 checks[path]={'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_bytes':len(before),'prefix_preserved':True,'appended_bytes':len(after)-len(before)}
assert Path('docs/ASSESMENT.md').read_text().rstrip().endswith('<!-- assessment-watch:'+run+' -->')
(art/'append_validation.json').write_text(json.dumps(checks,indent=2));print(now,validation,checks)
