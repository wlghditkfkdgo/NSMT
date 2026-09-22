from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');hs={r['path']:r['sha256'] for r in inv['files']}
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for r in inv['files']:
 p=Path(r['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=r['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 21 — NOW (예약 20260922T062001Z-d4a951fd)

### 관찰·증거

**실제 g11chk 실행의 G11 기록을 확인했고, 저장 checkpoint의 forward만으로 기록된 상태 초과를 독립 재현했다. 전체 batch clipping 지표의 CSV 보존도 확인했다.** HEAD **8ecf96ea7e1fb1badd112586b22134d8bf0d4536**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **15:20:49 KST**에 manifest·최신 감사/기억/§2H/canonical 기록·해당 checkpoint203파일을 `/tmp/nsmt_assessment_20260922T062001Z-d4a951fd`에 snapshot·hash했다. Trigger 불일치0, 연구 소스는 감사20과 동일하다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probe.py), [수치·기록 대조](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probes.json), [문서 수치 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/claim_checks.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/validation.json). Train SHA256 `TRAIN_HASH`, layers `LAYERS_HASH`. 검사한 checkpoint SHA256 **c798ba5d8636d7fbc1a35045e490c7eecaa7b0706d4edcf64ee277a6f2001dab**.

### (a) 구현 정확성 — 새 실제 기록의 검증 범위

**A05/G11-g11chk VERIFIED:** `g11chk-151001`의 `G11_violation.json`에서 run UUID **44e2b2349fc29e28**, zero-based epoch1/batch0, peak **332.54718017578125**, bound **305.0375175476074**, UTC06:10:06.157092를 확인했다. Run ID/UUID/mode/η1/QK True/ε.01/보정 파일은 저장 config와 모두 일치했고 bound도 지정 보정 JSON의 고정값과 정확히 일치했다. CSV는 완료된 epoch0 한 행이며 최종 성능 JSON은 없다. 따라서 이 run은 완료 성능으로 세지 않는다.

Epoch0의 best checkpoint를 CPU로 복원하고 원래 **train512sequence에 forward만** 수행했다. 표본 순서를 고정하고64개씩 처리했으며 모델/optimizer 갱신은 하지 않았다.

| 확인 항목 | 실제 값 |
|---|---:|
| Train sequence index381의 최대 상태 | **332.54718017578125** |
| 위반 JSON의 peak와 차이 | **0** |
| 전체512sequence 중 bound 초과 수 | **3** |
| 해당 checkpoint에서 전체 train 최대 상태 | **350.3652648925781** |

이는 기록된 상태 초과가 저장 모델·데이터에서 재현된다는 직접 근거다. 전체 학습/optimizer 경로 및 당시 shuffle된 batch0의 정확한 구성은 재현하지 않았다. 350.365는 고정 checkpoint로 모든 train 표본을 훑은 진단값이므로 기록 batch의332.547과 모순되지 않는다. 학습 중 모델이 계속 변하는 상황의 epoch 최대와도 구별한다.

**과거 사건과 구분:** 이번 run은 seed7/data_seed20260921, train/val/test **512/64/64**, 최대3epoch,batch64,g11_every10,softε.01/θ=.38146987702788376이다. 과거 qk2는 train2048/최대12epoch였으므로 같은 수치가 나왔다는 이유로 그 **별도 run의 원시 증거까지 복구됐다고 하지 않는다.** 이번 g11chk의 사건과 독립 forward 확인은 VERIFIED, 과거 qk2 사건의 정확한 원시 provenance는 별도 미확인 상태다. Suite/UUID를 포함해 구분한다.

### (b) 검증 적절성 — 전체 batch 통계의 실제 저장 확인, 인과·성능 판단 제한

**A09-CLIP-ALL-BATCHES의 CSV 보존 VERIFIED:** 실제 epoch0 CSV에 다음 네 필드가 존재한다.

| 필드 | CSV 값 |
|---|---:|
| grad_total_norm_pre_mean | 878689650.125 |
| grad_total_norm_pre_max | 3913563392 |
| clip_rate_all_batches | **1.0** |
| batches_seen | **8** |

512/64=8이며 batch 제한0과 일치한다. 감사20에서 수집·reducer를 검증했고 이번에는 실제 run의 CSV 저장을 확인했다. **이 새 run의 완료 epoch0에 한해서 8/8 batch가 clipping됐다는 기록 해석은 타당**하다. 과거 watch 표본 지표로 전체 batch를 일반화한 주장을 소급해 승인하지 않는다. Gradient 자체를 다시 backward로 계산한 것은 아니며 그 수치 재현은 not run이다. 중단으로 최종 결과 JSON이 없으므로 새 필드의 완료 JSON 보존도 아직 not run이다.

개별 원소의 epoch `max|grad|`와 post-clip norm은 여전히 별도 미구현/미확인이다. L2 norm의 batch 최대를 absmax 대용으로 부르지 않는다. G11 감시도 여전히 watch 시점에 한정돼 ‘모든 step에서 상태 상한이 보장된다’고 표현할 수 없다.

**Canonical15:10의 정정은 대체로 수용하되 두 문장을 보완할 것.**

- Hard/soft의 x≈0 Jacobian bound가 같다고 **전체 변환이 같거나 개선 원인이 ε뿐임이 증명되지는 않는다.** 같은 ε=.01, x=(.02,0,0,0)에서 hard 출력 첫 성분은1, soft는 **.8944271909999159**다. 동일ε에서 형태만 바꾼 통제 실험은 아직 not run이며, 형태 효과와 ε 효과의 분리는 계속 필요하다. 인과 단정을 철회한 취지는 유지한다.
- 고정η.2의 QK G는 **−.1233793961**, η.5는 **−.1771581586**이다. Canonical의 “0.2(−.1772)”는 η.5 수치와 바뀌어 인용됐다. 학습η G가 양수라는 정정은 맞다.

**성능:** 새 완료 recall/test 결과는 없다. 실제 상태 초과는 이 η1/seed/설정에서 안정성 기준에 걸렸다는 증거이며 아이디어 전체 실패나 QK 정규화의 보편적 실패가 아니다. 중단 checkpoint를 최종 성능으로 평가하지 않았다. 사전등록 임계값/분할/대조군 정책의 새 변경은 없고, 독립8seed/paired CI·효능 확증은 **not run**이다.

### (c) 다음 방향

고정 checkpoint의 동일 validation 표본에서 **score→p→pre-cap→post-cap 질량·순위를 연결**하는 기존 우선순위를 유지한다. 이 경로 진단에 상태 최대/G11 초과 여부와 전체 batch norm·clipping 기록을 함께 붙여, 검색 개선과 안정성 변화를 분리할 것. 이번 초과에 맞춰 보정 상한을 사후 확대하지 않는다. Score 표현이 병목이면③ causal key, p 변환에서 손실되면④ entmax를 앞당기는 조건부 규칙을 유지한다.

근거는 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간에 따른 gradient/clipping 분석, [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 정규화·거리/scale 연결을 재사용한다. 이번 새 판단은 직접 기록 대조와 forward 수치 검사에 기반하며 새 문헌 사실·효능 보장은 추가하지 않았다. A10-PROVENANCE·LOG/개별 absmax·A02 실제분포 연결/A07-REGEN 등 미검사 잔여는 유지한다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7. Epoch0 checkpoint 고정 forward와 문서 산술만 수행했다. 학습/backward/optimizer·GPU·환경 설치·모델/학습 소스 수정·git 변이·진행 프로세스 중단·다른 세션 대화 접근/메시지 전송은 하지 않았다. 기존 파일·checkpoint·문서 prefix를 보존하고 감사3문서에 append했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probe.py \
/tmp/nsmt_assessment_20260922T062001Z-d4a951fd \
f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T062001Z-d4a951fd/event_probe.log 2>&1
```

<!-- assessment-watch:20260922T062001Z-d4a951fd -->
'''.replace('NOW',now).replace('TRAIN_HASH',hs['f_lif_pop_v3/forecasting/train.py']).replace('LAYERS_HASH',hs['f_lif_pop_v3/forecasting/layers.py'])
mem=f'''

## 추적 갱신 — {now} (예약 감사21)

HEAD8ecf96ea7e1fb1badd112586b22134d8bf0d4536/snapshot15:20:49/203파일,manifest불일치0/소스변경0. G11실제g11chk UUID44e2b2349fc29e28 epoch1batch0 peak332.54718017578125>305.0375175476074/config·보정일치. Epoch0best고정 CPUtrain512forward에서sequence381 peak동일차0,3개초과/전체max350.3652649. 새g11chk사건VERIFIED,train2048옛qk2원시provenance대체아님. 실제CSV전체8batch premean878689650.125/max3913563392/clip1/count8확인,CSV보존VERIFIED;gradient재계산/완료JSON not run. 개별absmax/postclip잔여. Hard/softzero bound동일≠형태효과없음(xnorm.02 eps.01출력1vs.894427);canonical Gη.2는−.123379(−.177158은.5). 새성능결과없음/8seedCI not run. ASSESMENT감사21/results/assessment/20260922T062001Z-d4a951fd/.
'''
log=f'''

## {now} — 예약 추적 감사21: 실제 G11 기록·checkpoint 상태 초과 독립 확인

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD8ecf96ea7e1fb1badd112586b22134d8bf0d4536,snapshot15:20:49/203파일,manifest불일치0/소스변경0. 감사commit/tag없음.
- 실제g11chk-151001 UUID44e2b2349fc29e28,seed7/data_seed20260921,512/64/64,batch64,최대3epoch,η1/softε.01/θ.381469877/G11bound305.0375175. Eventepoch1batch0 peak332.54718017578125/config·보정일치. CPUtorch1.12/2threads에서epoch0best checkpoint고정train512 forward로sequence381동일값차0,3sequence초과/전체max350.3652649. 학습/optimizer 재현아님;새사건VERIFIED,옛2048seq qk2 원시근거대체아님.
- 실제epoch0 CSV pre-norm mean878689650.125/max3913563392/clip_rate_all_batches1/batches8로전체8/8기록확인. A09 CSV보존부분VERIFIED,실제gradient재계산/최종JSON보존not run. 개별absmax/postclip잔여유지.
- Canonical15:10보완:hard/softlocal bound같음≠동일ε형태효과없음,eps.01/xnorm.02 출력1vs.894427. Gη.2 −.1233793961,−.177158은η.5. 그외인과단정철회취지수용. 새성능평가/8seedCI not run,동일validation경로진단+안정성기록우선유지.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/(inventory,event_probe.py,event_probes.json,claim_checks.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/event_probe.log. Exactcommand·문헌은ASSESMENT감사21. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now,json.dumps(check,ensure_ascii=False))
