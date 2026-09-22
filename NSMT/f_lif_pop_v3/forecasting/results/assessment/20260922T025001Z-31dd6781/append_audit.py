from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781')
now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for row in inv['files']:
 p=Path(row['path']);h=row['sha256']
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=h:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=h:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 16 — NOW (예약 20260922T025001Z-31dd6781)

### 관찰 범위

**새 정책 oracle 두 조건과 학습η 결과의 재현을 확인했다. 탐색적 headroom은 확보됐으며, 학습된 선택자의 효능 확증은 아직 없다.** HEAD **7520087c6d5578c1ec640350fa233717ce78d980**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **11:50:51 KST**, manifest 대상·최신 감사/기억/사전등록/canonical log와 etagrid checkpoint 등160파일을 `/tmp/nsmt_assessment_20260922T025001Z-31dd6781`에 복사·hash했다. 감지 manifest 대비 hash 불일치0. 감사15 대비 연구 소스 변경0이며, 이미 재평가한 고정η 네 조건을 반복 실행하지 않았다. 사전등록 최신은 §2G이고 감사15의 상한 정정은 아직 반영되지 않았다.

Model SHA256 `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`. [전체 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probe.py), [수치·hash 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/validation.json).

### (a) 구현 정확성 — A12-ORACLE·A10-HASH의 새 실행 재현 확인

`etagrid-113240`의 새 완료 run3개를 현재 API로 복원하고 전체 test를 CPU 평가했다. Train/evaluate→model의 kind 전달과 copy에서는 uniform/recall에서는 answer-set 정책을 적용하는 현 코드가 유지됨을 확인했다. 새 oracle checkpoint는 **이 수정 정책에서 저장 MSE·M_eff가 정확히 재현**된다. 과거 정책 checkpoint를 새 정책으로 재평가한 결과와 구별한다.

| 새 완료 조건 | epochs | recall MSE | copy MSE | M_eff | 최종 c hit |
|---|---:|---:|---:|---:|---:|
| oracle η=.5 | 12 | .039580641913 | .143349488259 | .466552377101 | 1 |
| oracle η=1 | 12 | .034965688339 | .127822646497 | 1 | 1 |
| sparse 학습η | 12 | .272620149439 | .146173325183 | .132756536374 | 0 |

세 실행 모두 저장 전체 MSE/진단 M_eff와 재평가 차이0, evaluated parameter hash/checkpoint SHA256 일치, source hash 불일치0. Test 상태 finite 및 G11 범위 안이었다. 학습η 평가 checkpoint의 실제 eta는 약.02577485이며 마지막 epoch 통계 약.02629857과 구별한다. Oracle의 Q/K gradient0은 고정 정답 정책이 score를 대체한 결과이므로 gradient 연결 오류로 판정하지 않는다.

**VERIFIED 범위는 현 정책 checkpoint의 복원·수치 재현이다.** 종료 시 source hash만 기록하는 A10-PROVENANCE를 실행 시작 코드 고정까지 확인한 것으로 닫지 않는다. QK L2정규화 구현은 이 snapshot에 없으며 해당 후보 검증은 **not run**이다.

### (b) 검증 적절성 — 같은 데이터/예산 비교 확인, 한 seed의 탐색 수치

공통 seed7/data_seed20260921, train/val/test2048/256/256, batch64, spike, α=.7/K4/key3/scale8/θ5.561343350061557, 최대12epoch/patience10, 배치 제한0. 재평가는256sequence/8085recall query 전체이고 M_eff/hit은 query→sequence 평균이다. 재생성한 x/y/truth/kind tensor 전체를 hash해 **세 run의 각 split이 동일**함을 확인했다. Train/val/test는 생성 seed offset0/10000/20000을 쓰며 split별 hash는 서로 다르다. 이는 데이터 생성·동일 조건 확인이며 과거 test 반복 관찰의 영향을 없애 주지는 않는다.

감사15에서 재현한 full(η0) recall .276338636158에 대해 새 oracleη1과의 **탐색적 headroom=.241372947820**이다. `eta_grid.txt`의 공통 oracleη1 분모를 그대로 사용하면 G는 학습η **.0154055653**, 고정η.2 **−.1783491067**, .5 **−.6511322796**, 1 **−.5814403372**로 표와 일치한다. 이는 동일 seed에서 정답 정책의 큰 개선 여지가 있으나 학습 정책은 거의 회수하지 못한다는 근거다. **Oracle 성능은 실제 사용 시 정답에 접근할 수 있는 모델의 성능이 아니며 검색 성공을 대신하지 않는다.** G의 분모 정책을 명시하고 같은η oracle 대비 질량 진단과 구별한다.

- **A02/O7:** 새 oracleη.5의 M_eff .4665523771은 감사15의 동일 분포 계수 계산을 지지한다. Uniform oracle을 절대 상한으로 부르는 §2G/A02-REACHABILITY는 여전히 OPEN이다. 반례와 수정식은 감사15에 있으며 새 학습으로 그 오류가 해소되지 않는다.
- **A09/통계 해석:** 작업 기록의 chance≈.161은 이번 test 자체의 값이 아니다. 동일 test mask에 같은 query→sequence 집계를 적용한 uniform-slot 기대 hit는 **.156583749693**이다. 고정η1 c hit .21607055는 이 기대값보다 높지만, 독립 seed CI나 random-policy 학습 대조 없이 ‘질량 분포가 우연 수준’/원인 확정을 하지 않는다. Full-kernel mass .13032889와 uniform-slot chance .15658375는 다른 기준선이다.
- 학습η recall .272620149439는 이전 diag3와 동일한 seed·설정에서 같은 값이다. 이번 run의 재현성을 뒷받침하지만 **독립 seed 하나를 추가한 것으로 세지 않는다.** 학습η에서 효과가 ‘전혀 없다’는 해석도 기존 비영 효과 검사와 맞지 않는다.
- Canonical11:45가 탐색/미수렴/예산 공정성 미확인이라고 제한한 점은 적절하다. G14의 **이 seed에서 headroom 수치가 기준을 넘는 관찰**과 8seed/paired CI 기반 O7 확증을 분리한다. 이번 감사는 G14/O7 최종 성공·실패를 확정하지 않는다. 등록50epoch 조건의 성능 검증·다중seed·CI는 **not run**이다.

### (c) 개선 방향과 남은 조건

우선순위 **① 고정η 비교 → ② QK L2정규화 → ③ causal key 표현 → ④ entmax → ⑤ 별도 Gram → ⑥ 별도 delta**를 유지한다. ①의 이번 seed7/12epoch 격자와 새 정책 oracle 재현은 완료됐으므로 **②를 다음 탐색 대상으로 삼을 근거**가 확보됐다. 이는 정규화 최종 채택이나 효능 확인을 뜻하지 않는다.

η1에서 dense residual 없이도 낮은 M_eff가 남고 oracle은 크게 개선되므로, 단순 residual 삭제보다 **동일η에서 QK 크기 편향·score→p→c 변환·최적화 안정성을 분리**하는 방향이 타당하다. Norm ε/인과성/score scale을 명시하고 validation으로만 설정을 선택할 것. 큰 clipping 전 gradient와 absmax 평균 집계 문제(A09)가 남아 있으므로 진짜 최대·clipping 빈도·전후 norm도 함께 확인할 것. Score 순위는 괜찮고 p에서 질량이 손실되는 직접 근거가 생길 때만 entmax를 key 표현보다 앞당긴다.

이 방향은 앞서 원문 확인한 [Test-time regression §3 Eq.(35)–(36)](https://arxiv.org/html/2501.12352v1)의 정규화와 거리/scale 연결, [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 gradient/clipping 분석, [Adaptively Sparse Transformers](https://aclanthology.org/D19-1223.pdf)의 학습 가능한 entmax 근거를 재사용한다. 새로운 문헌 사실이나 효능 보장은 추가하지 않았으며 이번 우선순위 판단은 직접 재현 결과에 기반한다.

A09 absmax/KEY-GEOMETRY, A10-CAL·PROVENANCE·LOG, A07-REGEN 및 A02-REACHABILITY의 잔여 조건은 유지한다. 검토 소스가 동일하므로 이전 수정에 대한 전체 gate 재실행은 **not run**이다.

### 수행·보존·완료

CPU torch1.12.0+cu113/2threads/seed7, 새 checkpoint3개 forward 평가·split fingerprint·기존 결과의 G/동일 test chance 산술만 수행했다. 첫 파일 탐색에서 `rg --files -g AGENTS.md`가 일치 없음(exit1)을 반환한 것은 파일 탐색 결과이며 모델 검사 실패가 아니다. 모델/학습 소스 수정·학습/backward/optimizer·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송은 하지 않았다. 사본 hash와 기존 문서 prefix 보존 검사를 남기고3문서에 append했다. 감사 중 추가 변경은 다음 주기에 확인한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probe.py \
/tmp/nsmt_assessment_20260922T025001Z-31dd6781 \
f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T025001Z-31dd6781/completion_probe.log 2>&1
```

<!-- assessment-watch:20260922T025001Z-31dd6781 -->
'''.replace('NOW',now)
mem=f'''

## 추적 갱신 — {now} (예약 감사16)

HEAD7520087c6d5578c1ec640350fa233717ce78d980/snapshot11:50:51/160파일,manifest불일치0/소스변경0. 새oracleη.5/1 및학습η3checkpoint test256/8085query CPU재현,전체MSE/M_eff 차0/hash일치. Recall .039580641913/.034965688339/.272620149439, M_eff .466552377101/1/.132756536374. 새kind정책재현확인;실행시작provenance보장은아님. Headroom .241372947820,공통oracle1분모G학습η .0154055653;독립8seedCI not run. 같은test uniform-slotchance .156583749693(기존.161과표본다름);fullmass .130329와구별. Split tensorhash세run동일,split간상이. 학습η는이전diag3수치재현이지새독립seed아님. ①탐색격자재현완료→②QK정규화후보유지,구현/효능not run. §2G상한정정미반영/A02-REACHABILITY및기존OPEN유지. ASSESMENT감사16/results/assessment/20260922T025001Z-31dd6781/.
'''
log=f'''

## {now} — 예약 추적 감사16: 새 정책 oracle·학습η 완료 결과 재현

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD7520087c6d5578c1ec640350fa233717ce78d980.11:50:51 snapshot160파일,trigger불일치0,연구소스변경0,감사commit/tag없음.
- CPU torch1.12/2threads,seed7/data_seed20260921,2048/256/256,batch64,recall3keys/spike/α.7/K4/scale8/θ5.561343350061557,12epoch결과3개평가. Oracleη.5/1·학습η recall .039580641913/.034965688339/.272620149439,copy .143349488259/.127822646497/.146173325183, M_eff .466552377101/1/.132756536374. 전체test256/8085query,저장MSE·M_eff 차0/parameter·checkpoint hash일치,수정kind정책에서재현. Split tensorhash세run동일. 학습η결과는동일seed기존diag3재현으로독립표본추가아님.
- 탐색headroom .241372947820;oracle1공통분모G학습η .0154055653. §11:45표수치재현하되G14수치관찰과8seed/O7확증분리. 같은test uniform-slotchance .156583749693,fullkernelmass .13032889와구별;random-policy실험없이‘우연수준’인과단정금지. A02-REACHABILITY상한정정미반영,기존OPEN유지.
- 순위①seed7격자재현완료→②QK L2정규화탐색→③key→④entmax→⑤Gram→⑥delta유지. 새후보구현/학습/8seedCI not run. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/(inventory,completion_probe.py,completion_probes.json,validation,append_validation). Rawconsole forecasting/log/assessment/동일run/completion_probe.log. Exact command·문헌은ASSESMENT감사16. 소스수정/학습/GPU/설치/git변이/프로세스중단/다른세션대화열람·전송없음. 감사문서3개append·진단artifact만작성.
'''
checks=[]
for name,txt in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(txt)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(now);print(json.dumps(check,ensure_ascii=False))
