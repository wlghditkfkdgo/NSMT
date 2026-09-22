from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
out=Path('f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup');snap=Path('/tmp/nsmt_assessment_20260922-manual-followup');stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');inv=json.loads((out/'inventory.json').read_text())
body=f'''

## 추적 감사 14 — {stamp} (사용자 직접 요청: 작업 세션 추적)

### 관찰 결과

관찰 HEAD **4d67c5072b63bfeaf570be44c02c945d0f9aa8a8**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **11:33:02 KST**에123개 문서·소스·결과 파일을 `/tmp/nsmt_assessment_20260922-manual-followup`에 복사·hash했다. 감사13과 비교해 변경된 것은 감사자 기록을 포함한3문서이며, **모델/학습 소스·실행 결과의 새 변경은 확인되지 않았다.** 다른 세션 대화나 프로세스에는 접근하지 않았다. 작업의 진행 여부는 남겨진 문서/산출물 범위에서만 판단한다.

Canonical11:31 기록에서 **우선순위①고정η→②QK정규화→③key표현→④entmax→⑤Gram→⑥delta 수용**, residual 즉시 삭제 보류, 작은η의 효과 전무 주장 철회, 서로 다른 checkpoint 지표 비교 및 Hopfield 조건 표현 정정을 확인했다. 이는 문서상 결정 수용이며 고정η 실험이나 후보 구현이 완료됐다는 뜻은 아니다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/inventory.json), [직접 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probe.py), [수치 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/validation.json). Source hash는 감사13과 동일: model `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`.

### (a) 구현 정확성: 기존 판정 유지

새 구현 없음. A10-RESTORE-STATS 등 확인된 수정의 범위와 A09 epoch absmax 평균 집계/A10-CAL·PROVENANCE·LOG/key_norm 적합/A07-REGEN의 OPEN을 유지한다. 동일 모델·checkpoint 재평가는 **not run**이다. 아래는 새 문서 주장에 필요한 작은 대수/행렬 검사만 수행했다.

### (b) 검증: 타당한 rank 보완은 수용, 새 M_eff 일반화는 반례로 정정 요구

**A09-KEY-GEOMETRY — 표본 평균 Gram 설명 수용, 원 수치 재현은 OPEN.** 작업자가3.97은 개별 feature Gram이 아니라 **표본별 token Gram을 먼저 평균한 행렬**의 participation ratio라고 밝혔다. 이 경우 상한4가 적용되지 않는다는 지적은 맞다. 감사의 기존 상한4는 **각 개별 유닛/시퀀스의 key 행렬** 및 그 feature/token Gram에 한정한다. 평균을 먼저 하는 경우와 개별 PR를 먼저 계산해 평균하는 경우를 구분하도록 이 append에서 명확히 정정한다.

직접 CPU 예: seed7,단위 L2 무작위 key [64,42,4]에서 개별 token Gram rank≤4/평균 개별 PR **3.75866735**, **표본 평균 token Gram rank42/PR36.39041248**이었다. 이는 작업자가 말한 평균 방식의 가능성을 검증한 예이지, 원본3.97·35.67·38.88을 재현한 결과는 아니다. 그 수치들의 script/seed/표본/정규화/평균 축은 여전히 부족하다. 표본 평균 token Gram은 시점별 유사성이 여러 표본에서 공통되는 정도도 반영하므로, 개별 memory 용량·분리도와 동일시하지 않는다. 무작위 baseline은 좋은 key의 기준이나 Hopfield 위반 증명이 아니다.

**A02-REACHABILITY 신규 OPEN — ‘단일 정답이면 η<1에서 M_eff≥.5 불가능’은 틀림.** Canonical11:31은 η=.2/.5 표에서 모든η<1로 일반화했다. 현재 Selector의 oracle 경로와 cap을 그대로 사용해 α=.7,과거41칸,정답 한 칸의 p=1인 경우를 직접 계산했다.

| η | lag2의 post-cap M_eff | lag41의 post-cap M_eff |
|---|---:|---:|
| .92 | **.50704244** | **.50086126** |
| .93 | .54033787 | .53419064 |
| .95 | .62203030 | .61619958 |

모두 η<1이고 .5를 넘는다. 이는 성능 결과가 아니라 새 일반 명제에 대한 **확정된 수치 반례**다. 표본으로 확인한 .2/.5의 낮은 값과 ‘η<1 전체에서 불가능’은 다른 주장이다.

대수적으로 target이 cap에 도달한 구간은

`M_eff = b₀ / [b₀ + (1−η)(B−b_d)]`.

따라서 이 구간에서 **η ≥ 1−b₀/(B−b_d)**이면 M_eff≥.5다. 현재 b₀=1.1005474055,B=13.9614739001에서 lag2/41 경계는 각각 **.91771424/.91972394**이며, 그 경계에서 target cap 조건도 충족한다. 실제 Selector의 eta buffer float32 표현 때문에 식과 약1e-7 미만 차이는 생기지만 판정은 같다. η→1이면 residual이 사라져 이 단일 정답 정책의 M_eff→1이다.

따라서 도달 가능 영역을 그리자는 제안은 타당하지만 **영역이 비어 있다는 전제는 제거**해야 한다. 실과제의 여러 정답 칸·history 길이·lag 분포·정답 정책·cap에 따라 영역이 달라지며, 평균 lag나 평균 정답 칸 수만 대입해서 전체 sequence 평균 O7 판정을 대체하지 않는다. 임계값이나 등록된 η 격자를 이번 반례에 맞춰 자동 변경하지 않는다.

### (c) 개선 방향과 다음 확인 조건

채택 순위는 AUDIT-PRIORITY-01을 유지한다. 다음 의미 있는 근거는 **같은 정책 버전의 고정η 대조 완료 결과**와 **재현 가능한 key 기하/도달 가능성 진단**이다. 작업자는 rank 집계 코드를 artifact로 남기고, score→p→c 지표를 동일 checkpoint/표본/정답 mask에서 연결할 것. 도달 가능성 계산은 분석용이며, 높은η의 실제 학습 안정성·오차 개선을 증명하지 않는다.

문헌 방향은 이전에 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1) 및 [Hopfield Eq.(5)/Theorem4–5](https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf)의 적용 범위를 유지한다. 이번은 신규 문헌 주장 없이 직접 수치 검사를 추가했다. 새 학습 결과가 없으므로 **성능·효능 판정 보류**, 신규 학습/다중seed·CI는 **not run**이다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7,임의 작은 행렬 및 실제 Selector 계수 평가만 수행했다. 모델/학습 소스 수정·학습/optimizer·GPU·환경 설치·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람·전송 없음. 기존 파일/결과를 보존하고 감사 artifact와3문서 append만 작성했다. 사용자 직접 추적 요청이므로 예약 marker/ack는 추가하지 않았다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \\
/home/yschoi/.conda/envs/snn_recall/bin/python \\
f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probe.py \\
/tmp/nsmt_assessment_20260922-manual-followup \\
f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probes.json \\
> f_lif_pop_v3/forecasting/log/assessment/20260922-manual-followup/claims_probe.log 2>&1
```
'''
mem=f'''

## 추적 갱신 — {stamp} (감사14; 사용자 직접 요청)

HEAD4d67c5072b63bfeaf570be44c02c945d0f9aa8a8,11:33:02 snapshot123파일. 연구소스/결과변경없음;canonical11:31 우선순위수용·작은η/Hopfield/혼합checkpoint정정확인. 표본평균tokenGram은rank4초과가능하므로작업자보완수용,상한4는개별matrix에한정정정. 무작위64×42×4 예에서개별rank4,평균Gram rank42/PR36.3904;원3.97/35.67재현not run. 신규A02-REACHABILITY:‘단일정답η<1이면M_eff.5불가’반례,η.92 lag41 .50086126/lag2 .50704244. Cap구간경계η=1−b0/(B−bd),약.918–.920. 실제학습성능아님. 기존우선순위/OPEN유지,새학습not run. ASSESMENT감사14/results/assessment/20260922-manual-followup/.
'''
log=f'''

## {stamp} — 사용자 요청 추적 감사14: 결정 수용 확인·rank 보완 수용·새 질량 일반화 반례

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD4d67c5072b63bfeaf570be44c02c945d0f9aa8a8.11:33:02 snapshot123파일 /tmp/nsmt_assessment_20260922-manual-followup. 연구소스/결과hash는감사13과동일,문서상우선순위수용확인. 모델변경/신규학습/감사commit/tag없음.
- CPU torch1.12/2threads/seed7,실제Selector oracle계수와작은행렬검사. 표본평균tokenGram상한은4아님:64×42×4 unitnorm무작위예,개별rank4/PR평균3.75867,평균Gramrank42/PR36.39041. 작업자설명수용,감사상한4는개별matrix조건으로명확화. 원3.97/35.67 script/표본재현은OPEN.
- **11:31 §3 정정 요구(A02-REACHABILITY): 단일정답이어도η<1에서M_eff.5도달가능.** α.7/history41,η.92에서lag2 .50704244/lag41 .50086126. Cap구간M=b0/[b0+(1−η)(B−bd)],.5경계η=1−b0/(B−bd)≈.917714–.919724. .2/.5관찰을모든η<1에일반화하지말것. 높은η훈련안정성/효능을입증한것아님.
- 기존채택순위유지. 고정η새정책대조학습·8seedCI·성능판정not run. 원본/문서prefix보존;소스수정/학습/optimizer/GPU/설치/git변이/프로세스중단/다른세션대화접근없음. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/(inventory,claims_probe.py,claims_probes.json,validation),rawconsole forecasting/log/assessment/20260922-manual-followup/claims_probe.log. Exact command는ASSESMENT감사14. 수동요청으로watcherack없음.
'''
for name,text in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 with Path(name).open('a') as f:f.write(text)
v={'snapshot_mismatches':[x['path'] for x in inv['files'] if hashlib.sha256((snap/x['path']).read_bytes()).hexdigest()!=x['sha256']],'append_prefix':{n:Path(n).read_bytes().startswith((snap/n).read_bytes()) for n in ['docs/ASSESMENT.md','docs/PROJECT_LOG.md','docs/DOCS_REVIEW_MEMORY.md']},'research_file_changes_during_audit':[x['path'] for x in inv['files'] if not x['path'].startswith('docs/') and hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()!=x['sha256']]};assert not v['snapshot_mismatches'] and all(v['append_prefix'].values());(out/'validation.json').write_text(json.dumps(v,indent=2));print(stamp,v)
