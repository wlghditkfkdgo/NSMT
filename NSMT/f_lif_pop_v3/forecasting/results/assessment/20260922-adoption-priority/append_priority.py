from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
out=Path('f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority');snap=Path('/tmp/nsmt_assessment_20260922-adoption-priority');stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');inv=json.loads((out/'inventory.json').read_text());r=json.loads((out/'priority_probes.json').read_text())
sources=[{'title':'Test-time regression','url':'https://arxiv.org/html/2501.12352v1','checked':'§3 Eq.(3), feature covariance versus kernel Gram; Eq.(35)-(36), bandwidth and QKNorm; sufficient modeling assumptions do not automatically transfer to sparsemax fractional increments.'},{'title':'Hopfield Networks is All You Need','url':'https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf','checked':'Downloaded original manuscript; Eq.(5), Theorems4/5, Eq.(6)-(9): pattern-specific separation, beta, N, M, query distance. Mean cosine alone is insufficient. pdftotext recovered an xref warning and yielded these equations.','pdf_sha256':hashlib.sha256((snap/'hopfield.pdf').read_bytes()).hexdigest()},{'title':'Parallelizing Linear Transformers with the Delta Rule over Sequence Length','url':'https://arxiv.org/html/2406.06484v3','checked':'§3 Eq.(4), recurrent state update; does not replace remaining fractional-history summation automatically.'},{'title':'Adaptively Sparse Transformers','url':'https://aclanthology.org/D19-1223.pdf','checked':'§3 Proposition1, alpha Jacobian and learned sparsity; not a guarantee of scale/temperature independence.'}]
(out/'literature_checks.json').write_text(json.dumps({'checked_kst':stamp,'sources':sources},ensure_ascii=False,indent=2))
body=f'''

## 채택 우선순위 결정 — {stamp} (사용자 직접 요청; AUDIT-PRIORITY-01)

**권고 순서: ① 고정η 대조 → ② QK 정규화 → ③ key 표현 개선 → ④ α-entmax → ⑤ Gram 보정 → ⑥ delta rule.** 이는 다음 검증 대상으로 채택할 순서다. 아직 성능 개선을 입증한 최종 모델 선택은 아니다. ①의 고정η 격자는 기존 D-Y에 따라 먼저 진행하고, **dense residual의 즉시 삭제는 채택하지 않는다.** 기존 분수 기억 수식과 양의 계수·cap 계약을 보존하면서 원인을 분리할 수 있는 후보를 앞에 배치했다. 이 순위는 아래 수치·원문을 바탕으로 한 감사자의 공학적 판단이며 논문이 증명한 후보 간 우열은 아니다.

### 1. 이번 결정 전에 실제로 확인한 근거

HEAD **{inv['head']}**, exp/f-lif-pop-v3에서 소스·문서·diag3 sparse checkpoint19개 파일을 `/tmp/nsmt_assessment_20260922-adoption-priority`에 복사·hash한 뒤 CPU 검사했다. **학습/optimizer/GPU 실행 없음.** [소스·checkpoint inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/inventory.json), [실행 가능한 진단](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probes.json), [원문 확인 기록](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/literature_checks.json).

**동일 표본 비교:** diag3 spike/sparse의 같은 checkpoint, validation 첫64 sequence/2004 recall query, seed7/data_seed20260921, η=**0.02577485144**. 각 query의 모든32 unit을 평균하고 query→sequence 순으로 집계했다. Copy는 제외했으며 argmax 동률은 첫 인덱스를 택한다.

| 지표 | 이번 직접 계산 |
|---|---:|
| score argmax가 정답 칸인 비율 | **0.2715946141** |
| p argmax가 정답 칸인 비율 | **0.2715946141** |
| 최종 c argmax가 정답 칸인 비율 | **0.0000000000** |
| p의 정답 질량 | 0.2564991390 |
| post-cap c의 정답 질량 M_eff | **0.1370368662** |
| fractional kernel의 정답 질량 | **0.1344110089** |
| 동일 query에서 균등한 칸 선택의 정답 확률 | **0.1610314336** |

점수/p의 top1은 이 표본의 균등 기준보다 높지만, c의 정답 질량 증가는 약**0.002626**이다. 따라서 **정책이 계수에 전달되는 정도를 먼저 확인**할 근거는 있다. 이것만으로 η가 유일한 원인이라거나 readout에서 정보가 소실된다고 결론내리지는 않는다. 사용자 제시0.2924와3.7e-5를 그대로 같은 조건의 수치로 간주하지 않았다. 이전3.7e-5는 pilot-eta checkpoint에서 얻은 값이다.

**작은η의 효과는 0이 아니다.** 현재 α=.7,과거41칸에서 B=13.96147385,b₁=.68729711,b₀=1.10054743이다. 정답 한 칸에 p를 전부 주는 가상 정책에서, 정답 lag d가 최근 칸을 앞서는 cap 전 조건은

`η > (b₁ − b_d) / (B + b₁ − b_d)`  (d>1).

이때 최근 칸은 b₀보다 작아 target이 cap되더라도 아래 순위 비교가 유지된다. 실제 계수와 cap으로 검산했다.

| 정답 lag | 최근 칸을 앞서는 η 경계 | η=.026에서 정답이 최대 c인가 |
|---|---:|---|
| 2 | 0.007149 | 예 |
| 5 | 0.015867 | 예 |
| 10 | 0.021498 | 예 |
| 20 | 0.026224 | 아니오 |
| 41 | 0.030240 | 아니오 |

따라서 ‘η가 작으면 점수를 아무리 개선해도 드러나지 않는다’는 일반 명제는 **반례가 있다**. 또 η=.2에서 위 단일 정답 정책은 모두 최대 계수가 되지만, cap 후 정답 질량은 약.091–.093에 그친다. **Argmax 개선≠O7 M_eff 통과**다. 실제 정답 집합은 여러 칸이므로 이 표는 모델 성능 측정이 아닌 계수 경로의 통제된 수치 예다. η=.026에서 정책을 최근 칸↔가장 오래된 칸으로 바꾸면 계수 L1 차도 **0.7259966403**으로 0이 아니다. η·분포·cap을 함께 진단해야 한다.

**Rank 해석 정정:** 현재 유닛별 투영 key는4차원이다. 이번 shape [64,32,41,4]의 비중심 feature Gram participation ratio는 평균 **1.1747333**, 범위1.03329–1.25833, 이론적 상한**4**였다. 기존3.97/42와는 정의/표본이 달라 동일 수치의 재현으로 부르지 않는다. ‘42에 가까워야 한다’는 기준은4차원 key에는 불가능하다. 이번 값은 비중심 에너지의 편중을 보여주지만 의미적 회상 실패를 그 자체로 증명하지 않는다. Gram condition number2470도 token Gram/feature Gram, 중심화·정규화·작은 고유값 처리 정의가 먼저 필요하다.

### 2. 후보별 채택 순서와 통과 조건

| 순위 | 채택할 검증 대상 | 먼저 둘 이유 / 직접 근거 | 다음 단계로 채택할 조건·보류 조건 |
|---|---|---|---|
| **1** | **고정 η={{0,.2,.5,1}} + 학습η 대조**. Dense residual 재설계는 후속 조건부 | D-Y에 이미 있고 수식 변경이 작다. 같은 validation 표본에서 p hit.2716→c hit0, M_eff 증가.002626을 확인했다. 작은η의 lag별 제한과 cap 효과를 분리할 수 있다. | 같은 새 oracle 정책·seed·예산·보정 계약으로 **각 조건을 별도 학습**하고 spike를 주 결과로 둔다. Score/p/c/readout 및 pre/post-cap mass·kappa·G11·최대gradient를 함께 보고한다. η=1이 이미 dense residual 제거에 해당하므로 별도 삭제부터 시작하지 않는다. 기존 측정상 고정η가 자동 개선을 보장하지 않으므로 유효한 oracle/headroom부터 확인. |
| **2** | **투영 Q/K의 L2 정규화** + validation에서 정한 score scale/θ | 메모리 읽기와 cap을 유지하며 거리의 크기 편향을 분리하는 비교다. TTR Eq.(35)-(36)이 정규화 거리/내적 연결을 설명한다. | Norm=0 처리(ε), 인과성, gradient·cap 게이트 및 같은η 조건의 비교가 필요. 현 `key_norm=frozen` 입력 표준화와 **다른 연산**이다. 정규화만으로 θ 불필요/폭주 제거라고 주장하지 말 것. 기존 상태 크기의 유용한 신호가 사라질 가능성도 validation에서 확인. |
| **3** | **Key 표현 ablation**: 기존 [u,I] 대 causal Δu 추가 등 | 같은 모델/표본에서 score hit와 균등 기준의 간격은 있으나 완전 검색은 아니다. 현재 key의4차원/분포 편중을 고려하면 표현 변화는 직접 검사할 만하다. 기존 f_j/c 계약을 유지한다. | 갱신 전 정보만 사용하고 미래 state를 참조하지 않을 것. 차원/parameter 증가를 맞춘 대조와 정규화 조건을 둔다. 정답 집합 대비 score margin/랭킹 및 최종 c 질량·오차가 함께 나아져야 한다. 평균 cosine를 무조건 낮추는 목표는 금지—같은 값의 여러 정답 칸은 비슷해도 된다. |
| **4** | **α-entmax**: 먼저 고정 sparsity 지수, 이후 학습 지수 | 확률 p를 만드는 함수만 바꿔 현재 양의 계수 재가중에 연결할 수 있다. 원문 Proposition1은 학습 가능한 지수의 미분 근거다. 다만 현재 낮은η 전달 문제가 남으면 우선 효과를 식별하기 어렵다. | Fractional α와 구별해 지수를 γ 등으로 표기. Sparsemax/softmax와 score scale·η·예산을 맞추고 support·gradient·mass·recall을 비교. 미분 구현 검증 및 안정성이 필요. 지수 학습이 θ 역할을 자동 제거하지 않는다. |
| **5** | **Ridge/Gram 보정 읽기**를 별도 탐색 baseline으로 | TTR의 선형 최소제곱 해가 근거이나 현재 모델은 선형 K–V 회귀가 아니라 분수 증분의 재가중이다.4×4 solve라는 크기만으로 우선 도입할 근거는 부족하다. | K,V,f_j 대응을 수식으로 고정하고 λ·condition·수치안정성을 점검. Linear feature Gram4×4와 nonlinear kernel Gram T×T를 구분. 보정 가중치가 음수일 수 있어 기존 positivity/cap/G4 계약을 그대로 승계할 수 없다. 보정 score만 쓰는 whitening과 읽기 전체 교체도 별도 후보로 분리. |
| **6** | **Delta rule**을 별도 기억 구조 대조군으로 | DeltaNet의 recurrent update는 근거가 있지만, 기록/읽기 방식이 바뀌어 현재 아이디어의 작은 수정이 아니다. 검증 부담과 원인 해석 범위가 가장 크다. | K–V 정의·state 크기·parameter·실측 runtime/메모리·reset 계약을 새로 명시. O(T)는 고정 차원의 해당 recurrent 연산에 대한 말이다. 기존 f_j 전 이력 합산을 남기면 전체 모델이 자동 O(T)가 되지 않는다. 현재 모델의 교체 채택은 공정한 독립 성능/비용 비교 이후. |

**조건부 순위 조정:** ① 이후 score 순위는 좋지만 p 단계에서 질량이 사라지는 것으로 확인되면④를③보다 앞당긴다. Score 단계부터 정답을 구분하지 못하면③을 유지한다. Cap에서만 질량이 무너지면②–④를 한꺼번에 바꾸기보다 현 cap/η의 제약을 먼저 정량화한다. 안전 상한을 임의로 없애 성능만 맞추지 않는다. ⑤–⑥은 계산비가 작아 보여도 계약 변경이 커서 후순위다.

### 3. 논문 원문 확인 결과와 적용 한계

- [Test-time regression, §3 Eq.(3), (32), (35)–(36)](https://arxiv.org/html/2501.12352v1): 선형 최소제곱 해·feature covariance와 kernel Gram을 구분하며, QKNorm의 거리/내적 연결에도 bandwidth 선택이 있다. ‘모든 커널 읽기는 KᵀK=I일 때만 정확하다’는 형태로 일반화하지 않는다. 입력이4차원이라고 모든 비선형 기억 문제를4×4 역행렬로 해결할 수 있는 것도 아니다.
- [Modern Hopfield 원문, Eq.(5), Theorem4–5](https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf): 이번에는 원문 PDF를 직접 받아 읽었다(SHA256 **48cecc1d10cea553538fe8d1e2f1bf7378bed6b1233b2857384f5081ac2f7579**). 보장은 패턴별 `Δ_i=x_iᵀx_i−max_(j≠i)x_iᵀx_j`, β, 패턴 수/크기, query와 패턴의 거리 등에 의존한다. 평균 인접 cosine.758이나 rank만으로 조건 위반을 판정할 수 없다. 충분조건의 미확인은 실패의 필요조건 증명이 아니며, 현재 sparsemax/비대칭QK/분수가지에도 정리가 자동 적용되지 않는다. 감사12의 PDF 접근 미완 상태는 **이번 원문 확보·해당 정리 확인 범위에서 해소**했다.
- [DeltaNet, §3 Eq.(4)](https://arxiv.org/html/2406.06484v3): 현재 상태에서 key 방향의 연관을 수정하는 recurrent rule을 확인했다. 기존 분수 solver의 대체 가능성·우월성·전체속도는 별도 검증 대상이다.
- [Adaptively Sparse Transformers, §3 Proposition1](https://aclanthology.org/D19-1223.pdf): entmax 지수 미분과 head별 학습의 근거를 확인했다. 우리 회상/분수 동역학에서의 개선이나 score scale 독립성을 보장하는 정리는 아니다.

### 4. 공통 채택 기준과 이번 작업의 범위

먼저 checkpoint/config/source/정책 버전·보정 ID·평가 범위를 고정한다. **Validation으로만 후보와 hyperparameter를 선택**하고, 이미 반복 관찰한 test를 새로운 미관측 근거처럼 쓰지 않는다. 기존 사전등록의 독립 seed8개·paired CI·G14/O7 기준은 유지하며, 이번 탐색 수치에 맞춰 낮추지 않는다. 모든 후보에 동일 예산·통제 조건과 인과성/finite/G11·복원·원본 대조가 필요하되 수식이 바뀌는⑤–⑥은 변경된 게이트를 새로 정의해야 한다. 추가 비용은 단순 parameter 수가 아니라 전체 forward/backward·history 저장까지 측정한다. **효능이 미검증인 후보를 지금 ‘최종 채택’으로 표시하지 않는다.**

이번은 직접 validation forward1회와 계수 대수 검산·원문 확인이다. 새 학습, 고정η sweep 학습, 후보 구현, 다중 seed/CI 및 성능 우위 검증은 **not run**이다. 모델/학습 소스·사전등록 본문을 수정하지 않았고 기존 audit도 고치지 않고 append했다. canonical 기록과 문서 기억에는 이 결정만 추가한다. 예약 실행이 아니므로 watcher marker/ack를 새로 만들지 않는다.

재현 명령(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \\
/home/yschoi/.conda/envs/snn_recall/bin/python \\
f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probe.py \\
/tmp/nsmt_assessment_20260922-adoption-priority \\
f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probes.json
```

Raw console: `f_lif_pop_v3/forecasting/log/assessment/20260922-adoption-priority/priority_probe.log`. 수치 환경은 CPU torch1.12.0+cu113/2threads이며 문헌 PDF 추출의 xref 복구 경고는 원문 읽기 도구의 경고이지 모델 실패가 아니다.
'''
mem=f'''

## 우선순위 갱신 — {stamp} (AUDIT-PRIORITY-01; 사용자 직접 요청)

채택 검증 순서: 고정η→QK L2정규화→causal key표현→entmax→별도Gram→별도Delta. Residual즉시삭제아님,η1이이미제거대조. 같은diag3checkpoint validation64/2004query:score/p hit.2715946,c hit0,p mass.256499,c mass.137037,kernel.134411,chance.161031. 작은η에서도lag2/5/10이상적정책은최근계수역전가능,전혀효과없다는명제기각. Cap/질량분리필요. 투영key4차원PR상한4,이번비중심PR1.17473. Hopfield원문PDF확보/Eq5·Thm4/5확인으로평균cosine만으로실패단정불가 재확인. 조건별채택기준/순위조정은ASSESMENT AUDIT-PRIORITY-01. Artifacts results/assessment/20260922-adoption-priority/. 학습/후보효능검증not run.
'''
log=f'''

## {stamp} — 사용자 요청: 검색 개선 후보 채택 우선순위 감사

- AUDIT-PRIORITY-01. Branch exp/f-lif-pop-v3,HEAD{inv['head']},base329183b94f65090cc6b337f464c5aa4d8e127ad7. 소스/문서/checkpoint19파일사본 /tmp/nsmt_assessment_20260922-adoption-priority 및hash고정. 감사자모델수정/학습/optimizer/GPU/git변이/commit/tag없음.
- CPU torch1.12/2threads,diag3 spike sparse checkpoint(seed7,data_seed20260921,기존12epoch2048train),validation첫64sequence/2004recall만forward. Score/p hit.2715946141,c hit0,p mass.256499139,c mass.137036866,kernel.134411009,chance.161031434,eta.0257748514. 같은unit/query/sequence집계. Key[64,32,41,4],비중심featureGram PR평균1.174733/상한4.
- 실제b와cap로η.026 onehot정책검산:lag2/5/10은정답c최대가능,20/41은불가. ‘점수개선효과전무’단정은성립안함. η.2에서도단일정답mass약.09로argmax와O7질량구분. 후보검증순서 고정η→QK정규화→key표현→entmax→Gram별도기억→Delta별도기억. 최종효능채택아님. Validation선택/8seedpairedCI·G14/O7유지.
- TTR원문Eq3/32/35/36,DeltaNetEq4,entmax명제1확인. Hopfield원문PDF직접확보SHA48cecc1d10cea553538fe8d1e2f1bf7378bed6b1233b2857384f5081ac2f7579,Eq5/Thm4–5확인. 평균cosine/rank로정리실패단정불가. 문헌링크·조건/제한은NSMT/docs/ASSESMENT.md AUDIT-PRIORITY-01에명시.
- Evidence NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/(inventory,priority_probe.py,priority_probes.json,literature_checks,validation). Raw log forecasting/log/assessment/20260922-adoption-priority/. Exact CPU command는동감사항목. 새학습/후보구현/다중seed효능검증not run. 예약ack없음.
'''
for name,text in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 with Path(name).open('a') as f:f.write(text)
v={'snapshot_hash_mismatches':[x['path'] for x in inv['files'] if hashlib.sha256((snap/x['path']).read_bytes()).hexdigest()!=x['sha256']],'original_prefix_checks':{name:Path(name).read_bytes().startswith((snap/name).read_bytes()) for name in ['docs/ASSESMENT.md','docs/PROJECT_LOG.md','docs/DOCS_REVIEW_MEMORY.md']},'non_document_original_changes':[x['path'] for x in inv['files'] if not x['path'].startswith('docs/') and hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()!=x['sha256']]}
assert not v['snapshot_hash_mismatches'] and all(v['original_prefix_checks'].values());(out/'validation.json').write_text(json.dumps(v,indent=2));print(stamp,'appended; validation',v)
