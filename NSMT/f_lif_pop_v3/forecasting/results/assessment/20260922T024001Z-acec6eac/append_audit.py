from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
base=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac')
now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
body=r'''

## 추적 감사 15 — NOW (예약 20260922T024001Z-acec6eac)

### 관찰 범위와 실제 증거

**고정η 네 조건의 저장 결과를 재현했으나 효능은 미확증이다. §2G의 uniform oracle ‘절대 상한’ 주장은 수치 반례로 정정이 필요하다.** 관찰 HEAD **2769c5c8100e367b965a47b20bd1924f0b0d18a1**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 11:40:55 KST에 실제148파일을 `/tmp/nsmt_assessment_20260922T024001Z-acec6eac`로 복사·SHA256 기록했다. Trigger manifest와 실제 파일을 대조했으며, 감지 때 완료 JSON이 없던 η=1도 **snapshot 때에는 완료**되어 포함했다. Snapshot 내 학습η 후속 run은 완료 JSON이 없어 재평가하지 않았다. 감사14 대비 연구 소스 변경은 없다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/inventory.json), [직접 CPU 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probe.py), [결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/validation.json). Model SHA256 `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`, prereg `1721c443859c85f48ddbc6ba3ad908016108cc2bd54f9fddfaa0807faddba823`.

### (a) 구현 정확성 — 고정η 배선·복원·결과 재현 확인, 기존 OPEN 유지

`etagrid-113240`의 완료된 네 checkpoint를 현재 API로 CPU 복원했다. 각각 실제 eta buffer가 설정값과 일치하고, **전체 test MSE·M_eff 저장값과 재평가 차이0**, evaluated parameter hash/checkpoint hash 일치, 기록된 source hash와 고정한 소스의 불일치0이었다. A06 고정η 및 A10-HASH의 기존 확인을 이 네 실행까지 확대한다. 이 사실만으로 A10-PROVENANCE의 실행 시작 source 고정 문제를 닫지는 않는다.

공통 조건은 seed7/data_seed20260921, train/val/test=2048/256/256, batch64, spike readout, α=.7,K=4, recall key3, scale8, θ=5.561343350061557, 최대12epoch/patience10, 배치 제한0이다. Test 전256sequence/8085 recall query를 평가했다. M_eff·hit은 query→sequence 평균이다.

| 고정η | 완료 epoch | recall MSE | M_eff | 최종 c hit | cap rate |
|---|---:|---:|---:|---:|---:|
| 0 | 12 | .276338636158 | .130328894393 | .000000 | .000000 |
| .2 | 12 | .319387285780 | .127181601094 | .020639 | .009919 |
| .5 | **11** | .433504353901 | .132870673176 | .096712 | .064035 |
| 1 | 12 | .416682604337 | .136898894082 | .216071 | .134492 |

η=.5는 CSV epoch0에서 validation loss 최저, 이후10epoch 개선 없음과 patience10/종료 코드가 일치한다. **11epoch를 미완성 또는 실행 실패로 판정하지 않는다.** η=0의 Q/K gradient0은 선택 경로를 끈 설정에 맞는다. Test 상태는 네 조건 모두 finite/G11 상한 안이었으나 이는 전체 학습 step의 보장을 추가로 증명한 것이 아니다.

### (b) 검증 적절성 — 결과 재현은 통과, 해석·상한 정의는 정정 필요

**A09-GRADIENT OPEN:** η=.5/1의 마지막 epoch `grad_absmax_all` 보고값은 각각 **1.0608e11 / 4.3705e12**다. 현 reducer가 표본 batch 최대값들을 **평균**하므로 epoch 최대라고 부를 수 없다. 그래도 clipping 전 매우 큰 gradient가 실제 보고됐다는 근거다. 매 step norm1 clipping 및 finite 검사를 쓰고 있어 곧바로 비유한 학습 실패를 뜻하지 않으며, 원인을 점수 함수 하나로 단정하지 않는다. 진짜 최대·표본 수·clipping 빈도와 전후 norm을 추가로 기록할 필요가 있다. 재검사 없이 A09를 VERIFIED로 닫지 않는다.

**A02-REACHABILITY OPEN(P1), §2G 정정 요청:**

1. Uniform-on-answer oracle은 **정의된 정책 기준선**이지 모든 p의 post-cap M_eff 상한이 아니다. 현재 실제 Selector에 α=.7/history41/정답 lag{41,1}/η=.2를 넣으면, uniform p=(.5,.5)는 **.1644974421**, cap을 고려한 p=(.9173828776,.0826171224)는 **.1744286360**이다. 같은 계수·η·cap에서 더 높은 유효 질량이 존재한다. 이 반례는 허용된 p 공간의 계수 주장 검증이며 실제 score 학습으로 그 p가 도달된다는 뜻은 아니다.
2. 양의 b_i, cap C=b₀, η∈[0,1], 비어 있지 않은 정답 집합 A에서, `B=Σ_i b_i`, `B_A=Σ_(i∈A)b_i`, `m=|A|`라 두면 **임의의 simplex p에 대한 계수 질량의 최대**는 아래와 같다. 현재 b_i≤C 조건에서 적용한다.

   `S_max = min((1−η)B_A + ηB, mC)`

   `U = S_max / [S_max + (1−η)(B−B_A)]`.

   이유: `q_i=b_i p_i/Σ_j b_j p_j`도 임의 simplex이며, adaptive 총량ηB를 정답 칸들의 남은 cap 용량에 배분하면 최대 정답 질량을 얻는다. 정답 밖 residual은 그대로 남는다. 정답 칸이 포화되면 잉여량은 정답 칸에 버릴 수 있다. 이것은 **자유로운 정책에 대한 계수 상한**이고, WQ/WK 표현 제약·인과적으로 얻을 수 있는 정보·학습 최적화의 성공을 보장하지 않는다.
3. 같은 test256sequence/8085query의 실제 history 길이·정답 mask에 식과 uniform oracle을 적용하고 등록된 query→sequence 순서로 집계했다. 학습/forward 성능 실험이 아닌 대수 검사다.

| η | uniform oracle M_eff | 자유 정책 상한 U의 평균 |
|---|---:|---:|
| 0 | .130328893571 | .130328893571 |
| .2 | .298202511703 | .298208777887 |
| .5 | .466552376117 | **.466613112585** |
| .655 | **.557888951227** | .557946562113 |
| .7 | .590493784894 | .590537123649 |
| .75 | .631612043147 | .631647159784 |
| 1 | 1 | 1 |

이 표본에서는 두 값의 차가 작고, **등록 격자{0,.2,.5,1} 중 .5 평균 질량에 도달 가능한 것은 η1뿐이라는 결론은 올바른 상한으로도 유지**된다. 다만 고정 history41/평균 lag21/정답 칸3–4 예시가 그 증거는 아니다. η.655의 uniform 값이 이미 .5를 넘으므로 ‘과제 평균 정답 수3.5이니 최소η≈.70–.75’도 이 실측 분포의 필요조건이 아니다. 미래 dataset/seed 분포 전체에 일반화하지 않는다. 등록 임계값·확증 격자는 변경하지 않는다.
4. §2G의 ‘lag5/21/35 차이<.01’은 인용한 원 artifact 자체와 불일치한다. 정답4칸·η.2의 .3088−.2613=**.0475**다. `meff_reachable.txt`를 재생성할 명령/코드와 정확한 배치 정의도 남겨야 한다. .4666을 .34–.41의 ‘상한과 정합’이라 부르지 않는다. 표본과 집계가 다르다.
5. `learned/oracle` 비율만으로 ‘선택자는 잘했다’고 판정하지 않는다. **η=0이면 선택자가 무엇을 하든 비율1**이며 uniform oracle이 진짜 상한도 아니다. Full kernel 기준 질량, p 자체의 질량/순위, pre/post-cap 질량, 실제 오차를 함께 보고한다. 이는 진단 보완이고 승인된 O7 기준을 바꾸는 제안이 아니다.

**통계·분리·진행 상태:** 이번 동일 seed의 η 증가 조건들은 η0보다 recall MSE가 컸고, η1에서도 M_eff가 .1369에 그쳤다. 이를 ‘η만 올리면 해결된다’는 주장의 지지로 쓸 수 없다. 반대로 한 seed·짧은 예산·서로 다른 최적화 상태만으로 아이디어의 최종 실패나 원인까지 결정하지 않는다. 독립8seed/paired CI, 등록 최대50epoch 조건의 확증, 새로운 후보 성능 검사는 **not run**이다. 이미 반복 관찰한 test는 탐색용으로 표시하고 후보/scale 선택은 validation에서 진행한다.

11:45 canonical 기록에 새 oracle η.5/.1이 아닌 **η.5/1** 결과와 G14/O7 관련 해석이 추가된 것을 후속 관찰했다(`project_log_late.txt` 별도 보존). **그 새 학습 결과와 학습η 후속 run은 이번 snapshot 범위 밖**이며 새 정책 checkpoint 재평가는 **not run/다음 주기 확인 대상**이다. 해당 기록의 탐색적이라는 제한은 수용하되, ‘통과/실패’는 탐색 수치의 기준 충족 여부와 확증 판정을 구분해야 한다. 이번 감사에서는 그 표만으로 G14 VERIFIED를 부여하지 않는다. 또한 작은 학습η에서 ‘선택이 사실상 작동하지 않는다’는 표현은 감사 PRIORITY-01의 비영 효과 반례를 반영해 제한해야 한다.

### (c) 개선 방향 — 우선순위 유지, 정규화의 검증 조건을 구체화

**① 고정η 결과·동일 정책 oracle 비교 정리 → ② QK L2정규화 → ③ causal key 표현 → ④ entmax → ⑤ 별도 Gram → ⑥ 별도 delta** 순서를 유지한다. ①의 네 조건은 확보됐고 같은 예산/정책 oracle 및 후속 학습η의 provenance 재평가가 남았다. η1은 dense residual이 없는 조건인데도 정답 질량 개선이 작으므로, dense residual 지배만으로 현재 전체 현상을 설명할 수 없다. 다만 score→p→c 변환·cap·gradient·학습 동역학이 여전히 얽혀 있어 score 품질 단독의 확정 인과판정은 아니다.

QK 정규화는 같은η·예산에서 크기 편향을 분리하는 **다음 탐색 후보**로 타당하다. Score scale은 validation에서 고정하고 zero norm ε 처리·인과성·finite·실제 gradient 최대/clip 통계를 확인할 것. 같은 checkpoint/validation 표본에서 score/p/c의 정답 질량과 순위를 각각 기록해 p 단계의 손실이면 entmax를 key 표현보다 앞당기는 기존 조건부 규칙을 적용한다. 최종 c hit만으로 score와 p의 손실 위치를 구분하지 않는다.

문헌은 앞서 원문 확인한 [Test-time regression §3 Eq.(35)–(36)](https://arxiv.org/html/2501.12352v1)의 QK 정규화–거리 연결과 bandwidth 잔존, [Pascanu et al.의 gradient/clipping 분석](https://proceedings.mlr.press/v28/pascanu13.pdf), [Adaptively Sparse Transformers §3](https://aclanthology.org/D19-1223.pdf)를 재사용했다. 이 문헌들이 우리 모델의 정규화 효능·폭주 제거·최적 순위를 보장하지는 않는다. 새 논문 주장은 없고 이번 결론은 위 직접 수치 검토에 기반한다.

### 남은 조건·수행 기록

A10-CAL(key_norm 적합/보정 ID), A10-PROVENANCE, A10-LOG(final CSV/동적 열), A07-REGEN, A09-KEY-GEOMETRY의 열린 조건은 이번 재현만으로 닫지 않는다. 11:46:49 보존 검사에서 snapshot148파일 hash 불일치0. 작업 세션은 canonical/학습η checkpoint·CSV 및 HEAD를 추가 변경했고, 이를 감사자 변경으로 취급하지 않았으며 다음 주기에서 확인한다.

CPU torch1.12.0+cu113/2threads/seed7로 완료 checkpoint4개 forward 평가와 계수 대수·Selector 소규모 검사만 수행했다. 신규 학습/backward/optimizer/GPU/환경 설치/소스 변경/진행 프로세스 중단/git 변이/다른 세션 대화 접근·메시지 전송은 하지 않았다. 기존 문서 본문과 raw artifacts를 보존하며3문서에 append한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probe.py \
/tmp/nsmt_assessment_20260922T024001Z-acec6eac \
f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T024001Z-acec6eac/etagrid_probe.log 2>&1
```

<!-- assessment-watch:20260922T024001Z-acec6eac -->
'''.replace('NOW',now).replace('η.5/.1이 아닌 **η.5/1**','**η.5/1**')
memory=f'''

## 추적 갱신 — {now} (예약 감사15)

Snapshot11:40:55/HEAD2769c5c8100e367b965a47b20bd1924f0b0d18a1/148파일. Etagrid 고정η0/.2/.5/1 checkpoint CPU재평가 test256/8085query, MSE·M_eff 차0/hash일치. Recall .276339/.319387/.433504/.416683, M_eff .130329/.127182/.132871/.136899. η.5는patience10에맞는11epoch종료;η1도완료. A09 clipping전큰gradient(마지막표본absmax평균 .5:1.06e11/1:4.37e12),최대집계OPEN. A02-REACHABILITY §2G uniformoracle상한명제반례 .164497→.174429. 자유정책상한 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)]. 실제test분포η.5U .466613/.655uniform .557889;평균lag/정답수로최소η.70–.75단정불가. η0 learned/oracle비율1은선택성공증거아님. 우선순위유지,새oracle/학습η후속은snapshot후기록으로다음주기;8seedCI/새학습not run. ASSESMENT감사15,results/assessment/20260922T024001Z-acec6eac/.
'''
log=f'''

## {now} — 예약 추적 감사15: η 격자 재현·uniform oracle 상한 정정 요청

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7,HEAD2769c5c8100e367b965a47b20bd1924f0b0d18a1.11:40:55 snapshot148파일. 연구소스변경없음,감사commit/tag없음.
- CPU torch1.12/2threads,seed7/data_seed20260921,recall3keys,spike,K4/α.7/scale8/θ5.561343350061557,2048/256/256,batch64,최대12epoch/patience10. 고정η0/.2/.5/1의 checkpoint test256/8085query 재현,저장MSE·M_eff 차0/parameter·checkpoint hash일치. Recall .276338636158/.319387285780/.433504353901/.416682604337. η.5 11epoch는earlystop과일치. η1잔여dense가없어도M_eff .136899;η만높이는해결책미지지. 같은seed탐색,효능확증아님.
- A02-REACHABILITY 정정 요청: §2G uniformoracle은상한아님. 실제Selector반례(history41,lag41/1,η.2) .164497→.174429. 올바른자유정책계수상한 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)]. 실제test분포η.5 U=.466613,η.655 uniform=.557889. 고정history/평균lag 예로과제전체최소η단정금지. Lag차<.01도artifact자체 .3088−.2613=.0475와충돌. η0 learned/oracle비율1이라도선택성공아님.
- A09 최종epoch absmax표본평균 η.5~1.06e11/η1~4.37e12,clipping전/epoch최대아님. 효능/안정성원인단정금지. 기존OPEN유지. 11:45작업기록의새oracle 및후속학습η는초기사본범위밖,checkpoint재검사다음주기. G14/O7확증8seedCI not run. 순위①→②QK정규화→③key→④entmax→⑤Gram→⑥delta유지.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/(inventory,etagrid_probe.py,etagrid_probes.json,validation,project_log_late.txt). Raw log forecasting/log/assessment/동일run/etagrid_probe.log. Exact command와문헌링크는ASSESMENT감사15. 모델/학습소스수정·학습/GPU/설치/git변이/프로세스중단/다른세션대화열람·전송없음.3문서append및감사artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',memory),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 after=p.read_bytes();assert after[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(base/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(now,'appended',len(body),'characters; original prefixes preserved')
