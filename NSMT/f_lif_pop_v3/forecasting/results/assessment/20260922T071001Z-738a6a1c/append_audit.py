from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');hs={r['path']:r['sha256'] for r in inv['files']}
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for r in inv['files']:
 p=Path(r['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=r['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 22 — NOW (예약 20260922T071001Z-738a6a1c)

### 관찰 범위·핵심 판정

**A09 absmax 수정 완료 주장은 현재 소스에서 재현되지 않았다. 확인 실행의 오차·checkpoint는 재현됐지만, 관찰1회라 집계 오류를 검출하지 못한다.** 관찰 HEAD **031fb758af0fc2dc9257acc100783b5cd44404fe**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **16:10:35 KST**에 manifest·최신 감사/기억/사전등록§2H/canonical 문서·absmaxchk checkpoint206파일을 `/tmp/nsmt_assessment_20260922T071001Z-738a6a1c`로 snapshot·hash했다. Trigger hash 불일치0. 연구 코드 변경은 train.py뿐이며 모델/평가 API는 이전과 동일하다.

Train SHA256 `TRAIN_HASH`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/source.diff), [직접 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/validation.json).

### (a) 구현 정확성 — A09-ABSMAX OPEN 유지, 관찰 횟수 결함 확인

Canonical16:07은 absmax/_max 필드를 np.max로 바꿨다고 기록했으나, 고정한 실제 reducer는 모든 finite 목록에 **`float(np.mean(finite))`**를 적용한다. 실제 변경은 `grad_observations`에1을 추가하는 부분과 이미 존재하던 pre-norm max의 float 변환이며, absmax reducer 변경은 없다. 실제 함수 AST에서 집계 블록만 추출해 optimizer 없이 실행했다.

| 입력/필드 | 실제 출력 | 요구되는 의미의 값 |
|---|---:|---:|
| grad_absmax_all [1,9] | **5** | 9 |
| grad_WQ_absmax [1,9] | **5** | 9 |
| grad_WK_absmax [2,10] | **6** | 10 |
| grad_observations [1,1] | **1** | 2회 |
| grad_total_norm_pre [1,9]의 max | **9** | 9 |
| eta [1,9]의 mean | **5** | 5 |

Pre-norm max는 내부에서 미리 np.max한 단일값이라 맞지만, **그 성공이 개별 absmax의 수정을 입증하지 않는다.** `grad_observations`도1의 평균이므로 관찰 횟수가 늘어도1이다. 실제 결과 JSON의 train 메타데이터에는 이 필드가 없고 CSV에만 나타난다. **A09-OBSERVATIONS OPEN**으로 추적한다. Post-clip norm도 여전히 없다.

수정 조건은 필드 의미에 맞춰 absmax를 max, 관찰 횟수를 count/sum, 평균 지표를 mean으로 분리하고 JSON/CSV 양쪽에 의미가 보존되게 하는 것이다. 이는 실제 소스와 직접 반례에 따른 확정 결함이며 학습 실패를 뜻하지 않는다. 진행 중 추가 수정이 있을 수 있으나 이 snapshot을 VERIFIED로 닫을 근거는 없다.

### (b) 확인 실행·검증 절차

`absmaxchk-160633`은 seed7/data_seed20260921, **1epoch**, train/val/test512/64/64,batch64,g11_every10,배치 제한0,비정규화 sparse 학습η,θ5.561343350061557이다. Test64sequence/2022recall query를 CPU 재평가했다.

- 전체 MSE **.3704960201865058**, recall MSE **.3698382646185705**, M_eff **.13170154071437468**.
- 저장 전체 MSE/M_eff와 재평가 차이0, evaluated parameter/checkpoint SHA256 일치, source hash 불일치0.
- 실제 CSV/JSON에 전체 batch pre-norm mean **1.8237197399**, max **2.9906373024**, clip_rate_all_batches1, batches_seen8이 저장됐다(CSV는6자리 반올림). **A09 전체 batch norm/clip 필드의 완료 JSON·CSV 보존은 이번 run까지 확인됐다.** 그 gradient 자체를 backward로 재생성한 것은 아니다.

그러나8batch/g11_every10에서는 **batch0만 한 번 관찰**한다. 관찰1개이면 mean=max이고 1의 평균=관찰 횟수1이므로, 이 실행은 두 집계 결함을 검출할 수 없다. 최소한 서로 다른 두 관찰값을 넣어 확인해야 한다. 이번 감사의 위 추출 블록 반례가 그 조건을 충족한다. 추가 학습 없이도 이 집계 결함은 검사 가능하다.

새 run은 logging 확인용 짧은 실행으로, 이전12epoch의 격자 성능과 직접 비교해 개선/퇴보를 판정하지 않는다. G11 test 진단은 finite/상한 이내였으나 전체 학습 안정성의 증명은 아니다. 신규 학습·gradient 재계산·독립8seed/paired CI·성능 우위 검증은 **not run**이다. Canonical의 G 인용 정정과 G11 새/과거 사건 구분 수용은 확인했으며 감사21의 제한을 유지한다.

### (c) 개선 방향

먼저 실제 reducer·관찰 count·결과 전달을 수정하고, 서로 다른 관찰값을 사용한 작은 검증으로 문서의 완료 주장과 소스를 일치시킬 것. 그 다음 동일 validation checkpoint/표본의 score→p→pre-cap→post-cap 지표에 정확한 gradient/clip 통계를 함께 연결한다. Score 표현이 병목이면③ causal key, p 변환에서 손실되면④ entmax를 앞당기는 기존 조건부 순위를 유지한다. 이번 logging 확인 결과로 QK 정규화 채택이나 모델의 효능을 판단하지 않는다.

Gradient 크기·시간별 변화와 clipping을 구분하라는 방향은 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf), 정규화와 score scale의 구분은 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 근거를 재사용한다. 이번 판정은 직접 수치 검토에 기반하며 새 문헌 주장은 없다. A02 실제분포 연결/A07-REGEN/A10-PROVENANCE·LOG 등 범위 밖 OPEN은 그대로 둔다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7, 실제 reducer만 실행한 합성 검사와 checkpoint1개 forward 평가만 수행했다. 모델 학습/backward/optimizer/GPU·설치·모델/학습 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송은 하지 않았다. 기존 문서 prefix와 파일 사본을 보존하고3문서 append·진단 artifact만 작성했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probe.py \
/tmp/nsmt_assessment_20260922T071001Z-738a6a1c \
f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T071001Z-738a6a1c/absmax_probe.log 2>&1
```

<!-- assessment-watch:20260922T071001Z-738a6a1c -->
'''.replace('NOW',now).replace('TRAIN_HASH',hs['f_lif_pop_v3/forecasting/train.py'])
mem=f'''

## 추적 갱신 — {now} (예약 감사22)

HEAD031fb758af0fc2dc9257acc100783b5cd44404fe/snapshot16:10:35/206파일,manifest불일치0. Canonical16:07absmax수정주장과실제train불일치:reducer여전히mean,실제AST [1,9]→5/WK[2,10]→6. A09-ABSMAX OPEN. 신규grad_observations [1,1]→1이며trainJSON미전달(A09-OBSERVATIONS OPEN). Absmaxchk1epoch512/64/64,g11_every10→8batch중1관찰로mean/max결함검출불가. CheckpointCPUtest64/2022query MSE .370496020187/recall .369838264619/M_eff .131701540714 재현차0/hash일치. 전체batch pre-norm/clip 완료JSON·CSV보존은확인. Postclip없음/8seedCI not run. ASSESMENT감사22/results/assessment/20260922T071001Z-738a6a1c/.
'''
log=f'''

## {now} — 예약 추적 감사22: absmax 수정 주장 미재현·관찰 count 결함

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD031fb758af0fc2dc9257acc100783b5cd44404fe,snapshot16:10:35/206파일,manifest불일치0. 감사commit/tag없음.
- **16:07 §1 정정 요구:** 실제train reducer는np.mean유지. AST직접검사 absmax[1,9]→5/WK[2,10]→6,grad_observations[1,1]→1. Pre-norm max9는기존별도np.max결과라absmax수정증거아님. 관찰count는CSV만있고trainJSON누락. A09-ABSMAX/OBSERVATIONS OPEN,postclip없음.
- Absmaxchk seed7/data_seed20260921,1epoch,512/64/64,batch64,g11_every10,비정규화 sparse학습η/θ5.56134335. CPUtorch1.12/2threads test64/2022query 재평가전체MSE .3704960201865058/recall .3698382646185705/M_eff .13170154071437468,저장차0/parameter·checkpoint hash일치. 관찰1회라mean=max및count오류가숨음. 전체batch norm평균1.82371974/최대2.99063730/clip1/count8의완료JSON·CSV보존확인. Gradient재계산/새학습/8seedCI not run.
- 우선실제reducer/count/JSON전달을서로다른두관찰값으로검증한뒤같은validation경로진단. 기존③key/④entmax조건부순위및범위밖OPEN유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/(inventory,source.diff,absmax_probe.py,absmax_probes.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/absmax_probe.log. Exactcommand·문헌은ASSESMENT감사22. 학습/backward/optimizer/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now,json.dumps(check,ensure_ascii=False))
