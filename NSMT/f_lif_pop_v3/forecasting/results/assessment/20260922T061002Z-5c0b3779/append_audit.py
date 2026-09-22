from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');hs={x['path']:x['sha256'] for x in inv['files']}
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for x in inv['files']:
 p=Path(x['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=x['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 20 — NOW (예약 20260922T061002Z-5c0b3779)

### 관찰 범위·증거

**ε 보정 호환성과 전체 batch norm/clip 집계의 수정을 직접 확인했다. 개별 gradient absmax와 과거 G11 중단 사건은 아직 별도 OPEN이다.** 관찰 HEAD **b6d4baf6a54ee950403df1f60d05f290073bee07**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 15:10:37 KST에 manifest 대상·최신 감사/기억/사전등록§2H/canonical 기록198파일을 `/tmp/nsmt_assessment_20260922T061002Z-5c0b3779`로 snapshot·SHA256 기록했다. Trigger hash 불일치0. 실제 변경은 train/calibrate와 새 보정 JSON·qk_norm_form.txt이며 모델 변환은 감사19와 같다.

Train SHA256 `TRAIN_HASH`, calibrate `CAL_HASH`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/source.diff), [직접 진단 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/validation.json).

### (a) 구현 정확성 — 확인된 수정의 범위

**A10-CAL-QK-EPS VERIFIED:** 보정 출력명과 조회 패턴에 `qknorm-eps{qk_eps:g}`가 들어가며 payload의 qk_eps도 대조한다. 새 `...qknorm-eps0.01_seed7_260922-151001.json`을 실제 조회해 ε=.01은 승인, ε=1은 대응 파일 없음(None)을 확인했다. 파일명 필터와 별도로 **ε1 요청에 ε.01 artifact를 강제로 연결해도 ValueError로 거부**했다. 감사18의 잘못된 ε 보정 승인은 이 경로에서 해소됐다.

새 보정의 scale8/θ=.38146987702788376/G11 bound305.0375175476074는 이전 softε.01 보정과 같으며 기록된 layers/calibrate/config/synthetic source hash16자리 모두 사본과 일치했다. 보정 계산 전체를 다시 실행한 것은 아니며, 이번 검증은 조회·내용 호환성이다. Artifact 전체 SHA 고정/실행 시작 source 보존 등 A10의 다른 잔여 조건은 별도 유지한다.

**A09-CLIP-ALL-BATCHES 부분 VERIFIED:** `pre_clip` 반환값과 clip flag의 추가가 watch 조건 밖으로 이동했다. 함수 AST에서 실제 무조건 append문2개와 실제 reducer만 추출해 검사했다. 학습 loop/loss/optimizer는 실행하지 않았다.

| 주입한8batch pre-norm | 실제 새 집계 |
|---|---|
| [.5,2,.9,1.2,1,.2,3,.6] | mean **1.175**, max **3**, clip_rate_all_batches **.375**, batches_seen **8** |

새 필드 `grad_total_norm_pre_mean/max`, `clip_rate_all_batches`, `batches_seen`은 현재 myModel 진단 경로에서 정확히 집계됐고 최종 JSON 메타데이터 전달 목록에도 들어 있다(전달 배선 정적 확인). **새 학습 run에서 JSON/CSV가 이 필드를 끝까지 보존하는 검사는 not run**이다. 이전 gcmp/qk2 기록은 예전 표본 통계 그대로이며 새 의미로 재해석하지 않는다.

**A09-ABSMAX OPEN 유지:** 기존 `grad_absmax_all=[1,9]`는 여전히 **5**로 나온다. 새로 생긴 것은 **전체 gradient L2 norm의 batch 최대**이고, 요구한 모든 원소의 epoch `max|grad|`와 다른 양이다. 개별 WQ/WK absmax도 watch 표본 평균 구조가 남는다. True absmax·관찰 횟수·post-clip norm을 구분해 보완할 것. Norm 최대 추가만으로 A09 전체를 닫지 않는다.

**A05/G11-RECORD scoped VERIFIED:** G11 guard 안에서 run ID/UUID, epoch, batch, peak, bound, 보정 파일, mode, η, QK/ε, UTC를 JSON에 쓰고 예외를 발생시키는 경로가 추가됐다. Epoch loop의 `args.current_epoch` 대입도 확인했다. 해당 guard 블록만 **합성 값 peak11/bound10/epoch3/batch7**로 실행해 JSON 저장 및 FloatingPointError를 확인했다. Artifact는 `synthetic_g11/G11_violation.json`, run ID는 **AUDIT_SYNTHETIC_G11_NOT_A_TRAINING_RUN**이다. 실제 학습의 위반 기록과 혼동하지 않는다.

### (b) 검증 적절성·남은 근거

G11의 새 writer는 **향후 위반의 기록 경로를 검증한 것**이다. 감사19에서 남긴 qk2η1의 332.547 사건은 여전히 원시 증거 미확인이다. 이번 합성 기록으로 그 과거 사건을 VERIFIED로 전환하지 않는다. G11 검사 자체는 여전히 watch 표본에서 실행되므로 기록 보완이 전체 학습 step의 상태 감시를 추가한 것도 아니다.

`qk_norm_form.txt`는 hard/soft 모두1/ε로 유계이며 soft 전환이 유계성을 새로 만들지 않았다고 정정하고, 동일ε hard 실험은 **not run**으로 명시했다. 이는 감사18의 작은 Jacobian 검사와 맞으며 문서상 정정을 수용한다. Soft total/key gradient 표는 새 재현 코드·정확한 표본 식별자가 없으므로 **작업자 관찰값**으로 남긴다. 그 표만으로 hard 대비 형태 효과나 시간 누적의 유일한 원인을 확정할 수 없다. 기존 canonical14:09/14:18의 과도한 인과·전체 clipping 해석에 대한 감사18/19 제한은 유지한다.

이번 snapshot에 새 완료 학습 성능 결과는 없다. 사전등록 임계값/격자·data split·다중seed 정책에도 새 개정이 없다. **새 성능 판단 근거 없음**, 모델/checkpoint 재평가·새 학습·보정 전체 재실행·과거 G11 사건 재현·독립8seed/paired CI는 **not run**이다. 검사 명령 실패를 모델 실패로 판정한 항목은 없다.

### (c) 다음 개선 방향

통계·보정 수정이 확보됐으므로 다음 실제 run에서 **전체 batch 수·clip 횟수/norm 최대와 JSON/CSV 보존을 확인**할 수 있다. 동시에 같은 validation checkpoint/표본의 score→p→pre-cap→post-cap 지표를 먼저 연결한다. Score 단계부터 분리가 약하면③ causal key 표현, p 변환에서 손실되면④ entmax를 앞당기는 기존 조건부 우선순위를 유지한다. 새 성능 근거 없이 QK 정규화를 최종 채택/기각하거나 임계값을 바꾸지 않는다.

문헌 방향은 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 gradient/clipping 분석, [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 정규화–거리/scale 연결, [Adaptively Sparse Transformers](https://aclanthology.org/D19-1223.pdf)의 p 변환 근거를 재사용한다. 이번 판정은 직접 코드 경로·수치 검사에 기반하며 새 문헌 사실이나 효능 보장은 추가하지 않는다.

### 수행·보존

CPU Python/NumPy·기존 환경에서 보정 조회와 추출한 진단/예외 블록만 실행했다. 모델 forward/backward/학습/optimizer·GPU·설치·연구소스 수정·git 변이·프로세스 중단·다른 세션 대화 열람/전송은 하지 않았다. 실제 task 경로 대신 감사 artifact 디렉터리에만 합성 위반 기록을 썼고, 기존 문서 prefix와 snapshot hash를 보존했다. A02 실제 표본 집계 연결, A07-REGEN, A10-PROVENANCE·LOG 등 이번 범위 밖 OPEN은 유지한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probe.py \
/tmp/nsmt_assessment_20260922T061002Z-5c0b3779 \
f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T061002Z-5c0b3779/fixes_probe.log 2>&1
```

<!-- assessment-watch:20260922T061002Z-5c0b3779 -->
'''.replace('NOW',now).replace('TRAIN_HASH',hs['f_lif_pop_v3/forecasting/train.py']).replace('CAL_HASH',hs['f_lif_pop_v3/forecasting/calibrate.py'])
mem=f'''

## 추적 갱신 — {now} (예약 감사20)

HEADb6d4baf6a54ee950403df1f60d05f290073bee07/snapshot15:10:37/198파일. A10-CAL-QK-EPS VERIFIED:새eps파일조회.01승인/1미발견,강제.01→요청1 ValueError. 새보정θ.381469877/bound305.0375동일/sourcehash일치. A09 전체batch norm/clip집계추출검사 VERIFIED범위:8입력 mean1.175/max3/clip.375/count8. 새학습JSONCSV보존not run. absmax[1,9]→5남아A09전체OPEN/postclip없음. G11구조화기록guard합성peak11/bound10/epoch3/batch7에서write+raise검증,과거332.547사건확인아님. qk_norm_form의hard/soft유계정정수용,실제hard동조건not run. 새성능판단근거없음/8seedCI not run. ASSESMENT감사20/results/assessment/20260922T061002Z-5c0b3779/.
'''
log=f'''

## {now} — 예약 추적 감사20: 보정 ε 수정·전체 batch clip 집계·G11 구조화 기록 검사

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADb6d4baf6a54ee950403df1f60d05f290073bee07,snapshot15:10:37/198파일,manifest불일치0. 감사commit/tag없음. 연구변경train/calibrate·새보정·qk_norm_form,모델변환동일.
- A10-CAL-QK-EPS VERIFIED:ε.01자연조회승인/ε1조회None/강제잘못된artifact ValueError. 보정scale8/θ.38146987702788376/G11bound305.0375175 및sourcehash일치. 보정전체재실행not run.
- 실제AST append/reducer검사(학습loop미실행):8norm [.5,2,.9,1.2,1,.2,3,.6]→mean1.175/max3/clip .375/batches8. 전체batch수집·집계부분VERIFIED,새학습JSONCSV end-to-end not run. 기존absmax[1,9]→5라A09잔여OPEN/postclip없음. 과거gcmp/qk2표본통계를새의미로재해석금지.
- G11writer만합성peak11/bound10/epoch3/batch7로실행해JSON후raise확인. Run ID AUDIT_SYNTHETIC_G11_NOT_A_TRAINING_RUN;과거qk2 332.547사건검증아님. qk_norm_form의hard/soft둘다1/ε정정수용,실제hard동조건실험not run.
- 신규성능근거없음,모델/checkpoint/학습/GPU/독립8seedCI not run. Samevalidation score→p→c진단후③key/④entmax분기유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/(inventory,source.diff,fixes_probe.py,fixes_probes.json,synthetic_g11/G11_violation.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/fixes_probe.log. Exactcommand·문헌은ASSESMENT감사20. 연구소스수정/설치/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now,json.dumps(check,ensure_ascii=False))
