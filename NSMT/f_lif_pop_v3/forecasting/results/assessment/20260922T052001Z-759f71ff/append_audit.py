from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');hs={r['path']:r['sha256'] for r in inv['files']}
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for row in inv['files']:
 p=Path(row['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=row['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 19 — NOW (예약 20260922T052001Z-759f71ff)

### 観察 범위와 증거

**Soft QK 정규화의 η=.5/학습η 완료 결과를 재현했다. 성능 개선의 원인과 η1 중단의 원시 증거는 별도 확인이 필요하다.** HEAD **b6d4baf6a54ee950403df1f60d05f290073bee07**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **14:20:35 KST**에 manifest·최신 감사/기억/§2H/canonical 문서·대상 checkpoint208파일을 `/tmp/nsmt_assessment_20260922T052001Z-759f71ff`에 snapshot·hash했다. Manifest hash 불일치0, 감사18 대비 연구 소스 변경0. Trigger에 포함된 η.2 JSON은 감사18 사본과 같아 반복 평가하지 않았다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probe.py), [수치·hash](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/validation.json). Source SHA256: layers `LAYERS_HASH`, train `TRAIN_HASH`. 전체 소스/결과 hash는 inventory에 기록했다.

### (a) 구현 정확성 — 새 checkpoint 복원·수치 재현 VERIFIED, 기존 OPEN 유지

현재 API/진단 정의는 감사18과 동일함을 hash로 확인했다. `qk2-140541`의 새 완료η.5와 학습η checkpoint만 CPU 복원·평가했다. 두 실행 모두 전체 test MSE 및 M_eff의 저장값과 차이0, evaluated parameter/checkpoint SHA256 일치, 기록된 source hash 불일치0였다. Test 상태도 finite/G11 범위 안이었다. 이 결과는 **현재 soft 변환에서의 재현성 확인**이고 정규화의 최종 효능 검증은 아니다.

A18-QKNORM의 작은 score/zero-input/인과성 검사는 감사18의 확인 범위를 유지하며 전체 gate를 재실행하지 않았다. A09 gradient 표본/absmax reducer, A10-CAL qk_eps, A10-PROVENANCE·LOG, A02 실제 분포 연결, A07-REGEN 등의 잔여에는 수정 소스가 없으므로 OPEN 유지한다. Canonical14:18의 “clipping 전 진짜 최대·빈도 항목 충족” 선언은 **감사18의 실제 reducer·관찰 빈도 검사와 맞지 않는다.** 필드가 존재하는 것과 요구한 통계가 구현된 것은 다르다.

### (b) 실행 결과와 검증 적절성

공통 seed7/data_seed20260921,2048/256/256 sequence,batch64,12epoch,spike,α=.7/K4/key3,soft ε=.01/θ=.38146987702788376. Test256sequence/8085recall query 전체, M_eff/hit은 query→sequence 평균이다.

| 새 완료 조건 | recall MSE | M_eff | 최종 c hit | support_p | test max|u| |
|---|---:|---:|---:|---:|---:|
| QK η=.5 | **.319099823128** | .130831132549 | .059328653690 | .525685444474 | 47.66791534 |
| QK 학습η≈.026293 | **.273289752622** | .133798540174 | 0 | .351997204125 | 22.05676270 |

η.5는 이전 비정규화 η.5의 .433504353901보다 오차가 낮고, full .276338636158보다 높다. 학습η는 비정규화 .272620149439보다 약.000670 높은 오차다. 이전 oracleη1을 공통 분모로 한 탐색 G는 각각 **−.1771581586 / +.0126314219**다. 따라서 “사용 가능한 모든η에서 G음수”는 학습η까지 포함하면 틀리며, **양의 고정η .2/.5**의 관찰로 제한할 것. Oracle 분모는 이전 확인된 기준이며 이번에 다시 학습한 값이 아니다.

**A09-INTERPRETATION — 동반 변화와 인과를 분리:** η.5에서 support 증가(.39065→.52569), c hit 감소(.09671→.05933), M_eff 감소(.13287→.13083)가 관찰됐다. 이는 **오차 개선과 검색 지표 개선이 함께 나타나지 않았다**는 근거다. 그러나 support는 양수 원소 비율이며 entropy·uniform까지의 거리나 full과의 실제 계수 차를 직접 재지 않는다. 각 조건의 가중치·학습 경로·θ/ε도 달라 **“더 균등해져 full에 가까워진 것이 성능 개선 원인”은 아직 가설**이다. 고정 checkpoint에서 정책/계수만 통제한 비교와 동일 표본의 score→p→c 측정이 필요하다.

Canonical14:18의 η.5 상태 비교 기준 **22.7은 비정규화 η0(full)**에 해당한다. 대응되는 비정규화η.5의 감사15 test peak는 **20.91826248**이므로 조건을 맞춰 **20.9183→47.6679**로 기록할 것. 상태 증가 관찰 자체는 유지되지만 이를 support 확대의 인과 효과로 단정하지 않는다. 현재 max값은 전체256 test에서의 peak이고 학습 전체 peak와도 구분한다.

**A05/G11 — η1 미완료와 보고된 중단을 구분:** `qk2` η1은 CSV epoch0–7의8행과 checkpoint가 있으나 **완료 JSON 없음**. `e1chk-141652`도 CSV1행/완료 JSON 없음이다. Canonical은 η1의 **332.547>305.038로 중단**을 보고한다. 현재 소스에는 G11 예외 guard가 있지만, 보존된 CSV에 그 실패 batch 값은 없고 task log의 .log/.txt/.out 검색에서도 해당 예외 원문을 찾지 못했다. 따라서 판정은 **작업 기록상 G11 중단 보고, 실행 이벤트의 원시 근거 미확인**이다. 훈련을 재실행하거나 미완료 checkpoint를 최종 결과로 평가하지 않았다. 해당 사건을 VERIFIED로 닫으려면 run ID/epoch/batch와 예외 stdout 또는 구조화된 실패 기록을 기존 산출물에 연결할 것. 실패 batch가 epoch CSV에 남지 않는 것은 코드상 가능하며 수치 조작이나 모델 전체 실패로 오인하지 않는다.

`qknorm_grid.txt` 말미의 `TypeError: unsupported format string passed to NoneType.__format__`는 **누락 결과의 표 출력 오류**다. η1의 실제 G11 예외와 별개의 사건이다. 누락 결과는 `not completed/reported G11 stop`으로 명시하고 요약을 끝까지 생성하도록 보완할 필요가 있다.

η.5 마지막 epoch의 pre-clip norm14277.378/clip_rate1.0과 학습η의 .501578/clip_rate0도 **32batch 중4회 관찰**의 평균/비율이다. 전체 epoch 최대·모든 batch clipping 빈도·post-clip norm은 여전히 미확인이다. 새 표를 근거로 A09 수정 완료로 닫지 않는다.

### (c) 다음 방향 — ③ 채택 전 같은 표본의 경로 진단

현재까지 **QK 정규화 설정 묶음은 일부 고정η에서 오차를 낮췄지만 검색 성공을 입증하지 못했다.** 한 seed·짧은 예산으로 최종 기각 또는 일반적 효능을 확정하지 않는다. 작업자가 계획한 **동일 validation checkpoint/표본의 score 순위와 p 질량 분리 측정**을 다음 판단 근거로 삼는 것은 타당하다. Score 단계부터 정답 분리가 약하면③ causal key 표현, score 신호가 p에서 사라진다면④ entmax를 앞당기는 기존 조건부 우선순위를 유지한다. c hit/support만으로 그 분기를 정하지 않는다.

정규화의 거리/scale 연결은 앞서 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1), p 변환 후보의 근거는 [Adaptively Sparse Transformers](https://aclanthology.org/D19-1223.pdf), 시간별 gradient 분석은 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)를 재사용한다. 이번 신규 사실은 직접 결과 재현에 근거하며 새 문헌 주장은 없다. Validation으로 후보와 hyperparameter를 결정하고 반복 본 test는 탐색으로 명시한다. 새학습·η1 성능 평가·원시 G11 실패 재현·독립8seed/paired CI·효능 확증은 **not run**이다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7, 새 완료 checkpoint2개 forward 평가만 수행했다. 학습/backward/optimizer·GPU·환경 설치·모델/학습 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/메시지 전송은 하지 않았다. 기존 기록과 checkpoint를 보존하고3문서 append·진단 artifact만 작성했다. 감사 중 추가 변경은 다음 주기로 넘긴다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probe.py \
/tmp/nsmt_assessment_20260922T052001Z-759f71ff \
f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T052001Z-759f71ff/completion_probe.log 2>&1
```

<!-- assessment-watch:20260922T052001Z-759f71ff -->
'''.replace('NOW',now).replace('観察','관찰').replace('test max|u|','test 최대 상태 크기')
body=body.replace('LAYERS_HASH',hs['f_lif_pop_v3/forecasting/layers.py']).replace('TRAIN_HASH',hs['f_lif_pop_v3/forecasting/train.py'])
mem=f'''

## 추적 갱신 — {now} (예약 감사19)

HEADb6d4baf6a54ee950403df1f60d05f290073bee07/snapshot14:20:35/208파일,manifest불일치0/소스변경0. Qk2새η.5·학습η checkpoint CPUtest256/8085query 저장MSE/M_eff차0/hash일치. Recall .319099823128/.273289752622,M_eff .130831132549/.133798540174,G −.177158/+ .012631. η.5support확대·c hit감소는관찰이나uniform접근이개선원인이라는단정불가. 대응비정규화η.5 testpeak20.9183(22.7은full)→47.6679. η1 qk2 CSV8epoch/e1chk1epoch·완료JSON없음. G11 332.547중단은canonical보고,원시예외미확인;검증not run. 요약txt TypeError는None표출력오류로G11과별개. A09표본clip/absmax평균및ε보정누락잔여유지. 같은validation score→p→c진단후③key/④entmax분기;8seedCI/새학습not run. ASSESMENT감사19/results/assessment/20260922T052001Z-759f71ff/.
'''
log=f'''

## {now} — 예약 추적 감사19: QK η.5·학습η 재현, 인과 해석·η1 사건 근거 분리

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADb6d4baf6a54ee950403df1f60d05f290073bee07,snapshot14:20:35/208파일,manifest불일치0/연구소스변경0. 감사commit/tag없음.
- CPU torch1.12/2threads/seed7/data_seed20260921,2048/256/256,batch64,12epoch,softε.01/θ.381469877/spike/α.7/K4/key3. 새완료η.5·학습ηtest256/8085query MSE·M_eff차0/hash일치. Recall .319099823128/.273289752622, M_eff .130831132549/.133798540174,기존oracle1공통분모G −.1771581586/+ .0126314219. 모든η에서G음수아님. η.5오차개선은있으나full미달.
- **14:18 해석 정정 요청:** support확대만으로uniform접근·개선원인단정불가. 대응비정규화η.5 testmax는20.9183이지full22.7아님. 진짜gradient최대/빈도충족선언은현watch/reducer와불일치(A09OPEN). θ/ε/학습경로교란유의.
- η1 qk2 CSV8행/e1chk1행·완료JSON없음. 332.547>305.038 G11중단은canonical보고,원시예외tasklog에서미확인;그사건VERIFIED아님. 요약qknorm_grid TypeError는None출력오류와구별. η1평가/중단재현/새학습/8seedCI not run. Samevalidation score/p/c진단후③key/④entmax분기권고.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/(inventory,completion_probe.py,completion_probes.json,validation,append_validation);rawlog forecasting/log/assessment/동일run/completion_probe.log. Exactcommand·문헌은ASSESMENT감사19. 학습/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now,json.dumps(check,ensure_ascii=False))
