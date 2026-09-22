from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');hashes={r['path']:r['sha256'] for r in inv['files']}
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for row in inv['files']:
 p=Path(row['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=row['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 18 — NOW (예약 20260922T051001Z-b66150e0)

### 관찰 범위·핵심 판정

**Soft QK 정규화의 새 완료 결과2개를 재현했다. 보정 ε 호환성 누락과 clipping 표본을 전체로 일반화한 해석은 정정이 필요하다.** 관찰 HEAD **1832bdc16cd075473c582bd386686da18775ece8**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **14:10:34 KST**에 소스·문서·결과·해당 checkpoint211파일을 `/tmp/nsmt_assessment_20260922T051001Z-b66150e0`에 snapshot·hash했다. Trigger 이후 qk2η.2 CSV가 바뀌었으며, 실제 사본에는 **η0/.2 결과 JSON이 모두 완료**돼 있었다. η.5는 config/checkpoint만 있고 완료 JSON이 없어 성능 평가에서 제외했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/source.diff), [직접 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probe.py), [수치 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probes.json), [run config](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/run_configs.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/validation.json).

관찰 source SHA256: layers `LAYERS_HASH`, train `TRAIN_HASH`, config `CONFIG_HASH`, calibrate `CAL_HASH`.

### (a) 구현 정확성

**A18-QKNORM — 현재 변환·배선의 제한된 검증 VERIFIED.** 설정이 Selector까지 전달되고, 투영 q/k에 `x/sqrt(||x||²+ε²)`를 적용한 뒤 기존 거리/scale→p→rho→cap 경로를 사용한다. 이는 norm=1을 정확히 강제하는 연산이 아니라 **soft 정규화**다. ε=.01, 작은 float64 입력에서 독립 계산한 거리와 실제 score 최대차0, zero 입력 finite를 확인했다. `score_scale()`도 동일한 변환을 반영하도록 수정됐다(정적 확인; 보정 전 과정 재실행은 not run). 실제 η.2 checkpoint에서2sequence의 미래 절반 입력을 바꿔도 앞21event 출력 최대차0이었다. 이 작은 인과성 검사만으로 모든 설정의 전체 gate를 통과했다고 확대하지 않는다.

**A02-REACHABILITY — 정확식 구현 수정 VERIFIED(범위 제한).** `free_bound`가 제거되고 `sampled_best`와 `exact_bound`가 분리됐다. 이전 구성 반례에서 exact_bound=.5625772591831768, 정책의 실제 질량과 차 **1.11e-16**이었다. 격자/연속 경계 표시도 코드에 구분됐다. 따라서 ‘무작위 탐색을 상한으로 사용’한 **계산 구현 결함은 해소**됐다. 다만 이 코드는 고정 history41/배치 예시이며 실제 query→sequence 분포 적용이 추가된 것은 아니다. §2H 운영 결론에 실제 표본 상한을 연결하는 잔여 조건은 유지한다. 작업자의 “계산 근거 잔여를 닫는다”는 선언을 O7 일반성·효능 검증으로 확장하지 않는다.

**A10-CAL — key_norm/qk_norm 적합성 확인, 신규 qk_eps 누락 OPEN.** Captured `...qknorm_seed7_260922-140541.json`을 명시해 검사했다. key_norm none→frozen 및 qk_norm True→False는 실제 거부돼 기존 key_norm 검사의 누락은 이 조건에서 수정 확인했다. 그러나 **보정 qk_eps=.01에 대해 요청 qk_eps=1.0을 그대로 승인**하며 θ=.38146987702788376을 반환한다. ε는 변환·score scale을 바꾸므로 보정 호환성 검사와 설정 식별에 포함해야 한다. 이번 정상 qk2 실행은 실제 ε=.01끼리 맞아 이 결함이 그 결과를 오염시켰다는 뜻은 아니다. 보정 artifact hash/시작 코드 고정 등 기존 잔여는 유지한다.

### (b) 실행·검증 적절성

**새 완료 결과 재현:** qk2-140541, seed7/data_seed20260921, train/val/test2048/256/256,batch64,12epoch,spike,α=.7/K4/key3,scale8,soft ε=.01/θ=.38146987702788376. 전체 test256sequence/8085recall query를 평가했다.

| 조건 | recall MSE | M_eff | 최종 c hit | 저장값과 차이 |
|---|---:|---:|---:|---|
| soft QK η0 | .276338636158 | .130328894393 | 0 | 전체MSE/M_eff 모두0 |
| soft QK η.2 | **.306119084689** | **.131735961409** | .041690452399 | 전체MSE/M_eff 모두0 |

두 run의 evaluated parameter/checkpoint SHA256도 일치했다. η0 결과는 이전 full과 같으며 정상 비활성 대조다. η.2는 이전 비정규화 η.2의 recall .319387285780/M_eff .127181601094보다 이 seed에서 개선됐지만, **full .276338636158보다 오차가 크고** kernel mass .130328893616 대비 추가 정답 질량은 약.001407에 그친다. 정규화·ε·θ 재보정이 함께 바뀐 비교이므로 정규화 형태 하나의 독립 효과로 부르지 않는다. η.5/1·학습η의 현재 soft 격자는 **진행/미완료**, 해당 성능 재검사는 **not run**이다.

**A09-GRADIENT/CLIP OPEN — 표본과 전체를 구분해야 한다.** `pre_clip`으로 변수명을 바꿔 손실 누적 `total` 덮어쓰기를 피한 코드와 qk2 완료 결과를 확인했다. 작업자가 기록한 초기 IndexError는 실행 구현 결함이며 모델 이론 실패로 세지 않는다. 그러나 새 로그도 `watch = (batch_index % g11_every == 0)`일 때만 gradient norm/clip flag를 수집한다.

- Gcmp6조건은 **512sequence/batch64=8batch**, g11_every10이므로 관찰은 **batch0 한 번**이다. 보고된 clip_rate1.0은 ‘그 표본이 clipping됨’이며 canonical14:09의 **“매 배치가 잘린다”는 결론을 지지하지 않는다.** 표의 pre-norm도 epoch 전체 평균이 아닌 첫 관찰 batch 값이다.
- Qk2는32batch 중 index0/10/20/30의 **4회 표본**이다. η.2 마지막 epoch pre-norm86.0009/clip_rate1.0은 그4회 평균/비율이다. η0의 pre-norm.45993/clip_rate0도 모든32batch가 clipping되지 않았음을 증명하지 않는다.
- 실제 reducer 부분만 AST로 추출해 optimizer 없이 실행했다. `grad_absmax_all=[1,9]`는 **5**, pre-norm `[.5,10]`은5.25, clip flag `[0,1]`는.5로 집계된다. **진짜 epoch absmax와 전체 clipping 빈도는 아직 구현되지 않았고 post-clip norm도 없다.** 매 batch의 전역 norm 반환값은 이미 있으므로 추가 모델 forward 없이 전체 횟수/분모를 집계할 수 있다. sampled 지표는 관찰 횟수를 명시하고 최대값은 max로 분리할 것.

**A09 원인 단정 제한:** `grad_localise.txt`는 파라미터군별 큰 gradient의 위치를 보여주지만 실행 script/정확한 입력·seed·source hash와 Jacobian 경로 분리가 없다. “원인을 정확히 특정”, “남은 폭주는 T-step 누적이며 정규화는 원리적으로 고칠 수 없다”는 단정은 현재 증거를 넘는다. Hard norm도 `x/max(||x||,ε)`이면 미분 가능한 영역에서 norm bound1/ε를 갖는다. 직접4차원 local Jacobian을 계산해 **hard/soft 모두 x=0에서 ε1e-6→1e6, ε.01→100**을 확인했다. Soft 전환이 유계성을 새로 만든 것이 아니며 smoothness와 ε 크기의 효과를 분리해야 한다. 한 스텝의1/ε 상한과 전체 BPTT gradient1e13의 ‘규모 일치’ 역시 같은 양의 직접 비교가 아니다. 이 작은 Jacobian 검사는 모델 학습/backward 재현이 아니다.

**A10-PROVENANCE:** Gcmp/qkchk2는 기록된 config/layers hash가 현재 soft 소스와 다르고, 옛 config에는 qk_eps가 없다. 현재 기본값을 채워 읽으면 당시 hard 변환과 달라질 수 있다. 이들의 수치는 **당시 기록 관찰**로 남기고 현재 soft 소스로 동일 의미의 재현을 했다고 부르지 않는다(이번 checkpoint 재평가 not run). Hard/soft 방식·ε·θ·source snapshot을 함께 보존할 것. 새 qk2 완료2건은 사본 소스에서 재현됐다.

### (c) 개선 방향·채택 판단

**② QK 정규화는 탐색 진행 단계이며 최종 채택/기각을 보류한다.** 같은η에서 hard/soft, ε, θ 보정 정책을 구분하고 validation 표본에서 score/p/pre-cap/post-cap 질량·순위와 실제 오차를 연결한다. 우선 gradient 통계를 전체/표본으로 바로잡고, 작은 norm 발생 빈도와 key/query 경로별 local Jacobian·시간 길이별 gradient 변화를 통제해 원인 가설을 확인할 것. 원인을 단정한 상태로 여러 구조를 동시에 바꾸지 않는다. 이후 key 표현→entmax의 순서는 기존 조건부 규칙을 유지하며 Gram/delta는 별도 구조 대조로 둔다.

앞서 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)는 정규화와 거리/scale 연결의 근거이며, soft ε 선택이나 폭주 제거를 보장하지 않는다. [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간에 따른 gradient 곱·clipping 분석을 통제 진단 근거로 재사용한다. 이번 새 수치 판단은 위 직접 검사와 고정 결과에 기반하며 신규 문헌 주장은 없다. 독립8seed/paired CI·등록 예산 확증·성능 우위는 **not run**이다. Test 반복 관찰은 탐색으로 기록하고 후보 선택은 validation으로 제한한다.

### 보존·수행 기록

CPU torch1.12.0+cu113/2threads/seed7. 완료 checkpoint2개 forward, 작은4차원 local Jacobian, 실제 reducer만 실행한 진단, 보정호환성·정확상한 계산을 수행했다. 모델 학습/optimizer·GPU·소스 수정·환경 설치·프로세스 중단·git 변이·타세션 대화 접근/전송은 하지 않았다. A07-REGEN 및 미검사 잔여는 유지한다. 진행 파일 변경은 다음 주기에 다루고 기존 감사 본문을 고치지 않고 append한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probe.py \
/tmp/nsmt_assessment_20260922T051001Z-b66150e0 \
f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T051001Z-b66150e0/current_probe.log 2>&1
```

<!-- assessment-watch:20260922T051001Z-b66150e0 -->
'''.replace('NOW',now)
for token,name in [('LAYERS_HASH','layers.py'),('TRAIN_HASH','train.py'),('CONFIG_HASH','config.py'),('CAL_HASH','calibrate.py')]:body=body.replace(token,hashes['f_lif_pop_v3/forecasting/'+name])
mem=f'''

## 추적 갱신 — {now} (예약 감사18)

HEAD1832bdc16cd075473c582bd386686da18775ece8/snapshot14:10:34/211파일. SoftQK ε.01 구현배선/score차0/zero finite/실제prefix미래교란차0 확인. qk2η0/.2완료test256/8085query 재현 MSE·M_eff차0/hash일치:recall .276338636158/.306119084689,M_eff .130328894393/.131735961409. η.2비정규화 .319387보다개선이나full보다나쁨,ε·θ함께변경. η.5진행,미완료성능not run. A02 exact_bound 반례차1.11e-16로계산수정VERIFIED,실제분포연결잔여. A10-CAL key_norm/qk_norm거부VERIFIED,ε.01보정→요청1승인 신규OPEN. A09 clip_rate는watch표본: gcmp8batch중1회,qk2 32중4회;‘매batch잘림’근거없음. absmax[1,9]→5유지/postnorm없음. Hard/soft zeroJacobian둘다1/ε,폭주원인특정단정제한. Gcmp당시hard소스/epsilon미기록을currentsoft로재현하지않음. QK탐색유지/8seedCI not run. ASSESMENT감사18/results/assessment/20260922T051001Z-b66150e0/.
'''
log=f'''

## {now} — 예약 추적 감사18: soft QK 결과 재현·gradient 표본 해석·보정 ε 누락

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD1832bdc16cd075473c582bd386686da18775ece8.14:10:34 snapshot211파일. 감사commit/tag없음. Sourcehash·정확명령은ASSESMENT감사18/inventory.
- CPU torch1.12/2threads/seed7,data_seed20260921,train/val/test2048/256/256,batch64,12epoch,α.7/K4/key3/spike,scale8,softε.01/θ.381469877. Qk2완료η0/.2의전체test256/8085query MSE·M_eff차0/hash일치. Recall .276338636158/.306119084689, M_eff .130328894393/.131735961409. η.2비정규화 .319387→.306119탐색개선이나full보다오차큼. η.5등미완료는성능판정not run.
- A18-QKNORM 작은score독립계산차0/zero finite/실제η.2prefix인과차0. A02 exact_bound 이전반례와차1.11e-16로계산수정VERIFIED(실제query분포연결잔여). A10-CAL key_norm/qk_norm변경거부확인,ε.01보정으로ε1요청승인OPEN.
- **14:09 §3 정정 요청:** gcmp512/batch64=8batch,g11_every10으로clip_rate는첫batch1회값,‘매배치clipping’근거없음. Qk2는32중4회. 실제reducer absmax[1,9]→5,postnorm없음(A09OPEN). Hardfloor/softzeroJacobian모두1/ε(1e-6→1e6,.01→100);유계성을soft만의효과로설명하지말것. Local gradient위치만으로원인확정/T누적은정규화로원리적해결불가단정제한.
- Gcmp/qkchk2당시config/layers해시상이,qk_eps없음;현재soft로당시hard의미재현not run. 새QK탐색진행/독립8seedCI not run. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/(inventory,source.diff,current_probe.py,current_probes.json,run_configs.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/current_probe.log. 모델학습/optimizer/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append와진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now);print(json.dumps(check,ensure_ascii=False))
