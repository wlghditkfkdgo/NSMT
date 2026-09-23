from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T085001Z-3622969b';a=Path('f_lif_pop_v3/forecasting/results/assessment')/run;i=json.loads((a/'inventory.json').read_text());s=Path(i['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');v={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':[]}
for p,h in i['files'].items():
 if hashlib.sha256((s/p).read_bytes()).hexdigest()!=h:v['snapshot_changed'].append(p)
 if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:v['live_changed'].append(p)
assert not v['snapshot_changed'];(a/'validation.json').write_text(json.dumps(v,indent=2))
e='''

## 추적 감사 28 — NOW (예약 20260922T085001Z-3622969b)

**다중 seed 학습의 완료 증거가5개로 늘었다. 실제 confirm 결과와 η 개입 효능의 새 판단 근거는 없다.** HEAD `ae0f4ff50e247b255269342dd8f6879d192cfae8`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사27·기억·사전등록§2I·canonical17:43 운영 기록을 확인했다. 새 skill 등록은 관찰한 문서·운영 변경이며 감사에 사용하거나 다른 세션을 호출하지 않았다.

**17:50:39 KST** 실제245파일을 `/tmp/nsmt_assessment_RUN`로 snapshot·hash했다. 감지 시점과 불일치한 파일은 진행 중 seed256 CSV1개이며 snapshot에는9행이다. 연구 Python 소스와 사전등록은 감사27과 동일하다. `eta_selection.py` SHA256 `e2c417ddfb58cdbd21e6ff3b220521b5ea8aa8293f230745b10d070c2178e0db`, 사전등록 SHA256 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [메타데이터·checkpoint 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/metadata_probe.py), [검사 결과](../f_lif_pop_v3/forecasting/results/assessment/RUN/metadata_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현 정확성

새 모델/학습/η 선택 구현 변경이 없다. 기존 scoped VERIFIED를 유지한다. **A13-ETA-PROTOCOL OPEN 유지:** 감사27에서 확인한8seed 완료 확인, finite/frozen bound 거부, G11 실패와 최종 판정 연결, 선택/1회 평가 기록, D-AH 항목, 비교 기준·후보 집합 모호성은 같은 소스/사전등록 상태다. 동일한 합성 반례를 반복 실행하지 않았다. A09-SUMMARY-CONDITION-MIX 및 범위 밖 OPEN도 추가 수정 근거 없이 닫지 않는다.

### (b) 검증 진행·재현성

`seeds-173737`의 완료 JSON5개와 대응 checkpoint/config/CSV를 직접 대조했다. 모델을 CPU에서 복원해 파라미터 hash만 검사했으며 forward·dataset 생성·평가를 하지 않았다.

| seed | 완료 epoch / CSV 행 | best_val_loss (전체 MSE) | 완료 상태 |
|---|---:|---:|---|
| 7 | 12 / 12 | .22758172031123372 | no-test 완료 |
| 13 | 12 / 12 | .23057719745612862 | no-test 완료 |
| 21 | 12 / 12 | .22943382663187445 | no-test 완료 |
| 42 | 12 / 12 | .22813501182615330 | no-test 완료 |
| 123 | 12 / 12 | .22735320831781883 | no-test 완료 |
| 256 | 완료 JSON 없음 / 9행 | 미판정 | snapshot 시점 미완료 |
| 512,1024 | 완료 JSON 없음 | 미판정 | 완료 근거 미확인 |

완료5개 모두 **test=None/test_skipped=True**, JSON의 checkpoint SHA와 실제 파일 일치, 복원 파라미터 hash와 기록 일치, recorded source11개와 snapshot source 일치였다. JSON best_val_loss와 CSV 최소값도6자리 반올림 범위에서 일치했다. 이는 **저장·복원 identity와 기록 일관성 확인**이며 학습 과정을 재현했다거나 실행 시작 소스를 고정했다는 증명은 아니다(A10-PROVENANCE 잔여).

검사한 공통 설정은 일치했다: data_seed20260921,train2048/val256/test256/confirm256,batch64,최대12epoch,learnedη+sparse,softQKε.01,θ=.38146987702788376,input_scale8,frozen input norm,key_norm none,α.7,τ[4,8,16,32],G11 bound305.0375175476074. Seed별 초기화/학습 난수만 달라지는 실험이며 독립 데이터 생성 seed 반복은 아니다.

CSV의 관찰 시점 최대 상태는 완료5개에서22.684006~25.388060으로 bound 이하다. 하지만 **g11_every10**, epoch32batch이므로 관찰은4회다. 이 기록을 전체 step 안정성이나 최종 η 개입의 안정성으로 일반화하지 않는다. 미완료 파일 증가나 누락은 학습 실패로 판정하지 않는다. 감사 도중 파일이 늘거나 갱신되는 것은 다음 주기에서 다룬다.

표의 validation loss는 checkpoint 선택용 **전체 MSE**로 recall-only/confirm 지표가 아니다. 새 checkpoint들의 η 선택 성능·실제 confirm 평가·8seed paired CI·성능 우위 판단은 **not run**이다. 실제 confirm 데이터는 생성·열람하지 않았다.

### (c) 다음 방향

새 완료 seed 기록만으로 후보 채택이나 개선 순위를 바꾸지 않는다. **Confirm 접근 전 A13 절차 보완 및 정확한8seed 완료·설정·고정 checkpoint 확인이 우선**이다. 특히 “8개 학습이 끝나면 현재 eta_selection.py를 바로 실행”하는 운영 계획은 감사27의 계약 검사 후로 두어야 한다. 주 비교를 원 학습η 또는η0 중 명확히 고정하고, score→p→w→pre→post·전체 상태/finite·사건별 오차를 기준선까지 보존한다.

η.2는 후보로 유지하며 독립 검증을 기다린다. 관련 방향의 문헌 근거는 앞서 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1), [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 기존 범위 그대로다. 이번에는 새 학술 주장·성능 판단을 추가하지 않았다.

### 수행·보존

CPU Python3.10/torch1.12.0+cu113/2threads로 config/JSON/CSV·checkpoint hash 및 CPU 복원 파라미터 hash만 확인했다. 학습·forward·backward·optimizer·GPU·설치·소스 수정·git 변이·프로세스 중단·타세션 대화 접근/전송은 하지 않았다. 감사3문서 append 및 텍스트 진단 artifacts만 작성하고 기존 문서 prefix를 보존했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/RUN/metadata_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/metadata_probes.json > f_lif_pop_v3/forecasting/log/assessment/RUN/metadata_probe.log 2>&1
```

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
m=f'''\n\n## 추적 갱신 — {now} (예약 감사28)\n\nHEADae0f4ff5/snapshot17:50:39/245파일,manifest불일치seed256진행CSV1개(9행). 소스·§2I동일/A13 OPEN유지. Seeds-173737 완료7/13/21/42/123각12epoch/CSV12행,no-test;checkpoint/복원parameter/sourcehash일치,CSVminval반올림일치. Seed256완료JSON없음,512/1024완료근거미확인,실패판정아님. 공통data20260921/2048·256·256·confirm256/QKε.01/θ.381469877동일. WholevalMSE .22758/.23058/.22943/.22814/.22735는recall/confirm아님. Watch4/32batch상태22.684~25.388<305.038는전step안전증명아님. 실제confirm·η선택·8seedCI·학습/forward not run. Confirm전A13보완최우선,기존판정유지. ASSESMENT감사28/results/assessment/{run}/.\n'''
l=f'''\n\n## {now} — 예약 추적 감사28: 5seed 완료 기록·checkpoint identity 확인 (학습 없음)\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADae0f4ff50e247b255269342dd8f6879d192cfae8,snapshot17:50:39/245파일. Manifest불일치seed256진행CSV1개(9행),연구소스/§2I변경없음. 감사commit/tag/push없음.\n- Seeds-173737 seed7/13/21/42/123 완료각12epoch/CSV12행,test=None/skipped=True. JSONcheckpoint·CPU복원parameter·sourcehash일치/CSVminval일치. Best전체valMSE 각각.227581720311/.230577197456/.229433826632/.228135011826/.227353208318;recall-only/confirm판정아님. Seed256완료JSON없음/CSV9행,512/1024완료근거미확인,미완료를실패로분류안함.\n- 공통설정data_seed20260921,train2048/val256/test256/confirm256,batch64,QKε.01/θ.381469877/inputscale8/frozen input norm一致. 32batch당4watch의기록statepeak22.684~25.388은전체step/G11개입안정성보장아님.\n- A13-ETA-PROTOCOL OPEN유지,confirm전절차·정확8seed완료/고정checkpoint검사필요. 새η선택/confirm/pairedCI·성능판단not run. 실제confirm데이터미생성·미열람. CPUtorch1.12/2threads메타데이터+파라미터복원hash검사만;학습/forward/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음.\n- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,metadata_probe.py,metadata_probes.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/metadata_probe.log. Exactcommand·문헌은ASSESMENT감사28. 감사3문서append·텍스트artifact만작성.\n'''
checks={}
for path,t in [('docs/ASSESMENT.md',e),('docs/DOCS_REVIEW_MEMORY.md',m),('docs/PROJECT_LOG.md',l)]:
 p=Path(path);b=p.read_bytes()
 with p.open('ab') as f:f.write(t.encode())
 assert p.read_bytes()[:len(b)]==b;checks[path]={'prefix_sha256':hashlib.sha256(b).hexdigest(),'prefix_bytes':len(b),'preserved':True}
(a/'append_validation.json').write_text(json.dumps(checks,indent=2));print(now,v)
