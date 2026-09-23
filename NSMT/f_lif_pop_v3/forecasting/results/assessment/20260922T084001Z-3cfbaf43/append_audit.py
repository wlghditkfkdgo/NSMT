from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T084001Z-3cfbaf43';a=Path('f_lif_pop_v3/forecasting/results/assessment')/run;i=json.loads((a/'inventory.json').read_text());s=Path(i['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');v={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':[]}
for p,h in i['files'].items():
 if hashlib.sha256((s/p).read_bytes()).hexdigest()!=h:v['snapshot_changed'].append(p)
 if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:v['live_changed'].append(p)
assert not v['snapshot_changed'];(a/'validation.json').write_text(json.dumps(v,indent=2))
e='''

## 추적 감사 27 — NOW (예약 20260922T084001Z-3cfbaf43)

### 범위·핵심 판정

**新§2I는 미사용 confirm 분할과 validation 선택 규칙을 추가했으나, 현재 eta_selection.py는 그 계약을 충분히 강제하지 않는다. Confirm을 실제로 열기 전에 seed 완료 확인·G11 탈락·보고 항목·주장할 차이의 정의를 보완해야 한다.** 검사는 실제 confirm 데이터나 모델 평가 없이 합성 입력으로 수행했다. 다중 seed 학습의 미완료 상태를 모델 실패로 분류하지 않는다.

HEAD `f98fab4fe9940c281355597a0c91741f0b96ec05`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사26·문서 기억·사전등록§2I(D-AE~AJ)·canonical17:31 이후를 읽고 **17:40:58 KST** 실제224파일을 `/tmp/nsmt_assessment_RUN`에 snapshot·hash했다. Trigger 불일치0. 새 seed7의 checkpoint/config/result와 당시 존재한 seed13 config를 포함했다.

| 대상 | SHA256 |
|---|---|
| 사전등록 | `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168` |
| eta_selection.py | `e2c417ddfb58cdbd21e6ff3b220521b5ea8aa8293f230745b10d070c2178e0db` |
| config.py | `b033d265b97e9b5db01a849495e2c22fd0b9c088e15ad000d3fcdf0bf4496ae8` |
| synthetic.py | `3ca3199101ffdcf58f4da4279048834a32f8c131fb67a04eec50af6939495965` |

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/RUN/source.diff), [합성 protocol probe](../f_lif_pop_v3/forecasting/results/assessment/RUN/protocol_probe.py), [수치/실행흐름 증거](../f_lif_pop_v3/forecasting/results/assessment/RUN/protocol_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현 정확성

**분할 seed 연결은 확인됐다.** Dataset_Recall 생성자의 count/offset/RNG 선택 부분까지만 추출하고 RNG를 seed 기록 stub으로 대체했다. Train/val/test/confirm은 data_seed20260921에서 각각 **20260921 / 20270921 / 20280921 / 20290921**을 선택한다. 실제 confirm 샘플은 생성하거나 읽지 않았다. 기존 세 분할의 offset은 유지됐다. seed별 학습 난수와 data_seed는 별도다.

**A13-ETA-PROTOCOL OPEN — 다음 경로는 현재 snapshot에서 직접 확인한 구현 결함/계약 누락이다.**

| 영역 | 실제 검사·근거 | 남은 조건 |
|---|---|---|
| 8seed 완료 확인 | 실제 main을 fake run/metric으로 실행: seed7 하나만 있어도 confirm 진입. n=1의 CI `[nan,nan]` 뒤 “CI가0을 포함”이라고 출력 | 정확한8seed 집합·각 run 완료·설정/분할/보정 및 고정 checkpoint 확인을 **confirm 접근 전에** 수행. n=1은 CI 계산불가이며0 포함과 다름 |
| G11 finite | `row(..., diag.finite=False, peak5, bound10)`가 **within_bound=True** | finite 여부·값의 유한성·positive finite frozen bound를 명시적으로 검사 |
| G11 bound 누락 | bound=None도 **within_bound=True** | 보정 상한 없으면 확인 절차 진입 거부 또는 미판정. 정상통과로 분류하지 않음 |
| 진단 범위 | evaluate는 전체, selection_diagnostics는 **batches=4**. 16batch fake loader로 차이를 확인 | 전체 confirm/val에 대한 실제 max/finite 판정 또는 범위를 명시해 계약 고정 |
| 실패와 최종 결론 | 8seed 합성 confirm에서 모든 peak20>bound10으로 FAIL이어도 최종 **“개선이 유의함”** 출력 | 통계적 오차 감소와 안정성/채택 판정을 분리. G11 실패 seed를 사후 제외하지 말고 전부 보고하며 안정성 미통과를 최종 결론에 강제 |
| 위반 증거 | D-AG 요구 `G11_violation.json` 쓰기 경로 없음 | 위반마다 후보/seed/split/bound/peak/finite/hash를 구조화해 남길 것 |

**범위 구분:** config 기본 n_confirm=1000, batch64이면 앞256개만 진단되어 나머지744개는 상태 검사에서 빠진다. 그러나 새 seed7/13 저장 config에는 **n_confirm=256**이 명시돼 있으므로 이번 run 설정의4batch는 전체를 덮는다. 이번 run에서 실제744개가 누락됐다고 주장하지 않는다. 또한 synthetic finite=False 검사는 guard의 거부 동작을 확인한 것이며 실제 모델에서 비유한 상태가 발생했다는 뜻이 아니다. 본 감사는 실제 eta_selection CLI/confirm 평가를 실행하지 않았다.

**선택/1회 평가 보존:** 코드가 val 후보를 먼저 평가하고 그 뒤 confirm으로 넘어가는 순서는 맞다. 하지만 seed 디렉터리 존재만 수집하며, 학습 중 checkpoint와 완료 checkpoint를 구별하지 않는다. 중복 seed/다른 variant는 dict에서 마지막 경로로 조용히 덮어쓸 수 있다. 선택표·선택η·checkpoint/source/config hash를 confirm 이전에 고정 저장하는 단계도 없고 `--out`은 선택적이며 종료 후 기록된다. 중간 실패/재실행에서 confirm 재접근을 막거나 구분할 상태 기록이 없다. 이것이 이미 재관찰됐다는 증거는 아니며, “선택 뒤 한 번만”이라는 계약의 재현성을 확보할 남은 조건이다.

**D-AH 보고 항목 미충족:** 현재 row에는 recall/copy MSE, m_eff/hit/kernel/support/peak만 있다. score_top1/rank/rank_chance,p_mass,precap_mass,uniform_slot,recall-first MSE가 빠지고 confirm에서는 기준선의 상세진단도 저장하지 않는다. 분석이 진행 중인 것은 감안하되 현재 구현을 “§2I exactly”로 승인할 수 없다.

### (b) 사전등록·실행·통계 적절성

§2I에서 η1을 G11 실패로 제외하고 val recall 하나로 선택하며 seed7에서 한 번 고른 η를 공유하는 방향은 기존 감사 요구와 맞는다. 다만 **확증 평가 이전에 다음 모호성과 estimand 불일치를 먼저 해소**해야 한다.

- **D-AE 후보 수:** 후보표는 `{0,.1,.2,.3,.5}` 5개이나 바로 아래에서0은 후보가 아닌 기준선이라고 한다. 실제 eligible은 **.1/.2/.3/.5 네 개**다. 어떤 집합을 선택 대상으로 고정하는지 문서를 일치시킬 것. 이는 η0를 자동으로 선택시켜야 한다는 요구가 아니라 계약의 모순 정정이다.
- **D-AI vs D-AJ:** 보고할 CI는 selected−η0인데 주장 범위는 “원 학습η 대비 개선”이다. 코드도 η0 차이의 CI만 계산한다. **원 학습η 대비 차이/CI를 주 분석으로 할지, η0 대비를 주 분석으로 하고 주장을 바꿀지** confirm을 보기 전에 정하고 다른 비교는 보조로 표시해야 한다. η0 대비 CI만으로 원 학습η 대비 유의성을 주장할 수 없다.
- **w와 pre는 다르다:** §2I의 “precap_mass가 w 역할”은 감사25의 요구를 충족하지 않는다. `w=bp/Σbp`, `M_pre=(1−η)M_kernel+ηM_w`다. 예: b=[1,2],p=[.8,.2],정답첫칸,η=.2에서 w질량2/3이나 pre질량은.4다. w_mass를 별도 필드로 보고할 것. 원 궤적과 변경η 전체 forward의 구분도 유지한다.
- **통계 범위:** 8seed일 때 t 임계값2.365는 df7의 양측95% 근사이나, 코드의 n≠8→1.96 fallback은 미완료 seed 집합의 CI를 정당화하지 않는다. 정확한8seed 완료를 강제하고 안정성 실패/비유한 값/결측 처리 규칙을 사전에 고정할 것. 실제 seed가 공유하는 data_seed20260921은 초기화·학습 난수 반복을 뜻하며 데이터 생성 seed8개의 반복은 아니다. 추론 범위를 그에 맞춰 밝힌다.

**관찰 실행 상태:** `seeds-173737` seed7의 완료 JSON은12epoch, train2048/val256/test256/confirm256,batch64,softQKε.01이며 **test=None, test_skipped=True**다. 새로운 confirm 또는 test 성능 결과는 이번 snapshot에 없다. Seed13은 config/logargs까지 있어 진행 중 실행으로 취급한다. seed7 마지막epoch gradient norm mean.5955587104/max1.2339459658/postmax.9999993057,clip rate.03125/32batch,absmax 관찰4회로 기록됐다. 이는 기록 대조이며 gradient 자체를 재계산하지 않았다. Validation best_val_loss .2275817203은 전체 MSE이므로 recall-only나 confirm 성능으로 바꿔 부르지 않는다.

이번 판단은 **미사용 분할 도입과 선택 제어의 준비 상태**에 관한 것이다. 실제 후보 선택/실제 confirm 평가/8seed 완료/paired CI·효능 판정은 **not run**이며 모델 실패나 최종 성능 미달로 판정하지 않는다. 감사26의 통합 요약 조건 혼용은 이번 snapshot에서 별도 수정 증거가 없어 OPEN 유지한다.

### (c) 다음 우선순위

**현재 최우선은 confirm을 사용하기 전 절차를 완성하는 것**이다. 위4개 핵심 계약(선택 집합/주 비교, 정확한 seed 완료, 전체범위 G11·finite, 선택/평가 기록 보존)을 작은 합성 검증으로 먼저 확인할 것. 이어 모든 후보·기준선의 score→p→w→pre→post와 사건별 오차·실제 상태를 동일 분할에서 기록한다. η.2는 계속 후보 중 하나이며 이번 무결과 상태에서 채택을 확정하지 않는다. G11 실패의 성능이 좋아도 통과시키거나 실패 seed를 제거해서 유의성을 만들지 않는다.

기존 구조·QK 진단 뒤 causal key→entmax→별도 Gram/Delta의 조건부 순위는 유지한다. 커널 가중/혼합 구분은 앞서 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1), 실제 순환 안정성 점검은 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 기존 근거를 재사용한다. 새 문헌 주장이나 모델 효능 보장은 추가하지 않았다.

### 수행·보존

CPU Python3.10/torch1.12.0+cu113/2threads. 실제 AST 함수와 fake 모델/loader/metric으로 제어 흐름을 검사하고 dataset 생성자도 RNG seed 선택까지만 추출했다. 합성 n=1의 NumPy 자유도 경고/NaN은 확인 대상 경로의 증거이며 실제 연구 수치가 아니다. 실제 confirm 데이터 생성·열람·모델 평가, 학습/backward/optimizer/GPU·설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 접근/전송을 하지 않았다. Snapshot을 보존하고 감사3문서 append·텍스트 artifacts만 작성했다. 감사 도중 새 변경은 validation.json에 남겨 다음 주기로 넘긴다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/RUN/protocol_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/protocol_probes.json > f_lif_pop_v3/forecasting/log/assessment/RUN/protocol_probe.log 2>&1
```

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
m=f'''\n\n## 추적 갱신 — {now} (예약 감사27)\n\nHEADf98fab4f/snapshot17:40:58/224파일/manifest불일치0。新§2I/eta_selection.py/confirm offset30000検査。実confirm未生成未閲覧。A13-ETA-PROTOCOL OPEN:fake主処理seed1でもconfirm→CI NaN、finiteFalse/無boundもOK、全8seed G11 FAILでも有意改善表示。診断4batch固定:default1000では256のみ、実seeds config confirm256なら全件。8seed完了/同config固定checkpoint/重複排除/一回性・選択事前記録/G11記録不足。D-AE0候補5vs実4曖昧、D-AI eta0 CIとD-AJ原学習η主張不一致、D-AH不足/w≠pre。Seed7新12epoch no-test完了、seed13進行中、実confirm/性能/CI not run。候補選択前protocol完成優先、既存OPEN維持。詳細ASSESMENT감사27/results/assessment/{run}/.\n'''
# Keep the persistent memory Korean like the surrounding record.
m=f'''\n\n## 추적 갱신 — {now} (예약 감사27)\n\nHEADf98fab4f/snapshot17:40:58/224파일/manifest불일치0. 새§2I/eta_selection/confirm offset30000연결검사,실제confirm미생성·미열람. A13-ETA-PROTOCOL OPEN:합성main seed1도confirm진입→CI NaN,finiteFalse/상한없음도OK,8seed전원G11 FAIL이어도유의개선출력. 진단4batch고정(default1000이면256만,실제run confirm256은전체). 8seed완료·동일config/고정checkpoint·중복seed/1회성·선택사전기록/G11기록보완필요. D-AE0포함5후보vs코드4모호,D-AI eta0 CI와D-AJ원학습η주장불일치,D-AH필드부족/w≠pre. 새seed7 12epoch no-test완료/seed13진행중;실제confirm/성능/8seedCI not run. 미사용분할접근전protocol완성우선/기존OPEN유지. ASSESMENT감사27/results/assessment/{run}/.\n'''
l=f'''\n\n## {now} — 예약 추적 감사27: §2I 선택·confirm 절차의 구현 계약 검사 (학습 없음)\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADf98fab4fe9940c281355597a0c91741f0b96ec05,snapshot17:40:58/224파일,manifest불일치0. 감사commit/tag/push없음.\n- A13-ETA-PROTOCOL OPEN. 실제AST+fake main 검사에서seed1도confirm접근후NaN CI를0포함으로출력;diag finite=False/상한None도OK;8seed전원confirm G11 FAIL에도최종유의개선표시. 진단batches4고정은default1000에부족하나실제seeds config n_confirm256은전체. 8seed완료/고정checkpoint·설정일치/중복seed·선택사전기록/1회성·G11위반JSON·전체보고항목검사필요.\n- 사전등록보완:후보0포함5개vs기준선제외코드4개명확화;주비교D-AI eta0 vs D-AJ원학습ηCI불일치해결;w=bp/Σbp는pre=(1−η)kernel+ηw와다름. 모든후보/기준선score→p→w→pre→post+recall-first등D-AH필드추가.\n- Seeds-173737 seed7 12epoch2048/256/256/confirm256,batch64,no-test완료(test=None,skipped=True),seed13config는진행중. 새실제confirm/성능/pairedCI는not run. Offset의분리만RNG stub으로확인(20260921/20270921/20280921/20290921),미사용confirm을생성·열람·평가하지않음. 조건부초기화seed반복과data_seed반복구분필요.\n- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,source.diff,protocol_probe.py,protocol_probes.json,validation,append_validation);rawlog forecasting/log/assessment/동일run/protocol_probe.log. CPUtorch1.12/2threads합성검사만,학습/backward/optimizer/GPU/환경설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음. 미사용분할접근전선택protocol완성최우선,기존OPEN/후보순위유지. 상세수치·명령·문헌은ASSESMENT감사27.\n'''
checks={}
for path,t in [('docs/ASSESMENT.md',e),('docs/DOCS_REVIEW_MEMORY.md',m),('docs/PROJECT_LOG.md',l)]:
 p=Path(path);b=p.read_bytes()
 with p.open('ab') as f:f.write(t.encode())
 assert p.read_bytes()[:len(b)]==b;checks[path]={'prefix_sha256':hashlib.sha256(b).hexdigest(),'prefix_bytes':len(b),'preserved':True}
(a/'append_validation.json').write_text(json.dumps(checks,indent=2));print(now,v)
