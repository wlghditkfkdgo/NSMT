from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
run='20260922T015001Z-bff8435c';out=Path('f_lif_pop_v3/forecasting/results/assessment')/run;stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');inv=json.loads((out/'inventory.json').read_text());h={r['path']:r['sha256'] for r in inv['files']}
body=f'''

## 추적 감사 12 — {stamp} (예약 `{run}`)

### 관찰과 고정 증거

기억·감사11·사전등록 §2F·canonical 기록을 복구했다. **10:50:38 KST**, HEAD **520bb56872c6e7c68c48f65cd70601979c22a1c5**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7에서139개 파일을 `/tmp/nsmt_assessment_{run}`에 snapshot/hash했다. Trigger 파일 hash는 모두 일치했다. 모델 SHA256 `{h['f_lif_pop_v3/forecasting/model.py']}`, train `{h['f_lif_pop_v3/forecasting/train.py']}`, key_geometry `{h['f_lif_pop_v3/analysis/key_geometry.txt']}`. 감사 중 canonical10:50 검색 후보 문서가 추가되어 읽고 postscript로 별도 보존했다. 사전등록은 §2F 그대로다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/{run}/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/{run}/changes.diff), [독립 probe](../f_lif_pop_v3/forecasting/results/assessment/{run}/restore_grad_probes.json), [후속 문서](../f_lif_pop_v3/forecasting/results/assessment/{run}/document_postscript_inventory.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/{run}/validation.json). Raw console: forecasting/log/assessment/{run}/restore_grad_probe.log.

### (a) 구현 판정

**A10-RESTORE-STATS, VERIFIED(보고된 누락 승인 결함 해소).** 임시 checkpoint 사본을 사용하는8조건에서 정상 파일 승인, input_norm=frozen의 norm 통계 누락 거부, key_norm=frozen의 key 통계 누락 거부, 고정η의 eta_value 누락 거부, 일반 weight 누락 거부를 확인했다. Norm/key 설정이 none인 경우와 학습η(None)의 선택적 buffer 누락만 허용됐다. 기존 legacy pilot도 정상 로드되고 recall MSE **0.26989367121801305**가 재현됐다. 이는 누락 검사 수정의 확인이며 잘못 짝지어진 config나 모든 내용 변조를 검출하는 schema 검증까지 의미하지 않는다. 원본 checkpoint에는 fault를 주입하지 않았다.

**A09-GRAD, 부분 수정 / 최대값 집계 OPEN.** Q/K absmax와 모델 전체 absmax를 clipping 전에 수집하고 L2 norm 이름을 분리한 것은 확인했다. 그러나 공통 reducer가 **여전히 np.mean(finite)**다. 고정 소스의 집계 AST만 실행한 주입 검사에서 Q absmax [1,9]→**5**, 전체 absmax [2,10]→**6**으로 기록됐다. 따라서 이름이 absmax여도 epoch 최대값이 아니라 **관측 batch 최대값의 평균**이다. max reducer와 관측 batch 수를 명시하고, 전체 epoch 최대를 주장하려면 모든 batch에서 수집해야 한다. 현재 g11_every 표본 관측을 전체 최대라고 부르지 말 것.

**A09-NONFINITE, VERIFIED(JSON 전달 배선):** 같은 집계에 NaN1개와 유한값1개를 넣어 생성된 score_std_nonfinite=1이 실제 payload assignment를 거쳐 보존됨을 확인했다. Train 함수/optimizer는 실행하지 않았다. 실제 nonfinite 학습 중단 뒤 진단이 저장되는 경로와 동적 CSV 열 문제는 이번 확인 범위가 아니다.

**A10-THETA-JSON, VERIFIED:** 새 gradchk 결과에 theta **5.561343350061557**, calibrated_fields **[input_scale,theta]**가 직접 저장돼 있다. Config에만 저장되던 이전 제한은 이 결과부터 해소됐다. 보정 artifact ID/hash·호환 계약과 실행 시작 source snapshot 요구는 여전히 OPEN이다.

### (b) 새 실행·분석의 검증 수준

**gradchk-104646:** seed7/data_seed20260921, r2/k3, train/val/test512/64/64,batch64,2epoch,제한batch0,g11_every10의 기능 실행이다. 저장 checkpoint 재평가 전체 MSE **0.3377994264108825**, recall **0.34776720240825**로 기록과 차0이며 evaluated parameter/checkpoint hash가 일치했다. CSV2행에 absmax 열이 존재한다. 전체 gradient 표기는 epoch0 **1.441656**, epoch1 **0.385658**(CSV 반올림), JSON 최종0.385657817125다. **epoch당8batch 중 watch는 i=0 한 번**이라 이 실행만으로 다중 표본 reducer 오류가 드러나지 않는다. 새 성능 우위/안정성 일반화 근거로 보지 않는다.8seed/CI·정식 O7은 **not run**이다.

**A09-KEY-GEOMETRY 신규 OPEN(재현 정보·해석):** key_geometry.txt는 숫자와 test256 문구만 담고 명령/코드·checkpoint hash·정확한 표본·mask·집계식이 없다. Canonical10:50은 diag3 sparse/test64라고 설명하지만 이를 연결한 실행 가능한 진단 파일은 확인하지 못했다. 수치 재생성은 **not run**이다. 현재 모델의 유닛별 raw ξ는5차원, 투영 key는4차원이다. 따라서 유닛별42개 투영 key 행렬의 rank는 **최대4**이며 participation ratio3.97을 ‘42에 가까워야 하는데 실패’라고 해석할 수 없다. 다른 단위로 flatten/평균했다면 행렬 shape·중심화·정규화·고유값 cutoff부터 밝혀야 한다. 특히 42×42 token Gram과4×4 feature Gram을 구분할 것.

정답 최근접 확률0.2924와 계수 hit3.73e-5 비교도 현재 서로 같은 checkpoint·recall mask·unit 평균·sequence 평균인지 입증되지 않았다. 3.73e-5는 감사10에서 **pilot-eta checkpoint**에 얻은 값이고 diag3 sparse 전체 split의 hit는0이었다. 같은 표본으로 score순위→정책p→post-cap계수c→readout을 단계별로 계산해야 한다. 정답 집합에 여러 칸이 있으므로 rank의 무작위 기준50%도 순위 정의/집합 크기/후보 수에 맞춰 명시할 것.

**문서 정정 수용과 남은 과장:** canonical10:47의 singleton/폭주·소멸/η원인 단정 철회는 확인했다. 그러나10:40의 ‘약한 신호만 soma에서 사라진다/병목이 조건부 성립’ 및10:50의 ‘점수 함수를 아무리 개선해도 드러나지 않는다’는 여전히 확인 범위를 넘는다. End-to-end readout 비교는 서로 학습된 표현이 달라 같은 표현에 대한 인과 절제 실험이 아니다. ‘drive는 스파이크가 전혀 없다’도 계산 경로 기술로는 부정확하다: 현재 forward는 soma/spike를 계속 계산하되 drive 출력이 이를 우회한다. 비스파이킹 진단 readout이라는 범위로 쓰는 것이 정확하다.

### (c) 새 문헌 원문 대조와 개선 방향

후속10:50 문서에 새 논문·보장 주장이 있어 실제 원문을 확인했다. 아래는 도입을 승인하는 것이 아니라 **각 후보를 검증 가능한 가설로 좁히는 권고**다.

- **QK 정규화/Gram 보정:** [Wang·Shi·Fox, Test-time regression §3](https://arxiv.org/html/2501.12352v1)는 선형 최소제곱의 feature covariance 보정과 비선형 kernel Gram 보정을 구분하며, QKNorm 설명에도 **bandwidth B**가 남는다. 이를 근거로 우리 sparsemax의 θ 선택이 불필요해지거나 recurrent gradient 폭주 원인이 제거된다고 보장할 수 없다. 현재 정규화 거리도 `-‖q−k‖²/(d_q θ)`여서 θ 의존성이 남는다.4×4 선형 ridge solve는 탐색 후보지만 기존 분수 증분/양의 계수/cap 계약을 자동 보존하지 않는다. 실제 읽기 식과 안전 조건을 먼저 도출할 것.
- **Delta rule:** [Yang et al. 원문](https://arxiv.org/html/2406.06484v3)의 recurrent memory update는 비교 후보의 근거다. 이를 현재 모델에 일부 붙이는 것만으로 분수 history 전체 계산이 O(T)가 되는 것은 아니다. 교체할 상태·연산·값의 의미를 명시한 별도 baseline으로 검증할 것.
- **학습 희소성:** [Correia et al. §3/Proposition1](https://aclanthology.org/D19-1223.pdf)의 α 미분은 entmax 희소성 학습 후보를 뒷받침한다. θ/score scale 영향까지 제거한다는 보장은 없으므로 기존 sparsemax와 동일 예산에서 비교하고, 분수차수 α와 다른 기호를 사용할 것.
- **Hopfield:** [원 논문 초록](https://arxiv.org/abs/2008.02217)은 확인했으나 이번 원문 PDF/HTML 접근은 실패했다. 정리의 전체 가정 재검증은 **not run**. 평균 cosine만으로 해당 정리의 분리 조건 위반을 확정한 것으로 기록하지 않는다.

우선순위는 **진단 재현 스크립트·표본/shape/기준 고정 → D-Y 고정η와 동일 정책 oracle/Full 비교 → 필요 시 정규화/희소성 후보 검증**이다. Gram/delta 구조 변경은 기존 계약과 다르므로 별도 탐색으로 구분한다. 기존에 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단 원칙도 유지한다. 결과를 반복 관찰한 test에 맞춰 후보를 선정하지 말고 validation에서 규칙을 고정한 뒤 독립 평가할 것.

A10-CAL·PROVENANCE·LOG(final CSV/동적 열), key_norm=frozen 적합 배선, A07-REGEN, 다중 seed·통계는 OPEN 유지. 신규 학습/논문 후보 구현·효능 비교는 **not run**이다.

### 실행·보존

CPU snn_recall Python3.10/torch1.12.0+cu113/2threads/seed7. 기존 checkpoint2개 평가, 임시 복원fault8조건, 실제 소스의 집계·payload AST만 검사했다. 모델/학습 소스 수정·train/optimizer·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송 없음. 사본139개 hash와 캡처 checkpoint 보존을 확인했다. 후속 canonical append만 별도 보존했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \\
/home/yschoi/.conda/envs/snn_recall/bin/python \\
f_lif_pop_v3/forecasting/results/assessment/{run}/restore_grad_probe.py \\
/tmp/nsmt_assessment_{run} \\
f_lif_pop_v3/forecasting/results/assessment/{run}/restore_grad_probes.json \\
> f_lif_pop_v3/forecasting/log/assessment/{run}/restore_grad_probe.log 2>&1
```

<!-- assessment-watch:{run} -->
'''
mem=f'''

## 추적 갱신 — {stamp} (예약 감사12)

- Snapshot10:50:38/HEAD520bb568/139파일. A10-RESTORE-STATS 누락거부8조건+legacy정상복원 VERIFIED. Input/key frozen 및 fixedeta buffer누락거부,none/학습eta허용,weight누락거부.
- A09 absmax추가했지만np.mean집계유지:주입[1,9]→5,[2,10]→6. epoch최대아님;OPEN. *_nonfinite JSON전달과theta/calibrated_fields 실제JSON확인 VERIFIED.
- gradchk2epoch512/64/64,batch64,g11_every10:전체MSE.337799426411/recall.347767202408,hash/MSE재현. epoch8batch중1watch라최대집계문제안드러남. 성능통계not run.
- key_geometry 재생성코드/hash/정의미비(A09-KEY-GEOMETRY OPEN). 유닛별raw5/투영key4차원으로rank상한4;3.97/42를그자체붕괴증거로삼지말것. hit3.73e-5는pilot-eta이지diag3아님,동일표본비교필요.
- canonical10:47 과도gradient인과주장철회확인.10:40 readout원인단정/10:50 QKNormθ불필요·폭주제거/Gram해석과장잔여. TTR원문§3(bandwidth잔존),DeltaNet원문,entmaxPDF명제1실제확인;Hopfield초록만확인/PDF실패정리검증not run. ASSESMENT 감사12에직접링크와제한기록.
'''
log=f'''

## {stamp} — v3 예약 감사12: 복원 결함 해소·gradient 집계 잔여·문헌 적용 범위

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD520bb56872c6e7c68c48f65cd70601979c22a1c5.10:50:38 snapshot139파일 /tmp/nsmt_assessment_{run}. 감사자모델변경/학습/commit/tag없음.
- CPU Python3.10/torch1.12/2threads. 복원fault8조건에서frozenstats/fixedeta누락거부+legacy pilot recall.269893671218재현:A10-RESTORE-STATS VERIFIED. 새gradchk seed7,data_seed20260921,r2k3,512/64/64,batch64,2epoch,theta5.56134335,scale8기존checkpoint전체MSE.337799426411/recall.347767202408,hash/MSE차0.
- A09 absmax는여전히batch최대의평균:고정AST주입[1,9]→5,[2,10]→6. g11_every10,epoch8batch중1관측이므로이run으로전체max주장불가. nonfiniteJSON전달및theta/calibrated_fields저장확인.8seedCI/새학습not run.
- 10:50 문헌후보검토: key_geometry재현코드/명령/hash/mask/shape부족. 유닛별key4차원rank상한4이므로3.97/42를붕괴로단정불가. 0.2924와기존pilot hit3.73e-5를서로다른checkpoint/집계로비교하지말것. TTR원문bandwidthB잔존,QKNormθ불필요/폭주제거보장없음. Gram/delta는다른memory계약,현재분수history가자동O(T)되지않음. Entmax학습가능성만확인,온도독립보장아님. Hopfield원문PDF접근실패,정리가정검증not run. 직접링크/세부수정권고는ASSESMENT 감사12.
- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,changes.diff,restore_grad_probe.py,restore_grad_probes,postscript,validation),rawlog forecasting/log/assessment/{run}/. Exact command는ASSESMENT 감사12; OMP/MKL2·snn_recall python·frozenroot로실행. Train/optimizer/GPU/설치/git변이/프로세스중단없음,원본/문서prefix보존. A09/CAL/PROVENANCE/LOG/keynormfit/REGEN잔여유지.
'''
for name,text in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 with Path(name).open('a') as f:f.write(text)
print(stamp,'appended')
