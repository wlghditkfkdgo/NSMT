from pathlib import Path
import json,hashlib,datetime,subprocess
run='20260922T083001Z-36837373';a=Path('f_lif_pop_v3/forecasting/results/assessment')/run;i=json.loads((a/'inventory.json').read_text());s=Path(i['snapshot']);now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST');v={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'snapshot_changed':[],'live_changed':[]}
for p,h in i['files'].items():
 if hashlib.sha256((s/p).read_bytes()).hexdigest()!=h:v['snapshot_changed'].append(p)
 if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:v['live_changed'].append(p)
assert not v['snapshot_changed'];(a/'validation.json').write_text(json.dumps(v,indent=2))
e='''

## 추적 감사 26 — NOW (예약 20260922T083001Z-36837373)

**새 모델 실행·성능 판단 근거 없음. 다만 새 canonical 통합 요약에 기존 조건/수치가 잘못 합쳐진 부분이 있어 정정을 요구한다.** HEAD `0314e9319891dbe212b1a153b402db75143d170c`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 감사25 이후 commit diff는 canonical PROJECT_LOG의303행 추가뿐이며 그중 감사자의24·25차 append는 연구 변화로 세지 않았다. 최신 기억·감사25·사전등록§2H·canonical17:27 통합 요약과17:28 운영 기록을 확인했다.

**17:31:14 KST** 실제216파일을 `/tmp/nsmt_assessment_RUN`에 snapshot·hash했다. Trigger 불일치0, 감사25 이후 연구 소스/분석/결과 hash 변경0. Canonical SHA256 `a4819eff7eeebe49827cda1ecd34e17e2aa968f6cdab300a424ed6a042289230`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/RUN/inventory.json), [문서 diff](../f_lif_pop_v3/forecasting/results/assessment/RUN/project_log.diff), [산술 probe](../f_lif_pop_v3/forecasting/results/assessment/RUN/summary_probe.py), [정밀 계산](../f_lif_pop_v3/forecasting/results/assessment/RUN/summary_arithmetic.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/RUN/validation.json).

### (a) 구현 정확성

새 구현 변경 없음. 감사25의 logger/absmax·관찰 수·post-clip 저장과 rank 기준에 대한 scoped VERIFIED를 유지하며 재실행하지 않았다. A07-REGEN, A10-PROVENANCE·LOG 잔여, A02 실제분포 연결, A05-ETA-INTERVENTION-BOUND 등 OPEN은 그대로다. **통합 요약 §8의 “전부 수정 완료”는 §11의 OPEN 목록 및 최신 감사와 충돌**한다. 수정 완료 범위를 해당 이슈·버전·재검사 조건으로 제한해야 한다. 기존 모델을 다시 실행한 것은 **not run**이다.

### (b) 검증·결과 보고 적절성 — A09-SUMMARY-CONDITION-MIX OPEN

통합 요약 §5.1은 비QK etagrid MSE에 QK 조건의 G를 혼합했다. Snapshot JSON의 동일 etagrid full/oracle1을 사용해 `G=(E_full−E_condition)/(E_full−E_oracle1)`을 직접 재계산했다. E_full=.2763386361581949, E_oracle1=.03496568833854037이다. 이는 이전 탐색 표와 같은 공통 분모 산술이며 확증 G14 통과 판정은 아니다.

| 비QK 조건 | 표의 recall MSE | 요약 G | 재계산 G |
|---|---:|---:|---:|
| 학습η | .272620149439 | +.012 | **+.015405565340** |
| 고정η.2 | .319387285780 | −.123 | **−.178349106686** |
| 고정η.5 | .433504353901 | −.177 | **−.651132279581** |
| 고정η1 | .416682604337 | −.581 | −.581440337231 |

+.0126/−.1234/−.1772는 기존 **QK** 학습η/.2/.5 수치와 대응한다. 따라서 §5.1과 §6의 비QK “학습η +.012”를 고쳐야 한다. §5.3의 “G는 여전히 음수”도 고정η 조건으로 한정해야 하며, QK 학습η의 G는 **+.0126314**였다(감사19·23). “모든 결과가12epoch”라는 요약도 부정확하다. 동일 JSON에서 비QK η.5는 early stopping으로 **11epoch**, logging 확인 실행은1epoch이며, η 분리 분석은 새 학습 없는 checkpoint 평가다.

아래는 새 계산 결과가 아니라 **이미 감사25가 지적했는데 통합 요약에 다시 남은 제한**이다.

- §5.5 궤적 고정의 G11 ‘OK’는 원 학습η 궤적 peak22.07에 대한 값이다. 바뀐 계수로 실행한 시스템의 안정성은 그 행에서 **not run/N/A**다. 전체 forward η1은656.13으로 불합격이다.
- §5.2의 “readout은 full·oracle에서 무관”, “약한 신호가 스파이크를 통과하지 못한다”는 표만으로 원인을 확정하는 표현이다. 기존 readout 비교의 관찰 범위와 oracle 정책/소스 버전을 유지해야 한다(A08/A10 과거 해석 제한).
- §7의①·② “완료”는 단일 seed 탐색·재현을 완료했다는 범위로 표시해야 한다. η 선택 규칙·독립seed·미사용 평가자료 검증은 여전히 **not run**이다. §4의 MDE 상대1.2%도 v2 측정치를 v3 recall에서 확인한 검정력처럼 읽히게 해서는 안 된다.

통합 요약에 탐색/확증 구분과 OPEN 목록을 둔 점은 적절하지만, 이것이 개별 표의 조건 혼용을 상쇄하지는 않는다. 새 데이터/대조군/통계 실행 증거는 없으므로 성능 우열 판정과 기존 이슈 상태를 추가로 바꾸지 않는다.

### (c) 개선 방향·수행

먼저 요약표를 **suite/정규화/η/분할/실제 epoch/공통 분모**로 연결해 정정하고, 기존 **η.2 후보의 사전등록된 선택 규칙과 독립 검증** 우선순위를 유지할 것. 새 결과가 없어 후보 순위를 변경할 근거는 없다. 학술 근거는 감사25에서 확인해 인용한 [Test-time regression](https://arxiv.org/html/2501.12352v1)과 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 범위 그대로이며 새 문헌 주장 없음.

이번에는 표준 Python으로 JSON 산술만 수행했다. 모델 forward/학습/backward/optimizer/GPU·새 통계·웹 추가 조사 **not run**. Canonical의 commit+push 운영 기록은 관찰했으나 이 예약 감사의 명시적 git 변이 금지에 따라 감사자는 commit/push하지 않았다. 원격 게시의 독립 확인도 not run. 연구 소스/기존 artifact/프로세스/타세션 대화는 건드리지 않고 감사3문서 append·텍스트 증거만 작성했다.

```bash
/usr/bin/python3 f_lif_pop_v3/forecasting/results/assessment/RUN/summary_probe.py /tmp/nsmt_assessment_RUN f_lif_pop_v3/forecasting/results/assessment/RUN/summary_arithmetic.json > f_lif_pop_v3/forecasting/log/assessment/RUN/summary_probe.log 2>&1
```

<!-- assessment-watch:RUN -->
'''.replace('NOW',now).replace('RUN',run)
m=f'''\n\n## 추적 갱신 — {now} (예약 감사26)\n\nHEAD0314e931/snapshot17:31:14/216파일/manifest불일치0. 연구소스·결과hash변경없음,새성능근거없음. Canonical통합요약신규A09-SUMMARY-CONDITION-MIX OPEN:비QK MSE에QK G혼용. 동일JSON공통full/oracle1재계산G 학습η+.015405565/.2−.178349107/.5−.651132280(요약+.012/−.123/−.177오류). 비QK.5실제11epoch,모든실행12epoch아님. 전부수정완료/①②완료단정범위제한;고정궤적G11 OK·readout원인단정·v2MDE이식잔여. 기존VERIFIED/OPEN·η.2독립검증우선순위유지. 모델재실행/학습/통계not run. ASSESMENT감사26/results/assessment/{run}/.\n'''
l=f'''\n\n## {now} — 예약 추적 감사26: 통합 요약의 조건 혼용 정정 요청 (학습 없음)\n\n- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD0314e9319891dbe212b1a153b402db75143d170c,snapshot17:31:14/216파일,manifest불일치0. 연구소스·결과변경없음. 새성능판정근거없음. 감사commit/tag/push없음.\n- **통합요약§5.1·§6 정정:** 동일etagrid JSON 공통full .2763386361581949/oracle1 .03496568833854037에서비QK G는학습η+.015405565340/.2−.178349106686/.5−.651132279581. 현재표+.012/−.123/−.177은QK조건과혼용(A09-SUMMARY-CONDITION-MIX OPEN). §5.3 G음수는고정η에한정,QK학습η는+.0126314. 비QK.5실제11epoch/로그확인1epoch/분리분석새학습없음이므로전부12epoch표기정정.\n- §8 전부수정완료는§11OPEN과모순. §7①②완료는탐색에한정. §5.5고정궤적G11은원궤적peak이며개입시스템안전not run/N/A;§5.2readout인과단정·v2MDE1.2%v3이식제한유지. 기존VERIFIED/OPEN그대로,η.2선택규칙사전등록·독립검증우선순위유지.\n- 표준Python JSON산술만수행;모델forward/학습/backward/optimizer/GPU·독립seedCI·원격게시확인not run. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/{run}/(inventory,project_log.diff,summary_probe.py,summary_arithmetic.json,validation,append_validation),raw log forecasting/log/assessment/동일run/summary_probe.log. 상세근거·명령은ASSESMENT감사26. 예약감사git변이금지준수,3문서append·텍스트artifact만작성.\n'''
checks={}
for path,t in [('docs/ASSESMENT.md',e),('docs/DOCS_REVIEW_MEMORY.md',m),('docs/PROJECT_LOG.md',l)]:
 p=Path(path);b=p.read_bytes()
 with p.open('ab') as f:f.write(t.encode())
 assert p.read_bytes()[:len(b)]==b;checks[path]={'prefix_sha256':hashlib.sha256(b).hexdigest(),'prefix_bytes':len(b),'preserved':True}
(a/'append_validation.json').write_text(json.dumps(checks,indent=2));print(now,v)
