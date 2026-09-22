from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,subprocess
art=Path('f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e');inv=json.loads((art/'inventory.json').read_text());snap=Path(inv['snapshot']);now=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
check={'checked_kst':now,'snapshot_hash_mismatches':[],'live_changes_after_snapshot':[],'head_now':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
for row in inv['files']:
 p=Path(row['path'])
 if hashlib.sha256((snap/p).read_bytes()).hexdigest()!=row['sha256']:check['snapshot_hash_mismatches'].append(str(p))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:check['live_changes_after_snapshot'].append(str(p))
(art/'validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
body=r'''

## 추적 감사 17 — NOW (예약 20260922T043001Z-65597c1e)

### 관찰·증거

**§2H의 문서 정정과 정책 표 재현은 확인했다. 다만 새 `free_bound()`는 실제 상한이 아니므로 A02-REACHABILITY를 전체 종료하지 않는다.** HEAD **8b94404b25b4c2285dec59680b282a3a154de0d1**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 13:30:52 KST에 manifest 대상과 최신 감사/기억/canonical 문서147파일을 `/tmp/nsmt_assessment_20260922T043001Z-65597c1e`로 snapshot·SHA256 기록했다. Trigger hash 불일치0. 감사16 이후 실제 연구 변경은 prereg §2H, `analysis/meff_reachable.py/.txt`이며 모델/학습 소스·성능 결과 변경은 없다. 감사자의 이전 append를 연구 진전으로 세지 않았다.

새 분석 코드 SHA256 **1ae6191e99f03ea3636e46c0913fe5479f40f13a0d3f26ac4c7f50bcfa4a5c3d**. Model `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`는 이전과 동일. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/inventory.json), [직접 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/reachability_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/reachability_probes.json), [구성한 반례 정책](../f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/constructive_bound.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/validation.json).

### (a) 구현 정확성 — 모델 판정 유지, 분석 함수의 의미 정정 필요

모델 변경이 없으므로 기존 구현 확인과 잔여 OPEN을 유지한다. 새 분석 파일은 T42/α.7/history41, 정답 index를 `N−lag`부터 연속 배치한다. `uniform_policy()`를 직접 호출해 **저장 표16행·96개 수치 전부**가 4자리 출력까지 일치함을 확인했다. 분석 함수의 API와 파일 쓰기 부작용이 없음을 먼저 확인했고, 긴 전체 탐색 대신 표 재계산과 경계 주변의 제한된 CPU 검사를 수행했다.

`free_bound()`는 uniform 후보와 Dirichlet 4,000개 후보 중 가장 큰 값을 반환한다. 함수 docstring도 **“Dirichlet search, not a proof”**라고 명시한다. 유한 탐색으로 얻는 값은 가능한 최댓값의 **하한/달성값**이며 상한이 아니다. .5를 넘으면 해당 정책의 도달 가능성은 보일 수 있지만, 못 넘었다고 불가능성을 증명할 수 없다.

**기본 4,000회 탐색에 대한 직접 반례:** 과거41칸의 zero-based 정답 index `[3,5,15,17,21,25,29,30]`, η=.4681929352566433에서 함수는 **.5588662774934673**을 반환했다. 같은 b·η·cap 아래 남은 cap 용량에 adaptive 질량을 배분한 유효 simplex 정책을 직접 구성해 같은 `m_eff()`에 넣으면 **.5625772591831768**, 차이 **.0037109816897095**다. 구성 정책 전체와 합1을 artifact에 저장했다. 이는 범용 `free_bound` 상한 명제의 반례이며, 합성 회상 과제의 실제 정답 배치나 성능 결과로 주장하지 않는다.

### (b) 검증·문서 상태 — 정정 일부 VERIFIED, 상한/적용 범위 OPEN

**A02-REACHABILITY 부분 VERIFIED:** §2H와 canonical13:21에서 다음 정정을 실제 확인했다: uniform oracle은 정책 기준선, lag 영향<.01 일반화 철회, 최소η .70–.75 필요조건 철회, η0에서 learned/oracle비율1인 퇴화 인정, full-kernel mass와 uniform-slot chance 분리. 이전 ‘G14 통과’를 이 seed의 탐색 관찰로 제한하고, 작은η 효과 없음/우연 수준 단정도 철회했다. 이는 문서상 정정의 검증이며 새 모델 효능 확인은 아니다.

**잔여 OPEN — ‘자유 정책 상한’ 계산 근거:** §2H의 운영상 결론을 뒷받침하려면 유한 탐색 대신 감사15의 정확한 계수 상한을 사용해야 한다. 양의 b_i≤C, A≠∅에서 `B=Σb_i`, `B_A=Σ_A b_i`, `m=|A|`라 두면

`S=min((1−η)B_A+ηB, mC)`, `U=S/[S+(1−η)(B−B_A)]`.

새 코드의 고정 lag21/연속 배치에 이 식을 독립 적용했다. 정답 수1/2/3/4/6/10에 대한 **.005 격자의 최초 .5 도달 η**는 .920/.840/.750/.655/.455/.345로 저장 표와 일치한다. 각 경계와 직전 격자에서 원본 4,000회 탐색도 재실행해 이 12점은 일치했다. 따라서 **제시된 격자 수치는 맞지만 ‘탐색 최댓값=상한’이라는 방법론은 틀리다.**

표의 ‘최소η’는 연속값의 정확한 최소가 아니라 **탐색 격자 해상도 .005에서의 최초 도달값**으로 명시할 것. 예를 들어 정답3/4칸의 정확한 상한이 .5에 닿는 연속 경계는 각각 **.7465586758 / .6535398318**이다.

또한 고정 history41·정답3/4칸 표만으로 과제 전체 O7 평균을 판정할 수 없다. 실제 test256sequence의 query→sequence 집계에서 η.5 상한 **.466613112585**였다는 감사15 증거를 명시적으로 연결할 것. 등록 격자 중 η1만 가능한 결론은 **그 표본·집계·계수 계약 범위에서 유지**되며 미래 seed/dataset 전체의 증명으로 확대하지 않는다. 구현을 정확식으로 바꾸고 표본별 집계를 연결하거나, 함수/열 이름을 ‘sampled best’로 바꾸고 불가능성 주장에는 별도 정확식을 인용해야 이 잔여 이슈를 닫을 수 있다.

### (c) 개선 방향 — 새 성능 판단 근거 없음

이번 변경은 분석·문서 보완이며 새 학습 성능 결과는 없다. **성능 판정 보류**, checkpoint 재평가·QK 정규화 구현/효능 검사·새 학습·독립8seed/paired CI는 **not run**이다. 기존 순위 고정η→QK L2정규화→causal key 표현→entmax→별도 Gram→별도 delta를 유지한다. 당장 필요한 보완은 상한 분석에 정확식·격자 해상도·실제 표본 집계와 seed/hash를 남기는 일이다.

QK 정규화는 다음 탐색 후보이며 [앞서 확인한 Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 정규화–거리 연결과 scale 선택 범위를 유지한다. Gradient/clipping 진단은 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 원문 근거를 재사용한다. 이번에는 새 문헌 주장을 추가하지 않았고 위 직접 수치 검사로 판단했다. Canonical13:21의 진짜 최대·clipping 전후 norm 기록 계획은 아직 구현이 없어 **계획으로만** 인정한다. A09 absmax/KEY-GEOMETRY, A10-CAL·PROVENANCE·LOG, A07-REGEN 등은 재검사 근거 없이 닫지 않는다.

### 수행·재현

CPU NumPy 대수와 제한된 후보 탐색만 실행했다. 모델/학습 소스·기존 로그 수정, 학습/GPU/설치/프로세스 중단/git 변이/다른 세션 대화 열람·전송은 하지 않았다. 사본 hash와3문서 기존 prefix를 보존했다. 분석 원본 `main()`의 전체 무작위 sweep는 **not run**이고, 저장 정책 표 전 항목·상한 격자·경계12점·반례를 독립 검사했다.

아래 두 명령은 cwd NSMT, `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib` 환경에서 `/home/yschoi/.conda/envs/snn_recall/bin/python`으로 실행했다.

```text
f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/reachability_probe.py /tmp/nsmt_assessment_20260922T043001Z-65597c1e f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/reachability_probes.json
f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/constructive_probe.py /tmp/nsmt_assessment_20260922T043001Z-65597c1e f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e
```

Raw stdout는 `f_lif_pop_v3/forecasting/log/assessment/20260922T043001Z-65597c1e/{reachability_probe,constructive_probe}.log`에 보존했다.

<!-- assessment-watch:20260922T043001Z-65597c1e -->
'''.replace('NOW',now)
mem=f'''

## 추적 갱신 — {now} (예약 감사17)

HEAD8b94404b25b4c2285dec59680b282a3a154de0d1/snapshot13:30:52/147파일,manifest불일치0. 모델/성능결과변경없음. §2H·canonical13:21 uniform상한/lag/η필요조건/비율·chance/과잉표현정정확인(부분VERIFIED). 새meff_reachable.py uniform표16행96값재현,정확식 .005격자 경계 .920/.840/.750/.655/.455/.345 일치. 그러나free_bound는Dirichlet4000후보최대일뿐상한아님:유효정책반례 .558866277493→.562577259183. A02-REACHABILITY잔여OPEN,정확식 U+실제표본집계필요. 3/4칸연속η경계 .746558676/.653539832,표는격자근사. 새학습/성능재평가/8seedCI not run,②QK정규화다음탐색유지. ASSESMENT감사17/results/assessment/20260922T043001Z-65597c1e/.
'''
log=f'''

## {now} — 예약 추적 감사17: §2H 정정 부분 확인·무작위 최댓값을 상한으로 쓰는 잔여 문제

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD8b94404b25b4c2285dec59680b282a3a154de0d1.13:30:52 snapshot147파일/trigger불일치0,모델·학습·성능결과변경없음. 감사commit/tag없음.
- CPU NumPy,α.7/T42/history41. 새분석uniform정책16행96값전부재현. 정확상한 U와 .005격자의최초도달η .920/.840/.750/.655/.455/.345 일치,원함수경계주변12점의4000후보탐색확인. 연속경계3/4칸 .7465586758/.6535398318라격자근사명시필요.
- §2H·13:21문서정정(uniform상한/lag/최소η/비율퇴화/chance혼용/과잉성능표현)은부분VERIFIED. **free_bound는sampled best**이므로 A02-REACHABILITY잔여OPEN. 기본4000후보 .558866277493에대해같은정답집합·η에서구성정책 .562577259183확인. 정확식 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)] 및실제test query→sequence집계를운영결론에연결할것.
- 새성능판단근거없음. 신규학습/전체main sweep/checkpoint재평가/QK효능/8seedCI not run,기존우선순위와A09/A10/A07잔여OPEN유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/(inventory,reachability_probe.py,reachability_probes.json,constructive_probe.py,constructive_bound.json,validation,append_validation). Raw log forecasting/log/assessment/동일run/각probe.log. Exact command·문헌은ASSESMENT감사17. 소스수정/학습/GPU/설치/git변이/프로세스중단/타세션대화열람·전송없음,감사3문서append와진단artifact만작성.
'''
checks=[]
for name,content in [('docs/DOCS_REVIEW_MEMORY.md',mem),('docs/PROJECT_LOG.md',log),('docs/ASSESMENT.md',body)]:
 p=Path(name);before=p.read_bytes()
 with p.open('a') as f:f.write(content)
 assert p.read_bytes()[:len(before)]==before
 checks.append({'path':name,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),'prefix_preserved':True})
(art/'append_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(now,json.dumps(check,ensure_ascii=False))
