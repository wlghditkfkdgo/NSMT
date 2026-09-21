# f_lif_pop_v3 감사 기록

**최초 감사: 2026-09-21 KST · 상태: 수정 및 재검증 필요 · 학습 성능 판정: 아직 불가**

이 파일명은 사용자 요청대로 `ASSESMENT.md`다. 다른 작업 세션은 아래 이슈 ID와 마지막 후속 기록을 확인하고, 조치 결과를 끝에 추가한다. 이 문서 생성만으로 자동 감시나 다른 세션의 주기적 확인이 설정되지는 않는다.

## 1. 대상과 판정 범위

| 항목 | 감사 기준 |
|---|---|
| 브랜치 | `exp/f-lif-pop-v3` |
| 감사 대상 HEAD | `df3a3407b9ae653b4b1031320c4e8240b4c943a0` |
| v3의 실제 출발 commit | `329183b94f65090cc6b337f464c5aa4d8e127ad7` — 완료된 v2 TSMixer 기록 |
| 설계 기준 | 사전등록의 최신 §2A·§3A·§9A·§2B, IDEA_LOG rev.1, canonical PROJECT_LOG의 후속 결정 |
| 검사 대상 | `f_lif_pop_v3/forecasting`의 뉴런·선택자·config·생성기·loader·calibration·게이트 및 저장 결과, 관련 analysis |
| 문서별 기억 | [DOCS_REVIEW_MEMORY.md](DOCS_REVIEW_MEMORY.md) — docs 전체의 개별 요약·대체 관계·역사적 결과 |
| 증거 위치 | [results/assessment/20260921-2327-kst](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/) |
| 감사 방식 | 읽기 전용 코드 검토, 고정 사본의 CPU 수치 재검사, 기존 JSON 분석, 1차 문헌 확인 |

**종합 판정:** 최신 아이디어의 핵심 뉴런 구조는 대체로 구현되었다. 그러나 질량 대조군, cap 이후 판정식, 회상 과제의 난이도, 안정성 보정값에 확인된 문제가 있다. 현재 자료로 “검증 완료” 또는 “아이디어의 성능이 입증됨”을 선언할 수 없다. 아래 P1을 해결하고 실행 계약을 고정한 뒤 확인 실험을 수행하는 것이 타당하다. 이는 진행 중 프로세스를 강제 중지하라는 지시가 아니라, 현재 결과가 뒷받침할 수 있는 주장 범위의 판정이다.

확인 시점에는 `ours.py/model.py/train.py/test.py/utils.py`와 학습 실행기, 학습 완료 metric/checkpoint가 이 경로에 없었다. **미완성된 파이프라인과 이미 구현된 코드의 오류를 구분한다.** 다른 세션의 대화나 내부 계획은 접근하지 않았고, 파일·commit·결과에 드러난 작업만 감사했다. 작업 소스와 기존 결과를 수정하거나 새 학습을 실행하지 않았다. 공유 HEAD와 index도 변경하지 않았다.

## 2. 제대로 구현된 부분과 현재 실행 결과

`layers.py`는 동일 전류를 K개 가지에 넣고 `f_n=(I_n-u_n)/tau`를 기록한다. 과거 **막전위 자체를 더하는 v2와 달리 dynamics history를 재적분**한다. Query는 갱신 전 `[u_n,I_n]`, 후보는 `j<n`, 소마는 갱신 후 `u_(n+1)`을 읽는다. 가지는 reset하지 않고 소마만 standard soft reset하며 논리 뉴런마다 spike 하나를 낸다. 현재 고정값 tau=[4,8,16,32], g=0/pi=1, theta=1, eta_hat=-4, surrogate scale5, soma weight 초기1/K는 최신 결정과 대응한다.

Full 경로, eta=0 중립극한, alpha=1 Euler 환원, homogeneous 조화평균 tau≈8.533, window별 상태 초기화, shared selector도 구현되어 있다. Frozen normalization은 train sample의 projection 통계로 적합하고 이후 forward에서 batch 통계를 읽지 않는 구조다. 다만 이를 저장·복원하고 모든 학습 조건에 적용하는 전체 파이프라인 검증은 아직 없다.

독립 CPU 재실행은 기존 결과와 같은 **17 passed / 0 failed / 1 not run(G4)**이었다. G2/G6/G7/G9의 관측 오차는0, G3의 loop–유효 연산자 차이는5.00e-16이었다. 이 검사는 정의와 작은 입력의 일관성을 지지한다. Source parity, 학습 안정성, semantic recall을 대신하지 않는다. [구조화한 gate 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/gates.json)

기존 calibration JSON에서 선택된 값은 다음과 같다. 모두 **초기 모델의 train 입력 보정**이며 학습 후 정확도가 아니다.

| 입력/정규화 | seed | scale | 평균 spike rate | dead unit 비율 | max abs branch state |
|---|---:|---:|---:|---:|---:|
| recall k3 / frozen | 7 | 8 | 0.228051 | 0 | 18.4044 |
| recall k3 / frozen | 13 | 8 | 0.227020 | 0 | 17.6590 |
| recall k3 / frozen | 21 | 8 | 0.228973 | 0 | 16.8580 |
| recall k3 / frozen | 42 | 8 | 0.229696 | 0 | 17.0673 |
| recall k3 / none | 7 | 17 | 0.215098 | 0.3125 | 9.6437 |
| ETTh1 / frozen | 7 | 6 | 0.186099 | 0 | 19.6876 |

Frozen normalization은 이 표본에서 침묵 unit을 줄이고 목표 발화율을 만들었다. 4 seed의 scale 일치는 유용한 관찰이지만 8 seed·다른 alpha·code encoding·다층·학습 후 동작까지 보장하지 않는다. Full precision과 원본 hash는 [calibration inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/calibration_inventory.json)에 있다.

ETTh1 loader는 실제 CSV로 별도 검사했다. Train만으로 fit한 평균이 독립 계산과 정확히 같았고 첫 target 값도 일치했다. H96 window 수8209/2785/2785, H720은7585/2161/2161이었다. Validation target 시작8640, test 시작11520, 마지막 target 경계14400으로 올바르다. 이 결과는 loader 범위의 검사이며 아직 없는 trainer의 test 누출 방지까지 확인한 것은 아니다. [loader 증거](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/ett_loader.json)

## 3. 이슈 목록

P1은 해당 메커니즘/성공 판정 전에 해결해야 할 문제, P2는 필요한 검증·재현성 보완이다. 아래 상태는 최초 감사 시점이며 후속 조치로 닫으려면 §8에 증거를 추가한다.

| ID | 우선도 | 상태 | 내용 |
|---|---|---|---|
| A01 | P1 | 재현됨 | `mass_matched`가 cap 이전 질량을 맞춰 Full로 퇴화 |
| A02 | P1 | 수식 반례 재현됨 | O7의 실효 질량 공식이 R3 cap 이후에는 성립하지 않음 |
| A03 | P1 | 생성기·표본에서 확인 | 3→5→8이 동시에 유지·회상할 key 수를 늘리지 않음 |
| A04 | P1 | 재계산 불일치 | Chance mass 분모가 실제 query 시점의 history 길이가 아님 |
| A05 | P1 | 코드·전류 비교로 확인 | G11 보정이 실제 projected/normalized current를 재지 않음 |
| A06 | P1 | 배선 누락 확인 | `--eta_fixed`, `--no-cap`이 뉴런 생성자에 전달되지 않음 |
| A07 | P2 | 부분 검사 | G4 skip 사유 부정확, G7b/G15가 등록된 검사와 다름 |
| A08 | P2 | 미구현/인터페이스 부족 | 실제 데이터 G13·기전 진단과 학습용 analog 경로 부족 |
| A09 | P2 | 문서 충돌 | eta/w/reset/O7/보고서 상태·일부 과장된 설명 정리 필요 |
| A10 | P2 | 재현성 부족 | 보정 덮어쓰기·설정/통계 누락·실행 결과 불일치 |
| A11 | P2 | 코드와 완료 기록 불일치 | `rng_codes` 비복원 추출 수정이 현재 코드에 없음 |
| A12 | P1(진입 조건) | not run | 학습·oracle·GRU·통계·checkpoint 검증은 아직 없음 |

### A01. mass_matched — 확정된 구현 오류

위치: [layers.py](../f_lif_pop_v3/forecasting/layers.py)의 `Selector.forward`, 감사 snapshot 178–185행.

현재 순서는 `raw=b*rho` → `raw.sum()/B`로 질량 대조 생성 → cap이다. 그런데 cap 전에는 정의상 `sum(b*rho)=B`이므로 이 대조는 거의 항상 `b`가 된다. R3 때문에 감소한 **실제 선택 모델의 질량**을 맞추지 못한다.

동일 bank, alpha=.7, 과거41칸, eta=.5, 한 오래된 slot에 sparse p가 집중하는 반례에서:

- 선택 모델 kappa = **0.569804778811738**.
- mass_matched kappa = **1.0000000000000002**.
- mass_matched와 Full의 최대 계수 차이 = **1.11e-16**.
- 두 조건의 post-cap 질량 차이 = **6.006159352561638**.

수정 계약은 먼저 `c_sel=min(b*rho,b0)`를 구하고 `kappa=sum(c_sel)/B`, `c_control=kappa*b`로 시간 방향을 평탄화하는 것이다. `kappa<=1`, `b_j<=b0`이므로 이 대조는 cap에도 맞는다. 단 kappa 자체는 선택자가 만든 상태 의존량이므로 “내용 정보가 전혀 없다” 대신 **slot 배분을 제거하고 총량만 보존한 대조**라고 부르는 편이 정확하다.

통과 조건: cap 미작동 시 Full과 일치, 작동 시 동일 bank에서 `sum(c_control)==sum(c_sel)`이고 계수 배열은 다름. 수치 허용오차를 명시한다. 고정 bank 개입과 처음부터 따로 학습한 control을 모두 구분하고, kappa의 gradient 유지/detach 규칙도 고정한다. 따로 학습한 두 모델의 궤적·kappa가 매번 같을 필요는 없다.

### A02. cap 뒤 실제 M_eff를 정의해야 함

위치: 사전등록 D-H/§9A 및 IDEA_LOG의 `M_eff=(1-eta)*m0+eta` 설명.

이 식은 **cap 이전의 완벽한 oracle p**에서 성립한다. 일반 learned p의 식도 아니며, R3 이후의 oracle에도 일반적으로 성립하지 않는다. 실제 보고값은 정답 집합 A_n에 대해 다음이어야 한다.

```text
M_eff(n,b,d) = sum_{j in A_n} c(n,b,d,j) / sum_{j<n} c(n,b,d,j)
c = min(b*rho, b0)
```

A01의 oracle 반례에서 cap 전 공식은 **0.5090226726020335**, 실제 cap 후 값은 **0.13834115533070288**이다. 같은 데이터 생성기의 완벽한 oracle·eta=.5에서도 query 평균은 k3/k5/k8 각각 약.468/.434/.394였다. 이는 oracle-trained의 학습 결과가 아니라 **계수 배분의 수치 진단**이다.

통과 조건: p 질량, cap 전 `b*rho` 질량, cap 후 c 질량을 독립 계산해 대조하고 O7은 마지막 값을 쓴다. Empty history/정답 없는 event/분모0 처리, D/층/query/sequence/seed의 평균 순서를 먼저 고정한다. 0.5 기준은 이미 승인된 기준이므로 이 감사가 임의로 낮추지 않는다. 정의 수정과 도달 가능성 확인은 사전등록에 날짜부로 추가한다. “큰 정답 계수 질량”은 signed `f_j`의 실제 유용한 기여나 raw event의 인과적 중요도와 같지 않다.

### A03. 현재 key 수 증가는 기억 용량 축이 아님

위치: [synthetic.py](../f_lif_pop_v3/forecasting/data_provider/synthetic.py), `make_run_sequence` 19–33행.

생성기는 모든 fresh key를 먼저 도입한 뒤, 마지막 등장으로부터 간섭 run이1–3개인 key만 후보로 둔다. 따라서 n_keys=8에서 먼저 도입된4개 key는 recall이 시작될 때 이미 후보 범위를 벗어난다. 정상 범위에서는 최근 후보가 계속 존재하므로 오래된 key를 복구하는 fallback이 작동하지 않는다. 이런 key는 다시 질의되지 않는다.

고정 data RNG20260921, 각 난이도300 sequence, 공통 code encoding으로 재계산했다.

| n_keys | recall event 비율 | 실제 다시 질의한 고유 key 수/sequence | 첫 도입 중 만료된 key의 재질의 |
|---|---:|---:|---:|
| 3 | .75278 | 2.5233 | 해당 없음 |
| 5 | .58635 | 2.5867 | 0 |
| 8 | .33690 | 2.5467 | 0 |

모든300 sequence에 recall은 있었지만 질의 key 수는 거의 같았다. 따라서 도입 distractor 증가·query 위치 이동·copy/recall loss 비중 변화가 섞인다. 특히 전체 event loss의 validation으로 모델을 고르면 key8에서 증가한 copy 비중이 회상 개선 없이도 점수를 좋게 만들 수 있다.

또한 평균 run 길이만으로 “key16에서는 재등장이 수학적으로 불가능”하다고 결론내릴 수는 없다. 최소 run 길이2이면16개 도입은32 event에 가능하다. 표본에서0%를 관측한 것과 생성 가능한 경우가 없는 것은 다르다. 다만 이 경우에도 충분하고 균형 잡힌 질의를 제공하는 과제는 아니다.

수정 방향: 소개와 질의를 명시적으로 배치하여 각 key가 질의될 기회가 있고, 오래된 key도 다시 필요하게 한다. Key 수·정답 lag·간섭 수·query 수를 가능한 한 분리해 변화시킨다. 이를 위해 gap 규칙이나 T를 바꾸면 새 task revision으로 기록하고 이전 결과와 합치지 않는다. 기존 과제를 유지한다면 “용량 검증” 주장을 내리고 **근거리 regime 재등장 진단**으로 한정한다.

통과 조건: split별 queried-key coverage, key별 query 횟수, 정답 source lag, 간섭 run 수, first-query 비율, recall/copy 비중을 저장한다. GRU가 풀 수 없는 과제라고 가정하지 말고 실제 비교한다. Ground truth가 “값이 실제 제시된 첫 run”인지 “직전 재등장 run”인지도 최신 문서에 명시한다. 현재 코드는 전자를 사용한다.

### A04. Chance mass의 계산 오류

위치: `synthetic.py:sanity_check` 181–201행, 사전등록 D-O, PROJECT_LOG 23:16 항목.

현재 보고는 평균 source 수를 `T/2=21`로 나눈다. Recall query는 전체 시점에 균등하게 분포하지 않고 n_keys가 커질수록 뒤로 이동한다. 균등한 과거 slot 선택의 실제 정답 확률은 **query마다 `|A_n|/n`**이다. `E[r/n]`을 `E[r]/(T/2)`로 대체할 수 없다.

위 A03의 같은 표본에서 sequence 안 query 평균 후 sequence 평균:

| n_keys | 기존 방식 근사 | 실제 uniform-slot chance |
|---|---:|---:|
| 3 | .166217 | **.157525** |
| 5 | .164876 | **.125956** |
| 8 | .163975 | **.101110** |

또 uniform p는 변환 후 Full의 fractional 계수를 만든다. 그러므로 **uniform slot chance와 Full의 정답 커널 질량은 별개**다. 이 표본의 Full query 평균 질량은 .130611/.111149/.100469였다(이 세 값은 query pooling이므로 위 sequence 평균과 집계도 구분해야 한다).

통과 조건: 두 기준을 각각 정확히 계산하고 집계 단위를 통일하여 보고한다. “세 난이도의 chance가 같으므로0.5의 의미도 같다”는 D-O 근거를 정정한다. 0.5를 유지하더라도 chance 대비 차이/비율을 함께 보고할 수 있으며, 성공 기준을 결과에 맞춰 사후 조정해서는 안 된다.

### A05. G11이 실제 전류를 감시하지 않음

위치: [calibrate.py](../f_lif_pop_v3/forecasting/calibrate.py) 76·88–103·151–160행, `Embedding.current`.

`max_abs_drive`는 `max(abs(raw_patch))*scale`이다. 실제 가지 입력은 **Linear→frozen mean/std→scale**을 지난 값이다. 독립256-sequence probe에서 앞의 값은8.0, 실제 `max|I|`는 **30.6887283**이었다. 상태 최대값17.4722는 이 probe에서 폭주하지 않았지만, 로그의 `10*max|I|`라는 명칭과 측정 대상은 틀렸다.

게다가 `choose`는 rate와 가지 평균의 간단한 조건만 보고, 출력한 G11 bound 초과나 sparse 초기 보정 실패를 실패 exit로 만들지 않는다. 아직 trainer가 없으므로 “학습 중 상시 G11이 켜졌다”는 증거도 없다.

통과 조건: 뉴런에 들어가는 실제 전류를 층별로 직접 측정하고 branch별 max/finite flag를 저장한다. Train calibration에서 고정한 절대 기준과 학습 중 `max|u|/max|I|`를 함께 추적하여 projection drift가 기준을 자동으로 느슨하게 만들지 않도록 계약을 정한다. 선언 상한·NaN/Inf 초과 시 run 실패와 사유를 남긴다. 실제 current를 사용해 기존 보정을 새 artifact로 다시 수행하고 sparse 확인 결과도 저장한다.

R3+tau 이동의 기존 sweep는 유한한 입력/정책/길이의 관측이다. 최근 하나를 택하는 G11, greedy 단일 support, alpha=.7 검사만으로 모든 alpha·학습 정책·길이의 안정성 정리를 주장하지 않는다. “모든 eta/T에서 bounded” 같은 코드 설명도 측정 범위로 제한해야 한다.

### A06. 실험 이름과 실제 모델이 달라질 CLI 옵션

위치: [config.py](../f_lif_pop_v3/forecasting/config.py) 97–99·120–123·161–166행.

`--eta_fixed`와 `--no-cap`은 parser와 variant 이름에 존재하지만 `neuron_kwargs`에서 빠지고 `PopulationNeuron/Selector`에도 대응 인자가 없다. `eta_fixed=1, cap=False`를 준 감사 probe의 constructor payload에도 두 값은 없었다. 현재 이를 사용했다고 명명된 학습 결과는 없지만, 그대로 trainer를 연결하면 고정 eta/hard exclusion/no-cap이라는 라벨만 바뀐 실험이 될 위험이 있다.

통과 조건: 구현해 전달하거나 미지원 옵션을 명시적으로 거부한다. eta=1은 sigmoid의 큰 유한 logit으로 근사하지 말고 exact override가 가능해야 한다. CLI→config→모델→저장 config→fresh reload에서 실제 계수·gradient·parameter 학습 여부가 달라짐을 검증한다. 무제한 no-cap 설정은 별도의 탐색적 안정성 조건으로 식별한다.

### A07. 게이트의 pass/not-run 의미를 좁혀야 함

`check_model.py`의 G4는 원본 실행을 시도하지 않고 hardcoded NOT RUN을 기록한다. “torch>=2 CPU 환경 없음, snn_jelly CUDA 없음”이라는 사유는 정확하지 않다. 이번에 **snn_jelly torch2.11.0+cu130에서 CPU forward/backward가 실행**되고 `torch.compile` attribute가 존재함을 확인했다. 원본 spikeDE 설치·golden 실행 가능성 전체를 확인한 것은 아니지만 CUDA 불가는 CPU reference 불가의 근거가 아니다.

O9는 불가 시 reference 등급 하향을 허용하므로 G4 skip만으로 모든 개발을 중지할 필요는 없다. 현재 등급은 mathematical validation이다. Source parity를 주장하려면 원본 고정 commit의 scalar reference를 따로 실행해야 하며, reset을 바꾼 주 v3-A 전체가 원본과 같아야 하는 것은 아니다.

G7b는 사전등록의 bitwise 기준 대신 atol1e-12를 쓰며 관측 오차3.33e-16으로 통과했다. 수학적 문제라기보다 **허용오차 계약의 미기록 변경**이다. 허용오차를 날짜부로 명시하거나 bitwise를 실제 검사한다. G15는 현재 마지막 voltage 변화만 확인한다. Threshold를 실제 넘는 입력을 구성해 마지막 spike 변화와 이전 spike 불변도 확인하면 등록된 출력 정렬 검사가 된다.

현재 exit0은 `not_run=0`을 뜻하지 않는다. 후속 controller는 pass/fail/not_run과 허용된 reference 등급을 읽어 단계 진입을 결정해야 한다. Actual CPU/GPU·dtype parity는 이번 감사 not run이다.

### A08. 실제 데이터 진단과 analog 학습 경로

`Selector`에는 p/rho/score가 있지만 `PopulationNeuron`의 반환에서는 버려진다. 현재 aux는 `cap_rate, coeff, eta, kappa, spikes, state, voltage`다. G13은 별도 작은 입력에서 수치를 재구성할 뿐 실제 평가 전체의 네 support를 수집하지 않는다.

필요한 출력은 support(p/rho/b*rho/c), score/QK norm, eta, kappa, cap_rate, branch별 크기·상관/유효 rank, 선택 관련 gradient norm이다. Full 모드의 aux p는 코드상 `c/sum(c)`이므로 latent uniform p와 혼동하지 않도록 이름/정의를 분리한다. t=0은 history가 없어서 일반 시간 평균에서 별도 처리한다.

현재 state/voltage는 detach된 진단값이다. 이 값에 analog head를 연결하면 뉴런까지 end-to-end 학습되지 않는다. Frozen representation probe로는 타당하지만 spike/analog **별도 학습 비교**에는 gradient를 유지하는 명시적 readout 경로가 필요하다. 어떤 비교인지 결과에 표시한다.

### A09. 최신 문서 계약과 과장된 설명 정리

- IDEA_LOG의 w 초기1.0/eta=.5와 최신 §9A·코드의1/K/eta≈.018을 구분한다.
- 사전등록 초기 soma delayed-reset 식과 후속 standard soft reset 설명이 공존한다. 현재 코드의 post-charge reset은 후속 PROJECT_LOG O1·IDEA_LOG와 대응한다. 코드 오류로 단정하기보다 최종 식을 하나로 날짜부 확정한다.
- O7 cap 이전 공식, D-O chance 근거, 보고용 문서의 “구현 전/1.5배”는 정정 대상이다.
- 주 모델은 fractional subthreshold branches+ordinary soma다. 원본 f-SNN의 직접 재현, 일반적인 모든 SNN의 고정 기억, 유일하게 recall을 푸는 구조, 문헌 전체를 망라한 최초성으로 확대하지 않는다.
- 공유 selector는 교차 의존 경로를 만들지만 eta=0, sparsemax singleton support의 내부, cap 포화 등에서는 관련 미분이0일 수 있다. 항상 nonzero인 상호작용으로 서술하지 않는다.
- 실제 Full 경로도 현재는 Python history loop이며 fast_path는 검사 전용이다. 구현된 속도 이득이나 에너지 절감은 측정 전 주장할 수 없다.

역사 문서와 기존 실험 기록은 삭제/수정하지 않고 canonical log와 사전등록에 정정을 append한다. 통합 설명문에는 현재 계약으로 연결되는 안내를 둘 수 있다.

### A10. 보정과 분석의 provenance 보완

Calibration은 dataset/k/alpha/norm/seed 이름으로 같은 JSON을 덮어쓴다. Full args, data seed/hash, norm 통계, 선택 train sample IDs, 코드 hash, sparse check가 저장되지 않는다. Scale마다 shuffle loader를 새로 순회하므로 입력 subset도 바뀐다. 고정된 같은 train sample로 모든 후보 scale을 비교하고 원본 JSON을 보존해야 한다.

PROJECT_LOG에는 recall seed7 rate .2278/state17.81이지만 현재 JSON은 .2280505933/state18.4044304다. 원인을 추정해 덮지 말고 실행 명령·파일 hash와 함께 정정한다. `analysis/stability_sweep.py`에는 F1/F2 출력 코드가 있으나 저장된 `stability_sweep.txt`에는 그 부분이 없다. 현재 파일만으로 문서의 F1/F2 숫자 전체를 재구성할 수 없으므로 해당 실행 결과를 새 artifact로 보충한다.

`Config.run_id`는 alpha/tau/norm/scale 등의 차이를 모두 담지 않는다. 현재 exists 검사 덕분에 일부 중복은 거부되지만 여러 grid를 같은 suite에 넣을 준비는 부족하다. 전체 설정 hash와 run UUID, 소스/데이터 hash, split·seed·실행 명령·환경·checkpoint hash를 저장한다. `norm_mean/std`는 checkpoint에 포함되고 fresh reload에서 유지되는지 확인한다. 초기 projection으로 적합한 frozen 통계는 projection 학습 후 다시 표준정규가 된다는 보장이 없으므로 drift도 진단한다.

### A11. rng_codes 수정 완료 기록과 현재 코드가 다름

PROJECT_LOG 23:16과 HEAD commit 메시지는 부호 패턴의 비복원 추출로 수정했다고 적었다. 현재 `synthetic.py` 103–106행은 여전히 **전체 code 행렬을 중복이 없어질 때까지 다시 뽑는 while loop**다. n_keys=16/cue_dim4는 매우 비효율적이고, n_keys>16이면 종료 조건이 불가능하다. 이번 감사에서는 hang을 유발하는 실행을 하지 않았다. 3/5/8의 현재 probe는 완료됐다.

통과 조건: 가능한 부호 패턴에서 비복원 추출하고 `n_keys<=2**cue_dim`, `cue_dim+1<=patch_size` 등 입력을 검증한다. 3/5/8에서 codebook이 기존 것과 달라지면 dataset revision/hash가 바뀌므로 재보정·재평가한다. Sanity check가 출력만 하는 run/gap/coverage 범위도 실제 조건 검사로 보완한다. 마지막 잘린 run의 길이는 완전한 run과 구분한다.

### A12. 학습 파이프라인이 갖춰야 할 판정 조건

아래는 현재 **not run/미확인**이며 이미 실패한 학습 결과라는 뜻이 아니다.

| 검증 범위 | 필요한 증거 |
|---|---|
| 실제 과제 배선 | Recall의 per-event causal `Linear(D,1)`; ETT head 별도; 미래 target/truth가 learned mode에 입력되지 않음 |
| Oracle 계약 | oracle-trained/test-time/generator lookup 분리; oracle이 없는 첫 event·copy event의 p fallback 명시; 동일 cap/eta/normalization |
| 최소 훈련 확인 | Train/val loss 감소, 관련 parameter gradient, best-val 선택/복원, 저장 후 fresh reload, partial batch metric 집계 |
| 비교군 | Full/dense/sparse/recent/수정된 mass-matched/oracle-trained/GRU/scalar/alpha1/용량 대조; 공통 초기화와 활성 parameter 수 기록 |
| 데이터 독립성 | Split별 독립 생성 seed와 dataset hash; 같은 training-seed pair에 같은 데이터·order; normalization fit은 train만 |
| 성능 지표 | Recall/copy/first-query MSE 분리, 전체 raw element 집계, seed별 paired 값; ETT 전체 MSE/MAE 및 naive/ridge 참조 |
| O7 통계 | Cap 후 M_eff, headroom>MDE일 때만 G, 평균 상대개선 및 paired CI; 실패/중단 seed도 기록 |
| 저장 | neorecall 형식 CSV/events/logargs/config/best model, full precision JSON, manifest/completion, source/data/config hash |
| 실행 안정성 | Actual current 기반 G11, dtype/device 검사, 완료 행렬 누락·중복·source drift 검사 |

Synthetic recall은 ETT와 MSE 단위/분산이 다르다. v2 ETT에서 가져온 MDE .005를 그대로 synthetic oracle headroom의 확정 기준으로 쓰면 안 된다. 먼저 해당 과제의 paired 변동과 실제적인 최소 효과를 정의한다. 8 seeds는 반복 단위이며 D개 뉴런·42 events·겹치는 ETT window를 독립 학습 반복처럼 세면 안 된다. 탐색 pilot에서 SD를 재추정했다면 본 결과를 이미 본 시점과 변경 사항을 남긴다.

## 4. 최종 아이디어의 현재 타당성

구조는 구현 가능하고 검증할 가치가 있다. 가지를 reset하지 않는 것은 기억 dynamics와 발화를 분리하며, shared selector는 population state를 사용해 사건별 배분을 바꿀 수 있다. 그러나 아직 **fractional 기억이 더 좋다**, **population이 선택을 더 잘 학습한다**, **선택이 성능에 도움이 된다** 중 어느 것도 v3 학습으로 입증되지 않았다.

연구의 설득력은 기능을 늘리는 데서 나오기보다 아래 대비를 성공시키는 데서 나온다.

1. Sparse가 Full과 수정된 mass-matched를 이겨야 slot 선택의 이득을 질량 감쇠와 구분할 수 있다.
2. 같은 조건의 alpha=.7 대1, homogeneous 대heterogeneous를 통해 차수와 다중 시간척도를 구분한다. 고정 tau4개의 alpha1 baseline은 fitted fractional SOE와 동일한 모델이 아니다.
3. Oracle-trained가 Full을 이기는 여지가 있는지 확인해야 학습 selector의 실패를 올바르게 해석할 수 있다. Oracle이 못 이겨도 곧바로 모든 retrieval 구조가 무용하다는 뜻은 아니다.
4. GRU 등 압축 상태가 과제를 쉽게 풀면, 현재 과제의 성공은 뉴런 내부 선택 기전의 진단으로 한정한다. 장기 기억의 우월성을 주장하려면 실제 key coverage·lag를 늘려야 한다.

## 5. 관련 연구에 근거한 수정 방향

아래 논문 사실과 **이번 모델에 대한 제안/추론**을 구분한다. 조회일은2026-09-21이다. 문헌의 개선율을 이 모델의 예상 개선율로 옮기지 않는다.

### 5.1 가장 먼저: 회상 과제의 판별력 확보

Zoology는 쉬운 synthetic associative recall을 푸는 모델도 더 복잡한 회상에서 차이를 보일 수 있음을 분석하고 MQAR를 제안한다. 현재 생성기에 필요한 것은 논문 이름을 붙이는 것이 아니라 다양한 query 위치·거리·key 수를 분리하고 압축 상태 기준선을 실제 측정하는 것이다. **제안:** A03/A04를 고친 뒤 long-delay와 key coverage를 고정한 held-out task를 만들고, 같은 데이터에서 GRU와 causal attention 참조를 비교한다. [Arora et al., Zoology, 원문](https://arxiv.org/abs/2312.04927)

### 5.2 Oracle는 성공하고 learned selector만 실패할 때

Sparsemax는 simplex projection으로 정확한0을 만들지만 support에 따라 Jacobian이 달라진다. Singleton support 내부에서는 score gradient가0이며, 현재 모델에서는 eta≈.018 및 cap 포화도 selector gradient를 줄일 수 있다. 이는 확인할 수 있는 수학적 경로이지 실제 학습 실패 원인이 이미 관측됐다는 뜻은 아니다. [Martins & Astudillo, ICML 2016](https://proceedings.mlr.press/v48/martins16.html)

**제안:** 먼저 실제 batch의 Q/K gradient, singleton support 비율, cap_rate, eta 이동량, score 분산을 기록한다. 문제가 관측될 때만 고정 temperature 민감도, dense warm-up, `Δu` key ablation을 별도 탐색으로 수행한다. Entmax는 softmax와 sparsemax를 포함하는 조절 가능한 희소 변환이므로 대안 비교 근거가 있다. Fractional 차수와 혼동하지 않게 `alpha_entmax`를 별도 이름으로 둔다. [Peters et al., ACL 2019](https://aclanthology.org/P19-1146/)

사후 min cap에서 많은 gradient가 막히는 경우 constrained sparsemax 계열의 **제약을 포함한 배분**도 후속 후보가 될 수 있다. 다만 총질량 B와 계수 상한 b0를 동시에 요구하면 support 크기에 따라 불가능할 수 있으므로 feasible mass/support를 먼저 정의해야 한다. 해당 논문은 번역 attention의 근거이며 fractional signed feedback의 안정성을 증명하지 않는다. [Malaviya et al., ACL 2018](https://aclanthology.org/P18-2059/)

### 5.3 Oracle도 도움이 없을 때: 저장 내용과 소마 병목 분리

DH-SNN은 여러 dendritic branch의 timing factor를 학습하는 다중 구획 모델로 다양한 시간척도를 처리한다. 이는 공유 소마·이질 가지의 근접 선례이며 그 조합 자체만으로 신규성을 주장하기 어렵다는 근거다. **제안:** oracle-conditioned branch state의 frozen linear probe와 end-to-end analog head를 구분하여 측정한다. 전자는 정보 존재, 후자는 spike 출력/학습 경로의 병목을 살핀다. 둘의 결과 없이 가지 수·소마 가중치·tau를 동시에 바꾸지 않는다. [Zheng et al., Nature Communications 2024, 원문 PDF](https://www.nature.com/articles/s41467-023-44614-z.pdf)

현재 value는 raw target 값이 아니라 signed dynamics `f_j`다. 올바른 사건의 slot을 골라도 그 f_j가 필요한 값을 그대로 담는 것은 아니다. **추론:** oracle도 못 푸는 경우 먼저 저장 표현의 value decodability와 signed cancellation을 확인하고, input-only retrieval 등은 기억 의미가 바뀌는 별도 모델로 선언한다.

### 5.4 Fractional 고유 기여와 비용을 분리

f-SNN은 fractional neuronal dynamics의 출발점이다. 현재 주 모델은 별도의 가지–소마 구조와 content selector/cap을 더했으므로 원 논문의 결과·정리를 그대로 승계하지 않는다. [Ge et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/80b4df828ee59926a5f2422f1c072d88-Abstract-Conference.html)

LongSpike는 fractional state-space dynamics에 SOE 근사를 사용하는 관련 모델이다. 확인한 문서는2026-06-11 arXiv v1 **preprint**이며 심사 완료 논문으로 표시하지 않는다. **제안:** 우선 같은 soma/입력/학습 예산의 alpha1 이질 baseline을 유지한다. 그다음 필요한 경우 실제 fractional kernel 또는 유효 응답을 맞춘 SOE 대조를 추가해 approximation error와 recall 성능·wall time·메모리를 함께 본다. 고정 tau4개를 곧바로 fitted SOE라고 부르지 않는다. [He et al., LongSpike, 원문 PDF](https://arxiv.org/pdf/2606.12895)

NvoFDE는 hidden-state에 따른 variable-order fractional dynamics의 선례다. 이는 사건별 slot 선택과 다른 방식으로 기억 커널을 조절한다. 현재 버그를 해결하는 직접 처방이 아니라, 선택 자체의 기여가 확인된 후 “개별 사건 선택이 차수 조절보다 무엇을 더 하는가”를 묻는 비교 후보다. [Cui et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33769)

### 5.5 선택 weight와 인과적 효용을 분리

Attention weight를 그대로 설명으로 읽는 데 대한 비판과, 어떤 검증이 있어야 설명력이 있는지를 다룬 후속 논쟁이 있다. 이 모델에서도 높은 M_eff만으로 인과적인 기억 사용을 증명하지 않는다. **제안:** 같은 bank의 정답/오답 slot 교체, key shuffle, coefficient shuffle을 비교하고, 재학습 Full/recent/mass-matched도 함께 보고한다. 고정 bank 개입과 입력을 바꿔 이후 전체 궤적까지 달라지는 개입은 다른 결과로 저장한다. [Jain & Wallace, NAACL 2019](https://aclanthology.org/N19-1357/), [Wiegreffe & Pinter, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1002/)

## 6. 권장 진행 순서

1. A01/A02/A05/A06/A11의 확정 오류와 배선을 수정하고 재현 probe를 새 artifact에 실행한다. 기존 결과를 덮지 않는다.
2. A03/A04의 task 의미·chance·평가 mask·집계 단위와 oracle fallback을 확정하여 사전등록에 append한다.
3. 수정된 동일 train sample로 보정하고 frozen 통계와 실제 current bound를 저장한다. Actual data의 support·cap·gradient도 확인한다.
4. A12의 짧은 end-to-end smoke와 fresh reload를 통과시킨다. Pilot 정확도를 최종 결과로 쓰지 않는다.
5. Full/oracle-trained/sparse/GRU로 먼저 과제와 경로의 판별력을 확인한다. Pilot에서 설계를 바꾸면 exploratory revision으로 분리하고 confirmatory seeds/test를 다시 고정한다.
6. 고정된 비교군·8 seeds·paired 분석으로 O7을 평가한다. 그 뒤 사전등록된 ETT H96 및 보조 H720로 진행한다. 미세한 개선이나 oracle headroom 부족은 불확실성으로 보고한다.

## 7. 이번 감사의 재현 자료와 제한

재사용 가능한 읽기 전용 probe: [scripts/audit_f_lif_pop_v3.py](../scripts/audit_f_lif_pop_v3.py). 결과: [probes.json](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/probes.json). 이 파일은 알려진 오류 출력을 유지시키는 회귀 테스트가 아니라 **수정 전후 값을 비교하는 측정 도구**다.

감사 당시 실행 명령(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3.py /tmp/nsmt_assessment_20260921
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python /tmp/nsmt_assessment_20260921/f_lif_pop_v3/forecasting/check_model.py --phase all
/home/yschoi/.conda/envs/snn_jelly/bin/python -c 'import torch; x=torch.tensor([1.,2.],requires_grad=True); x.square().sum().backward(); print(torch.__version__, x.grad.tolist(), hasattr(torch,"compile"))'
```

수정 후 현재 소스를 검사할 때 첫 명령의 마지막 인자를 `.`으로 바꾸고 **새 결과 경로**에 저장한다. 원래 사본이 없으면 위 HEAD를 별도 임시 디렉터리에 추출해 검사한다. 실행 중인 작업 트리를 reset/switch하지 않는다. 최초 snapshot과 검토 종료 전 hash를 비교했고 감사 대상 소스·기존 문서·calibration에 외부 변경은 없었다(이 감사가 append한 canonical 기록 제외).

환경은 Python3.10.18, torch1.12.0+cu113, CPU thread2, torch seed7이다. 합성 데이터 probe seed20260921, 각 key 조건300 sequences, 전류 진단256 sequences, alpha=.7/T42/K4를 사용했다. Gate runner는 파일에 고정된 각 gate seed를 따른다. ETT loader 검사는 conda lib를 `LD_LIBRARY_PATH`에 두고 원본 ETTh1 CSV를 읽었다. [부가 검사의 exact commands](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/commands.md)

이번에 실행하지 않은 것: optimizer step/새 훈련, checkpoint 성능 재평가, 원본 spikeDE golden, 실제 GPU parity, 전력/속도 benchmark, 미래 결과 자동 감시. 기존 calibration/stability 숫자는 저장 artifact 분석이며 전체 sweep 재실행이 아니다. 모든 finite sample 검사는 보편적인 안정성 증명이 아니다. 웹 검색은 원문 중심의 관련 연구 검토이며 완전한 systematic novelty search는 아니다.

## 8. 작업 에이전트의 후속 기록 규칙

이 문서의 최초 판정과 증거를 지우지 않는다. 수정 구현, smoke 전, confirmatory launch 전, 결과 집계 후에 이 파일의 마지막 기록을 다시 확인한다. 주기적으로 확인하도록 하는 실제 지시는 사용자가 해당 세션에 전달한다.

각 조치는 다음 형식으로 **끝에 append**한다.

```text
날짜/시각(KST):
대응 ID: A01, ...
구현 commit / 실행 당시 source hash:
변경 파일과 실제 동작:
사전등록 변경 여부 / 이미 관측한 결과:
정확한 실행 명령, 데이터 revision/hash, seeds:
수정 전 수치 → 수정 후 수치:
증거 artifact 경로:
상태: FIXED-PENDING-REVIEW / VERIFIED / OPEN / DEFERRED
남은 제한과 다음 단계:
```

“수정했다”는 설명이나 gate 총 pass 수만으로 이슈를 닫지 않는다. 각 항목의 통과 조건을 실제로 확인한 결과가 있어야 한다. A12는 학습 완료 결과가 생긴 뒤 별도 감사로 갱신한다. 본 문서와 함께 [canonical PROJECT_LOG](PROJECT_LOG.md)에도 실험/정정 이력을 append한다.

### 후속 기록

아직 없음. 최초 감사 시점의 이슈는 모두 위 상태다.

---

2026-09-22 00:10 KST (작업 에이전트)
대응 ID: A01, A02, A03, A04, A05, A06, A07, A11 (수정) · A08, A09, A10 (부분) · A12 (미착수)
구현 commit / 실행 당시 source hash: 아래 커밋. 파일 sha256[:12] — 8d623509002f layers.py;26ec5eddb67c check_model.py;dcfb6b72d099 calibrate.py;548933ad7e39 config.py;656bd4302131 data_provider/synthetic.py;
변경 파일과 실제 동작:
- `layers.py` — mass_matched를 상한 적용 후 질량 기준으로(A01), `eta_fixed`/`cap` 생성자 배선(A06), aux에 support 4종·`has_history` 추가, full 모드의 가짜 `p`를 `None`으로(A08)
- `data_provider/synthetic.py` — run layout을 revision r2로(A03), `rng_codes` 비복원 추출(A11), `task_stats`로 chance 두 종·coverage·lag·first-query 계산(A04)
- `calibrate.py` — 실제 전류 기반 G11·유한성 검사·sparse 초기 실패 처리(A05), 고정 표본으로 전 scale 비교·hash/args/norm 통계 저장·덮어쓰기 방지 파일명(A10)
- `config.py` — `neuron_kwargs`에 `eta_fixed`/`cap` 포함(A06), `--gap_range` → `--min_gap`
- `check_model.py` — G4 사유 정정, G7b 허용오차 명시, G15를 스파이크 수준으로(A07), G13을 실제 실행 aux 기반으로(A08), 신규 G16(A01 통과조건)·G17(A06 통과조건)
사전등록 변경 여부 / 이미 관측한 결과: `Population_fLIF_v3_prereg_KO.md` §2C에 D-P~D-V를 날짜부로 append. 학습 결과는 아직 없으므로 결과를 본 뒤의 기준 변경은 없다. O7-①의 0.5는 감사 권고대로 유지했다.
정확한 실행 명령, 데이터 revision/hash, seeds:
```bash
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
cd NSMT/f_lif_pop_v3/forecasting
/home/yschoi/.conda/envs/snn_recall/bin/python check_model.py --phase all
/home/yschoi/.conda/envs/snn_recall/bin/python calibrate.py --task recall --n_train 1024 --cpu
/home/yschoi/.conda/envs/snn_recall/bin/python calibrate.py --task ett --data ETTh1 --cpu
```
데이터 revision r2, data_seed 20260921, 각 난이도 300 sequences, torch seed 7, probe 표본 sha256[:16]은 보정 JSON의 `probe_sample_sha256_16`.
수정 전 수치 → 수정 후 수치:
| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| A01 mass_matched kappa | 1.000000000000000 (= full, `\|Δ\|`=1.11e-16) | 선택 모델과 동일 0.589396241989, `\|control−full\|`=0.2957, `\|Σc_control−Σc_sel\|`=1.78e-15 |
| A02 M_eff (동일 반례) | 공식값 0.5090226726020335 | 실측 0.1383411553307029 (정의를 실측으로 교체) |
| A03 재질의 고유 key | 2.52 / 2.59 / 2.55 (k=3/5/8) | coverage 99.3% / 86.4% / 48.2% |
| A04 chance (k=3/5/8) | 0.1662 / 0.1649 / 0.1640 | uniform-slot 0.1569 / 0.1263 / 0.1031, full-kernel 0.1305 / 0.1080 / 0.0913 |
| A05 max\|I\| (recall) | 8.0 (raw patch × scale) | **30.504** (Linear→norm→scale 후), G11 bound 305.0, 관측 max\|u\| 19.49 |
| A06 eta_fixed/cap | 모델에 미전달 | eta_fixed 0→0.0, 1→1.0 정확 덮어쓰기·eta_hat 동결; no-cap max c 7.107 > b₀ 1.101 |
| A07 G4 사유 | "torch≥2 CPU 환경 없음" (오류) | "spikeDE 소스 부재". snn_jelly torch 2.11.0+cu130 CPU 동작 확인 |
| A07 G15 | 전압 변화만 | 스파이크 8개 변화, 이전 시점 0 |
| A11 rng_codes(16,4) | 거부 샘플링 (사실상 무한) | 0.0001s, 16개 distinct; (17,4)는 ValueError |
| 게이트 총계 | 17 passed / 1 not run | **19 passed / 0 failed / 1 not run** |
| 보정 (recall, r2) | scale 8.0, rate 0.2278 | scale 8.0, rate 0.1848 (r2 과제 + 고정 표본) |
| 보정 (ETTh1) | scale 6.0, rate 0.1861 | scale 6.0, rate 0.1896, max\|I\| 50.756 |
증거 artifact 경로: `f_lif_pop_v3/analysis/check_model_phaseAC.txt`, `f_lif_pop_v3/analysis/recall_task_sanity.txt`, `f_lif_pop_v3/forecasting/results/calibration/*_260921-2348.json` 외 신규 timestamp JSON
상태: A01 A02 A03 A04 A05 A06 A07 A11 = FIXED-PENDING-REVIEW · A08 A09 A10 = OPEN(부분) · A12 = OPEN(not run)
남은 제한과 다음 단계:
- **A08 잔여:** aux에 support 4종과 `has_history`는 넣었으나 score/QK norm, branch 상관·유효 rank, 선택 gradient norm은 아직 없다. **gradient를 유지하는 analog readout 경로**는 `ours.py`가 없어 미구현이다. 현재 `state`/`voltage`는 detach된 진단값이며 여기에 head를 붙이면 end-to-end 학습이 되지 않는다는 지적은 유효하다.
- **A09 잔여:** 사전등록 §2C로 O7·chance·G4 사유는 정정했으나 `IDEA_LOG.md`(w 초기값·η·cap 후 M_eff·속도 이득 표현)와 `IDEA_SUMMARY_FOR_REPORT.md`의 과장 표현은 아직 미정리.
- **A10 잔여:** 덮어쓰기 방지·고정 표본·source/표본 hash·full args·norm 통계·sparse 확인 저장은 완료. `Config.run_id`의 전체 설정 hash와 run UUID, checkpoint에 `norm_mean/std` 포함 및 fresh reload 확인, `stability_sweep.txt`의 F1/F2 출력 보충은 미완. PROJECT_LOG 23:16의 rate .2278/state 17.81과 JSON .2280505933/state 18.4044의 불일치 **원인은 확인됐다**: 그 사이에 `fit_norm`을 1배치에서 8배치로 바꾸었고 로그는 이전 실행값이다. 두 값 모두 r2 재보정으로 대체되었다.
- **A03 잔여 제한:** r2에서도 T=42의 구간 수(약 12.4)가 상한이라 `n_keys=8`의 coverage는 48.2%다. 주 난이도 축을 `{3, 5}`로 두고 8은 stress 조건으로만 보고하도록 D-R에 명시했다.
- **다음 단계:** A12의 진입 조건(`ours.py`/`train.py`/`test.py`, oracle 3종 분리, GRU 대조군, 학습 중 실제 전류 G11, neorecall 형식 저장)을 구현한 뒤 smoke → pilot → confirmatory 순으로 진행한다.
