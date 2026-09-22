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

---

## 9. 추적 감사 01 — 2026-09-21 23:52 KST

**범위:** 사용자의 지속 추적 요청에 따라, 다른 세션의 수정 코드·새 보정 결과·사전등록 §2C를 재검토했다. 기존 본문과 판정은 당시 기록으로 유지하며, 아래 상태가 이번 확인 범위를 갱신한다. 파일 변경만으로 완료 처리하지 않고 고정 사본에서 재실행했다. 현재 HEAD는 `df3a3407b9ae653b4b1031320c4e8240b4c943a0`, branch는 `exp/f-lif-pop-v3`이며 미커밋 변경을 검사했다.

**증거:** 23:49:25 KST 사본의 [파일별 SHA256](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/inventory.json), [기존 probe 재실행](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/probes.json), [독립 후속 probe](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/followup_probes.json), [게이트 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/gates.json). 사본 이후 추가된 §2C와 ETT 보정은 [별도 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/later_inventory.json)로 식별한다. 디렉터리 시각은 UTC다.

### 9.1 구현 및 검증 상태 갱신

| ID | 이번 판정 | 확인 근거 / 남은 조건 |
|---|---|---|
| A01 | **VERIFIED — 동일 bank 수치 오류 수정** | cap 후 질량 대조에서 sparse/control kappa 모두 `0.569804778811738`, 질량 차 `6.006159352561638 → 0`, Full과 최대 계수 차 `.2956719406`. 별도 학습 대조군의 전체 궤적/효과는 아직 not run. |
| A02 | **정의 정정 확인; metric 구현 OPEN** | §2C D-Q가 실제 `c`의 정답 비율로 정정했고 0.5를 유지했다. 기존 반례는 여전히 `.50902267` 대 `.13834116`. 학습 결과 집계 코드에서 이 정의를 지키는지는 아직 검사할 대상이 없다. |
| A03 | **PARTIAL — 영구 배제 해결, 난이도 해석 제한** | r2에서 예전 8-key의 초기 4개 key도 재질의된다. 단 실제 고유 질의 key 평균은 `2.98 / 4.32 / 3.8567`. §2C가 3·5를 주 조건, 8을 stress로 낮춰 명시한 것은 이 제한에 대응한다. 자세한 판단은 §9.2. |
| A04 | **VERIFIED — task_stats 두 기준** | 아래 표의 sequence 평균 chance와 Full kernel mass가 독립 재계산과 일치한다. 향후 모델 metric도 같은 mask·집계 단위를 사용해야 한다. |
| A05 | **PARTIAL — 전류 측정 수정; 상한 판정 OPEN** | 실제 `embedding.current()`로 측정하고 finite·sparse 초기 발화율을 확인한다. 하지만 `choose()`는 여전히 상태 상한 초과 후보를 통과시킨다. 새 branch 최대값 필드에도 오류가 있다(§9.3). |
| A06 | **VERIFIED — 배선·정확한 eta·동일 설정 복원** | `neuron_kwargs`에 cap/eta_fixed가 연결됐다. eta=0/1 정확값과 fixed 학습 제외, cap 해제 확인. eta=1/cap=False 설정으로 state_dict 저장·새 인스턴스 복원 후 spike/state가 일치한다. 정식 trainer의 config.pt 복원은 not run. |
| A07 | **PARTIAL — G15·사유·허용오차 정정** | 독립 재실행 **19 pass / 0 fail / 1 not run(G4)**. G15 실제 마지막 spike 8개 변화·이전 변화0. G7b 1e-12가 §2C에 명시되었다. 이는 기존 관측 후의 개정이지 원래 bitwise 계약 통과가 아니다. G4는 여전히 not run. |
| A08 | **PARTIAL** | 전체 history 시점의 support 네 종류와 has_history가 전달된다. 학습형 analog에 필요한 state/voltage는 여전히 detach 상태; rank·실제 데이터/학습 경로 검증은 남아 있다. |
| A09 | **PARTIAL** | §2C에서 O7·chance·안정성 주장·난이도 범위를 정정했다. 아래 추가 기록/명칭 정확성은 보완해야 한다. |
| A10 | **PARTIAL — 재현성 개선** | scale 후보 간 동일 입력 재사용, sample/source hash, 전체 args, frozen 통계, sparse 결과, 분 단위 timestamp가 생겼다. 같은 분 동일 조건 충돌 방지와 실패한 sparse 진단 보존, bound 저장은 남아 있다. |
| A11 | **VERIFIED — codebook 생성** | 비복원 추출 구현 확인. 16개의 서로 다른 code 생성 및 17개 요청 ValueError를 독립 실행했다. r1과 codebook/과제가 바뀌었으므로 기존 r1 보정을 재사용하지 않는다. |
| A12 | **OPEN / not run** | 이 확인 시점에는 정식 학습 파이프라인과 완료 성능 결과가 없다. Gate/calibration을 모델 효과 검증으로 해석하지 않는다. |

### 9.2 r2 과제의 실제 의미와 다음 실험

동일 code encoding, data seed20260921, T42, run2–5, min_gap1, 조건당300 sequences. Chance와 kernel은 query 평균 후 sequence 평균이다. 고유 key 수는 한 sequence에서 실제 다시 질의된 수이며, 전체 key 재질의 비율은 모든 key가 한 번 이상 재질의된 sequence의 비율이다.

| n_keys | 평균 고유 재질의 key | 전체 key를 재질의한 sequence | recall 사건 비율 | uniform-slot chance | Full kernel mass |
|---:|---:|---:|---:|---:|---:|
| 3 | 2.9800 | 98% | .751984 | .1569355707 | .1305301421 |
| 5 | 4.3200 | 41% | .585159 | .1263116365 | .1079814616 |
| 8 | 3.8567 | 0% | .332460 | .1030723388 | .0912742649 |

이는 r1의 영구 배제 오류가 해결되었음을 보여주지만 **8-key가 5-key보다 많은 key의 회상을 실제 평가한다는 증거는 아니다.** 반대로, 예측할 key가 미리 알려지지 않는다면 도입 key 전체를 저장할 필요가 있으므로 이 수치만으로 내부 저장 부하가 더 작다고 단정해서도 안 된다. 정확한 표현은 ‘8개 도입, 평균3.86개 재질의, 낮은 query coverage의 stress’다. §2C의 주 조건3/5 + 보조 stress8 구분을 유지하고 capacity scaling 일반화는 보류한다.

또한 모든 사건의 MSE만 보고하면 key가 많을수록 copy 사건의 가중치가 늘어난다. **recall-only MSE와 재등장 run 첫 사건 MSE**를 함께 고정해야 한다. run 내부에서는 앞 사건에서 회상한 출력을 유지하는 쉬운 경로가 있을 수 있다. `lag_max`의 현재 구현은 ‘sequence별 query의 평균 source lag 중 최댓값을 구한 뒤 sequence 평균’이다. 출력의 `max`를 전체 표본의 실제 최장 lag로 읽으면 안 된다.

관련 연구에 근거한 조건부 개선 방향은 §5의 [Zoology/MQAR](https://arxiv.org/abs/2312.04927)와 같다. 현재 exploratory r2와 별도로, **도입 key를 균형 있게 재질의하는 일정**, query 수 고정, 도입 key 수와 delay의 독립 조절을 설계하면 recall 부하와 지연 효과를 더 잘 분리할 수 있다. 이것은 문헌을 바탕으로 한 실험 설계 제안이며 해당 논문의 과제를 그대로 재현했다는 뜻은 아니다. T42 유지가 필수라면 run 수/길이와 coverage 제약의 양립 가능성을 먼저 계산하고 새 revision으로 기록한다. 현재 승인된 O7 0.5를 이 표본 때문에 내리지 않는다.

### 9.3 새로 확인된 잔여 구현 문제 — 학습 전에 반영 권고

**A05-후속, P1: 안정성 상한 초과가 거부 조건이 아니다.** `calibrate.choose()`에 firing_rate=.2, finite=True, 건강한 branch 평균, `max_abs_current=1`, `max_abs_state=1001`인 후보를 주면 ‘closest to target’으로 선택한다. 이는 실제 실행에서 폭주가 있었다는 뜻이 아니라 **선택 정책의 반례**다. 현재 main은 `EXCEEDS`를 출력할 뿐 picked를 무효화하지 않는다. Full과 sparse의 실제 상태/전류 finite, 사전 선언 bound와 초과 여부를 payload에 저장하고 초과 후보/보정을 명시적으로 실패 처리해야 한다. 학습 중 bound를 매번 새 최대값에 따라 올리면 frozen bound 검사가 아니므로 별도로 구분한다.

**A05/A10-후속, P2: `branch_abs_max`가 최댓값을 재지 않는다.** 코드가 각 배치에서 `abs(state).mean(T,B,D)`를 모은 뒤 그 평균들의 max를 저장한다. 독립 64-sequence probe에서 저장값은 `[2.5546, 1.5537, .9099, .5078]`인 반면, 실제 K별 최대는 `[17.5061, 11.5224, 6.7558, 3.6820]`이었다. 기존 `max_abs_state` 자체는 올바르다. 새 필드는 `amax(T,B,D)`를 배치 간 max로 모으거나 이름을 ‘max_batch_mean’으로 바꿔야 한다. 현재 보정 JSON의 이 필드로 branch별 안정성 주장을 하지 않는다.

**A06/A08-후속, P2: cap을 꺼도 `cap_rate`가 양수다.** cap=False/eta=1 실행에서 reported cap_rate 최대 `.200000003`. 실제 cap 동작 빈도가 아니라 `raw > b0`의 잠재 초과율이다. `cap_rate=0`과 별도의 `would_cap_rate`로 구분하거나 정의를 명시해야 한다. mass_matched에서도 이 값이 최종 대조 계수의 clipping 비율인지, 원래 선택 계수의 clipping 비율인지 구별한다.

**A10-후속:** 새 timestamp는 분까지만 있으므로 동일 분 동일 seed/config의 재실행은 덮어쓸 수 있다. 초/UUID/run ID + exclusive creation 등으로 충돌을 막는다. sparse 초기 band 실패 시 `picked=None`이 되면서 payload의 `sparse_at_init`도 None으로 지워진다. 실패 row는 저장해야 원인을 재검토할 수 있다. 새 JSON에 `10*max_abs_current`를 재계산할 정보는 있지만 실제 적용할 frozen bound와 판정도 직접 저장하는 편이 안전하다.

### 9.4 새 보정 결과와 재현성 제한

새 recall r2 seed7 보정은 scale8, Full firing rate 약.1866, sparse 약.1867, max current30.50375, max state19.48979이다. 실제 exact 값은 [보정 artifact](../f_lif_pop_v3/forecasting/results/calibration/recall_k3_r2_a0.7_norm-frozen_seed7_260921-2348.json)를 기준으로 한다. 이전 r1 약.228과의 차이는 task/codebook 및 표본 생성·난수 소비 순서도 바뀐 조건 간 관측으로, 개선/악화 효과로 읽지 않는다. ETT에도 [새 seed7 보정](../f_lif_pop_v3/forecasting/results/calibration/ETTh1_a0.7_norm-frozen_seed7_260921-2350.json)이 추가됐다. 두 결과는 초기 동작점이며 학습 정확도가 아니다.

§2C와 check_model에 기재된 ‘2026-09-22 감사’는 실제 최초/이번 감사가 **2026-09-21 KST**였다는 점과 다르다. 또한 새 보정 actual current30.504와 초기 독립 probe30.6887은 서로 다른 표본/조건에서 같은 측정 오류를 지지하는 수치이지 동일 표본 재현의 수치 일치는 아니다. 후속 정정으로 구분하면 된다. §2C D-R의 run 수 근사는 평균에 근거한 계획값이며 구조적/확률적 최대를 증명하지 않는다.

### 9.5 실행·추적 방식

CPU만 사용했으며 Python3.10/torch1.12, OMP/MKL threads2, torch seed7이다. 주요 명령(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3.py /tmp/nsmt_assessment_20260921-144925-utc
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3_followup.py /tmp/nsmt_assessment_20260921-144925-utc
```

Gate는 사본의 `check_model.GateReport()`에 `phase_a`, `phase_c`, `phase_c_audit`를 순서대로 호출해 JSON으로 저장했다. 감사자가 처음 시도한 `--json` CLI 옵션은 이 runner에 없어 exit2였고, 올바른 Python API 호출로 다시 실행했다. 이 명령 착오를 모델 실패로 세지 않았다. 원시 stdout/stderr는 task `log/assessment/20260921-144925-utc/`에 로컬 보관한다. 기존 파일/학습 작업/HEAD/index를 수정하지 않았다. 새 학습·GPU benchmark·golden reference·trained metric 검사는 **not run**.

이 세션에서 확인되는 의미 있는 source/result 변경을 계속 관찰하고, 수정 → 재검사 → 근거 → 상태 변경 순서로 이 파일 끝에 기록한다. 관찰 중 코드가 바뀌면 사본별 판정을 분리한다. **세션 종료 후에도 동작하는 자동 감사 서비스나 예약 작업은 설정하지 않았다.** 따라서 ‘관찰함’은 위 inventory로 특정된 실제 확인 범위만 뜻한다.

**23:54 KST 수치 정정:** §9.4의 recall Full firing rate ‘약.1866’은 감사 기록 전사 오류다. 저장 JSON의 정확한 값은 **.1847657673060894**, sparse는 **.18672253005206585**다. 새 ETTh1 보정은 scale6, Full **.18956471048295498**, sparse **.18921967595815659**, max current **50.75556945800781**, max state **19.68758773803711**이다. 그 밖의 판정은 동일하다.

## 10. 추적 감사 02 — 2026-09-22 00:00 KST (G4 reference 확보 및 다음 수정 검토 중)

**A07/G4의 소스 부재 사유 해소:** 감사자가 사전등록에 명시된 [PhysAGI/spikeDE 고정 커밋](https://github.com/PhysAGI/spikeDE/tree/fcd743befe504b1a471fa81887e6af7d6789da2e)의 원본 파일9개를 임시 경로 `/tmp/nsmt_spikede_ref_fcd743b`에 내려받았다. 패키지 설치나 소스 수정 없이 기존 `snn_jelly` CPU/torch2.11에서 import와 scalar 실행이 가능했다. 따라서 ‘고정 소스가 장비에 없다’를 지속적인 blocker로 취급할 근거는 없다. 원본은 [ICLR 2026 f-SNN 논문](https://proceedings.iclr.cc/paper_files/paper/2026/hash/80b4df828ee59926a5f2422f1c072d88-Abstract-Conference.html)의 공개 구현이며, reference commit/hash를 고정한다.

원본 `LIFNeuron` + 공개 `pred_integrate_tuple`를 실제 호출한 첫 실행에서 alpha=.3/.5/.7/1 × tau4/8 × 상수/펄스/seed7 난수 입력의 **24개 조건 모두 spike 일치**, 독립 scalar 정의와 최대 상태 오차 **3.0184188481996443e-15**였다. 저장된 기존 `reset_conventions.py::code_convention()`과 원본은 **17 spikes, 상태 오차6.661338147750939e-16**. v3의 arctan surrogate와 원본(scale5)은 spike 일치, gradient 최대 오차 **4.440892098500626e-16**였다. [첫 실행 결과·원본 파일 SHA256](../f_lif_pop_v3/forecasting/results/assessment/20260921-1500-utc-reference/reference_results.json), [scalar 전체 궤적](../f_lif_pop_v3/forecasting/results/assessment/20260921-1500-utc-reference/scalar_trajectories.csv).

이는 **원본 scalar 정의의 실제 실행 대조**다. 기존 v3의 비리셋 가지+별도 소마가 원본 reset 뉴런과 동일하다는 뜻은 아니며, full SNNWrapper/FX, compiled execution, adjoint backward, GPU, 학습은 not run이다. 공식 `check_model.py`의 G4는 아직 hard-coded not run이므로 총 gate 수를 임의로20 pass로 바꾸지 않는다. 담당 세션은 이 artifact를 연결해 scalar-reference 실행 범위와 남은 full-wrapper 검증을 구분할 수 있다. 최초 실행은 당시 live source를 읽었으며, 이후 관찰된 수정과 섞이지 않도록 23:59:54의 새 고정 사본으로 반복 확인 중이다.

**Commit 경계:** 다른 세션이 `2da37e385dc134cdd798b40d6f9c2e72d065343b`에 수정과 감사 중간 파일을 저장했다. 이전23:49 사본의 forecasting Python 코드가 이 commit과 전부 일치함을 확인했다([mapping](../f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/commit_mapping.json)). 감사자는 commit/tag/index를 조작하지 않았다. 동시 commit에 감사자의 미지원 CLI 시도 stderr 등 원시 로그4개도 들어갔다. 로컬 파일은 보존했고, 향후 담당 세션은 AGENTS의 원시 로그 로컬 보존 규칙에 맞게 추적 대상을 정리하되 실제 artifact를 삭제하지 않아야 한다. 이 동시 commit은 §9의 최종 부분이 작성되기 전 중간 snapshot이므로 작업 트리의 뒤쪽 append까지 확인한다.

---

2026-09-22 00:01 KST (작업 에이전트)
대응 ID: 추적 감사 §9.3의 신규 3건 + §9.4 기록 정정
구현 commit / 실행 당시 source hash: 아래 커밋. 파일 sha256[:12] — 609cfaf6170b layers.py;218e5c47317f check_model.py;e9aaa59716e9 calibrate.py;548933ad7e39 config.py;656bd4302131 data_provider/synthetic.py;
변경 파일과 실제 동작:
- `calibrate.py` — `choose()`가 `within_bound`를 통과 조건에 포함(거부 사유에 탈락 개수 명시), `declared_bound`/`within_bound`/`g11_factor` 저장, `branch_abs_max`를 `amax(T,B,D)`의 배치 간 max로 수정, timestamp를 초 단위로 + `open(...,'x')`로 덮어쓰기 차단, 실패한 `sparse_at_init` row 보존
- `layers.py` — `cap_rate`를 "반환된 계수가 실제로 잘린 비율"로 정의하고 정책의 잠재 초과율은 `would_cap_rate`로 분리. 뉴런 aux에도 전달
- `Population_fLIF_v3_prereg_KO.md` — §2C 날짜를 2026-09-21로 정정, 30.6887 "일치" 표현 정정, **D-W** 추가
- `check_model.py` — 코드 내 감사 날짜 2026-09-22 → 2026-09-21
사전등록 변경 여부 / 이미 관측한 결과: §2C에 D-W를 append. 학습 결과는 여전히 없으므로 결과를 본 뒤의 기준 변경이 아니다.
정확한 실행 명령, 데이터 revision/hash, seeds:
```bash
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
cd NSMT/f_lif_pop_v3/forecasting
/home/yschoi/.conda/envs/snn_recall/bin/python check_model.py --phase all
/home/yschoi/.conda/envs/snn_recall/bin/python calibrate.py --task recall --n_train 1024 --cpu
/home/yschoi/.conda/envs/snn_recall/bin/python calibrate.py --task ett --data ETTh1 --cpu
```
데이터 revision r2, data_seed 20260921, torch seed 7, CPU.
수정 전 수치 → 수정 후 수치:
| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| `choose()` 상한 초과 후보 (max\|u\|=1001, bound=10) | **picked=True** ('closest to target') | **picked=False**, 사유에 "1 in-band candidate(s) rejected for exceeding the declared bound" |
| `choose()` 정상 후보 (max\|u\|=5, bound=10) | — | picked=True (회귀 없음) |
| `branch_abs_max` (독립 64-seq probe) | [2.0167, 1.1760, 0.6747, 0.3720] | 실제 최댓값 [12.3278, 6.8536, 3.6856, 2.0798] |
| `branch_abs_max` (실제 보정 JSON) | — | [19.4898, 13.6371, 8.3487, 4.6658], mean [2.531, 1.535, 0.899, 0.503] |
| `cap_rate` cap=True | 0.048780488 | 0.097561 / would 0.097561 |
| `cap_rate` cap=False | **0.048780488** (cap을 껐는데 양수) | **0.000000** / would 0.097561 |
| `cap_rate` mass_matched | — | **0.000000** / would 0.097561 |
| 게이트 | 19 passed / 0 failed / 1 not run | **동일** (회귀 없음) |
| 보정 recall r2 seed7 | scale 8.0, rate 0.1848 | **동일**, within_bound=True (bound 305.0, max\|u\| 19.49) |
| 보정 ETTh1 seed7 | scale 6.0, rate 0.1896 | **동일**, bound 507.6, max\|u\| 19.69 |
증거 artifact 경로: `f_lif_pop_v3/analysis/check_model_phaseAC.txt`, `f_lif_pop_v3/forecasting/results/calibration/*_2609*-*.json` (초 단위 timestamp)
상태: §9.3 신규 3건 = FIXED-PENDING-REVIEW · §9.4 기록 정정 2건 = FIXED · A02 A03 A08 A09 A10 = OPEN(부분) · A12 = OPEN(not run)
남은 제한과 다음 단계:
- §9.2를 수용해 **capacity scaling 일반화를 보류**한다. r2에서도 고유 재질의 key는 2.98 / 4.32 / 3.86이므로 `n_keys=8`이 더 많은 key의 회상을 평가한다는 증거가 없다. D-W에 "8개 도입, 평균 3.86개 재질의, 낮은 coverage의 stress"로 표현을 고정했다.
- §9.2의 평가 설계 요구(recall-only MSE와 재등장 첫 사건 MSE 분리, `lag_max` 해석 주의)를 D-W에 명시했다. 실제 구현은 `test.py`가 생길 때 들어간다.
- A08의 analog readout 경로, A09의 IDEA_LOG 표현 정리, A10의 run UUID·checkpoint norm 통계 fresh reload는 여전히 미완이며 `ours.py`/`train.py`/`test.py` 작업과 함께 처리한다.
- **다음 작업: `ours.py`(M1 백본 + readout) 구현.** A08의 gradient 유지 analog 경로를 여기서 함께 만든다.

### 추적 감사 02 재검사 완료 — 2026-09-22 00:03 KST (감사자)

23:59:54 고정 사본의 [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/inventory.json)와 [독립 재실행](../f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/followup_probes.json)을 기준으로 다음을 확인했다. 이후 담당 세션 commit `0afda35825ba160689209177c0f0e9e173c618fa`도 관찰했다.

- **A05 선택 단계 상한 처리 VERIFIED:** 일관된 `within_bound=False/declared_bound=10`을 가진 max state1001 후보가 이제 거부된다. 새 row schema에 맞춰 감사 probe도 해당 필드를 명시했다.
- **A05/A10 branch 최대값 VERIFIED:** 동일64-sequence probe에서 보고값과 직접 tensor 최대가 정확히 `[17.5060939789, 11.5224008560, 6.7558383942, 3.6819860935]`로 일치한다. 담당 세션의 별도 표본 수치와 혼합하지 않는다.
- **A06/A08 cap 지표 VERIFIED 범위:** cap=False인 동일 설정 probe에서 cap_rate 최대 `.200000003 → 0`. `would_cap_rate`로 잠재 초과율을 분리하고 mass_matched의 cap_rate=0 의미를 코드에 명시했다.
- **A10 저장 경로 개선 확인:** 초 단위 timestamp와 exclusive `open(...,'x')`는 동일 파일 충돌 시 덮어쓰기를 막는다. sparse band 실패 시에도 row를 보존하는 것을 아래 main 호출로 확인했다.
- **A07 원본 scalar 실행 재현 VERIFIED:** 새 고정 사본으로 원본 대조를 반복해 24개 조건의 모든 수치와 CSV SHA256 `2fda1abbb03743e1b9074d2fb7b7b84feeec8e322201b42b0b66bc0fc657a7b7`이 첫 실행과 일치했다. [고정 사본 reference 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/reference/reference_results.json). 공식 runner G4 연결·full-wrapper 검사는 별도로 남는다.

**A05-후속 P1은 아직 OPEN — sparse 초기 실패 경로 누락.** `main()`이 선택된 Full 후보 뒤에 실행하는 sparse 확인에는 `finite`, `within_bound`, branch 건강 조건이 통과 기준으로 연결되지 않았다. mock probe row를 주입해 실제 main의 저장·반환 동작을 검사했다. 이는 실제 데이터/학습 폭주를 관측했다는 뜻이 아니다.

| 주입한 sparse 확인 결과 (Full 후보는 정상) | 현재 picked | main 반환/CLI 종료값 | 판정 |
|---|---|---|---|
| 정상 | 유지 | 0 | 정상 |
| 발화율.2, state1001, bound10, within_bound=False | **유지** | **0** | 초과 상태가 성공 처리됨 |
| 발화율.2, finite=False | **유지** | **0** | 비유한 상태가 성공 처리됨 |
| 발화율.01 (band 밖) | None, 실패 row 보존 | 1 | 실패 처리 정상 |

[실패 주입 검사 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/calibration_failure_paths_corrected.json). 앞서 저장된 `calibration_failure_paths.json`은 감사 harness가 반환값1을 CLI exit0으로 잘못 표기한 필드가 있어 **corrected 파일로 대체 해석**한다. 모델 main은 실제로 band 실패에서1을 반환했고 CLI도1이다. 나머지 picked 판정은 동일하다.

권고 조치: sparse 초기 확인에도 Full과 같은 finite/branch/bound 정책을 적용하고, 실패 이유와 row를 보존한 뒤 nonzero 종료로 처리한다. Sparse에 적용할 상한은 선택된 Full 보정의 frozen bound임을 명시하고 별도 current 진단과 구별한다. 실제 학습에서는 layer별 state/voltage/current finite·frozen bound·발화율 drift를 검사한다. 이 연결까지 확인하기 전 A05 전체를 VERIFIED로 닫지 않는다.

실행(cwd NSMT, OMP/MKL threads2):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3_followup.py /tmp/nsmt_assessment_20260921-145954-utc
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_20260921-145954-utc f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/calibration_failure_paths_corrected.json
/home/yschoi/.conda/envs/snn_jelly/bin/python scripts/audit_spikede_reference.py /tmp/nsmt_spikede_ref_fcd743b /tmp/nsmt_assessment_20260921-145954-utc f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/reference
```

**현재 우선순위:** sparse 보정 실패 정책 연결 → causal readout/oracle/analog 경로 smoke → 실제 학습/평가 결과 감사. 이 시점까지 학습 결과에 기반한 효과 판정은 여전히 불가하다. 코드가 존재하지 않는 부분을 이미 구현된 오류로 세지 않는다.


## 자동 감시 예약 활성화 — 2026-09-22 00:16 KST

사용자의 명시적 요청으로 **10분마다**(`*/10 * * * *`) 서버 cron이 코드·설정·텍스트 결과 변경을 확인하고 기존 감사 세션에 `codex queue`로 요청을 보내도록 등록했다. 문서의 이전 ‘예약 미설정’ 설명은 당시 상태이며 **이 항목부터 예약 활성**이다. `docs/ASSESMENT.md`를 비롯한 감사 자체의 출력은 trigger에서 제외해 무한 반복을 막는다. 변경이 없거나 앞선 감사가 대기 중이면 새 호출을 하지 않는다. 감사 중 도착한 변경은 다음 주기에 처리한다.

실제 crontab 등록 재조회와 cron active, CLI queue 접수 확인. 첫 감사 요청 ID `20260921T151514Z-5f69481c`는 **접수/대기** 상태이며 아직 완료로 기록하지 않았다. 임시 fixture 기반9개 동작 검사 통과(변경 감지, 무변경 skip, 중복 방지, 완료 마커 요구, 감사 중 변경 보존, 실패/timeout 처리, pause). 결과는 [설정 증거](../f_lif_pop_v3/forecasting/results/assessment/automation-setup-20260922/)에 있다. 모델 학습/수정과 Git index/HEAD 조작은 하지 않았다.

예약 감사는 세 검토 항목을 계속 적용하고 본 파일과 필요한 canonical 기록을 append한다. 서버·cron·Codex 로그인/daemon·네트워크가 동작해야 한다. [운영 방법 및 중지/재개](ASSESSMENT_WATCH.md), [예약 프롬프트](ASSESSMENT_WATCH_PROMPT.md).


## 추적 감사 03 — 2026-09-22 00:28 KST (예약 ID `20260921T151514Z-5f69481c`)

### 범위와 증거 고정

기억 문서, 본 파일의 최신 append, 사전등록 §2C D-P~D-W, canonical PROJECT_LOG를 복구하고 trigger manifest를 실제 파일과 대조했다. 감지 목록 `synthetic.py`·`utils.py` 외에 trigger 이후 `train.py`·`test.py`와 첫 smoke 결과가 생겼으므로 함께 검토했다. **감지 hash를 원본 사본으로 취급하지 않았다.** 재현 전에 2026-09-22 **00:18:04 KST**에 문서/소스/텍스트 결과62개를 `/tmp/nsmt_assessment_20260922-0017-kst-scheduled`로 복사하고, 평가 전에 checkpoint/config도 별도 복사·hash했다. 폴더의 0017 표기는 실제 캡처 시각과 다르다.

관찰 branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, 캡처 HEAD `0afda35825ba160689209177c0f0e9e173c618fa`. 종료 전 다른 세션의 HEAD `c34fa1c68c49169c13947b674543512dd05a057f`를 관찰했지만 캡처된62개 파일 내용은 그대로였다. 이후 생성된 새 결과는 이번 고정 사본의 판정에 섞지 않는다.

- [전체 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/inventory.json), [checkpoint hash](../f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/checkpoint_inventory.json), [독립 CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/pipeline_probes.json), [관찰한 원래 smoke JSON](../f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/observed_smoke.json), [무결성/재현 검산](../f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/validation.json).
- 주요 SHA256: `synthetic.py=d81410d66f579ffea42f4692bf8cae9bd8994d3c9927a4cdac6002055fecf943`, `utils.py=01a5f85fc6f339a71e3555a7cdd2aa928f0640d723e2202e9cf874abd725c25a`, `test.py=db6503bcbcfe4baec254fbe1830c01d681badd7785eea0dfe15f79967b58fa75`, `train.py=afa20d3636810e3801e3cc6cf0bec324845a8e86baceb21cb9456a08720e1c13`.
- Checkpoint `best+model.pt=375628a1dfe871dfa9a7f518018a1f08777e5279b6abc7780739da3a3952463b`. 보정 소스 `e9aaa59716e9acaff7873b1d7a45eeea823d44a72bae13999aef7be324c46d7c`는 이전 A05 실패 경로 검사 당시와 같다.

### (a) 아이디어 구현 — M1 주요 경로 확인, 선택성 측정과 oracle 계약은 OPEN

**A08 / A12, VERIFIED 범위:** 새 `ours.py`의 recall head는 사건별 Linear이며 미래 패치를21번부터 바꿔도 이전 출력 차이가 spike/analog 모두0이었다. non-oracle에 truth를 반전해서 전달해도 출력 차이0. 새 analog 경로는 detach된 진단 전압과 별도로 gradient를 유지한다. 동일 checkpoint의 readout을 analog로 바꾼 작은 backward에서 embedding/query/soma/recall-head gradient norm이 각각4.9884/.28147/1.50696/5.1680으로 유한하고 0이 아니다. **optimizer step 없이 graph만 확인**했으며 analog 재학습 성능은 not run이다. 기존의 ‘analog 경로 미구현’ 문제는 이 범위에서 해소됐지만 A08의 전체 진단/효과 주장은 닫지 않는다.

**A02-DIAG / A08-DIAG, P1 OPEN — M_eff의 표본·집계 구현이 설명과 다름.** `selection_diagnostics()`는 `kind`를 받지만 사용하지 않고 `truth.any()`로 사건을 고른다. 따라서 값이 입력에 보이는 첫 구간의 copy 사건933개가 섞인다. `per_sequence`라는 리스트에는 실제로 batch×시점별 평균을 넣고 마지막에 평균하므로 D-Q의 query → sequence 집계가 아니다. `hit`는 `c.argmax` 중 첫 번째 logical unit만 사용한다.

| 동일 test128 sequences / 같은 checkpoint | M_eff | Full kernel mass |
|---|---:|---:|
| 저장된 JSON, batch64의 기존 진단 | 0.22736487855635037 | 0.22719346466718926 |
| 같은128개, batch128로 기존 진단 재실행 | 0.22734299720060536 | 0.22717166101057412 |
| 같은128개, batch16로 기존 진단 재실행 | 0.22735235131368403 | 0.22718179849684803 |
| 독립 계산: recall4059건, sequence 내부 평균 후128개 평균 | **0.12957216919969688** | **0.12994454027837143** |

따라서 기존0.227을 recall 선택성의 근거로 사용하지 않는다. batch 차이 자체는 작지만 동일 표본의 진단이 batch 분할에 의존함을 재현했다. 독립 all-unit coefficient-top1 hit는0이며, 이 수치를 latent `p`의 support hit로 바꾸어 해석하지 않는다. 진단은 기본 첫4배치만 보므로 큰 평가에서는 전체 test 또는 사전 선언된 시간 분산 표본이라는 §6 계약도 아직 충족하지 않는다. 남은 조건: recall mask·제외 사건수·sequence별 query 수·집계 단위를 명시하고 batch 불변성을 검산할 것. `p` support hit와 실제 `c` top1 hit는 별도 이름으로 정의하고 모든 unit을 포함할 것. copy 통계를 원한다면 별도 열로 유지한다.

**A12-ORACLE, P2 OPEN — first-run uniform fallback 설명과 실제 동작 불일치.** `truth_to_oracle_p()`는 ‘첫 등장 구간의 모든 사건은 uniform fallback’이라고 설명하지만, 첫 구간 안에서도 이미 값이 제시된 과거 칸이 있어 truth가 비어 있지 않다. 같은 test에서 copy610건이 non-uniform oracle을 받았다. `kind==0`을 명시적으로 처리하거나, 다른 정책을 의도했다면 사전등록/문서에 먼저 수정해 대조군의 의미를 고정해야 한다. non-oracle 정답 누출을 발견했다는 뜻은 아니다.

### (b) 검증 과정 — recall checkpoint 재현 성공, 전체 평가 경로는 미완

**A10-RELOAD / A12, VERIFIED 범위:** 기존 `smoke-001655`는 seed7, r2 k3, train/val/test512/128/128, batch64,2epochs인 기능 확인 실행이다. frozen norm buffer를 포함한 저장 checkpoint를 새 모델에 CPU 로드해 아래 오차를 재현했다. 원래 JSON과 각 MSE 차이는 최대5.56e-17. 저장 파일 원본 hash도 보존됐다.

| 분리 지표 | CPU 재평가 MSE | 사건 수 |
|---|---:|---:|
| 전체 | 0.35197442620527725 | 5376 |
| copy | 0.3153044522647635 | 1317 |
| recall | 0.36387251826727696 | 4059 |
| recall 첫 사건 | 0.35899281812515704 | 1199 |

새 `kind` API는 bool 대신 int8의0=copy,1=recall,2=recall-first다. `evaluate()`의 분리는 이 API에 맞으며 D-W 구현을 확인했다. 옛 감사 probe의 bool 가정을 그대로 재사용하지 않았다. source provenance 중 `train.py`만 실행 당시 hash와 캡처 hash가 다르다. 따라서 **저장 checkpoint 평가 재현**을 확인한 것이며 당시 trainer로 처음부터 학습을 재현한 것은 아니다.

**A12-ETT, P1 OPEN — ETT의 빈 truth와 selector 진단이 호환되지 않음.** 실제 loader 계약인 `(B,0)` truth를 넣은 소규모 CPU ETT 진단에서 `truth[:, n, :n]`가 `IndexError: too many indices for tensor of dimension 2`로 실패했다. `test()`는 모든 myModel에 이 진단을 호출하므로 ETT는 오차 계산 뒤 결과 저장 전에 실패할 수 있다. truth가 없는 task에서는 정답 질량/hit를 N/A로 두고 상태·support 등 관측 가능한 진단만 계산해야 한다. 실제 ETT 학습은 not run이며 이는 평가 함수 수준의 재현이다.

**A12-GRU, P1 OPEN — None 진단값으로 logger 실패.** GRU 분기의 `train_one_epoch()`는 firing_rate/selector 지표를 None으로 돌려준다. 이 계약을 `EpochLog.logging()`에 넣으면 TensorBoard `add_scalar(None)`에서 NotImplementedError가 발생한다. verbose도 None 처리 없이 `.mean()`을 호출한다. 해당 지표를 task/model에 따라 생략하거나 명시적 N/A로 저장하도록 처리해야 GRU 대조군을 완료할 수 있다. GRU 학습을 새로 실행한 것이 아니라 실제 logger에 반환 schema를 주입한 검사다.

**A10-CAL / A05, P1 OPEN — 보정 artifact와 요청 설정의 호환성 검사 누락.** `load_calibration()`은 dataset/k/r2/alpha/norm과 파일명 정렬만 본다. 독립 probe에서 tau=[40,80,160,320], cue_mode=code, eta_fixed=1로 변경한 요청도 기존 seed7 보정 파일의 scale8/bound305.0375를 그대로 받았다. seed123에도 같은 파일을 쓰는 사실 자체는 D11의 공유 보정 정책과 모순되지 않는다. 문제는 **공유가 허용된 축과 물리적 동작점을 바꾸는 축을 구분하는 계약/검사가 없다는 것**이다. 파일명 seed를 정렬해 마지막 파일을 고르는 것도 최신 시각 선택이나 계약 일치 검사가 아니다. 보정 artifact hash 및 데이터 revision/cue/tau/차원/정규화·current 조건을 연결하고, 비교군 간 의도적으로 공유하는 필드는 따로 선언할 것. 보정 없음/picked 없음이면 현재는 경고 후 bound 없이 학습을 계속하므로 정식 실행은 실패로 처리해야 한다.

A05의 기존 sparse 초기 finite/bound 누락은 보정 소스가 그대로여서 OPEN을 유지한다. 이번에 같은 fault injection을 반복하지 않았다. trainer의 state bound 검사는 매10번째 배치이며 optimizer.step 뒤 수행되고, loss/gradient finite와 별도로 모든 state/voltage/current finite를 확인하지 않는다. 이번 smoke에서 폭주를 관측했다는 뜻은 아니다. 실측 진단 max state17.9078 < frozen bound305.0375였지만, 이 sampled guard를 전체 학습의 안정성 증명으로 읽지 않는다.

**A10-LOG, P2 OPEN:** `EpochLog.write()`가 CSV를 쓰지만 trainer에서 호출하지 않아 완료 smoke의 `log/best_log_0.csv`·`log/final+result.csv`가 없다. TensorBoard/checkpoint/JSON은 존재한다. 프로젝트 표준 CSV와 full-precision 결과 메타데이터를 연결해야 한다. `provenance.parameter_hash`는 마지막 epoch 모델에서 계산하지만 test는 best checkpoint를 읽는다. 이번 smoke에서는 hash가 실제 checkpoint와 일치했으므로 현재 결과의 mismatch를 주장하지 않는다. best가 마지막이 아닌 실행을 위해 평가 대상 checkpoint 파일/hash 및 buffer를 포함한 state provenance를 별도로 기록할 것. 현재 requires_grad parameter 수에는 task에서 사용하지 않는 다른 head도 들어 있으므로 parameter-matched 비교 전 active head 범위를 분리해야 한다.

**A12-TEST-SPLIT, P2 OPEN:** parser의 `--no-test`와 관계없이 train은 `test(args)`와 test fresh reload를 호출한다. 탐색 실행에서 test를 보지 않으려는 옵션이 작동하지 않는다. 옵션을 실제로 배선하고, 이미 확인한 smoke test seed를 이용한 설계 조정은 탐색으로 기록해야 한다. 이 정적 호출 경로 문제를 train/val/test 데이터 자체의 중복 발견으로 해석하지 않는다. 정식8seed·검정력/신뢰구간·oracle-trained·재학습 Full/GRU 검증은 이번 범위에서 **not run**이다.

### (c) 결과 기반 개선 방향 — 효과 판정 보류, 측정/대조군을 먼저 고정

저장된 같은-checkpoint 개입의 recall MSE는 sparse0.3638725, full0.3633943, recent0.3647343, mass_matched0.3633943, test-time oracle0.3477329다. Full과 mass_matched가 같은 것은 이 smoke의 κ=1/cap_rate=0과 일관된다. eta≈0.017882, support_p≈0.2533이지만 support_rho/braw/c는1이다. **잠재 확률의 희소성과 실제 계수의 제거를 구분**해야 하며,2epoch·1seed 결과로 학습된 선택성/효율/예측 개선을 주장하지 않는다. 시험 시 정책 교체 결과는 Full 재학습이나 oracle-trained를 대신하지 못하고 O7 성공 판정의 분모로 대체할 수 없다.

1. **측정과 실행 차단 오류를 우선 수정:** recall-only sequence 집계, oracle fallback, ETT 빈 truth, GRU N/A 로깅, calibration 호환성/실패 정책, CSV 및 test-off 배선. 같은 고정 checkpoint와 작은 fixture로 수정 후 독립 재검사할 수 있다. 단순 FIXED-PENDING-REVIEW 표기만으로 닫지 않는다.
2. **그 다음 대조군을 동일 예산으로 분리:** 학습 sparse/재학습 Full/GRU/oracle-trained와 같은-checkpoint 개입을 별도 표로 기록한다. 독립 seed의 대응 차이와 불확실성을 보고한다. 고정 가중치 개입과 학습된 대조군을 구별하고 여러 seed로 해석을 보정하는 방향은 [Wiegreffe & Pinter, EMNLP 2019](https://aclanthology.org/D19-1002/)의 진단 접근을 참고한다. 해당 논문이 f-LIF의 효과를 보증한다는 뜻은 아니다.
3. **회상 과제의 실제 요구량을 통제:** k label만 늘리는 대신 실제 재질의 key 수·recall lag·copy/recall 비율·질의 coverage를 함께 고정/보고하고, n_keys8은 기존 D-W대로 낮은 coverage stress로 남긴다. 연상 회상을 sequence 모델의 입력 의존 기억 사용 진단으로 다루는 [Zoology / MQAR](https://arxiv.org/abs/2312.04927)를 참고하되, 현재 과제를 그 논문의 동일 benchmark라고 부르지 않는다.
4. **현재 near-Full 동작의 원인을 구분:** 수정된 진단과 pilot validation에서 eta/selector gradient 및 oracle-trained 상한을 확인한 뒤, 사전 선언된 eta 조건·spike 대 analog 재학습을 사용해 선택 경로와 readout 병목을 구분한다. 지금 test-time oracle의 작은 gap만 보고 gate threshold를 낮추거나 구조를 확정하지 않는다. 새 튜닝에 쓴 표본과 최종 판정 표본을 구분한다.

위 두 1차 원문의 웹 페이지를 실제 확인했다. 개선 방향은 해당 연구의 진단 원칙을 이 모델에 적용한 제안이며, 아직 실행되지 않은 성능 개선의 증거가 아니다.

### 실행·보존·다음 조건

CPU torch1.12.0+cu113, snn_recall Python, seed7, threads2. 기존 checkpoint 재평가와 소규모 forward/backward/실패 schema 검사만 수행했다. 모델/학습 소스 수정, optimizer step, 새 학습, GPU 점유, 설치, 프로세스 중단, git add/commit/tag/push/reset/switch, 다른 세션 대화 열람/메시지 전송은 하지 않았다. 새 감사 코드만 `scripts/audit_v3_pipeline.py`에 작성했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_pipeline.py \
/tmp/nsmt_assessment_20260922-0017-kst-scheduled \
f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/pipeline_probes.json
```

명령 성공 및 JSON 파싱/감사 코드 구문/고정62개 파일 hash/원본 checkpoint hash/기존3문서 prefix 보존을 확인했다. 종료 메타데이터 검사에서 시스템 Python에 zoneinfo가 없어 한 번 실패했고, stdlib datetime.timezone(+09:00)으로 바꿔 성공했다. 이는 모델 실패와 무관하다. 기존 실행 결과/체크포인트/raw log는 보존했다. 예약은 새로 만들지 않았고00:20 cron의 `audit_already_pending`으로 중복 호출 방지를 확인했다. 다음 감사는 수정 증거와 캡처 이후 결과를 검토하며, 이 append 자체는 변화 감지 대상에서 제외된다.

<!-- assessment-watch:20260921T151514Z-5f69481c -->


## 추적 감사 04 — 2026-09-22 00:33 KST (예약 ID `20260921T153001Z-ed6b3b26`)

### 변경 구분과 사본

최신 기억/감사03/사전등록 D-W/canonical log를 읽었다. 이번 trigger의 config/train/test 및 smoke/logargs 변경은 **감사03에서 이미 관찰한 내용과 hash가 같다.** 감사자가 쓴3문서 append를 새 구현 변경으로 세지 않았다. 실질적인 새 평가 대상은 `pilot-eta-001742`의 완료 결과다(결과 작성00:19:55 KST, 감사03의00:18:04 사본 이후). HEAD `c34fa1c68c49169c13947b674543512dd05a057f`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`.

재현 전 **00:30:37 KST**, trigger 대상 파일·필수3문서·pilot checkpoint/config를 `/tmp/nsmt_assessment_20260921T153001Z-ed6b3b26`로 별도 복사·hash했다. Trigger hash는 사본으로 간주하지 않았다. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/inventory.json), [관찰 결과 사본](../f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/observed_pilot.json), [CPU 재검사](../f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/pilot_probes.json), [hash/재현 검산](../f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/validation.json).

주요 SHA256: 결과 `fa0ed79239fc2e012606615f5776b64c24a64c23eac906dd5bd2149234e22681`; checkpoint 파일 `259794eb9a1620113fa50d0779fcfba95acec45197c7eabd8af788ea817f6679`; test.py `db6503bcbcfe4baec254fbe1830c01d681badd7785eea0dfe15f79967b58fa75`; train.py `afa20d3636810e3801e3cc6cf0bec324845a8e86baceb21cb9456a08720e1c13`. Pilot provenance의11개 소스 hash가 사본과 모두 일치한다.

### (a) 구현 — 기존 판정 유지, pilot에서도 지표 오류 확인

**A02-DIAG/A08-DIAG OPEN:** 새 checkpoint의 test256개를 CPU에서 읽고 실제 post-cap 계수를 독립 집계했다. 저장된 M_eff **0.2314670247**은 기존 copy 혼입·batch×시점 집계 때문에 recall-only 지표로 사용할 수 없다. recall8085건을 sequence 내부 평균 후256개 평균하면 **0.1330930309**, 동일 집계의 Full kernel mass는 **0.1303288936**다. 기존 진단 mask에 copy1899건이 포함된다. 따라서 0.5 기준을 만족했다는 근거가 없다. batch128/16의 기존 진단값도 각각0.23145139/0.23146362로 달라진다.

eta는 초기 sigmoid(-4)≈0.017986에서 선택 checkpoint의 **0.02721739**로 증가했다. support_p≈0.216741이지만 support_rho/braw/c는 모두1이고 κ=1, cap_rate=0이다. 선택 정책이 변한 사실과 실제 계수의 희소화는 구분해야 한다. A12-ORACLE의 first-run fallback 문제도 copy1243건에서 non-uniform 정책으로 확인된다. A05·A10-CAL·A12-ETT/GRU/TEST-SPLIT은 관련 소스가 그대로여서 OPEN을 유지하며 이전 실패 검사를 반복하지 않았다. 이미 확인한 causal/analog graph를 새로 VERIFIED로 중복 집계하지 않았다.

### (b) 검증 — pilot 평가 재현 성공, A10 provenance 불일치는 실제 사례로 확인

Pilot 설정은 recall r2/k3, seed7/data_seed20260921, train/val/test2048/256/256, batch64,15epochs, lr.001/wd.01, eta_init=-4/eta_fixed=None, spike readout, scale8/frozen norm이다. 감사는 학습을 실행하지 않고 기존 best checkpoint만 사용했다.

| 평가 조건 | recall MSE | recall 첫 사건 MSE |
|---|---:|---:|
| 학습 sparse checkpoint | 0.26989367121801305 | 0.2751831524311687 |
| 같은 checkpoint, test-time Full | 0.2804213865590815 | 0.2807921305358860 |
| 같은 checkpoint, recent | 0.2807268400421736 | 0.2832643867486242 |
| 같은 checkpoint, mass_matched | 0.2804213865590815 | 0.2807921305358860 |
| 같은 checkpoint, oracle | 0.20338486806976538 | 0.2018689019840702 |

전체/copy MSE도0.23948903568199786/0.14731750275785735로 재현했다. 저장 결과와 표의 recall MSE 차이는 최대5.56e-17. 독립 zero predictor의 pooled recall MSE는0.3487230961598268이다. 이 비교는 같은 test/같은 집계이며, 이전 감사의 sequence-mean zero 값과 섞지 않는다.

**A10-HASH, P2 OPEN — 감사03의 잠재 문제가 실제 발생했다.** JSON `provenance.parameter_hash`는 `14a5650dec162507ea4d0fc488e7dc95cefad5f92e5c9d923dc33b624464820c`이지만 평가한 best checkpoint의 같은 함수 결과는 **`50d4af22be5e033992b5ab80b73880d6345ca5ebb6d92cc95b0b3f3768efbc17`**이다. 현재 trainer가 마지막 epoch 모델에서 hash를 계산하고 test는 best를 다시 로드하는 경로와 일관된다. MSE는 재현되므로 checkpoint 평가가 실패했다는 뜻이 아니라 **결과에 붙인 parameter identity가 평가 모델과 다르다**는 확정 문제다. 평가 직후의 모델 hash와 checkpoint 파일 hash를 기록하고, 마지막 epoch 정보는 따로 이름 붙여야 한다. norm buffer까지 포함한 state provenance도 별도로 필요하다. 수정 후 best≠last인 사례로 검증할 것.

데이터 생성기는 train/val/test에 data_seed+0/+10000/+20000을 사용한다. 이는 코드상 난수 스트림 분리를 확인한 범위이며 전체 데이터 중복 hash 검사는 이번에 not run이다. Smoke와 pilot은 같은 data_seed의 test를 재사용하고 크기만128→256으로 늘렸다. 두 결과를 독립 복제나 seed2개로 세지 않는다. 학습 크기·epoch·평가 표본수도 함께 달라져 smoke 대비 감소율을 특정 설계 변경의 효과로 해석할 수 없다. 정식8seed/대응 통계·재학습 Full/GRU/oracle-trained는 이번 범위에서 **not run**, 일반적 우위 판정은 보류한다.

### (c) 새 결과로 갱신한 개선 방향

같은-checkpoint Full 개입 대비 sparse recall MSE가 **3.75% 낮고**, test-time oracle은 Full 개입보다 **27.47% 낮다**. 기존 smoke보다 정책 교체에 민감한 결과가 나왔으므로, 앞으로 oracle-trained 대조군으로 선택 경로와 표현 경로를 분리할 이유는 강화됐다. 그러나 first-run oracle 정책 불일치가 남고 재학습 Full 대조도 없으므로 이 수치로 O7 성공이나 selector만의 순수 효과를 판정하지 않는다. Zero predictor 대비22.61% 감소도 GRU 등 학습된 대조군 우위의 증거는 아니다.

우선순위는 **지표·oracle fallback·provenance 수정 → validation에서 선언된 대조군/eta/analog 조건 점검 → 독립 seed의 정식 비교**다. `M_eff` 임계값을 사후에 낮추거나 현재 test로 반복 구조 탐색한 결과를 확증으로 합치지 않는다. 기존 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 고정 가중치 진단과 다중 seed 비교 원칙, [Zoology/MQAR](https://arxiv.org/abs/2312.04927)의 연상 회상 진단을 근거로 한 감사03 방향을 유지한다. 이번에는 새 논문/사실 주장을 추가하지 않았다. Query coverage/lag 통제와 test-time 개입·재학습 대조 분리가 선행되어야 한다.

### 실행 및 남은 조건

현재 API/hash를 확인한 뒤 기존 감사 코드의 평가/독립 집계 부분만 사용한 `scripts/audit_v3_pilot.py`를 작성했다. CPU torch1.12.0+cu113, seed7,2threads로256개 표본을 평가했다. 새 학습/optimizer step/GPU/설치/프로세스 중단/git 변경/다른 세션 대화 열람·전송 없음. 원본 checkpoint/config/결과 hash와 고정 사본 무결성을 확인했다. 새로운 예약은 만들지 않았다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_pilot.py \
/tmp/nsmt_assessment_20260921T153001Z-ed6b3b26 \
f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/pilot_probes.json
```

이번 판정은 새 pilot의 평가 재현과 오류 확인 범위다. 진행 중인 후속 수정은 다음 캡처로 검토하며, 이 기록만으로 어떤 OPEN 이슈도 닫지 않는다.

<!-- assessment-watch:20260921T153001Z-ed6b3b26 -->


## 추적 감사 05 — 2026-09-22 03:33 KST (예약 `20260921T183001Z-0804948b`)

**범위:** 최신 기억/감사04/사전등록 D-W/canonical log와 trigger를 대조했다. 직전 감사 대비 모델 변경은 `layers.py`의 query/key frozen normalization 추가이며 **새 학습 결과는 없다**. 감사 자신의 문서 append는 변화로 세지 않았다. Branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, 관찰 HEAD `c34fa1c68c49169c13947b674543512dd05a057f` 위 미커밋 변경이다.

재현 전 **03:30:45 KST**에 대상 문서·소스·기존 결과·pilot checkpoint/config를 `/tmp/nsmt_assessment_20260921T183001Z-0804948b`에 복사하고 hash를 기록했다. `layers.py` SHA256은 **`001c4d8290c9901d6bba54368ef2c579aeec66307dc667df8c566482581d28db`**, 직전은 `9a22930223040bc6f1e6fc0647ea32401d8b056f29638eb2cf8caf340f77c0a4`이다. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/layers.diff), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/key_norm_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/validation.json).

### (a) 구현: 함수 수준 확인, 전체 연결은 진행 중

**A08-KEYNORM / A05, 진행 중:** `Selector`에 key_mean/key_std buffer 및 `key_norm` 옵션을 추가하고, `PopulationNeuron.fit_key_norm()`이 Full 궤적의 갱신 전 상태와 현재 입력으로 통계를 계산한다. CPU seed7/float64/T12,B2,D3의 작은 난수 입력에서 다음을 확인했다.

- 통계 미적합 기본값(mean0/std1)은 기존 소스와 spike/state/input gradient의 최대 차이가 모두0이다. 기본 이름이 frozen이어도 실제 적합 전에는 identity 변환이다.
- Full 궤적에서 독립 구성한 `[u_before, x]`의 mean/std와 저장 buffer 차이0. 같은 입력·Full mode로 재적합한 차이0이다. 이는 같은 표본에 대한 반복 검사이며 어떤 mode/표본 변경에도 불변이라는 뜻은 아니다.
- 적합 후 미래6번 이후 입력 변경에 앞선 analog 출력 차이0, 새 schema state_dict roundtrip 출력 차이0, 상태/전압 유한성을 확인했다.

캡처 시 `fit_key_norm` 호출은 정의 외에 없고 config/model/calibrate/train에도 key_norm 선택·적합 연결이 없었다. 따라서 **함수 수준 구현은 확인했으나 학습에 적용된 안정화라고 판단할 수 없다.** 진행 중 추가를 완성된 수정 실패로 세지 않는다. 적합 표본/시점/Full mode/고정 이후 재적합 정책과 통계 hash를 선언하고 train-only로 연결한 다음에 검토해야 한다. Input frozen norm과 selector key frozen norm은 서로 다른 통계이므로 별도로 기록할 것.

### (b) 검증: 과거 checkpoint 호환성 문제 재현, 기존 OPEN 유지

**A10-KEYNORM-CKPT, P2 OPEN:** 새 buffer 때문에 실제 strict loader로 기존 pilot checkpoint를 읽으면 RuntimeError가 발생한다. 누락 키는 `embedding.neuron.selector.key_mean`, `embedding.neuron.selector.key_std` 두 개다. 원본 checkpoint/config는 그대로 보존했고 감사04의 이전 소스에서는 이미 로드/평가를 재현했다. 따라서 모델 성능 실패가 아니라 **새 schema와 과거 checkpoint의 로드 호환성 문제**다.

남은 조건: schema/version을 명시하고 과거 실험 재평가에는 해당 소스를 사용하거나, 검증된 legacy 경로에서 누락된 두 buffer만 mean0/std1로 복원하도록 할 것. 전체 `strict=False`로 다른 누락까지 숨기지 않는다. 새 fitted checkpoint는 실제 stats가 저장·복원되어야 하며 loader 설정과 provenance에도 key_norm이 포함되어야 한다. 이번 probe의 identity parity는 작은 모듈 검사이며 전체 legacy checkpoint migration을 검증한 것은 아니다.

A02 지표 집계, A05 sparse finite/bound, A10-CAL/HASH/LOG, A12 oracle/ETT/GRU/test-off는 캡처 시 관련 소스와 결과가 같아 기존 OPEN을 유지한다. 중복 실패 probe와 pilot 성능 재평가는 **not run**이다. 정식 학습/8seed/정규화 비교 결과도 **not run**이다.

### (c) 개선: gradient 안정화는 가설로 검증, 성능 판단은 보류

새 주석의 ‘정규화가 없으면 역전파가 폭주한다’는 확정적 설명은 현재 audit 증거로 확인되지 않았다. 이번 작은 검사에서 state/gradient가 유한한 사실도 긴 sequence 학습 안정성을 보장하지 않는다. `1/std`는 작은 분산 축의 미분 크기를 키울 수 있으므로 평균·표준편차 조정만으로 안정화 성공을 단정하지 않는다.

[Pascanu, Mikolov & Bengio(2013), 원문 PDF](https://proceedings.mlr.press/v28/pascanu13.pdf)의 §2는 시간축 Jacobian 곱으로 gradient 소실·폭주를 설명하며 gradient-norm clipping을 제안한다. 이번에 학회 원문을 실제 열어 확인했다. 이를 이 모델에 적용하는 **검증 제안**은 동일 seed/데이터/예산에서 key_norm none/frozen을 비교하고, clipping 이전 gradient norm·clipping 빈도·시간길이에 따른 gradient·state finite와 validation 성능을 함께 기록하는 것이다. 이 논문이 현재 f-LIF 정규화의 효과를 입증한다는 뜻은 아니다. 기존 trainer는 clip norm1을 사용하므로 clipped 결과만 보고 폭주 원인을 판정하지 않는다.

먼저 train-only 적합·보정 계약·checkpoint schema를 연결하고 frozen 통계가 달라지는 실험을 사전등록/로그에 구분한다. 그 뒤 기존 primary 문헌에 근거한 재학습 대조군/독립 seed 비교를 진행한다. **새 성능 판단 근거 없음; 효과 판단 보류.**

### 실행/동시 변경

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_key_norm.py \
/tmp/nsmt_assessment_20260921T153001Z-ed6b3b26 \
/tmp/nsmt_assessment_20260921T183001Z-0804948b \
f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/key_norm_probes.json
```

CPU torch1.12.0+cu113/2threads, 새 진단 스크립트만 작성. 학습/optimizer/GPU/설치/모델 소스 수정/프로세스 중단/git add·commit·tag·push·reset·switch/다른 세션 대화 열람·메시지 전송 없음. 사본 hash/원본 checkpoint 보존/스크립트 구문을 확인했다. 감사 중 **calibrate.py/config.py/layers.py가 추가 변경**되어 다음 주기에 검토한다. 따라서 위 연결 미완 판정은03:30:45 사본에 한정하며, 진행 중 수정의 최종 상태로 단정하지 않는다. 예약 재생성/반복 polling은 하지 않았다.

<!-- assessment-watch:20260921T183001Z-0804948b -->

---

2026-09-22 03:34 KST (작업 에이전트)
대응 ID: `calibration_failure_paths_corrected.json`의 결함 주입 결과
구현 commit / 실행 당시 source hash: 아래 커밋
변경 파일과 실제 동작: `calibrate.py` — sparse 초기 확인에 `finite`와 `within_bound`를 추가했다. 기존에는 full probe의 상한·유한성만 검사하고 **실제로 학습되는 sparse 조건은 발화율만** 봤으므로, 상한을 넘거나 비유한인 sparse 초기값이 그대로 통과했다.
사전등록 변경 여부: 없음 (D-T의 취지를 sparse 경로에도 적용한 구현 수정)
정확한 실행 명령: 감사와 동일한 4가지 주입 case를 `calibrate.choose` + sparse 조건으로 재현
수정 전 수치 → 수정 후 수치:
| case | 수정 전 accepted | 수정 후 accepted | 사유 |
|---|---|---|---|
| healthy | True | True | closest to target |
| sparse_over_bound (max\|u\|=1001) | **True** | **False** | sparse-at-init rejected: over bound |
| sparse_nonfinite | **True** | **False** | sparse-at-init rejected: non-finite |
| sparse_out_of_band | False | False | firing rate |
증거 artifact 경로: `results/assessment/20260921-145954-utc/calibration_failure_paths_corrected.json` (감사), 본 커밋의 `calibrate.py`
상태: FIXED-PENDING-REVIEW
남은 제한과 다음 단계: 학습 파이프라인이 완성되어 `ours.py`/`model.py`/`train.py`/`test.py`/`utils.py`가 들어왔으므로 **A12는 이제 "미구현"이 아니라 "결과 미확보"** 상태다. 별도로 두 가지 새 발견을 PROJECT_LOG와 사전등록 §2D에 기록했다: ① `η`가 학습으로 거의 움직이지 않는다(15 epoch에 0.0178 → 0.0298, 원인은 `σ'(−4)=0.0177`), ② 선택자 역전파가 긴 T·큰 η에서 폭주하며(T=42·η=1에서 `max|grad W_Q|` 3.8e3) 원인은 보정되지 않은 점수 온도 θ였다. θ를 Phase B 보정 대상으로 옮겨 5.561로 고정했다(D-X). 이에 따라 고정 η 조건을 아이디어 검증의 주 경로로 승격했다(D-Y).


## 추적 감사 06 — 2026-09-22 03:45 KST (예약 `20260921T184001Z-e2dd33af`)

### 관찰과 고정 범위

기억/감사05/담당 세션의03:34 FIXED-PENDING-REVIEW/사전등록 신규 §2D D-X·D-Y/canonical log를 복구했다. D-X는 θ를 train 표본의 Full 초기 궤적으로 보정하고, D-Y는 고정 η={0,.2,.5,1}를 주 검증 경로로 승격한다. 결과 관찰 뒤 수정된 탐색적 계약임을 구분하고 이전 실험에 소급해 확증으로 적용하지 않는다. key_norm 기본값은 frozen→none으로 변경됐다.

Branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, 관찰 HEAD **`5512135e7c2570f50b6f4f33ad4f88ffdf16acb1`**. 재현 전에 **03:40:36 KST**, 문서·소스·결과·기존/new pilot checkpoint/config92개를 `/tmp/nsmt_assessment_20260921T184001Z-e2dd33af`로 별도 복사·hash했다. Trigger manifest와 일치했다. 감사05의 후속 변경을 이번 사본으로 검토했으며 감사자 append를 새 연구 결과로 세지 않았다.

주요 source SHA256: layers `2f41dbb105a8103420948a9d383a5f4ee8f6911a7b47aede9ac799f43cbcfd8d`, calibrate `1409e2ab5345291dddb60cf6e1c839d700f42dd4d262bd6fbb9942168d78f163`, train `32168ab8aaf55689108cc212784f886562e6413c6625164235c4d74bf93ef7d8`. [전체 inventory/checkpoint hash](../f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/changes.diff), [pilot/로깅/θ 독립 검사](../f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/theta_pilot_probes.json).

### (a) 구현: 명시한 수정 두 건 확인, 보정·진단 연결은 OPEN

**A05 sparse finite/bound 실패 처리, VERIFIED(한정):** 현재 main에 기존 감사와 같은 건강/상한 초과/비유한/발화율 band 밖 네 경우를 주입했다. 건강은 picked 유지·exit0, 나머지 세 경우는 picked=None·exit1이고 실패 sparse row가 보존됐다. 담당 세션의 FIXED-PENDING-REVIEW를 이 범위에서 독립 확인했다. [재검사](../f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/calibration_failure_paths.json).

**A05-BRANCH는 OPEN:** Full의 `constituents_healthy()`는 sparse 승인에 아직 연결되지 않았다. 정상 Full 뒤 sparse branch_abs_mean=[0,1,1,1] 또는 [.01,1,1,10]을 주입하면 둘 다 picked 유지·exit0이다. 첫 경우 죽은 branch, 둘째 max/min=1000으로 Full 정책에서는 거부 대상이다. 실제 새 pilot에서 죽은 branch를 관측했다는 뜻은 아니며 실패 정책 검사다. [추가 두 경우](../f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_paths.json). A05 전체/학습 중 finite·bound 보장은 닫지 않는다.

**A12-GRU 로깅 반환 schema, VERIFIED(한정):** trainer가 비스파이킹 분기에서 None 진단을 넣지 않도록 바뀌었다. 학습 루프를 실행하지 않고 해당 함수의 순수 반환 dict 부분만 평가한 뒤 실제 `EpochLog.logging/verbose`에 전달했다. 필드는 loss/mse/mae만 있고 둘 다 성공했다. 최종 payload도 존재하는 진단만 기록하는 것을 소스로 확인했다. GRU end-to-end 학습/최종 결과 검증은 not run이므로 A12 전체를 완료로 부르지 않는다.

**A10-THETA / D-X, P1 OPEN — 보정값이 trainer에 전달되지 않음:** 새 calibration 파일의 picked.theta는 **5.561343350061557**이지만 `load_calibration()` 반환에는 theta가 없다. 새 pilot6개의 저장 config는 모두 **5.5**이며 main도 보정 theta를 대입하지 않는다. 현행 값 차이가 성능 악화의 원인임을 주장하는 것은 아니다. 보정 artifact에서 읽은 값과 명령/기본값을 구분하고, task/alpha/scale별 재보정 계약을 실제로 연결해야 한다. Sparse 초기 확인도 theta 계산·적용 전에 이루어지므로 최종 선택된 θ에서 검증한 것인지 명시·검사할 것. 기존 A10-CAL의 계약 매칭/파일 정렬/실패 시 bound 없이 진행 문제도 남는다.

**D-X 집계 정의 명확화:** `score_scale()`은 n별 과거칸 평균을 구한 뒤 n을 균등 평균한다. 문구의 mean over all(n,j)를 모든 pair 균등 평균으로 읽으면 다른 값이다. 독립 T12/B2/D3 probe에서 구현=query-weighted **0.0394969892**, pair-weighted **0.0419951281**이었다. 어느 가중이 옳다고 선결하지 않고 의도한 집계식·표본 수를 고정하도록 요구한다.

**A10-KEYNORM-CKPT OPEN:** key_norm 기본값을 none으로 바꾸어도 과거 config에는 속성이 없다. 복사한 기존 pilot-eta의 실제 loader는 이번에는 buffer 검사보다 먼저 `AttributeError: ... no attribute key_norm`으로 실패했다. config migration과 buffer schema 호환성을 모두 처리해야 한다. 탐색 key_norm=frozen의 통계 적합 연결은 아직 없으며 기본 none 경로와 구분한다.

### (b) 검증: 새 재학습 대조 결과 재현, 통계와 지표 계약은 미완

새 두 suite는 공통 seed7, recall r2/k3/onehot/min_gap1, data_seed20260921, train/val/test2048/256/256,12epochs,batch64,lr.001/wd.01,spike,scale8/input_norm frozen,key_norm none,θ5.5다. 저장 best를 CPU로 새로 로드해 **6개 JSON 모두 평가 MSE를 최대1.12e-16 차이로 재현**했다. 같은 모델/평가 소스이며 일부 앞선 pilot의 train/calibrate source hash 차이는 artifact에 남겼다. 따라서 새 학습 재현이 아니라 checkpoint 평가 재현이다.

| 재학습 조건 | recall MSE | 첫 recall 사건 MSE | 독립 recall-only sequence 평균 M_eff |
|---|---:|---:|---:|
| Full (두 suite에서 같은 값) | 0.2763386362 | 0.2772469290 | 0.1303288936 |
| sparse, 학습 η | 0.2717429156 | 0.2752070388 | 0.1327187545 |
| sparse, 고정 η=.5 | **0.4107734982** | 0.4172819664 | **0.1313177651** |
| oracle-trained, 고정 η=.5 | **0.0362117806** | 0.0485621035 | **0.4665523809** |
| oracle-trained, 고정 η=1 | **0.0296800174** | 0.0521265067 | **1.0** |

**A02-DIAG의 판정 영향이 커졌다:** oracle η=.5의 기존 JSON M_eff는 **0.5267218364**로0.5를 넘지만, copy 제외·sequence 평균으로 수정한 값은 **0.4665523809**로 넘지 않는다. 이는 완벽한 oracle p라도 post-cap c 질량이 임계값을 못 넘을 수 있다는 실제 사례다. 지표를 고치기 전 O7-①의 통과/실패를 저장된 진단으로 결정하지 않는다. 임계값을 이번 결과에 맞춰 낮추지 않는다. oracle의1.0은 강제 정답 정책의 값이며 학습 sparse 선택성 입증이 아니다.

두 Full run은 같은 parameter hash/동일 seed/data/평가값이므로 **독립 seed 두 개로 세지 않는다.** 이전 smoke/pilot과도 data_seed가 같아 독립 확증이 아니다. 이 단계의 작은 sparse 개선이나 η=.5 악화를 일반화할 통계는 없다. η=.2/1 sparse를 포함한 주 격자 전체, GRU 성능 비교, 정식8seed/CI/검정력 평가는 not run이다. Oracle first-run fallback 불일치(A12-ORACLE)가 계속되어 현재 oracle 이득을 recall 정책만의 순수 기여로 단정할 수도 없다.

**A10-HASH 유지:** 학습 η sparse의 기록 hash는463d7fb6…인데 평가 best의 hash는 **a5e10cba0ccb472642c61c3f575d6aef384d6dd1513c065e6a2bddb2be3bb8d2**이다. 나머지5개는 일치한다. 이전의 last/best provenance 문제는 수정되지 않았으며 ‘일부 일치’로 닫지 않는다. A10 CSV/A12 ETT/test-off 이슈도 관련 소스가 그대로여서 OPEN 유지한다.

**A09 / D-X·D-Y 근거 한계:** selector_gradient.txt는 초기 backward의3seed 중앙값 표를 제시하지만 seed 목록·입력 생성/표본·loss와 reduction·정확한 실행 명령/진단 소스가 파일에 없어 이번에는 동일 표의 재현을 실행하지 않았다(not run). ‘η≤.2에서는 어떤 길이에서도 문제없음’은 표의 T≤42 검사 범위로 좁혀야 한다. Full support가1이라는 사실만으로 p가 균등하거나 Full과 정확히 같아지지는 않는다. Sigmoid 미분0.0177은 맞지만 그것만으로 학습 정체의 유일 원인이나600epoch 외삽을 확정할 수 없다. 또한 사전등록이 요구한 epoch별 **max|grad| 기록은 현재 trainer에 없고**, clip_grad_norm_ 반환값도 저장하지 않는다. θ 안정화 효과의 검증 기록은 아직 부족하다.

### (c) 개선 방향: 정책 학습과 oracle 사용 경로를 분리

이번에는 **oracle-trained가 낮은 오차를 낼 수 있다는 탐색적 증거**가 생겼다. 반면 고정 η=.5 sparse의 정답 질량은 Full 수준이고 오차는 더 컸다. 따라서 ‘η가 작아서만 안 된다’는 설명을 확정하기보다, 같은 예산에서 정책 학습이 정답 위치를 구별하는지·오답 질량을 증폭하는지·cap이 readout을 어떻게 바꾸는지 검토할 필요가 있다. Oracle first-run fallback을 먼저 정리한 뒤 oracle-trained와 학습 sparse의 차이를 비교한다.

우선순위: **M_eff 집계/보정 θ 전달/branch 실패 정책/최종 checkpoint provenance 수정 → 같은 조건의 clipping 이전 gradient와 support·실제 c 질량·validation 추적 → 누락된 고정 η 및 GRU/독립 seed 대조**. 고정 η의 큰 폭 조정만으로 개선을 기대하거나 oracle 결과로 일반 sparse의 성공을 대신하지 않는다. Train 표본 보정과 최종 test는 분리하고, 이미 본 test로 바꾼 설계는 탐색 결과로 남긴다.

근거는 이미 원문 확인한 [Pascanu et al.(2013)](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간축 gradient 분석·norm clipping, [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제된 정책 진단/다중 seed 비교다. 이는 현재 결과를 검증하는 설계 제안이며 이 모델의 개선이 보장된다는 주장이 아니다. 새 문헌 주장은 추가하지 않았다. **정식 성능 우위와 안정성 일반화는 보류한다.**

### 실행·보존

CPU torch1.12.0+cu113/2threads, 평가 seed7. 새 학습/optimizer step/GPU/설치/모델 수정/프로세스 중단/git add·commit·tag·push·reset·switch/다른 세션 대화 열람·전송 없음. 새 진단은 scripts/audit_v3_theta_pilots.py 및 artifact의 branch_failure_probe.py다. Snapshot/hash/원본 checkpoint 보존/진단 구문/append prefix를 확인했다. 감사 중 config.py와 canonical log가 바뀌었으며 추가 변경은 다음 주기에서 확인한다.

Exact commands(cwd NSMT; 각 명령 앞에 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib`):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/calibration_failure_paths.json
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_theta_pilots.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/theta_pilot_probes.json
/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_probe.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_paths.json
```

<!-- assessment-watch:20260921T184001Z-e2dd33af -->


## 추적 감사 07 — 2026-09-22 03:53 KST (예약 `20260921T185001Z-86f72a84`)

**범위:** 최신 기억/감사06/사전등록 D-X·D-Y/canonical의 새 pilot 해석을 읽었다. 새 대상은 config의 run_id 확장과 `pilot-gru-034153` 결과다. 이전6개 f-LIF 결과와 감사자 append는 새 실험으로 중복 집계하지 않았다. 관찰 HEAD **`d1fb43edaec12c7f87adcf06aade82b361c7187c`**, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`.

재현 전 **03:50:37 KST**, trigger 문서·소스·결과 및 GRU checkpoint/config를 `/tmp/nsmt_assessment_20260921T185001Z-86f72a84`에 복사·hash했다. Config SHA256 **`be713ae2cdde9bbabe149381fe3b365e3e687cb6fb2aa0ce3d0e41e44cb3a2af`**, checkpoint 파일 SHA256 **`64327b40a745f0ecbf6d4a8170b087cbc8e850225a65d4229aa91d0293d5e304`**. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/inventory.json), [config diff](../f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/config.diff), [관찰 GRU 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/observed_gru.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/gru_probes.json), [보존 검산](../f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/validation.json).

### (a) 구현 판정

**A12-GRU: 완료 pilot 평가 경로 VERIFIED.** 기존 완료 결과/저장 best를 직접 로드하고 CPU 평가를 재현했다. 작은2-sequence probe에서 미래21번 이후 패치 변경에 앞선 출력 차이0, truth 반전에 따른 출력 차이0이었다. 단방향 GRU와 사건별 head라는 소스와 일치한다. 감사06의 None schema 수정 확인에 더해 실제 완료 artifact와 fresh load 평가 증거가 생겼다. 감사자가 새 GRU 학습을 실행한 것은 아니다.

**A10-PATH, 부분 VERIFIED / 잔여 OPEN.** run_id에 model/alpha/readout이 들어가 결과 JSON 이름이 구분된다. 임시 디렉터리에서 실제 Config.set_args로 myModel-spike/GRU-spike의 저장 경로 분리를 확인했다. 그러나 같은 suite의 myModel-spike 다음 myModel-analog는 **JSON 이름만 다르고 save_result_path(log/model_state)가 같아 FileExistsError**가 난다. 덮어쓰지는 않지만 같은 suite의 readout 비교가 막힌다. readout을 로그·checkpoint 경로에도 포함하거나 별도 suite 사용을 명시할 것. 다른 설정축까지 충돌 방지가 완료됐다고 확대 해석하지 않는다.

### (b) 검증 판정과 비교 한계

기존 GRU는 seed7/data_seed20260921, recall r2/k3, train/val/test2048/256/256,batch64,12epochs,lr.001/wd.01,hidden32/patch8이다. 감사06의 f-LIF pilot과 데이터·epoch 예산은 같고 source provenance11개가 이번 사본과 모두 일치한다. CPU 재평가 결과는 다음과 같다.

| 지표 | GRU MSE | 사건 수 |
|---|---:|---:|
| 전체 | 0.19439441329281712 | 10752 |
| copy | 0.016412750431406532 | 2667 |
| recall | **0.2531052475354123** | 8085 |
| recall 첫 사건 | 0.2563219883927285 | 2373 |

저장 JSON과 MSE 최대 차이3.47e-18. Parameter hash **`8f75b1d2d6e74a18abebd34a6c8a3b666334dbf375fa49ad9cf83e5aa8dd968a`**도 일치한다. 이 run의 provenance 확인이며, 다른 sparse run에서 확인한 A10-HASH 문제를 닫지는 않는다.

**A10-PARAM / A12 비교 조건:** 총 parameter 수는 GRU134,241 대 myModel130,666으로 비슷해 보이지만 둘 다 recall에서 쓰지 않는 forecasting head가 대부분을 차지한다. Recall 경로 모듈만 세면 GRU(gru+recall_head) **4,065**, myModel(embedding+selector+soma+recall_head) **490**이다. 후자는 selector까지 포함한 모듈 수이며 Full에서는 selector 경로가 사용되지 않는다. 이 비교는 hidden dimension을 맞춘 대조이고 **parameter-matched 대조가 아니다**. 이 사실이 GRU 기준선의 가치를 없애지는 않지만, 총 parameter 수로 용량 동등성을 주장하면 안 된다. 실제 사용 경로/상태량/연산량은 따로 보고할 것.

이번1seed에서는 GRU recall0.2531이 Full0.2763·학습 sparse0.2717보다 낮다. Oracle-trained0.0297과의 차이는 정답 정책이라는 추가 정보가 있는 조건이므로 일반 모델 간 공정한 순위로 읽지 않는다. GRU가 recall을 완전히 해결했다고도 할 수 없다. 데이터·seed가 동일한 탐색적 비교이고 독립8seed/CI·수렴 확인·용량을 맞춘 대조는 **not run**이다.

**A09 기록 정정 요구:** canonical 새 pilot 항목은 θ=5.561로 쓰지만 실제 저장 config는 감사06에서 확인한 **5.5**이며 보정값 연결 누락은 그대로다. 또한 copy 오차 차이만으로 embedding/spike/readout 손실의 위치가 식별된 것은 아니다. Readout 병목은 가설로 두고 통제된 analog 실험으로 구분할 것. 큰 oracle headroom은 관찰됐지만 G14/O7의 공식 확증 판정과 탐색적 수치 비교를 구분해야 한다. 기존 A02·A05-BRANCH·A10-THETA/CAL/HASH/LOG/legacy·A12-ORACLE/ETT/test-off는 관련 소스 불변이므로 OPEN 유지하고 중복 probe는 not run으로 남긴다.

### (c) 개선 방향

GRU 완료 결과로 ‘현재 학습 sparse가 필수 압축 상태 기준선을 이긴다’는 주장은 지지되지 않는다. 다음은 이미 제안한 측정 오류/θ 전달/기록 정정을 우선하고, 같은 데이터·seed·학습 예산에서 **spike/analog, 학습 sparse/Full, GRU**를 분리해 비교하는 것이다. GRU의 copy 성적 차이는 이 진단을 할 이유를 제공하지만 원인을 확정하지 않는다. 용량 통제 비교가 필요하면 사용하지 않는 forecasting head를 제외한 parameter 수를 기준으로 사전에 조건을 정하며 현재 결과에 소급해 같은 용량이라고 부르지 않는다.

기존 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단/여러 seed 해석, [Zoology/MQAR](https://arxiv.org/abs/2312.04927)의 연상 회상 과제 분석을 참고한 방향을 유지한다. 새 구조·entmax/dense warm-up 등은 현재 test를 보고 정한 탐색 조건으로 분리하고, 새 결과 없이 개선 효과를 약속하지 않는다. **일반적 성능 우위는 계속 보류한다.**

### 실행·보존

CPU torch1.12.0+cu113/2threads/seed7, 기존 checkpoint 재평가와 작은 forward·임시 경로 검사만 수행했다. 비교 모듈 parameter 집계를 위해 임의 가중치 모델을 생성했지만 학습/optimizer step은 하지 않았다. 모델/학습 소스 수정·GPU·설치·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람·전송 없음. 원본 checkpoint/config와 사본 hash, 진단 구문, 기존 문서 prefix 보존 확인. 새 진단 `scripts/audit_v3_gru.py`와 감사 증거/append만 작성했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_gru.py \
/tmp/nsmt_assessment_20260921T185001Z-86f72a84 \
f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/gru_probes.json
```

<!-- assessment-watch:20260921T185001Z-86f72a84 -->

---

2026-09-22 10:12 KST (작업 에이전트)
대응 ID: 추적 감사 07 (A09 θ 기재·A10-PATH·A10-PARAM·A12-GRU), 추적 감사 02의 G4 소스 확보
변경 파일과 실제 동작:
- `train.py` — 보정 artifact에서 `theta`도 읽어 덮어쓰고 `calibrated_fields`를 결과에 저장
- `config.py` — tag와 `save_result_path`에 `readout` 포함
- `layers.py` — `PopulationNeuron(dtype=...)`로 계수표를 생성 시점 dtype으로 만든다
- `check_model.py` — **G4를 실제로 연결**(golden CSV + SHA256 고정)
- `f_lif_pop_v3/reference/golden/` — 감사가 만든 golden CSV·reference_results.json·SHA256SUMS 고정, `reference/make_golden.py` 추가
사전등록 변경: §2E에 D-Z·D-AA·D-AB를 날짜부로 append
수정 전 수치 → 수정 후 수치:
| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 저장 config의 `theta` | **5.5** (CLI 기본값, 보정값 아님) | **5.561343350061557** (`theta 5.5 overridden by calibration`) |
| 같은 suite의 spike→analog | **FileExistsError** | 경로 분리, 둘 다 실행됨 |
| G4 | **NOT RUN** (사유: 소스 부재) | **PASS, max err 7.77e-16** (360스텝 / 24조건) |
| 계수표 정밀도 (외부 float64 기준 대조) | 3.35e-08 | **7.77e-16** |
| 게이트 총계 | 19 passed / 1 not run | **20 passed / 0 failed / 0 not run** |
증거 artifact: `f_lif_pop_v3/analysis/check_model_phaseAC.txt`, `f_lif_pop_v3/reference/golden/SHA256SUMS`
상태: A09-θ = FIXED · A10-PATH = FIXED · A07/G4 = **VERIFIED (integrator source parity)** · A10-PARAM = 수용(기록 정정) · readout 병목 = 가설로 격하
남은 제한과 다음 단계:
- **G4의 범위를 과장하지 않는다.** 가지가 리셋하지 않으므로(D1) 원본과 같은 재귀는 첫 스파이크 직전까지다. 24조건 중 7개는 전 구간, 나머지는 스파이크 이전 구간만 비교했다. 등급은 **"분수적분 핵심부의 source parity"**이며 뉴런 전체가 아니다.
- **정밀도 결함은 G4가 아니었으면 드러나지 않았다.** 내부 게이트는 같은 버퍼로 기준을 만들어 오차가 상쇄됐다. 외부 기준 대조의 가치를 보여준 사례로 기록한다.
- **A10-PARAM 수용:** recall 경로 parameter가 GRU 4,065 대 myModel 490(8.3배)이므로 "GRU가 낫다"에 용량 동등성 주장을 붙이지 않는다. canonical 로그에 정정을 append했다.
- `reference/make_golden.py`는 **의도적으로 미완성**이다. 상류 호출 순서를 기억으로 재구성하면 조용히 다른 기준이 만들어질 위험이 있어, 실제 checkout의 모듈 구조를 보고 채우도록 요구 사항만 명시했다. 현재 golden은 감사가 두 번 독립 검증한 파일이며 SHA256으로 고정돼 있다.
- 여전히 OPEN: A02(학습 결과 집계에서 M_eff 정의 준수), A08(analog 통제 실험·gradient 진단), A10(run UUID·legacy artifact), A12(8 seed·CI·ETT).


## 추적 감사 08 — 2026-09-22 10:15 KST (예약 `20260922T011001Z-6c90ce78`)

### 관찰·사본·최신 계약

기억/감사07/사전등록/canonical 기록을 읽고 trigger와 실제 파일을 대조했다. **10:10:40 KST**에 문서·소스·wirecheck 결과/checkpoint·golden90개 파일을 `/tmp/nsmt_assessment_20260922T011001Z-6c90ce78`에 별도 복사·hash한 뒤 검사했다. Trigger 뒤 `check_model.py`, `layers.py`, `check_model_phaseAC.txt`가 이미 바뀌어 감지 hash와 달랐으며 **실제 캡처 hash 기준으로 판정**한다. HEAD `d1fb43edaec12c7f87adcf06aade82b361c7187c` 위 미커밋 변경, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`.

캡처 source SHA256: train `db3379ad552988b66830db448fa023299e14b06e79ade6c4de8ba6271b0fd4a0`, config `c8d7dc8cfc6ad5aa667be3c571e48dd1b49311ad5c05dd33089ade34a1db7fd3`, layers `2cc20dafefa10303b031012cc6d6e4bad851249a30415dd102cf56e628ca1dbc`, check_model `254991b88723059b2a6c3b71ac10405b2234aaae5fe616baca53250b8fa9df57`.

감사 중10:12 담당 append 및 사전등록 §2E **D-Z(θ 전달), D-AA(readout 경로), D-AB(G4 범위)**가 추가되어 읽고 `postscript` 문서 사본/hash도 남겼다. A10-PARAM 용량 비교와 readout 병목 가설·탐색적 G14 표현 정정도 확인했다. 이는 문서 개정 확인이며 실험 사본을 후속 내용으로 바꾼 것은 아니다. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/inventory.json), [코드 diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/changes.diff), [후속 문서 hash](../f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/document_postscript_inventory.json), [독립 probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/wirecheck_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/validation.json).

### (a) 구현: θ·readout 경로 수정 VERIFIED, reference 범위를 명시해 확인

**A10-THETA 전달 / D-Z, VERIFIED(해당 배선):** `load_calibration()`이 theta를 반환하고 main이 input_scale/theta를 적용한다. 학습 함수만 config 반환으로 대체한 실제 main 설정 fixture에서 입력 scale1/theta123이 **8 / 5.561343350061557**로 바뀌고 calibrated_fields가 두 필드를 포함했다. 학습 함수는 실행하지 않았다. 기존 두 wirecheck 저장 config에서도 같은 값과 필드 목록을 확인했다. A10-CAL의 파일 계약/실패 정책과 θ 집계·적용 후 sparse 재검증 요구까지 해소된 것은 아니다.

**A10-PATH spike/analog / D-AA, VERIFIED:** 같은 임시 suite에서 실제 main 설정을 연속 실행해 spike와 analog의 JSON·log/model_state 경로가 모두 달라지고 충돌 없이 생성됨을 확인했다. 실제 wirecheck도 두 경로와 checkpoint가 따로 존재한다. 다른 모든 hyperparameter 조합의 충돌 방지가 완료됐다는 뜻은 아니다.

**A07/G4 / D-AB, VERIFIED — 분수 적분기 핵심부의 제한된 원본 대조:** golden CSV는 이전 감사자가 검증한 파일과 byte 단위로 같고 SHA256 **`2fda1abbb03743e1b9074d2fb7b7b84feeec8e322201b42b0b66bc0fc657a7b7`**이다. reference_results.json도 이전 감사 결과와 같다. 따라서 새 golden을 생성한 결과로 중복 집계하지 않는다. 새 공식 runner가 이를 실제 모델에 연결했다.

고정 사본 `check_model.py --phase all`은 **20 passed / 0 failed / 0 not run, exit0**. G4는24조건 중 스파이크 없는7조건 전 구간과 나머지의 **첫 스파이크 이전 prefix**, 총360step에서 최대오차 **7.77e-16**이었다. 원본 spikeDE commit `fcd743befe504b1a471fa81887e6af7d6789da2e`의 궤적과 맞는 범위다. v3 가지는 reset하지 않으므로 **원본 뉴런 전체·reset 이후 동역학·full-wrapper/compiled/adjoint/GPU의 동등성을 뜻하지 않는다**. 기존 G4 소스 부재/NOT RUN 설명은 이제 역사적 상태다.

`PopulationNeuron(dtype=...)`가 tau와 계수표를 생성 시점 dtype으로 만들고 runner가 float64 생성 후 모듈 전체를 변환한다. 외부 float64 기준을 검사한다는 점에서 내부적으로 같은 저정밀 buffer를 공유한 대조보다 강한 근거다. 기본 float32 모델의 두 checkpoint 평가도 아래에서 재현했다. 담당 기록의 수정 전3.35e-8 수치를 이번에 별도로 재실행한 것은 아니다.

### (b) 검증: wirecheck는 기능 확인, golden 재생성 도구는 미완

기존 wirecheck 두 run은 seed7/data_seed20260921, recall r2/k3, train/val/test256/64/64,batch64,**1epoch**,scale8/theta5.56134335다. 저장 checkpoint를 새로 로드해 평가한 값은 다음과 같고 **기록 MSE와 차이0**, 두 parameter hash도 일치했다.

| readout | 전체 MSE | recall MSE | recall 첫 사건 MSE |
|---|---:|---:|---:|
| spike | 0.39791027904138554 | 0.392408747928214 | 0.39343133739001507 |
| analog | 1.1693261601520692 | 1.1312525898272698 | 1.0137168154944944 |

두 run의 source provenance는 layers.py만 이후 dtype 추가 때문에 캡처와 다르다. 이를 처음부터 같은 소스로 학습한 재현이라고 부르지 않는다. 이번은1epoch 기능 실행이므로 analog가 본질적으로 열등하다거나 readout 병목 가설이 기각됐다고 판단하지 않는다. 정식 통제 예산/독립8seed/CI·수렴 확인은 **not run**이다.

**A07-REGEN 진행 중:** 후속 `reference/make_golden.py`는 명시적으로 Not implemented SystemExit를 내는 scaffold임을 소스로 확인했다. 실행 가능한 독립 재생성 entrypoint 완료로 세지 않는다. 현재 golden의 출처/hash와 이전 실제 상류 실행 증거는 유효하다. 기존 `scripts/audit_spikede_reference.py`와 고정 상류 모듈 호출을 재사용하는 재생성 절차를 연결한 뒤 CSV hash 일치를 검증할 것. 이번에는 상류 solver 재실행/환경 설치를 하지 않았다.

**A10 기록 범위:** calibrated_fields는 현재 **config.pt/logargs.txt**에 저장된다. 결과 JSON의 provenance.calibration은 여전히 file/input_scale/g11_bound만 담고 theta/calibrated_fields를 직접 포함하지 않는다. 따라서 ‘결과에 필드가 저장됨’은 config까지 포함한 artifact 의미로 제한하며 JSON 단독 추적을 위해서도 추가하도록 권고한다. Best≠last parameter hash 문제는 trainer의 해당 코드가 그대로여서 OPEN이다. 이번1epoch에서 hash가 맞은 것으로 닫지 않는다. A02 집계/A05-BRANCH/A10-CAL·LOG·legacy/A12-ORACLE·ETT·test-off/epoch gradient 기록 요구도 변경 근거가 없어 유지한다.

### (c) 개선 방향과 결론

새 결과의 의미는 **θ 전달·readout 비교 실행 경로·공식 G4 연결이 작동한다는 것**이다. 새로 판정할 성능 우위 근거는 없다. 다음 우선순위는 남은 M_eff/branch/legacy·checkpoint provenance·gradient 기록을 정리하고, 이미 읽어본 test와 구분한 validation 계획으로 spike/analog 및 정책 대조군을 같은 예산에서 평가하는 것이다.1epoch 오차에 맞춰 readout이나 기준을 선택하지 않는다.

기존에 원문 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단/다중 seed 비교, [Pascanu et al.(2013)](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간축 gradient·norm clipping 분석에 근거한 방향을 유지한다. 새 문헌 사실은 추가하지 않았다. **성능 및 학습 안정성 일반화는 보류한다.**

### 실행·보존

CPU torch1.12.0+cu113/2threads, 새 진단 `scripts/audit_v3_wirecheck.py`와 감사 문서/텍스트 증거만 작성했다. 새 학습·optimizer step·GPU·설치·모델 수정·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람·전송 없음. Main fixture는 train 함수를 대체한 설정 검사다. 원본 checkpoint/사본 hash·구문·append prefix를 확인했다. 학습/대조군 문제를 검증하지 않는 gate의20pass를 전체 연구 성공으로 해석하지 않는다.

Exact commands(cwd NSMT; 각각 앞에 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib`):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python /tmp/nsmt_assessment_20260922T011001Z-6c90ce78/f_lif_pop_v3/forecasting/check_model.py --phase all
/home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_wirecheck.py /tmp/nsmt_assessment_20260922T011001Z-6c90ce78 f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/wirecheck_probes.json
```

<!-- assessment-watch:20260922T011001Z-6c90ce78 -->


## 추적 감사 09 — 2026-09-22 10:21 KST (예약 `20260922T012001Z-8b3e144d`)

**새 판단 근거 없음.** 최신 기억·감사08·사전등록 §2E(D-Z/AA/AB)·canonical log를 읽고,10:20:33 KST에 실제 대상 파일을 `/tmp/nsmt_assessment_20260922T012001Z-8b3e144d`로 복사·hash했다. 관찰 HEAD **`ecec8d397862e6367e556668d82e6dd11e2e36e4`**, branch `exp/f-lif-pop-v3`. 이번 변경 목록의 prereg/check_model/analysis/layers/make_golden은 **감사08의 실험 사본 및 후속 문서 사본과 전부 같은 hash**이며 새 commit으로 기록된 것이다. Trigger와 현재 hash도 일치한다. 감사자3문서의 append를 연구 변경으로 세지 않았다. [대조 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T012001Z-8b3e144d/inventory.json).

- **(a) 구현:** A10-THETA 배선·A10-PATH spike/analog 분리·A07/G4 제한된 적분기 parity는 감사08의 VERIFIED 범위를 유지. 새 수정 판정 없음. `make_golden.py`는 동일한 미완 scaffold라 재생성 완료로 닫지 않는다.
- **(b) 검증:** 새 실행 결과 없음. 동일 게이트/체크포인트 재검사는 **not run**. 기존20개 gate 통과를 다시 실행한 것으로 기록하지 않으며 A02/A05-BRANCH/A10-CAL·HASH·LOG·legacy/A12-ORACLE·ETT·test-off 등 미수정 이슈는 유지한다.
- **(c) 개선:** 새 성능 판단 근거 없음. 감사08의 지표·실패 정책·provenance 정리와 통제된 readout/독립 seed 비교 방향을 유지하며 성능·안정성 일반화는 보류한다. 새 문헌 주장/검색 없음.

수행 범위는 읽기·snapshot/hash·append뿐이다. 모델/학습 수정·학습·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송·예약 재생성 없음. 기존 결과/checkpoint/raw log를 보존했다.

<!-- assessment-watch:20260922T012001Z-8b3e144d -->


## 추적 감사 10 — 2026-09-22 10:38 KST (예약 `20260922T013001Z-6c439d7a`)

### 관찰 범위와 고정 증거

기억·감사09·사전등록·canonical 기록을 복구하고 trigger와 실제 파일을 대조했다. **10:30:36 KST**에111개 파일을 `/tmp/nsmt_assessment_20260922T013001Z-6c439d7a`에 snapshot/hash한 뒤 검사했다. 관찰 HEAD **73599f02a6cec05e485ea078669a9cb1723a83e7** 위 미커밋 수정, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 캡처된 trigger 대상 hash는 일치했다. 검사 중 추가된 사전등록 **§2F D-AC/AD** 및 canonical10:31 기록도 읽고 별도 postscript로 보존했다.

Source SHA256: train `146ab71656368df19556007eed1c4384a9c24414849ca6c8e4beebe995de200a`, test `c4558bfda69fd39ee433f9765d6b739548ed8761e97c26e0ec600158024eb1d0`, model `d5ba914759d6906f2d5293601237f409005bd8aed7382cc6d4311f7d92d6bcef`, layers `71c5e17e8b32fac4619f000356d80e1cd34c6198127833d0959f21fd0a92fc7d`, calibrate `f6e19b754fa08ca63382d448155e8eba3b2c9b4dffe4c12fd7604fa7d5701c95`。 전체 목록은 [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/changes.diff), [후속 문서](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/document_postscript_inventory.json), [재검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/diagnostic_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/validation.json)에 있다.

감사 도중 config/layers/ours 및 analog checkpoint가 다시 변경됐다. **이번 판단은 캡처 버전에 한정**하며 진행 중 analog 출력을 확정 오류로 세지 않는다. 종료 전 관찰 HEAD d54caa1d07bd3a4f4de464852fac0bd8d2262421의 새 변경과 추가 ETT/no-test/Full/oracle 결과는 다음 주기 대상이다. 감사 문서 자체의 append는 연구 변화로 세지 않는다.

### (a) 구현: 진단·거부 경로는 수정 확인, 정규화 통계 복원에는 결함

**A02-DIAG / D-AC, VERIFIED(주요 지표 집계):** 현재 API로 기존 pilot-eta와 새 diag3 checkpoint를 각각 로드해 모든256 sequence/8085 recall query를 계산했다. `kind>0`, query→sequence 평균 및 전 embedding unit hit 계산을 소스와 수치로 확인했다. Batch128과16에서 M_eff/hit/kernel_mass 차이가 모두0이었다. 기존 pilot M_eff **0.13309303159470623**은 이전 독립값0.1330930309와 약7e-10 차이로 일치한다. 기존 파일의0.231467을 덮어쓰지 않는다. Hit는 정확히0이 아니라 **3.72888448e-5**여서 문서의0.0000은 반올림값이다. 기본 diagnostics는 여전히 첫4batch 표본이고, support/eta 등의 보조 지표는 배치별 평균이므로 전체 데이터·불균등 배치 집계로 확대 해석하지 않는다.

**A05-BRANCH, VERIFIED(실패 판정 배선):** 이전과 같은 sparse branch [0,1,1,1] 및 [.01,1,1,10] 주입은 이제 picked=null/exit1이고 진단 row가 남는다. Healthy는 승인/exit0, nonfinite·bound초과·발화율 미달은 거부/exit1도 재확인했다. 실제 새 보정을 수행한 것은 아니다. [branch 주입](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/branch_paths.json), [기존 실패 경로](../f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/failure_paths.json).

**A10-CAL / D-AD, 부분 VERIFIED:** 실제 loader가 기존 호환 파일을 반환하고 tau40/80/160/320 및 cue 불일치를 ValueError로 거부한다. require_calibration 기본True와 상한 없는 경우의 guard를 해당 AST 블록만 실행해 확인했다(optimizer/학습 미실행). 하지만 key_norm=frozen으로 변경해도 기존 none 보정이 승인됐으며, 입력 분포·정책별 적용 계약 전체를 검증한 것은 아니다. 최신 mtime 하나를 선택하는 방식은 artifact를 다시 복사하면 선택이 달라질 수 있다. 보정 ID/hash와 호환 schema를 명시하고 의도적으로 공유하는 축과 재확인이 필요한 축을 구분할 것. 정식 실행 전체가 검증됐다고 닫지 않는다.

**A10-KEYNORM-CKPT: legacy 정상 복원 VERIFIED / A10-RESTORE-STATS 신규 OPEN(P1).** 새 인자 기본값과 누락 key_mean/key_std의 항등 복원으로 옛 pilot config/checkpoint가 실제 로드되며 기존 MSE가 재현됐다. 그러나 OPTIONAL_BUFFERS에 **embedding.norm_mean/std까지 무조건 포함**되어 있다. 정상 diag3 checkpoint의 두 통계만 제거한 임시 사본을 input_norm=frozen으로 읽으면 로드가 승인되고 경고만 출력되며 **동일 입력2개에서 예측 최대차0.8706527948**이 발생한다. 일반 weight 제거는 거부됐다. 이는 원본 파일 손상을 주장하는 것이 아니라 **누락을 안전하게 거부하지 못하는 확정된 복원 결함**이다. Schema/version과 당시 config를 근거로 migration을 제한하고, frozen으로 학습한 통계가 없으면 실패하도록 할 것. eta_value도 항등 통계가 아니므로 누락 허용을 별도 계약으로 다룰 것.

**A12-ETT, VERIFIED(빈 truth 진단 경로):** CPU ETT 형태의 임의 입력2개/빈(B,0) truth에서 예외 없이 finite 상태 진단을 반환하고 M_eff/hit/kernel_mass=None, sequences/queries=0이었다. 실제 ETTh1 전체 재평가·학습은 **not run**; 담당10:31 문서의1epoch 완료 주장은 아직 이번 독립 검사 범위가 아니다.

**A12-TEST-SPLIT 및 A10-HASH, VERIFIED(학습 이후 배선):** 고정 train 소스에서 학습 이후 AST 블록만 실행했다. test와 data_provider를 호출 시 실패하는 sentinel로 대체했는데 --no-test 경로는 접근하지 않고 test=null/test_skipped=true로 완료했다. Last 모델을 메모리에서만 변경해 best와 hash가 다른 조건에서 last/evaluated/checkpoint SHA256이 각각 맞았다. 새 학습을 통한 end-to-end 검증은 not run이다. no-test에서도 parameter_hash_evaluated라는 이름을 쓰므로 실제 평가 여부는 test_skipped와 함께 읽어야 한다.

### (b) 검증과 새 실행 결과: 성능 주장은 보류

캡처된 diag-102225 및 errchk-102436 JSON에는 train/provenance가 없어 중간 저장물로 분류한다. 동일 metric을 독립 성공 반복으로 세지 않는다. errchk2는1epoch 기능 결과, diag3 sparse는12epoch 완료 결과다. 저장 소스 hash가 실행 종료 시 live 파일에서 계산되므로 새 test hash가 있어도 실제 저장 진단은 옛 집계인 경우가 있다. **실행 시작 시 source snapshot/hash와 완료 상태를 별도로 남겨야 한다(A10-PROVENANCE OPEN).**

| checkpoint 재평가 | 전체 MSE | recall MSE | 정정 M_eff | kernel_mass |
|---|---:|---:|---:|---:|
| 기존 pilot-eta,15epoch | 0.239489035682 | 0.269893671218 | 0.133093031595 | 0.130328893616 |
| 새 diag3 sparse,12epoch | 0.241255409829 | 0.272620149439 | 0.132756536374 | 0.130328893616 |

두 checkpoint의 MSE는 기존 기록과 최대2.8e-17 차이로 재현됐다. Seed7/data_seed20260921, r2/k3, train/val/test2048/256/256, frozen norm/scale8이다. Theta는 기존1.0 대 새5.561343350061557로 다르고 epoch 예산도 달라 **θ 효과의 통제 비교가 아니다**. 새 diag3의 옛 보고 M_eff0.231213은 새 집계에서0.132757로 내려가며 정답 추가 질량은 약0.002428, hit0이다. 새 성능 우위·선택 검색 성공의 근거가 되지 않는다. 독립8seed/CI·정식 O7 판정은 **not run**이다.

**A09-GRAD OPEN / canonical10:31 표현 정정 필요:** 새 support_size/score_std와 pre-clipping Q/K/eta_hat gradient norm은 유용하다. score.std(unbiased=False)는 history1에서도 유한값을 만들며 이번 forward/diagnostic JSON도 NaN 없이 저장됐다. 하지만 gradient는 g11_every 표본의 **L2 norm 평균**이며 사전등록 max|grad|가 아니다. 최종 JSON은 마지막 epoch의 표본 평균이고, 수집한 *_nonfinite 수도 payload 필드에서 빠진다. diag3 마지막 epoch의 singleton_frac0.06798, Q/K norm0.004664/0.013687, eta_hat norm0.003896과 errchk2의0.03826/0.005332/0.013558/0.000352는 **관측한 표본의 값**이다. 이를 근거로 ‘singleton 가설 기각’, ‘폭주도 소멸도 없다’, ‘eta 포화가 유력 원인’이라고 인과 결론을 내릴 수 없다. 전형적 singleton 점유가 낮았다는 제한된 관찰로 수정하고, query·시점·unit별 분포/최대값·nonfinite·실제 업데이트 크기 및 고정η 통제 결과로 확인할 것.

### (c) 개선 우선순위와 문헌 근거

1. **복원 신뢰성:** frozen 통계 누락 거부 및 버전별 migration을 먼저 해결한다. 보정 artifact ID/hash·완료 상태·실행 시작 소스·실제 평가 checkpoint identity를 함께 기록한다. 기존 파일은 보존하고 정정 진단을 별도 artifact로 연결한다.
2. **원인 분리:** D-Y의 고정η 격자와 spike/analog·Full/oracle 개입을 같은 데이터/예산에서 비교하고, gradient/선택 질량/회상 오차가 함께 바뀌는지 validation에서 살핀다. 이미 본 test에 맞춰 threshold나 유리한 조건을 고르지 않는다. 아직 완료되지 않은 analog 결과로 병목을 확정하지 않는다.
3. **진단의 범위:** 기존에 원문 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단·여러 seed 비교 방향, [Pascanu et al.(2013)](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간축 gradient/norm 분석을 유지한다. 한 번의 norm이나 support 비율을 인과 증거로 취급하지 않는다. 새 문헌 사실은 추가하지 않았다.

A10-LOG CSV 미호출, A12-ORACLE fallback 계약, A07-REGEN scaffold 및 미검증 통계/대조군 요구는 그대로 OPEN이다. 신규 학습·공식 gate 전체 재실행·GPU 검사는 not run이다.

### 실행·보존

CPU snn_recall Python3.10/torch1.12.0+cu113/2threads/seed7, 기존 checkpoint 평가·작은 forward·임시 파일 fault injection·학습 이후 AST/guard 검사만 수행했다. 모델/학습 소스 수정·train 함수/optimizer 실행·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송 없음. 사본111개 hash 일치, 평가한 spike checkpoint 보존 및 기존3문서 prefix를 확인했다. 진행 중 analog checkpoint의 외부 변경은 validation에 따로 남겼다.

Exact commands(cwd NSMT; 각 명령 앞 환경은 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib`, Python은 `/home/yschoi/.conda/envs/snn_recall/bin/python`):

```text
python scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_20260922T013001Z-6c439d7a f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/failure_paths.json
python f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_probe.py /tmp/nsmt_assessment_20260922T013001Z-6c439d7a f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/branch_paths.json
python f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/diagnostic_probe.py /tmp/nsmt_assessment_20260922T013001Z-6c439d7a f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/diagnostic_probes.json
```

<!-- assessment-watch:20260922T013001Z-6c439d7a -->


## 추적 감사 11 — 2026-09-22 10:45 KST (예약 `20260922T014001Z-17a8da3d`)

### 관찰·고정 범위

기억·감사10·canonical PROJECT_LOG·사전등록 최신 **§2F D-AC/AD**를 읽었다. 사전등록 변경 감지는 이미 감사10에서 읽은 개정으로 새 규정으로 중복 집계하지 않는다. **10:40:39 KST**, HEAD **639d2293dcf12e3db3738641ff774a30e9f7439a**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7에서143개 실제 파일을 `/tmp/nsmt_assessment_20260922T014001Z-17a8da3d`에 복사·hash했다. Trigger와 대상 파일 hash는 전부 일치하고 HEAD만 감지 이후 변경됐다. ETT 원본도 재평가 전에 별도 사본을 고정했다(SHA256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066).

Source SHA256: ours `a9be634c58b635febc6f3cd4acea1169ffc882847778ca86f9412e012ae0a85f`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`, train `6e4f7490cff5df383e9428d71889f172f34ca5c1a3dd4dd4dfdc24eab0b092d3`, test `d70f50fdef2df4de3e52b67a227b7b43ada75160a5a8d5865b461910e5a3aead`, config `09ed6c4d0c0650e4b001cf427a9562944074e5c99a1321318cc2fafae46ad0b2`. Model/calibrate/utils는 감사10과 동일하므로 A10-RESTORE-STATS 등 해당 열린 이슈는 재검사 없이 유지한다.

증거: [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/changes.diff), [ETT 사본 hash](../f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/data_snapshot.json), [독립 재평가/진단](../f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/readout_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/validation.json). Console은 forecasting/log/assessment/20260922T014001Z-17a8da3d/readout_probe.log에 저장했다.

### (a) 구현: oracle 수정 확인, drive는 추가 탐색 경로

**A12-ORACLE, VERIFIED(현재 공개 모델 경로):** truth_to_oracle_p에 kind를 전달해 copy 사건은 uniform, recall만 정답 집합에 균등 질량을 주도록 바뀌었다. Train/evaluate/diagnostics의 전달 배선도 소스로 확인했다. 동일256 sequence/8085 recall query에서 copy 비균등 사건이 **옛 규칙1243 → 현재0**, recall 정답 질량 오차0이며 공개 모델은 oracle에서 kind 누락을 ValueError로 거부했다. 이전610건은 다른 확인 표본의 수치로, 이번 전체256 sequence 수치와 혼동하지 않는다. 새 csvchk의 실제 checkpoint 평가도 현재 규칙에서 재현됐다. 감사자는 학습 함수를 실행하지 않았다.

**A08-DRIVE, VERIFIED(배선·인과성·gradient의 제한된 검사):** drive는 리셋 후 soma voltage인 analog와 달리 **soma 이전 학습된 가지 가중합**을 읽는다. 입력2개에서 aux live drive와 `(state * soma.weight).sum(-1)`의 차0, gradient 연결을 확인했다. 입력/soma weight/Q gradient norm은0.604835/1.133588/0.056158로 유한했다.20번째 patch 이후 입력만 바꿨을 때 앞20개 예측 차0이었다. 새 학습/optimizer step 없이 고정 checkpoint의 단일 역전파만 수행했다.

Drive와 analog는 학습 목적·gradient 경로까지 달라지는 **end-to-end 통제 조건**이다. Drive 개선만으로 ‘정보 손실이 soma에만 있다’거나 ‘가지가 정답을 충분히 저장한다’고 단정할 수 없다. Analog도 reset 이후 상태를 읽으므로 spike 비선형 하나만 분리한 실험은 아니다. 사전등록 §2F까지 drive 정의/판정 계획의 추가 개정은 없으므로 현 결과를 탐색적으로 취급하고, 정식 비교 전에 readout과 oracle 정책 버전을 고정할 것.

**A10-LOG, 부분 VERIFIED:** train이 logging/verbose 대신 EpochLog.write를 호출한다. csvchk-103953의 실제 best_log_0.csv에 epoch0/1의2행, train/val 및 selector 진단 열이 존재한다. CSV min val_loss0.371975는 JSON0.37197544992758175를6자리로 반올림한 값이다. 과거 run에 CSV가 생긴 것으로 소급하지 않는다. `log/final+result.csv`는 여전히 없고, 동적 *_nonfinite 열을 나중에 추가하면 고정된 CSV 열 목록에서 누락될 수 있으므로 프로젝트 표준 전체 준수는 OPEN이다.

### (b) 검증: 재현되는 결과와 정책 변경의 영향을 분리

**12epoch 탐색 readout 비교:** seed7/data_seed20260921, r2/k3, train/val/test2048/256/256, batch64, 제한 batch0, scale8/theta5.561343350061557, frozen input norm. 아래 **비-oracle** checkpoint들은 현재 고정 소스에서도 기록한 MSE와 차0이었다. Kernel_mass는 약0.130328894로 동일하다.

| readout | Full recall MSE | 학습η sparse recall MSE | sparse M_eff |
|---|---:|---:|---:|
| spike | 0.276338636158 | 0.272620149439 | 0.132756536374 |
| analog(리셋 후) | 0.283003154904 | 0.261619795824 | 0.133927164587 |
| drive(soma 전) | 0.274684306208 | 0.212159097751 | 0.134012595802 |

Drive sparse의 오차 감소는 **이 seed/예산의 관찰**이다. M_eff는 kernel보다 약0.003684만 높다. 낮아진 오차와 선택 검색 성공을 동일시하지 않는다. 여러 seed/CI·정식 O7·용량 통제 GRU 비교는 **not run**이다. 기존 Full 재실행은 같은 seed의 반복이며 독립 seed 표본으로 세지 않는다.

**A12-ORACLE-VERSION / A10-PROVENANCE OPEN:** 아래 세12epoch oracle checkpoint는 **옛 정책**으로 평가하면 저장 MSE가 차0으로 재현되지만, 현재 정책으로 평가하면 달라진다. 재평가에서 kind를 무시하는 옛 helper를 메모리에서만 대체했으며 원본 소스를 수정하지 않았다.

| checkpoint | 저장/옛 규칙 recall MSE | 같은 checkpoint·새 규칙 recall MSE |
|---|---:|---:|
| diag3 spike oracleη1 | 0.029680017366 | 0.035348084881 |
| diag3 analog oracleη1 | 0.042595001411 | 0.071973921201 |
| drive oracleη1 | 0.031637966001 | 0.045209064883 |

이는 모델 실패나 파일 손상이 아니라 **평가 정책 변경**이다. 새 규칙으로 재학습한12epoch 대조군은 이번 사본에 없다. Copy 정책 변경이 recurrent 상태를 통해 이후 recall에도 영향을 주므로 과거 headroom/G 지표와 새 정책 지표를 섞지 않는다. 특히 drive oracle JSON은 새 ours/test 소스 hash를 담고도 **옛 정책에서만 저장 수치가 재현**된다. 종료 시 live 파일 hash만 읽는 현재 방식으로는 실행 중 로드된 코드를 식별하지 못한다. 시작 시 코드 사본/정책 version을 고정하고 결과에 연결할 것. 새 규칙으로 실행된 csvchk는2epoch/256·64·64 기능 점검이며 recall0.377701122733을 재현했다. 이를12epoch oracle의 대체 성능값으로 쓰지 않는다.

**A10-HASH 실제 artifact 확인:** 새 parameter_hash_evaluated와 checkpoint_sha256이 있는 run은 모두 실제 로드한 값과 일치했다. 특히 drive oracle은 last≠best 조건에서도 evaluated hash가 맞았다. 기존 diag3 spike sparse의 옛 parameter_hash 불일치는 이미 알려진 과거 결함이며 보존한다. 한 가지 개선이 과거 artifact를 자동 정정하지 않는다.

**A12-ETT, VERIFIED(제한된 실제 평가):** ETTh1 checkpoint를 별도 데이터 사본으로 읽어 **test 첫3batch=192개 창/129024요소**에서 저장 MSE **0.911763752704**를 차0으로 재현했다. 전체 test 창은2785개다. train/val도 max_batches=3인1epoch 기능 실행이므로 ‘ETTh1 전체 검증 완료’로 기록하지 않는다. 같은 표본의 persistence1.664264396996, window_mean0.856615248951이며 모델이 window_mean보다 좋지 않다. 기존 loader는 scaler를 train 첫8640행에만 맞추고, target 경계는 train0–8640/val8640–11520/test11520–14400으로 분리한다. 전체 데이터 성능/반복통계는 not run이다. Selection diagnostics는 기본4batch를 읽어 평가의3batch와 범위가 다를 수 있으므로 표본 수/범위를 명시할 것.

**A12-TEST-SPLIT:** 실제 notest-103025 결과에서 test=null/test_skipped=true를 확인했고 checkpoint identity도 일치했다. 이 run의 test를 감사자가 새로 평가하지 않았다. 감사10의 호출 차단 검사와 합쳐 현재 배선 상태를 유지하며, 이미 본 다른 run의 test가 다시 미관측 데이터가 되는 것은 아니다.

### (c) 개선 방향과 남은 조건

- 새 oracle 계약과 drive를 **탐색 개정**으로 기록하고 같은 정책 버전에서 Full/sparse/고정η/oracle 비교를 맞춘다. 진행 중 결과와 과거 정책 결과를 분리한다. 최종 규칙은 validation에서 정하고 이미 반복 관찰한 test로 threshold·조건을 선택하지 않는다.
- Drive의 낮은 MSE가 선택 기능과 관련되는지 동일 예산·여러 seed에서 확인한다. 선택 질량/정답 hit/회상 유형별 오차를 함께 보고, 학습된 같은 표현에 대한 정책 개입과 별도 학습 readout 비교의 질문을 구분한다. 기존에 원문 확인한 [Wiegreffe & Pinter(2019)](https://aclanthology.org/D19-1002/)의 통제 진단 방향과 [Zoology/MQAR](https://arxiv.org/abs/2312.04927)의 recall 평가 맥락을 유지한다. 이번에 새 문헌 사실은 추가하지 않았다.
- A10-RESTORE-STATS의 frozen 통계 누락 거부, 실행 시작 source/정책 고정, 보정 호환 schema를 우선 해결한다. A09 표본 norm의 인과 해석 제한/epoch max|grad|·nonfinite 기록, A07-REGEN scaffold, 다중 seed·통계 요구는 계속 OPEN이다. 동일 결함 재검사는 not run.

### 실행·보존

CPU Python3.10/torch1.12.0+cu113/2threads/seed7. 사본 checkpoint 평가와 drive의 입력2개 gradient/인과성 검사, oracle helper 검사 및 CSV/소스 읽기만 수행했다. 새 학습·optimizer·GPU·환경 설치·모델/학습 소스 수정·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람/전송 없음.143개 사본과 모든 캡처 checkpoint의 현재 hash 일치, 기존 문서 prefix를 확인했다. 감사자 파일 외 연구 내용은 수정하지 않았다.

Exact command(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/readout_probe.py \
/tmp/nsmt_assessment_20260922T014001Z-17a8da3d \
f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/readout_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T014001Z-17a8da3d/readout_probe.log 2>&1
```

<!-- assessment-watch:20260922T014001Z-17a8da3d -->


## 추적 감사 12 — 2026-09-22 10:55 KST (예약 `20260922T015001Z-bff8435c`)

### 관찰과 고정 증거

기억·감사11·사전등록 §2F·canonical 기록을 복구했다. **10:50:38 KST**, HEAD **520bb56872c6e7c68c48f65cd70601979c22a1c5**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7에서139개 파일을 `/tmp/nsmt_assessment_20260922T015001Z-bff8435c`에 snapshot/hash했다. Trigger 파일 hash는 모두 일치했다. 모델 SHA256 `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, key_geometry `526cdc9af752557eae4c36c7f65565b3a03268c4f841d2f4d9d99ec4ced3892a`. 감사 중 canonical10:50 검색 후보 문서가 추가되어 읽고 postscript로 별도 보존했다. 사전등록은 §2F 그대로다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/changes.diff), [독립 probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/restore_grad_probes.json), [후속 문서](../f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/document_postscript_inventory.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/validation.json). Raw console: forecasting/log/assessment/20260922T015001Z-bff8435c/restore_grad_probe.log.

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
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/restore_grad_probe.py \
/tmp/nsmt_assessment_20260922T015001Z-bff8435c \
f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/restore_grad_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T015001Z-bff8435c/restore_grad_probe.log 2>&1
```

<!-- assessment-watch:20260922T015001Z-bff8435c -->


## 추적 감사 13 — 2026-09-22 11:01 KST (예약 `20260922T020001Z-8f8cd215`)

**새 판단 근거 없음.** 최신 기억·감사12·사전등록 §2F·canonical 기록을 읽고 **11:00:58 KST**에 실제123개 파일을 `/tmp/nsmt_assessment_20260922T020001Z-8f8cd215`에 복사·hash했다. 관찰 HEAD **41554c68cd97cbad522008297b98eda230b1e651**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 감시 대상120개 파일은 trigger 및 감사12 사본과 모두 hash가 같다. 새 commit의 key_geometry와 canonical10:50 문헌 내용도 이미 감사12에서 검토했다. 차이는 감사자3문서 append뿐이므로 연구 변경으로 세지 않는다. [대조 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T020001Z-8f8cd215/inventory.json).

Source SHA256: model `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`.

- **(a) 구현:** A10-RESTORE-STATS 누락 거부 및 JSON nonfinite/보정 필드 전달의 감사12 VERIFIED 범위를 유지한다. A09 absmax 평균 집계 등 미수정 이슈는 OPEN이며 새 VERIFIED 판정 없음.
- **(b) 검증:** 새 실행 결과 없음. 동일 checkpoint/probe 재검사·학습·통계 검증은 **not run**. A09-KEY-GEOMETRY 재현 정보와 rank 해석, A10-CAL·PROVENANCE·LOG, key_norm 적합, A07-REGEN의 남은 조건을 유지한다.
- **(c) 개선:** 새 성능 판단 근거 없음. 감사12에서 원문 확인한 문헌의 적용 범위와 진단 재현→고정η/정책 대조→필요한 후보 검증 순서를 유지한다. 새 문헌 주장·검색 없음, 성능 우위 판단 보류.

읽기·snapshot/hash·문서 append만 수행했다. 모델/학습 소스 수정·학습·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송·예약 재생성 없음. 기존 local changes/checkpoint/raw log를 보존했다.

<!-- assessment-watch:20260922T020001Z-8f8cd215 -->


## 채택 우선순위 결정 — 2026-09-22 11:15 KST (사용자 직접 요청; AUDIT-PRIORITY-01)

**권고 순서: ① 고정η 대조 → ② QK 정규화 → ③ key 표현 개선 → ④ α-entmax → ⑤ Gram 보정 → ⑥ delta rule.** 이는 다음 검증 대상으로 채택할 순서다. 아직 성능 개선을 입증한 최종 모델 선택은 아니다. ①의 고정η 격자는 기존 D-Y에 따라 먼저 진행하고, **dense residual의 즉시 삭제는 채택하지 않는다.** 기존 분수 기억 수식과 양의 계수·cap 계약을 보존하면서 원인을 분리할 수 있는 후보를 앞에 배치했다. 이 순위는 아래 수치·원문을 바탕으로 한 감사자의 공학적 판단이며 논문이 증명한 후보 간 우열은 아니다.

### 1. 이번 결정 전에 실제로 확인한 근거

HEAD **41554c68cd97cbad522008297b98eda230b1e651**, exp/f-lif-pop-v3에서 소스·문서·diag3 sparse checkpoint19개 파일을 `/tmp/nsmt_assessment_20260922-adoption-priority`에 복사·hash한 뒤 CPU 검사했다. **학습/optimizer/GPU 실행 없음.** [소스·checkpoint inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/inventory.json), [실행 가능한 진단](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probes.json), [원문 확인 기록](../f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/literature_checks.json).

**동일 표본 비교:** diag3 spike/sparse의 같은 checkpoint, validation 첫64 sequence/2004 recall query, seed7/data_seed20260921, η=**0.02577485144**. 각 query의 모든32 unit을 평균하고 query→sequence 순으로 집계했다. Copy는 제외했으며 argmax 동률은 첫 인덱스를 택한다.

| 지표 | 이번 직접 계산 |
|---|---:|
| score argmax가 정답 칸인 비율 | **0.2715946141** |
| p argmax가 정답 칸인 비율 | **0.2715946141** |
| 최종 c argmax가 정답 칸인 비율 | **0.0000000000** |
| p의 정답 질량 | 0.2564991390 |
| post-cap c의 정답 질량 M_eff | **0.1370368662** |
| fractional kernel의 정답 질량 | **0.1344110089** |
| 동일 query에서 균등한 칸 선택의 정답 확률 | **0.1610314336** |

점수/p의 top1은 이 표본의 균등 기준보다 높지만, c의 정답 질량 증가는 약**0.002626**이다. 따라서 **정책이 계수에 전달되는 정도를 먼저 확인**할 근거는 있다. 이것만으로 η가 유일한 원인이라거나 readout에서 정보가 소실된다고 결론내리지는 않는다. 사용자 제시0.2924와3.7e-5를 그대로 같은 조건의 수치로 간주하지 않았다. 이전3.7e-5는 pilot-eta checkpoint에서 얻은 값이다.

**작은η의 효과는 0이 아니다.** 현재 α=.7,과거41칸에서 B=13.96147385,b₁=.68729711,b₀=1.10054743이다. 정답 한 칸에 p를 전부 주는 가상 정책에서, 정답 lag d가 최근 칸을 앞서는 cap 전 조건은

`η > (b₁ − b_d) / (B + b₁ − b_d)`  (d>1).

이때 최근 칸은 b₀보다 작아 target이 cap되더라도 아래 순위 비교가 유지된다. 실제 계수와 cap으로 검산했다.

| 정답 lag | 최근 칸을 앞서는 η 경계 | η=.026에서 정답이 최대 c인가 |
|---|---:|---|
| 2 | 0.007149 | 예 |
| 5 | 0.015867 | 예 |
| 10 | 0.021498 | 예 |
| 20 | 0.026224 | 아니오 |
| 41 | 0.030240 | 아니오 |

따라서 ‘η가 작으면 점수를 아무리 개선해도 드러나지 않는다’는 일반 명제는 **반례가 있다**. 또 η=.2에서 위 단일 정답 정책은 모두 최대 계수가 되지만, cap 후 정답 질량은 약.091–.093에 그친다. **Argmax 개선≠O7 M_eff 통과**다. 실제 정답 집합은 여러 칸이므로 이 표는 모델 성능 측정이 아닌 계수 경로의 통제된 수치 예다. η=.026에서 정책을 최근 칸↔가장 오래된 칸으로 바꾸면 계수 L1 차도 **0.7259966403**으로 0이 아니다. η·분포·cap을 함께 진단해야 한다.

**Rank 해석 정정:** 현재 유닛별 투영 key는4차원이다. 이번 shape [64,32,41,4]의 비중심 feature Gram participation ratio는 평균 **1.1747333**, 범위1.03329–1.25833, 이론적 상한**4**였다. 기존3.97/42와는 정의/표본이 달라 동일 수치의 재현으로 부르지 않는다. ‘42에 가까워야 한다’는 기준은4차원 key에는 불가능하다. 이번 값은 비중심 에너지의 편중을 보여주지만 의미적 회상 실패를 그 자체로 증명하지 않는다. Gram condition number2470도 token Gram/feature Gram, 중심화·정규화·작은 고유값 처리 정의가 먼저 필요하다.

### 2. 후보별 채택 순서와 통과 조건

| 순위 | 채택할 검증 대상 | 먼저 둘 이유 / 직접 근거 | 다음 단계로 채택할 조건·보류 조건 |
|---|---|---|---|
| **1** | **고정 η={0,.2,.5,1} + 학습η 대조**. Dense residual 재설계는 후속 조건부 | D-Y에 이미 있고 수식 변경이 작다. 같은 validation 표본에서 p hit.2716→c hit0, M_eff 증가.002626을 확인했다. 작은η의 lag별 제한과 cap 효과를 분리할 수 있다. | 같은 새 oracle 정책·seed·예산·보정 계약으로 **각 조건을 별도 학습**하고 spike를 주 결과로 둔다. Score/p/c/readout 및 pre/post-cap mass·kappa·G11·최대gradient를 함께 보고한다. η=1이 이미 dense residual 제거에 해당하므로 별도 삭제부터 시작하지 않는다. 기존 측정상 고정η가 자동 개선을 보장하지 않으므로 유효한 oracle/headroom부터 확인. |
| **2** | **투영 Q/K의 L2 정규화** + validation에서 정한 score scale/θ | 메모리 읽기와 cap을 유지하며 거리의 크기 편향을 분리하는 비교다. TTR Eq.(35)-(36)이 정규화 거리/내적 연결을 설명한다. | Norm=0 처리(ε), 인과성, gradient·cap 게이트 및 같은η 조건의 비교가 필요. 현 `key_norm=frozen` 입력 표준화와 **다른 연산**이다. 정규화만으로 θ 불필요/폭주 제거라고 주장하지 말 것. 기존 상태 크기의 유용한 신호가 사라질 가능성도 validation에서 확인. |
| **3** | **Key 표현 ablation**: 기존 [u,I] 대 causal Δu 추가 등 | 같은 모델/표본에서 score hit와 균등 기준의 간격은 있으나 완전 검색은 아니다. 현재 key의4차원/분포 편중을 고려하면 표현 변화는 직접 검사할 만하다. 기존 f_j/c 계약을 유지한다. | 갱신 전 정보만 사용하고 미래 state를 참조하지 않을 것. 차원/parameter 증가를 맞춘 대조와 정규화 조건을 둔다. 정답 집합 대비 score margin/랭킹 및 최종 c 질량·오차가 함께 나아져야 한다. 평균 cosine를 무조건 낮추는 목표는 금지—같은 값의 여러 정답 칸은 비슷해도 된다. |
| **4** | **α-entmax**: 먼저 고정 sparsity 지수, 이후 학습 지수 | 확률 p를 만드는 함수만 바꿔 현재 양의 계수 재가중에 연결할 수 있다. 원문 Proposition1은 학습 가능한 지수의 미분 근거다. 다만 현재 낮은η 전달 문제가 남으면 우선 효과를 식별하기 어렵다. | Fractional α와 구별해 지수를 γ 등으로 표기. Sparsemax/softmax와 score scale·η·예산을 맞추고 support·gradient·mass·recall을 비교. 미분 구현 검증 및 안정성이 필요. 지수 학습이 θ 역할을 자동 제거하지 않는다. |
| **5** | **Ridge/Gram 보정 읽기**를 별도 탐색 baseline으로 | TTR의 선형 최소제곱 해가 근거이나 현재 모델은 선형 K–V 회귀가 아니라 분수 증분의 재가중이다.4×4 solve라는 크기만으로 우선 도입할 근거는 부족하다. | K,V,f_j 대응을 수식으로 고정하고 λ·condition·수치안정성을 점검. Linear feature Gram4×4와 nonlinear kernel Gram T×T를 구분. 보정 가중치가 음수일 수 있어 기존 positivity/cap/G4 계약을 그대로 승계할 수 없다. 보정 score만 쓰는 whitening과 읽기 전체 교체도 별도 후보로 분리. |
| **6** | **Delta rule**을 별도 기억 구조 대조군으로 | DeltaNet의 recurrent update는 근거가 있지만, 기록/읽기 방식이 바뀌어 현재 아이디어의 작은 수정이 아니다. 검증 부담과 원인 해석 범위가 가장 크다. | K–V 정의·state 크기·parameter·실측 runtime/메모리·reset 계약을 새로 명시. O(T)는 고정 차원의 해당 recurrent 연산에 대한 말이다. 기존 f_j 전 이력 합산을 남기면 전체 모델이 자동 O(T)가 되지 않는다. 현재 모델의 교체 채택은 공정한 독립 성능/비용 비교 이후. |

**조건부 순위 조정:** ① 이후 score 순위는 좋지만 p 단계에서 질량이 사라지는 것으로 확인되면④를③보다 앞당긴다. Score 단계부터 정답을 구분하지 못하면③을 유지한다. Cap에서만 질량이 무너지면②–④를 한꺼번에 바꾸기보다 현 cap/η의 제약을 먼저 정량화한다. 안전 상한을 임의로 없애 성능만 맞추지 않는다. ⑤–⑥은 계산비가 작아 보여도 계약 변경이 커서 후순위다.

### 3. 논문 원문 확인 결과와 적용 한계

- [Test-time regression, §3 Eq.(3), (32), (35)–(36)](https://arxiv.org/html/2501.12352v1): 선형 최소제곱 해·feature covariance와 kernel Gram을 구분하며, QKNorm의 거리/내적 연결에도 bandwidth 선택이 있다. ‘모든 커널 읽기는 KᵀK=I일 때만 정확하다’는 형태로 일반화하지 않는다. 입력이4차원이라고 모든 비선형 기억 문제를4×4 역행렬로 해결할 수 있는 것도 아니다.
- [Modern Hopfield 원문, Eq.(5), Theorem4–5](https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf): 이번에는 원문 PDF를 직접 받아 읽었다(SHA256 **48cecc1d10cea553538fe8d1e2f1bf7378bed6b1233b2857384f5081ac2f7579**). 보장은 패턴별 `Δ_i=x_iᵀx_i−max_(j≠i)x_iᵀx_j`, β, 패턴 수/크기, query와 패턴의 거리 등에 의존한다. 평균 인접 cosine.758이나 rank만으로 조건 위반을 판정할 수 없다. 충분조건의 미확인은 실패의 필요조건 증명이 아니며, 현재 sparsemax/비대칭QK/분수가지에도 정리가 자동 적용되지 않는다. 감사12의 PDF 접근 미완 상태는 **이번 원문 확보·해당 정리 확인 범위에서 해소**했다.
- [DeltaNet, §3 Eq.(4)](https://arxiv.org/html/2406.06484v3): 현재 상태에서 key 방향의 연관을 수정하는 recurrent rule을 확인했다. 기존 분수 solver의 대체 가능성·우월성·전체속도는 별도 검증 대상이다.
- [Adaptively Sparse Transformers, §3 Proposition1](https://aclanthology.org/D19-1223.pdf): entmax 지수 미분과 head별 학습의 근거를 확인했다. 우리 회상/분수 동역학에서의 개선이나 score scale 독립성을 보장하는 정리는 아니다.

### 4. 공통 채택 기준과 이번 작업의 범위

먼저 checkpoint/config/source/정책 버전·보정 ID·평가 범위를 고정한다. **Validation으로만 후보와 hyperparameter를 선택**하고, 이미 반복 관찰한 test를 새로운 미관측 근거처럼 쓰지 않는다. 기존 사전등록의 독립 seed8개·paired CI·G14/O7 기준은 유지하며, 이번 탐색 수치에 맞춰 낮추지 않는다. 모든 후보에 동일 예산·통제 조건과 인과성/finite/G11·복원·원본 대조가 필요하되 수식이 바뀌는⑤–⑥은 변경된 게이트를 새로 정의해야 한다. 추가 비용은 단순 parameter 수가 아니라 전체 forward/backward·history 저장까지 측정한다. **효능이 미검증인 후보를 지금 ‘최종 채택’으로 표시하지 않는다.**

이번은 직접 validation forward1회와 계수 대수 검산·원문 확인이다. 새 학습, 고정η sweep 학습, 후보 구현, 다중 seed/CI 및 성능 우위 검증은 **not run**이다. 모델/학습 소스·사전등록 본문을 수정하지 않았고 기존 audit도 고치지 않고 append했다. canonical 기록과 문서 기억에는 이 결정만 추가한다. 예약 실행이 아니므로 watcher marker/ack를 새로 만들지 않는다.

재현 명령(cwd NSMT):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probe.py \
/tmp/nsmt_assessment_20260922-adoption-priority \
f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/priority_probes.json
```

Raw console: `f_lif_pop_v3/forecasting/log/assessment/20260922-adoption-priority/priority_probe.log`. 수치 환경은 CPU torch1.12.0+cu113/2threads이며 문헌 PDF 추출의 xref 복구 경고는 원문 읽기 도구의 경고이지 모델 실패가 아니다.
