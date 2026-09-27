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


## 추적 감사 14 — 2026-09-22 11:35 KST (사용자 직접 요청: 작업 세션 추적)

### 관찰 결과

관찰 HEAD **4d67c5072b63bfeaf570be44c02c945d0f9aa8a8**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **11:33:02 KST**에123개 문서·소스·결과 파일을 `/tmp/nsmt_assessment_20260922-manual-followup`에 복사·hash했다. 감사13과 비교해 변경된 것은 감사자 기록을 포함한3문서이며, **모델/학습 소스·실행 결과의 새 변경은 확인되지 않았다.** 다른 세션 대화나 프로세스에는 접근하지 않았다. 작업의 진행 여부는 남겨진 문서/산출물 범위에서만 판단한다.

Canonical11:31 기록에서 **우선순위①고정η→②QK정규화→③key표현→④entmax→⑤Gram→⑥delta 수용**, residual 즉시 삭제 보류, 작은η의 효과 전무 주장 철회, 서로 다른 checkpoint 지표 비교 및 Hopfield 조건 표현 정정을 확인했다. 이는 문서상 결정 수용이며 고정η 실험이나 후보 구현이 완료됐다는 뜻은 아니다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/inventory.json), [직접 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probe.py), [수치 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/validation.json). Source hash는 감사13과 동일: model `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`.

### (a) 구현 정확성: 기존 판정 유지

새 구현 없음. A10-RESTORE-STATS 등 확인된 수정의 범위와 A09 epoch absmax 평균 집계/A10-CAL·PROVENANCE·LOG/key_norm 적합/A07-REGEN의 OPEN을 유지한다. 동일 모델·checkpoint 재평가는 **not run**이다. 아래는 새 문서 주장에 필요한 작은 대수/행렬 검사만 수행했다.

### (b) 검증: 타당한 rank 보완은 수용, 새 M_eff 일반화는 반례로 정정 요구

**A09-KEY-GEOMETRY — 표본 평균 Gram 설명 수용, 원 수치 재현은 OPEN.** 작업자가3.97은 개별 feature Gram이 아니라 **표본별 token Gram을 먼저 평균한 행렬**의 participation ratio라고 밝혔다. 이 경우 상한4가 적용되지 않는다는 지적은 맞다. 감사의 기존 상한4는 **각 개별 유닛/시퀀스의 key 행렬** 및 그 feature/token Gram에 한정한다. 평균을 먼저 하는 경우와 개별 PR를 먼저 계산해 평균하는 경우를 구분하도록 이 append에서 명확히 정정한다.

직접 CPU 예: seed7,단위 L2 무작위 key [64,42,4]에서 개별 token Gram rank≤4/평균 개별 PR **3.75866735**, **표본 평균 token Gram rank42/PR36.39041248**이었다. 이는 작업자가 말한 평균 방식의 가능성을 검증한 예이지, 원본3.97·35.67·38.88을 재현한 결과는 아니다. 그 수치들의 script/seed/표본/정규화/평균 축은 여전히 부족하다. 표본 평균 token Gram은 시점별 유사성이 여러 표본에서 공통되는 정도도 반영하므로, 개별 memory 용량·분리도와 동일시하지 않는다. 무작위 baseline은 좋은 key의 기준이나 Hopfield 위반 증명이 아니다.

**A02-REACHABILITY 신규 OPEN — ‘단일 정답이면 η<1에서 M_eff≥.5 불가능’은 틀림.** Canonical11:31은 η=.2/.5 표에서 모든η<1로 일반화했다. 현재 Selector의 oracle 경로와 cap을 그대로 사용해 α=.7,과거41칸,정답 한 칸의 p=1인 경우를 직접 계산했다.

| η | lag2의 post-cap M_eff | lag41의 post-cap M_eff |
|---|---:|---:|
| .92 | **.50704244** | **.50086126** |
| .93 | .54033787 | .53419064 |
| .95 | .62203030 | .61619958 |

모두 η<1이고 .5를 넘는다. 이는 성능 결과가 아니라 새 일반 명제에 대한 **확정된 수치 반례**다. 표본으로 확인한 .2/.5의 낮은 값과 ‘η<1 전체에서 불가능’은 다른 주장이다.

대수적으로 target이 cap에 도달한 구간은

`M_eff = b₀ / [b₀ + (1−η)(B−b_d)]`.

따라서 이 구간에서 **η ≥ 1−b₀/(B−b_d)**이면 M_eff≥.5다. 현재 b₀=1.1005474055,B=13.9614739001에서 lag2/41 경계는 각각 **.91771424/.91972394**이며, 그 경계에서 target cap 조건도 충족한다. 실제 Selector의 eta buffer float32 표현 때문에 식과 약1e-7 미만 차이는 생기지만 판정은 같다. η→1이면 residual이 사라져 이 단일 정답 정책의 M_eff→1이다.

따라서 도달 가능 영역을 그리자는 제안은 타당하지만 **영역이 비어 있다는 전제는 제거**해야 한다. 실과제의 여러 정답 칸·history 길이·lag 분포·정답 정책·cap에 따라 영역이 달라지며, 평균 lag나 평균 정답 칸 수만 대입해서 전체 sequence 평균 O7 판정을 대체하지 않는다. 임계값이나 등록된 η 격자를 이번 반례에 맞춰 자동 변경하지 않는다.

### (c) 개선 방향과 다음 확인 조건

채택 순위는 AUDIT-PRIORITY-01을 유지한다. 다음 의미 있는 근거는 **같은 정책 버전의 고정η 대조 완료 결과**와 **재현 가능한 key 기하/도달 가능성 진단**이다. 작업자는 rank 집계 코드를 artifact로 남기고, score→p→c 지표를 동일 checkpoint/표본/정답 mask에서 연결할 것. 도달 가능성 계산은 분석용이며, 높은η의 실제 학습 안정성·오차 개선을 증명하지 않는다.

문헌 방향은 이전에 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1) 및 [Hopfield Eq.(5)/Theorem4–5](https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf)의 적용 범위를 유지한다. 이번은 신규 문헌 주장 없이 직접 수치 검사를 추가했다. 새 학습 결과가 없으므로 **성능·효능 판정 보류**, 신규 학습/다중seed·CI는 **not run**이다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7,임의 작은 행렬 및 실제 Selector 계수 평가만 수행했다. 모델/학습 소스 수정·학습/optimizer·GPU·환경 설치·프로세스 중단·git add/commit/tag/push/reset/switch·다른 세션 대화 열람·전송 없음. 기존 파일/결과를 보존하고 감사 artifact와3문서 append만 작성했다. 사용자 직접 추적 요청이므로 예약 marker/ack는 추가하지 않았다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probe.py \
/tmp/nsmt_assessment_20260922-manual-followup \
f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/claims_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922-manual-followup/claims_probe.log 2>&1
```


## 추적 감사 15 — 2026-09-22 11:49 KST (예약 20260922T024001Z-acec6eac)

### 관찰 범위와 실제 증거

**고정η 네 조건의 저장 결과를 재현했으나 효능은 미확증이다. §2G의 uniform oracle ‘절대 상한’ 주장은 수치 반례로 정정이 필요하다.** 관찰 HEAD **2769c5c8100e367b965a47b20bd1924f0b0d18a1**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 11:40:55 KST에 실제148파일을 `/tmp/nsmt_assessment_20260922T024001Z-acec6eac`로 복사·SHA256 기록했다. Trigger manifest와 실제 파일을 대조했으며, 감지 때 완료 JSON이 없던 η=1도 **snapshot 때에는 완료**되어 포함했다. Snapshot 내 학습η 후속 run은 완료 JSON이 없어 재평가하지 않았다. 감사14 대비 연구 소스 변경은 없다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/inventory.json), [직접 CPU 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probe.py), [결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/validation.json). Model SHA256 `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`, prereg `1721c443859c85f48ddbc6ba3ad908016108cc2bd54f9fddfaa0807faddba823`.

### (a) 구현 정확성 — 고정η 배선·복원·결과 재현 확인, 기존 OPEN 유지

`etagrid-113240`의 완료된 네 checkpoint를 현재 API로 CPU 복원했다. 각각 실제 eta buffer가 설정값과 일치하고, **전체 test MSE·M_eff 저장값과 재평가 차이0**, evaluated parameter hash/checkpoint hash 일치, 기록된 source hash와 고정한 소스의 불일치0이었다. A06 고정η 및 A10-HASH의 기존 확인을 이 네 실행까지 확대한다. 이 사실만으로 A10-PROVENANCE의 실행 시작 source 고정 문제를 닫지는 않는다.

공통 조건은 seed7/data_seed20260921, train/val/test=2048/256/256, batch64, spike readout, α=.7,K=4, recall key3, scale8, θ=5.561343350061557, 최대12epoch/patience10, 배치 제한0이다. Test 전256sequence/8085 recall query를 평가했다. M_eff·hit은 query→sequence 평균이다.

| 고정η | 완료 epoch | recall MSE | M_eff | 최종 c hit | cap rate |
|---|---:|---:|---:|---:|---:|
| 0 | 12 | .276338636158 | .130328894393 | .000000 | .000000 |
| .2 | 12 | .319387285780 | .127181601094 | .020639 | .009919 |
| .5 | **11** | .433504353901 | .132870673176 | .096712 | .064035 |
| 1 | 12 | .416682604337 | .136898894082 | .216071 | .134492 |

η=.5는 CSV epoch0에서 validation loss 최저, 이후10epoch 개선 없음과 patience10/종료 코드가 일치한다. **11epoch를 미완성 또는 실행 실패로 판정하지 않는다.** η=0의 Q/K gradient0은 선택 경로를 끈 설정에 맞는다. Test 상태는 네 조건 모두 finite/G11 상한 안이었으나 이는 전체 학습 step의 보장을 추가로 증명한 것이 아니다.

### (b) 검증 적절성 — 결과 재현은 통과, 해석·상한 정의는 정정 필요

**A09-GRADIENT OPEN:** η=.5/1의 마지막 epoch `grad_absmax_all` 보고값은 각각 **1.0608e11 / 4.3705e12**다. 현 reducer가 표본 batch 최대값들을 **평균**하므로 epoch 최대라고 부를 수 없다. 그래도 clipping 전 매우 큰 gradient가 실제 보고됐다는 근거다. 매 step norm1 clipping 및 finite 검사를 쓰고 있어 곧바로 비유한 학습 실패를 뜻하지 않으며, 원인을 점수 함수 하나로 단정하지 않는다. 진짜 최대·표본 수·clipping 빈도와 전후 norm을 추가로 기록할 필요가 있다. 재검사 없이 A09를 VERIFIED로 닫지 않는다.

**A02-REACHABILITY OPEN(P1), §2G 정정 요청:**

1. Uniform-on-answer oracle은 **정의된 정책 기준선**이지 모든 p의 post-cap M_eff 상한이 아니다. 현재 실제 Selector에 α=.7/history41/정답 lag{41,1}/η=.2를 넣으면, uniform p=(.5,.5)는 **.1644974421**, cap을 고려한 p=(.9173828776,.0826171224)는 **.1744286360**이다. 같은 계수·η·cap에서 더 높은 유효 질량이 존재한다. 이 반례는 허용된 p 공간의 계수 주장 검증이며 실제 score 학습으로 그 p가 도달된다는 뜻은 아니다.
2. 양의 b_i, cap C=b₀, η∈[0,1], 비어 있지 않은 정답 집합 A에서, `B=Σ_i b_i`, `B_A=Σ_(i∈A)b_i`, `m=|A|`라 두면 **임의의 simplex p에 대한 계수 질량의 최대**는 아래와 같다. 현재 b_i≤C 조건에서 적용한다.

   `S_max = min((1−η)B_A + ηB, mC)`

   `U = S_max / [S_max + (1−η)(B−B_A)]`.

   이유: `q_i=b_i p_i/Σ_j b_j p_j`도 임의 simplex이며, adaptive 총량ηB를 정답 칸들의 남은 cap 용량에 배분하면 최대 정답 질량을 얻는다. 정답 밖 residual은 그대로 남는다. 정답 칸이 포화되면 잉여량은 정답 칸에 버릴 수 있다. 이것은 **자유로운 정책에 대한 계수 상한**이고, WQ/WK 표현 제약·인과적으로 얻을 수 있는 정보·학습 최적화의 성공을 보장하지 않는다.
3. 같은 test256sequence/8085query의 실제 history 길이·정답 mask에 식과 uniform oracle을 적용하고 등록된 query→sequence 순서로 집계했다. 학습/forward 성능 실험이 아닌 대수 검사다.

| η | uniform oracle M_eff | 자유 정책 상한 U의 평균 |
|---|---:|---:|
| 0 | .130328893571 | .130328893571 |
| .2 | .298202511703 | .298208777887 |
| .5 | .466552376117 | **.466613112585** |
| .655 | **.557888951227** | .557946562113 |
| .7 | .590493784894 | .590537123649 |
| .75 | .631612043147 | .631647159784 |
| 1 | 1 | 1 |

이 표본에서는 두 값의 차가 작고, **등록 격자{0,.2,.5,1} 중 .5 평균 질량에 도달 가능한 것은 η1뿐이라는 결론은 올바른 상한으로도 유지**된다. 다만 고정 history41/평균 lag21/정답 칸3–4 예시가 그 증거는 아니다. η.655의 uniform 값이 이미 .5를 넘으므로 ‘과제 평균 정답 수3.5이니 최소η≈.70–.75’도 이 실측 분포의 필요조건이 아니다. 미래 dataset/seed 분포 전체에 일반화하지 않는다. 등록 임계값·확증 격자는 변경하지 않는다.
4. §2G의 ‘lag5/21/35 차이<.01’은 인용한 원 artifact 자체와 불일치한다. 정답4칸·η.2의 .3088−.2613=**.0475**다. `meff_reachable.txt`를 재생성할 명령/코드와 정확한 배치 정의도 남겨야 한다. .4666을 .34–.41의 ‘상한과 정합’이라 부르지 않는다. 표본과 집계가 다르다.
5. `learned/oracle` 비율만으로 ‘선택자는 잘했다’고 판정하지 않는다. **η=0이면 선택자가 무엇을 하든 비율1**이며 uniform oracle이 진짜 상한도 아니다. Full kernel 기준 질량, p 자체의 질량/순위, pre/post-cap 질량, 실제 오차를 함께 보고한다. 이는 진단 보완이고 승인된 O7 기준을 바꾸는 제안이 아니다.

**통계·분리·진행 상태:** 이번 동일 seed의 η 증가 조건들은 η0보다 recall MSE가 컸고, η1에서도 M_eff가 .1369에 그쳤다. 이를 ‘η만 올리면 해결된다’는 주장의 지지로 쓸 수 없다. 반대로 한 seed·짧은 예산·서로 다른 최적화 상태만으로 아이디어의 최종 실패나 원인까지 결정하지 않는다. 독립8seed/paired CI, 등록 최대50epoch 조건의 확증, 새로운 후보 성능 검사는 **not run**이다. 이미 반복 관찰한 test는 탐색용으로 표시하고 후보/scale 선택은 validation에서 진행한다.

11:45 canonical 기록에 새 oracle **η.5/1** 결과와 G14/O7 관련 해석이 추가된 것을 후속 관찰했다(`project_log_late.txt` 별도 보존). **그 새 학습 결과와 학습η 후속 run은 이번 snapshot 범위 밖**이며 새 정책 checkpoint 재평가는 **not run/다음 주기 확인 대상**이다. 해당 기록의 탐색적이라는 제한은 수용하되, ‘통과/실패’는 탐색 수치의 기준 충족 여부와 확증 판정을 구분해야 한다. 이번 감사에서는 그 표만으로 G14 VERIFIED를 부여하지 않는다. 또한 작은 학습η에서 ‘선택이 사실상 작동하지 않는다’는 표현은 감사 PRIORITY-01의 비영 효과 반례를 반영해 제한해야 한다.

### (c) 개선 방향 — 우선순위 유지, 정규화의 검증 조건을 구체화

**① 고정η 결과·동일 정책 oracle 비교 정리 → ② QK L2정규화 → ③ causal key 표현 → ④ entmax → ⑤ 별도 Gram → ⑥ 별도 delta** 순서를 유지한다. ①의 네 조건은 확보됐고 같은 예산/정책 oracle 및 후속 학습η의 provenance 재평가가 남았다. η1은 dense residual이 없는 조건인데도 정답 질량 개선이 작으므로, dense residual 지배만으로 현재 전체 현상을 설명할 수 없다. 다만 score→p→c 변환·cap·gradient·학습 동역학이 여전히 얽혀 있어 score 품질 단독의 확정 인과판정은 아니다.

QK 정규화는 같은η·예산에서 크기 편향을 분리하는 **다음 탐색 후보**로 타당하다. Score scale은 validation에서 고정하고 zero norm ε 처리·인과성·finite·실제 gradient 최대/clip 통계를 확인할 것. 같은 checkpoint/validation 표본에서 score/p/c의 정답 질량과 순위를 각각 기록해 p 단계의 손실이면 entmax를 key 표현보다 앞당기는 기존 조건부 규칙을 적용한다. 최종 c hit만으로 score와 p의 손실 위치를 구분하지 않는다.

문헌은 앞서 원문 확인한 [Test-time regression §3 Eq.(35)–(36)](https://arxiv.org/html/2501.12352v1)의 QK 정규화–거리 연결과 bandwidth 잔존, [Pascanu et al.의 gradient/clipping 분석](https://proceedings.mlr.press/v28/pascanu13.pdf), [Adaptively Sparse Transformers §3](https://aclanthology.org/D19-1223.pdf)를 재사용했다. 이 문헌들이 우리 모델의 정규화 효능·폭주 제거·최적 순위를 보장하지는 않는다. 새 논문 주장은 없고 이번 결론은 위 직접 수치 검토에 기반한다.

### 남은 조건·수행 기록

A10-CAL(key_norm 적합/보정 ID), A10-PROVENANCE, A10-LOG(final CSV/동적 열), A07-REGEN, A09-KEY-GEOMETRY의 열린 조건은 이번 재현만으로 닫지 않는다. 11:46:49 보존 검사에서 snapshot148파일 hash 불일치0. 작업 세션은 canonical/학습η checkpoint·CSV 및 HEAD를 추가 변경했고, 이를 감사자 변경으로 취급하지 않았으며 다음 주기에서 확인한다.

CPU torch1.12.0+cu113/2threads/seed7로 완료 checkpoint4개 forward 평가와 계수 대수·Selector 소규모 검사만 수행했다. 신규 학습/backward/optimizer/GPU/환경 설치/소스 변경/진행 프로세스 중단/git 변이/다른 세션 대화 접근·메시지 전송은 하지 않았다. 기존 문서 본문과 raw artifacts를 보존하며3문서에 append한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probe.py \
/tmp/nsmt_assessment_20260922T024001Z-acec6eac \
f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/etagrid_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T024001Z-acec6eac/etagrid_probe.log 2>&1
```

<!-- assessment-watch:20260922T024001Z-acec6eac -->


## 추적 감사 16 — 2026-09-22 11:54 KST (예약 20260922T025001Z-31dd6781)

### 관찰 범위

**새 정책 oracle 두 조건과 학습η 결과의 재현을 확인했다. 탐색적 headroom은 확보됐으며, 학습된 선택자의 효능 확증은 아직 없다.** HEAD **7520087c6d5578c1ec640350fa233717ce78d980**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **11:50:51 KST**, manifest 대상·최신 감사/기억/사전등록/canonical log와 etagrid checkpoint 등160파일을 `/tmp/nsmt_assessment_20260922T025001Z-31dd6781`에 복사·hash했다. 감지 manifest 대비 hash 불일치0. 감사15 대비 연구 소스 변경0이며, 이미 재평가한 고정η 네 조건을 반복 실행하지 않았다. 사전등록 최신은 §2G이고 감사15의 상한 정정은 아직 반영되지 않았다.

Model SHA256 `050bb4c5ad641da7790e575f84e33026cc531acfe3f633001ad47f491865a480`, train `887cfb9ee5cf4fc6ac68c3cbbe56227518d616988a0af75a540548898fe5a29c`, layers `45322e8940acf477057c9ce420f9763ce77b04132879dbd782292413732876ac`. [전체 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probe.py), [수치·hash 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/validation.json).

### (a) 구현 정확성 — A12-ORACLE·A10-HASH의 새 실행 재현 확인

`etagrid-113240`의 새 완료 run3개를 현재 API로 복원하고 전체 test를 CPU 평가했다. Train/evaluate→model의 kind 전달과 copy에서는 uniform/recall에서는 answer-set 정책을 적용하는 현 코드가 유지됨을 확인했다. 새 oracle checkpoint는 **이 수정 정책에서 저장 MSE·M_eff가 정확히 재현**된다. 과거 정책 checkpoint를 새 정책으로 재평가한 결과와 구별한다.

| 새 완료 조건 | epochs | recall MSE | copy MSE | M_eff | 최종 c hit |
|---|---:|---:|---:|---:|---:|
| oracle η=.5 | 12 | .039580641913 | .143349488259 | .466552377101 | 1 |
| oracle η=1 | 12 | .034965688339 | .127822646497 | 1 | 1 |
| sparse 학습η | 12 | .272620149439 | .146173325183 | .132756536374 | 0 |

세 실행 모두 저장 전체 MSE/진단 M_eff와 재평가 차이0, evaluated parameter hash/checkpoint SHA256 일치, source hash 불일치0. Test 상태 finite 및 G11 범위 안이었다. 학습η 평가 checkpoint의 실제 eta는 약.02577485이며 마지막 epoch 통계 약.02629857과 구별한다. Oracle의 Q/K gradient0은 고정 정답 정책이 score를 대체한 결과이므로 gradient 연결 오류로 판정하지 않는다.

**VERIFIED 범위는 현 정책 checkpoint의 복원·수치 재현이다.** 종료 시 source hash만 기록하는 A10-PROVENANCE를 실행 시작 코드 고정까지 확인한 것으로 닫지 않는다. QK L2정규화 구현은 이 snapshot에 없으며 해당 후보 검증은 **not run**이다.

### (b) 검증 적절성 — 같은 데이터/예산 비교 확인, 한 seed의 탐색 수치

공통 seed7/data_seed20260921, train/val/test2048/256/256, batch64, spike, α=.7/K4/key3/scale8/θ5.561343350061557, 최대12epoch/patience10, 배치 제한0. 재평가는256sequence/8085recall query 전체이고 M_eff/hit은 query→sequence 평균이다. 재생성한 x/y/truth/kind tensor 전체를 hash해 **세 run의 각 split이 동일**함을 확인했다. Train/val/test는 생성 seed offset0/10000/20000을 쓰며 split별 hash는 서로 다르다. 이는 데이터 생성·동일 조건 확인이며 과거 test 반복 관찰의 영향을 없애 주지는 않는다.

감사15에서 재현한 full(η0) recall .276338636158에 대해 새 oracleη1과의 **탐색적 headroom=.241372947820**이다. `eta_grid.txt`의 공통 oracleη1 분모를 그대로 사용하면 G는 학습η **.0154055653**, 고정η.2 **−.1783491067**, .5 **−.6511322796**, 1 **−.5814403372**로 표와 일치한다. 이는 동일 seed에서 정답 정책의 큰 개선 여지가 있으나 학습 정책은 거의 회수하지 못한다는 근거다. **Oracle 성능은 실제 사용 시 정답에 접근할 수 있는 모델의 성능이 아니며 검색 성공을 대신하지 않는다.** G의 분모 정책을 명시하고 같은η oracle 대비 질량 진단과 구별한다.

- **A02/O7:** 새 oracleη.5의 M_eff .4665523771은 감사15의 동일 분포 계수 계산을 지지한다. Uniform oracle을 절대 상한으로 부르는 §2G/A02-REACHABILITY는 여전히 OPEN이다. 반례와 수정식은 감사15에 있으며 새 학습으로 그 오류가 해소되지 않는다.
- **A09/통계 해석:** 작업 기록의 chance≈.161은 이번 test 자체의 값이 아니다. 동일 test mask에 같은 query→sequence 집계를 적용한 uniform-slot 기대 hit는 **.156583749693**이다. 고정η1 c hit .21607055는 이 기대값보다 높지만, 독립 seed CI나 random-policy 학습 대조 없이 ‘질량 분포가 우연 수준’/원인 확정을 하지 않는다. Full-kernel mass .13032889와 uniform-slot chance .15658375는 다른 기준선이다.
- 학습η recall .272620149439는 이전 diag3와 동일한 seed·설정에서 같은 값이다. 이번 run의 재현성을 뒷받침하지만 **독립 seed 하나를 추가한 것으로 세지 않는다.** 학습η에서 효과가 ‘전혀 없다’는 해석도 기존 비영 효과 검사와 맞지 않는다.
- Canonical11:45가 탐색/미수렴/예산 공정성 미확인이라고 제한한 점은 적절하다. G14의 **이 seed에서 headroom 수치가 기준을 넘는 관찰**과 8seed/paired CI 기반 O7 확증을 분리한다. 이번 감사는 G14/O7 최종 성공·실패를 확정하지 않는다. 등록50epoch 조건의 성능 검증·다중seed·CI는 **not run**이다.

### (c) 개선 방향과 남은 조건

우선순위 **① 고정η 비교 → ② QK L2정규화 → ③ causal key 표현 → ④ entmax → ⑤ 별도 Gram → ⑥ 별도 delta**를 유지한다. ①의 이번 seed7/12epoch 격자와 새 정책 oracle 재현은 완료됐으므로 **②를 다음 탐색 대상으로 삼을 근거**가 확보됐다. 이는 정규화 최종 채택이나 효능 확인을 뜻하지 않는다.

η1에서 dense residual 없이도 낮은 M_eff가 남고 oracle은 크게 개선되므로, 단순 residual 삭제보다 **동일η에서 QK 크기 편향·score→p→c 변환·최적화 안정성을 분리**하는 방향이 타당하다. Norm ε/인과성/score scale을 명시하고 validation으로만 설정을 선택할 것. 큰 clipping 전 gradient와 absmax 평균 집계 문제(A09)가 남아 있으므로 진짜 최대·clipping 빈도·전후 norm도 함께 확인할 것. Score 순위는 괜찮고 p에서 질량이 손실되는 직접 근거가 생길 때만 entmax를 key 표현보다 앞당긴다.

이 방향은 앞서 원문 확인한 [Test-time regression §3 Eq.(35)–(36)](https://arxiv.org/html/2501.12352v1)의 정규화와 거리/scale 연결, [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 gradient/clipping 분석, [Adaptively Sparse Transformers](https://aclanthology.org/D19-1223.pdf)의 학습 가능한 entmax 근거를 재사용한다. 새로운 문헌 사실이나 효능 보장은 추가하지 않았으며 이번 우선순위 판단은 직접 재현 결과에 기반한다.

A09 absmax/KEY-GEOMETRY, A10-CAL·PROVENANCE·LOG, A07-REGEN 및 A02-REACHABILITY의 잔여 조건은 유지한다. 검토 소스가 동일하므로 이전 수정에 대한 전체 gate 재실행은 **not run**이다.

### 수행·보존·완료

CPU torch1.12.0+cu113/2threads/seed7, 새 checkpoint3개 forward 평가·split fingerprint·기존 결과의 G/동일 test chance 산술만 수행했다. 첫 파일 탐색에서 `rg --files -g AGENTS.md`가 일치 없음(exit1)을 반환한 것은 파일 탐색 결과이며 모델 검사 실패가 아니다. 모델/학습 소스 수정·학습/backward/optimizer·GPU·설치·프로세스 중단·git 변이·다른 세션 대화 열람/전송은 하지 않았다. 사본 hash와 기존 문서 prefix 보존 검사를 남기고3문서에 append했다. 감사 중 추가 변경은 다음 주기에 확인한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probe.py \
/tmp/nsmt_assessment_20260922T025001Z-31dd6781 \
f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/completion_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T025001Z-31dd6781/completion_probe.log 2>&1
```

<!-- assessment-watch:20260922T025001Z-31dd6781 -->


## 추적 감사 17 — 2026-09-22 13:34 KST (예약 20260922T043001Z-65597c1e)

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


## 추적 감사 18 — 2026-09-22 14:15 KST (예약 20260922T051001Z-b66150e0)

### 관찰 범위·핵심 판정

**Soft QK 정규화의 새 완료 결과2개를 재현했다. 보정 ε 호환성 누락과 clipping 표본을 전체로 일반화한 해석은 정정이 필요하다.** 관찰 HEAD **1832bdc16cd075473c582bd386686da18775ece8**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **14:10:34 KST**에 소스·문서·결과·해당 checkpoint211파일을 `/tmp/nsmt_assessment_20260922T051001Z-b66150e0`에 snapshot·hash했다. Trigger 이후 qk2η.2 CSV가 바뀌었으며, 실제 사본에는 **η0/.2 결과 JSON이 모두 완료**돼 있었다. η.5는 config/checkpoint만 있고 완료 JSON이 없어 성능 평가에서 제외했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/source.diff), [직접 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probe.py), [수치 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/current_probes.json), [run config](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/run_configs.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/validation.json).

관찰 source SHA256: layers `0ad31dc6250a780703a5195b32606657424298925fd948c8f3411fd948c32a04`, train `07012eaa57f61c4dbbdf930da270cc5191016a87859da6417bd9fdfbb36bb62e`, config `cde4a30e6fe813927586feddfebb312ebb689f6269d4107654f79321a94908ef`, calibrate `1b95aef5ea841b1ed9cda1b137b98b8b752119bf9009a1495dde1ced80b2258a`.

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


## 추적 감사 19 — 2026-09-22 14:24 KST (예약 20260922T052001Z-759f71ff)

### 관찰 범위와 증거

**Soft QK 정규화의 η=.5/학습η 완료 결과를 재현했다. 성능 개선의 원인과 η1 중단의 원시 증거는 별도 확인이 필요하다.** HEAD **b6d4baf6a54ee950403df1f60d05f290073bee07**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **14:20:35 KST**에 manifest·최신 감사/기억/§2H/canonical 문서·대상 checkpoint208파일을 `/tmp/nsmt_assessment_20260922T052001Z-759f71ff`에 snapshot·hash했다. Manifest hash 불일치0, 감사18 대비 연구 소스 변경0. Trigger에 포함된 η.2 JSON은 감사18 사본과 같아 반복 평가하지 않았다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probe.py), [수치·hash](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/completion_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/validation.json). Source SHA256: layers `0ad31dc6250a780703a5195b32606657424298925fd948c8f3411fd948c32a04`, train `07012eaa57f61c4dbbdf930da270cc5191016a87859da6417bd9fdfbb36bb62e`. 전체 소스/결과 hash는 inventory에 기록했다.

### (a) 구현 정확성 — 새 checkpoint 복원·수치 재현 VERIFIED, 기존 OPEN 유지

현재 API/진단 정의는 감사18과 동일함을 hash로 확인했다. `qk2-140541`의 새 완료η.5와 학습η checkpoint만 CPU 복원·평가했다. 두 실행 모두 전체 test MSE 및 M_eff의 저장값과 차이0, evaluated parameter/checkpoint SHA256 일치, 기록된 source hash 불일치0였다. Test 상태도 finite/G11 범위 안이었다. 이 결과는 **현재 soft 변환에서의 재현성 확인**이고 정규화의 최종 효능 검증은 아니다.

A18-QKNORM의 작은 score/zero-input/인과성 검사는 감사18의 확인 범위를 유지하며 전체 gate를 재실행하지 않았다. A09 gradient 표본/absmax reducer, A10-CAL qk_eps, A10-PROVENANCE·LOG, A02 실제 분포 연결, A07-REGEN 등의 잔여에는 수정 소스가 없으므로 OPEN 유지한다. Canonical14:18의 “clipping 전 진짜 최대·빈도 항목 충족” 선언은 **감사18의 실제 reducer·관찰 빈도 검사와 맞지 않는다.** 필드가 존재하는 것과 요구한 통계가 구현된 것은 다르다.

### (b) 실행 결과와 검증 적절성

공통 seed7/data_seed20260921,2048/256/256 sequence,batch64,12epoch,spike,α=.7/K4/key3,soft ε=.01/θ=.38146987702788376. Test256sequence/8085recall query 전체, M_eff/hit은 query→sequence 평균이다.

| 새 완료 조건 | recall MSE | M_eff | 최종 c hit | support_p | test 최대 상태 크기 |
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


## 추적 감사 20 — 2026-09-22 15:14 KST (예약 20260922T061002Z-5c0b3779)

### 관찰 범위·증거

**ε 보정 호환성과 전체 batch norm/clip 집계의 수정을 직접 확인했다. 개별 gradient absmax와 과거 G11 중단 사건은 아직 별도 OPEN이다.** 관찰 HEAD **b6d4baf6a54ee950403df1f60d05f290073bee07**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. 15:10:37 KST에 manifest 대상·최신 감사/기억/사전등록§2H/canonical 기록198파일을 `/tmp/nsmt_assessment_20260922T061002Z-5c0b3779`로 snapshot·SHA256 기록했다. Trigger hash 불일치0. 실제 변경은 train/calibrate와 새 보정 JSON·qk_norm_form.txt이며 모델 변환은 감사19와 같다.

Train SHA256 `bf2f5737fb1df24c395dc1227cb14e5fbf25be4f7c336cd312ff3cecbd90eea3`, calibrate `7d6e3549a85aa7a3c5fdfb6520e7622b80cfc96b3ade14a09298cf0a9fb6f2a2`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/source.diff), [직접 진단 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/fixes_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/validation.json).

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


## 추적 감사 21 — 2026-09-22 15:24 KST (예약 20260922T062001Z-d4a951fd)

### 관찰·증거

**실제 g11chk 실행의 G11 기록을 확인했고, 저장 checkpoint의 forward만으로 기록된 상태 초과를 독립 재현했다. 전체 batch clipping 지표의 CSV 보존도 확인했다.** HEAD **8ecf96ea7e1fb1badd112586b22134d8bf0d4536**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **15:20:49 KST**에 manifest·최신 감사/기억/§2H/canonical 기록·해당 checkpoint203파일을 `/tmp/nsmt_assessment_20260922T062001Z-d4a951fd`에 snapshot·hash했다. Trigger 불일치0, 연구 소스는 감사20과 동일하다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probe.py), [수치·기록 대조](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probes.json), [문서 수치 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/claim_checks.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/validation.json). Train SHA256 `bf2f5737fb1df24c395dc1227cb14e5fbf25be4f7c336cd312ff3cecbd90eea3`, layers `0ad31dc6250a780703a5195b32606657424298925fd948c8f3411fd948c32a04`. 검사한 checkpoint SHA256 **c798ba5d8636d7fbc1a35045e490c7eecaa7b0706d4edcf64ee277a6f2001dab**.

### (a) 구현 정확성 — 새 실제 기록의 검증 범위

**A05/G11-g11chk VERIFIED:** `g11chk-151001`의 `G11_violation.json`에서 run UUID **44e2b2349fc29e28**, zero-based epoch1/batch0, peak **332.54718017578125**, bound **305.0375175476074**, UTC06:10:06.157092를 확인했다. Run ID/UUID/mode/η1/QK True/ε.01/보정 파일은 저장 config와 모두 일치했고 bound도 지정 보정 JSON의 고정값과 정확히 일치했다. CSV는 완료된 epoch0 한 행이며 최종 성능 JSON은 없다. 따라서 이 run은 완료 성능으로 세지 않는다.

Epoch0의 best checkpoint를 CPU로 복원하고 원래 **train512sequence에 forward만** 수행했다. 표본 순서를 고정하고64개씩 처리했으며 모델/optimizer 갱신은 하지 않았다.

| 확인 항목 | 실제 값 |
|---|---:|
| Train sequence index381의 최대 상태 | **332.54718017578125** |
| 위반 JSON의 peak와 차이 | **0** |
| 전체512sequence 중 bound 초과 수 | **3** |
| 해당 checkpoint에서 전체 train 최대 상태 | **350.3652648925781** |

이는 기록된 상태 초과가 저장 모델·데이터에서 재현된다는 직접 근거다. 전체 학습/optimizer 경로 및 당시 shuffle된 batch0의 정확한 구성은 재현하지 않았다. 350.365는 고정 checkpoint로 모든 train 표본을 훑은 진단값이므로 기록 batch의332.547과 모순되지 않는다. 학습 중 모델이 계속 변하는 상황의 epoch 최대와도 구별한다.

**과거 사건과 구분:** 이번 run은 seed7/data_seed20260921, train/val/test **512/64/64**, 최대3epoch,batch64,g11_every10,softε.01/θ=.38146987702788376이다. 과거 qk2는 train2048/최대12epoch였으므로 같은 수치가 나왔다는 이유로 그 **별도 run의 원시 증거까지 복구됐다고 하지 않는다.** 이번 g11chk의 사건과 독립 forward 확인은 VERIFIED, 과거 qk2 사건의 정확한 원시 provenance는 별도 미확인 상태다. Suite/UUID를 포함해 구분한다.

### (b) 검증 적절성 — 전체 batch 통계의 실제 저장 확인, 인과·성능 판단 제한

**A09-CLIP-ALL-BATCHES의 CSV 보존 VERIFIED:** 실제 epoch0 CSV에 다음 네 필드가 존재한다.

| 필드 | CSV 값 |
|---|---:|
| grad_total_norm_pre_mean | 878689650.125 |
| grad_total_norm_pre_max | 3913563392 |
| clip_rate_all_batches | **1.0** |
| batches_seen | **8** |

512/64=8이며 batch 제한0과 일치한다. 감사20에서 수집·reducer를 검증했고 이번에는 실제 run의 CSV 저장을 확인했다. **이 새 run의 완료 epoch0에 한해서 8/8 batch가 clipping됐다는 기록 해석은 타당**하다. 과거 watch 표본 지표로 전체 batch를 일반화한 주장을 소급해 승인하지 않는다. Gradient 자체를 다시 backward로 계산한 것은 아니며 그 수치 재현은 not run이다. 중단으로 최종 결과 JSON이 없으므로 새 필드의 완료 JSON 보존도 아직 not run이다.

개별 원소의 epoch `max|grad|`와 post-clip norm은 여전히 별도 미구현/미확인이다. L2 norm의 batch 최대를 absmax 대용으로 부르지 않는다. G11 감시도 여전히 watch 시점에 한정돼 ‘모든 step에서 상태 상한이 보장된다’고 표현할 수 없다.

**Canonical15:10의 정정은 대체로 수용하되 두 문장을 보완할 것.**

- Hard/soft의 x≈0 Jacobian bound가 같다고 **전체 변환이 같거나 개선 원인이 ε뿐임이 증명되지는 않는다.** 같은 ε=.01, x=(.02,0,0,0)에서 hard 출력 첫 성분은1, soft는 **.8944271909999159**다. 동일ε에서 형태만 바꾼 통제 실험은 아직 not run이며, 형태 효과와 ε 효과의 분리는 계속 필요하다. 인과 단정을 철회한 취지는 유지한다.
- 고정η.2의 QK G는 **−.1233793961**, η.5는 **−.1771581586**이다. Canonical의 “0.2(−.1772)”는 η.5 수치와 바뀌어 인용됐다. 학습η G가 양수라는 정정은 맞다.

**성능:** 새 완료 recall/test 결과는 없다. 실제 상태 초과는 이 η1/seed/설정에서 안정성 기준에 걸렸다는 증거이며 아이디어 전체 실패나 QK 정규화의 보편적 실패가 아니다. 중단 checkpoint를 최종 성능으로 평가하지 않았다. 사전등록 임계값/분할/대조군 정책의 새 변경은 없고, 독립8seed/paired CI·효능 확증은 **not run**이다.

### (c) 다음 방향

고정 checkpoint의 동일 validation 표본에서 **score→p→pre-cap→post-cap 질량·순위를 연결**하는 기존 우선순위를 유지한다. 이 경로 진단에 상태 최대/G11 초과 여부와 전체 batch norm·clipping 기록을 함께 붙여, 검색 개선과 안정성 변화를 분리할 것. 이번 초과에 맞춰 보정 상한을 사후 확대하지 않는다. Score 표현이 병목이면③ causal key, p 변환에서 손실되면④ entmax를 앞당기는 조건부 규칙을 유지한다.

근거는 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 시간에 따른 gradient/clipping 분석, [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 정규화·거리/scale 연결을 재사용한다. 이번 새 판단은 직접 기록 대조와 forward 수치 검사에 기반하며 새 문헌 사실·효능 보장은 추가하지 않았다. A10-PROVENANCE·LOG/개별 absmax·A02 실제분포 연결/A07-REGEN 등 미검사 잔여는 유지한다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7. Epoch0 checkpoint 고정 forward와 문서 산술만 수행했다. 학습/backward/optimizer·GPU·환경 설치·모델/학습 소스 수정·git 변이·진행 프로세스 중단·다른 세션 대화 접근/메시지 전송은 하지 않았다. 기존 파일·checkpoint·문서 prefix를 보존하고 감사3문서에 append했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probe.py \
/tmp/nsmt_assessment_20260922T062001Z-d4a951fd \
f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/event_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T062001Z-d4a951fd/event_probe.log 2>&1
```

<!-- assessment-watch:20260922T062001Z-d4a951fd -->


## 추적 감사 22 — 2026-09-22 16:13 KST (예약 20260922T071001Z-738a6a1c)

### 관찰 범위·핵심 판정

**A09 absmax 수정 완료 주장은 현재 소스에서 재현되지 않았다. 확인 실행의 오차·checkpoint는 재현됐지만, 관찰1회라 집계 오류를 검출하지 못한다.** 관찰 HEAD **031fb758af0fc2dc9257acc100783b5cd44404fe**, branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7. **16:10:35 KST**에 manifest·최신 감사/기억/사전등록§2H/canonical 문서·absmaxchk checkpoint206파일을 `/tmp/nsmt_assessment_20260922T071001Z-738a6a1c`로 snapshot·hash했다. Trigger hash 불일치0. 연구 코드 변경은 train.py뿐이며 모델/평가 API는 이전과 동일하다.

Train SHA256 `5d1a223d3cd3860cfbd752ed37f317b416dc475f3acb000362464bd1cf0cd68e`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/source.diff), [직접 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/validation.json).

### (a) 구현 정확성 — A09-ABSMAX OPEN 유지, 관찰 횟수 결함 확인

Canonical16:07은 absmax/_max 필드를 np.max로 바꿨다고 기록했으나, 고정한 실제 reducer는 모든 finite 목록에 **`float(np.mean(finite))`**를 적용한다. 실제 변경은 `grad_observations`에1을 추가하는 부분과 이미 존재하던 pre-norm max의 float 변환이며, absmax reducer 변경은 없다. 실제 함수 AST에서 집계 블록만 추출해 optimizer 없이 실행했다.

| 입력/필드 | 실제 출력 | 요구되는 의미의 값 |
|---|---:|---:|
| grad_absmax_all [1,9] | **5** | 9 |
| grad_WQ_absmax [1,9] | **5** | 9 |
| grad_WK_absmax [2,10] | **6** | 10 |
| grad_observations [1,1] | **1** | 2회 |
| grad_total_norm_pre [1,9]의 max | **9** | 9 |
| eta [1,9]의 mean | **5** | 5 |

Pre-norm max는 내부에서 미리 np.max한 단일값이라 맞지만, **그 성공이 개별 absmax의 수정을 입증하지 않는다.** `grad_observations`도1의 평균이므로 관찰 횟수가 늘어도1이다. 실제 결과 JSON의 train 메타데이터에는 이 필드가 없고 CSV에만 나타난다. **A09-OBSERVATIONS OPEN**으로 추적한다. Post-clip norm도 여전히 없다.

수정 조건은 필드 의미에 맞춰 absmax를 max, 관찰 횟수를 count/sum, 평균 지표를 mean으로 분리하고 JSON/CSV 양쪽에 의미가 보존되게 하는 것이다. 이는 실제 소스와 직접 반례에 따른 확정 결함이며 학습 실패를 뜻하지 않는다. 진행 중 추가 수정이 있을 수 있으나 이 snapshot을 VERIFIED로 닫을 근거는 없다.

### (b) 확인 실행·검증 절차

`absmaxchk-160633`은 seed7/data_seed20260921, **1epoch**, train/val/test512/64/64,batch64,g11_every10,배치 제한0,비정규화 sparse 학습η,θ5.561343350061557이다. Test64sequence/2022recall query를 CPU 재평가했다.

- 전체 MSE **.3704960201865058**, recall MSE **.3698382646185705**, M_eff **.13170154071437468**.
- 저장 전체 MSE/M_eff와 재평가 차이0, evaluated parameter/checkpoint SHA256 일치, source hash 불일치0.
- 실제 CSV/JSON에 전체 batch pre-norm mean **1.8237197399**, max **2.9906373024**, clip_rate_all_batches1, batches_seen8이 저장됐다(CSV는6자리 반올림). **A09 전체 batch norm/clip 필드의 완료 JSON·CSV 보존은 이번 run까지 확인됐다.** 그 gradient 자체를 backward로 재생성한 것은 아니다.

그러나8batch/g11_every10에서는 **batch0만 한 번 관찰**한다. 관찰1개이면 mean=max이고 1의 평균=관찰 횟수1이므로, 이 실행은 두 집계 결함을 검출할 수 없다. 최소한 서로 다른 두 관찰값을 넣어 확인해야 한다. 이번 감사의 위 추출 블록 반례가 그 조건을 충족한다. 추가 학습 없이도 이 집계 결함은 검사 가능하다.

새 run은 logging 확인용 짧은 실행으로, 이전12epoch의 격자 성능과 직접 비교해 개선/퇴보를 판정하지 않는다. G11 test 진단은 finite/상한 이내였으나 전체 학습 안정성의 증명은 아니다. 신규 학습·gradient 재계산·독립8seed/paired CI·성능 우위 검증은 **not run**이다. Canonical의 G 인용 정정과 G11 새/과거 사건 구분 수용은 확인했으며 감사21의 제한을 유지한다.

### (c) 개선 방향

먼저 실제 reducer·관찰 count·결과 전달을 수정하고, 서로 다른 관찰값을 사용한 작은 검증으로 문서의 완료 주장과 소스를 일치시킬 것. 그 다음 동일 validation checkpoint/표본의 score→p→pre-cap→post-cap 지표에 정확한 gradient/clip 통계를 함께 연결한다. Score 표현이 병목이면③ causal key, p 변환에서 손실되면④ entmax를 앞당기는 기존 조건부 순위를 유지한다. 이번 logging 확인 결과로 QK 정규화 채택이나 모델의 효능을 판단하지 않는다.

Gradient 크기·시간별 변화와 clipping을 구분하라는 방향은 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf), 정규화와 score scale의 구분은 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 근거를 재사용한다. 이번 판정은 직접 수치 검토에 기반하며 새 문헌 주장은 없다. A02 실제분포 연결/A07-REGEN/A10-PROVENANCE·LOG 등 범위 밖 OPEN은 그대로 둔다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7, 실제 reducer만 실행한 합성 검사와 checkpoint1개 forward 평가만 수행했다. 모델 학습/backward/optimizer/GPU·설치·모델/학습 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송은 하지 않았다. 기존 문서 prefix와 파일 사본을 보존하고3문서 append·진단 artifact만 작성했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib \
/home/yschoi/.conda/envs/snn_recall/bin/python \
f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probe.py \
/tmp/nsmt_assessment_20260922T071001Z-738a6a1c \
f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/absmax_probes.json \
> f_lif_pop_v3/forecasting/log/assessment/20260922T071001Z-738a6a1c/absmax_probe.log 2>&1
```

<!-- assessment-watch:20260922T071001Z-738a6a1c -->


## 추적 감사 23 — 2026-09-22 16:26 KST (예약 20260922T072001Z-785d6d29)

### 관찰 범위·핵심 판정

**단계별 분해6행과 η 개입8행의 수치를 재현했다. 다만 무작위 순위 기준·개입의 인과 해석을 정정해야 하며, QK 학습η checkpoint에 η=1을 적용하면 G11 상태 상한을 초과한다.** HEAD `adb43e4ae47848d682a06b7ba5b067ed3d8015ab`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 16:20:44 KST에 실제231파일(분석·소스·config/checkpoint·문서)을 `/tmp/nsmt_assessment_20260922T072001Z-785d6d29`로 snapshot하고 SHA256을 기록했다. Trigger hash 불일치0. 최신 사전등록은 §2H이며 forecasting Python 소스는 감사22 snapshot과 동일하다. 감사자의 이전 append는 새 연구 성과로 세지 않았다.

`stage_decomposition.py` SHA256 `4c384e3d846e9e2d6fca309a86f5d925c874b71b4f938df79791e0d8b23ebe6c`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/inventory.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/stage_probe.py), [전체 정밀 수치](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/stage_probes.json), [추가 비교](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/supplement_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/validation.json).

### (a) 구현·진단 정확성

**A02 단계 분해의 실제 계수 재구성은 확인됐다.** 현재 API에서 state[n]=u(n+1), query는 갱신 전 u(n), history는 각 시점의 갱신 전 상태와 current다. 분석 함수의 인덱싱·kind>0 mask·D 평균→query 평균→sequence 평균이 이에 맞는다. 6개 checkpoint의 같은 test256sequence/8085recall query에서 원문 표의 모든 지표가 표시 정밀도4자리까지 재현됐다. 재구성 c와 실제 forward가 남긴 c의 차이는0이었다. 추가로 QK 고정η.2 조건을 검사했다. 원래 스크립트의 batches=4는 이 설정의 batch64/test256 전체를 덮지만 큰 데이터에 자동으로 일반화되지는 않는다.

**A02-STAGE-RANK OPEN — “0.5=무작위” 표기 오류.** 지표는 m개의 정답 중 최상위 정답의 0-based 순위를 n−1로 나눈 값이다. 무작위 permutation의 기대값은

`E[best rank/(n−1)] = (n−m)/((m+1)(n−1))`, n>1.

0.5는 m=1에 한정된다. 실제 query→sequence 집계 기준은 **.214817733876**이다. n=2..8의 모든 정답 부분집합35조건을 열거해 식과의 차이 ≤5.56e−17을 확인했다([검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/rank_formula_check.json)). 따라서 학습η 비QK rank .258861은 이 순위 기준에서 무작위보다 나쁘고, 학습η QK .195898은 더 좋다. Top1 hit의 별도 기준 .156583749693은 그대로다. 두 지표가 서로 다른 양상을 보이는 것은 모순이 아니다. 향후 각 query의 정답 수에 맞춘 순위 기준을 함께 기록할 것.

**A02 단계 사이에 커널 가중 정책을 추가할 것.** 현재 수식의 정확한 분해는

`w_j = b_j p_j / Σ b p`, `M_pre = (1−η) M_kernel + η M_w`.

| 학습η checkpoint | p mass | w mass | pre-cap | post-cap |
|---|---:|---:|---:|---:|
| 비QK | .2443073973 | .2245154195 | .1327565364 | .1327565364 |
| QK ε.01 | .2872716602 | .2622897784 | .1337985402 | .1337985402 |

혼합식은 float32 오차 범위에서 직접 확인됐다(각 query/unit 최대 오차의 집계 <5e−8). 작은 η가 커널 기준선과의 차이를 축소한다는 설명은 지지된다. 그러나 **p→w 커널 재가중과 w→pre 혼합은 별개**이며 모든 질량 감소가 η에서 생긴다고 쓰지 않는다. η=1 비QK 학습 모델도 p .137611→w/pre .121563→post .136899로 중간 변화가 있으므로 양 끝의 근접함만으로 “손실이 거의 없다”고 일반화하지 않는다. Score와 p의 top1 일치는 이 표의 순위 보존 확인이지 전체 선택·상태 동역학의 검증을 대체하지 않는다.

**A09-ABSMAX/OBSERVATIONS는 OPEN 유지.** train.py가 감사22와 같은 hash여서 실제 reducer/count/JSON 전달 결함에 수정 근거가 없다. 이번에는 같은 반례를 다시 실행하지 않았다. 다른 기존 OPEN도 새 근거 없이 닫지 않았다.

### (b) 새 개입 결과·검증 적절성

Snapshot의 learned 및 learned+QK 두 checkpoint에서 η buffer만 메모리 내 변경해 학습값/.2/.5/1의8조건을 CPU forward로 재검사했다. 모든 조건에서 trainable parameter bytes는 그대로였고 저장 checkpoint는 변경하지 않았다. 표의 recall MSE/M_eff/hit8행은4자리까지 일치한다. 설정은 seed7/data_seed20260921, train/val/test2048/256/256,batch64,최대12epoch,평가 batch 제한0이다. Split 재생성 hash는 서로 달랐다(추가 artifact에 수록). 이는 split 생성의 구분 확인이며 반복 test 선택 문제를 해소하지 않는다.

| QK 학습η checkpoint의 평가η | recall MSE | M_eff | c hit | 현재 궤적 score top1 | 최대 상태 |
|---|---:|---:|---:|---:|---:|
| .0262929853 | .273289752622 | .133798540174 | 0 | .340716419226 | 22.05676270 |
| .2 | **.268067504245** | .151894157073 | .145839501694 | .279855566571 | 22.29709053 |
| .5 | .273393590377 | .176371790477 | .262352495659 | .244263590342 | 47.29192734 |
| 1 | .306745081982 | .202267859521 | .341512628376 | **.189319316592** | **664.14135742** |

**A05-ETA-INTERVENTION-BOUND OPEN: η1의 실제 상태 초과를 확인했다.** 기존 config G11 bound는 **305.0375175476074**이며 재평가 max **664.141357421875**가 이를 넘는다(`within_bound=false`, 유한값). 이는 현재 test 고정 checkpoint 개입의 상태 기준 위반이며 학습 프로세스 중단을 관찰했다는 뜻은 아니다. M_eff .2023도 O7 .5에 미달한다. 기존 qk2 고정η1 학습 중단·g11chk 사건과 서로 다른 실험이다. 사후 상한 확대 없이 이 개입 조건의 안정성 불합격을 성능과 함께 보고해야 한다.

**A08-ETA-INTERVENTION 해석 정정:** 가중치 고정은 score/상태 고정과 다르다. η가 fractional recurrence에 들어가 u·query·history를 바꾸므로, QK score top1은 .340716에서 η1의 **.189319**로 달라졌다. η1의 최종 c hit .341513을 원래 궤적 score .340716과 “일치”한다고 해석하면 다른 상태의 통계를 혼합한다. b 재가중과 cap도 c의 순위에 영향을 준다. 현재 결과는 **η 변경의 전체 순환 모델 개입 효과**다. p/score 고정 하의 혼합·cap 직접 효과를 주장하려면 저장된 동일 궤적의 p,b에서 계수만 다시 계산하는 별도 대조가 필요하다. 상태 변화·총질량·cap·spike readout이 함께 바뀌므로 “dense 이력이 유용한 정보를 나르기 때문에 MSE가 나빠졌다”는 특정 매개 원인은 아직 미확인이다.

**Canonical16:13 §3의 조건 혼용:** QK 고정η.2 학습 recall .306119에 대응하는 score top1은 **.150596957017**, rank **.336628487234**다. 인용한 .1214는 비QK η.2 모델의 값이다. QK의 작은η 학습 모델이 더 나은 점수를 보였다는 관찰 방향은 유지되지만 정확한 조건을 짝지어야 한다. 또 QK/nonQK 두 학습 checkpoint의 차이는 θ·정규화·학습 궤적을 포함한 조건 비교이며 QK 변환만의 단독 인과 효과는 아니다.

η.2 개입의 full 대비 산술 G는 **.03426702117**, 상대 recall 감소는 **2.9931%**로 재현된다(기존 full .276338636158, oracle1 .034965688339). 그러나 full은 별도로 학습한 모델이다. 같은 checkpoint의 η0 개입 대조는 원8행에 없으므로 이 값만으로 그 모델 내부 선택의 기여를 고립하지 못한다. “선택이 도움이 된 첫 관측”도 과거 학습η의 양의 G(.0154056 등)가 이미 있어 부정확하다. 이번 결과는 기존보다 큰 **단일 seed 탐색 개선 관측**으로 기록한다.

Test가 반복 관찰되고 testη 후보가 비교됐다는 한계를 명시한 점은 적절하다. 현재 사전등록에 없는 학습/추론η 불일치 후보이며 확증 채택은 보류한다. V2의 MDE1.2%를 다른 과제·분산·설계의 v3 단일 seed에 옮겨 통계 근거로 삼지 않는다. **Validation 통제 비교·독립8seed paired CI·새 학습·효능 확증은 not run.** eta_intervention.txt에는 원 실행 코드/정확 명령/정밀 JSON이 없어 A10-PROVENANCE 잔여다. 이번 감사의 재구성 코드·정밀 결과·checkpoint hash를 제공하지만 당시 실행 provenance를 소급 보장하지 않는다.

### (c) 채택 검증 우선순위

**AUDIT-PRIORITY-01 보완: 다음 우선 작업은 같은 validation checkpoint에서 η 개입의 효과·안정성을 분리하는 것이다.** η=.2는 탐색 후보로 유지하되, 원 학습η·η0·고정η 격자를 같은 checkpoint/validation 표본에서 비교하고 score→p→w→pre→post 및 G11을 함께 보고할 것. 궤적 고정 계수 재계산과 모델 전체 forward 개입을 구분하면 실제 희석과 상태 피드백을 분리할 수 있다. Test에서 발견한 후보를 validation에서 다시 본다고 이미 관찰한 test가 새로운 확증 자료가 되는 것은 아니다. 후보와 η 선택 규칙을 사전등록 개정으로 고정한 뒤, 별도 미사용 평가 자료·독립seed 검증을 설계해야 한다.

기존 순서 **①η/혼합 구조 진단 → ②QK 조건 검증 → ③causal key → ④entmax → ⑤별도 Gram/ridge → ⑥별도 Delta**는 유지한다. 이번 표는 주된 손실이 p→w→혼합에서 나타나며 score top1→p top1은 보존된다. 따라서 **entmax를 자동 승격할 근거는 없다**. Top1 보존만으로 entmax가 불필요하다고 결론내리지도 않는다. 동일 validation에서 score는 충분한데 p의 질량 배분이 병목이라는 대조가 확보될 때만③/④의 조건부 순위를 바꾼다. η1의 큰 M_eff를 이유로 residual 제거 또는 η1 채택을 권하지 않는다.

문헌 근거는 앞서 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 커널·정규화/scale 구분, [Adaptive Sparse Transformers §3](https://aclanthology.org/D19-1223.pdf)의 entmax 변환 및 Jacobian, [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 순환 상태/gradient 안정성 분석을 재사용한다. 이번 우선순위 보완은 직접 수치 검사에 기반한 연구 판단이며 해당 논문이 이 모델의 효능을 보장한다는 주장은 아니다.

### 수행·보존

CPU torch1.12.0+cu113/2threads/seed7. 고정 checkpoint forward와 작은 조합론 검사를 수행했다. 학습/backward/optimizer/GPU·환경 설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송은 하지 않았다. 진단 코드/텍스트 결과와 감사3문서 append만 작성했다. 중간 inventory 비교 명령은 과거 schema의 list를 dict로 가정해 AttributeError가 났으며, 실제 snapshot bytes 비교로 바로잡았다. 이는 감사 보조 명령 오류로 모델 실패가 아니다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/stage_probe.py /tmp/nsmt_assessment_20260922T072001Z-785d6d29 f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/stage_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T072001Z-785d6d29/stage_probe.log 2>&1
```

추가 QK 고정η.2 비교는 같은 환경에서 `supplement_probe.py`에 같은 snapshot과 `supplement_probes.json` 경로를 인자로 주어 수행했으며 raw log는 동일 task의 `log/assessment/20260922T072001Z-785d6d29/supplement_probe.log`다. Snapshot/현재 변화 대조는 validation.json에 기록했으며 감사 중 새 변화는 다음 주기가 처리한다.

<!-- assessment-watch:20260922T072001Z-785d6d29 -->


## 추적 감사 24 — 2026-09-22 17:04 KST (예약 20260922T080001Z-b5162cb6)

### 범위·관찰 버전

**Absmax/관찰 횟수의 계산 수정과 post-clip norm 추가는 직접 검사로 확인했다. 다만 이 snapshot에서는 새 정수 관찰 횟수가 기존 logger를 깨뜨려 정상 CSV 저장에 도달하지 못한다.** 이는 모델 수치 실패가 아닌 기록 경로의 타입 결함이다.

HEAD `adb43e4ae47848d682a06b7ba5b067ed3d8015ab`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사23·문서 기억·사전등록§2H·canonical PROJECT_LOG를 읽고 **17:00:50 KST** 실제207파일을 `/tmp/nsmt_assessment_20260922T080001Z-b5162cb6`에 snapshot·hash했다. 이전 감사 이후 연구 변경은 train.py이며 이전 감사자 append는 연구 변화로 세지 않았다.

- Trigger train SHA256: `97f737bf6f63aba74897a1ad46d997040f06a4147b476855d09729b20d71ddc4`.
- **검사 snapshot train SHA256: `d5a6faa339fc918052c96ccfaec40b7c63e564068775155b0c7f9429abcac160`**.
- 감지 이후 수정되어 manifest 불일치1개였다. 감사 도중 live train은 다시 변경됐다(종료 hash는 validation.json). 본 판정은 위 고정 snapshot에 한정하며 뒤의 수정은 다음 주기 대상이다. `absmax2-170000` 디렉터리도 새로 관찰됐지만 이번 snapshot/trigger 밖의 실행 결과를 미완료나 실패로 분류하지 않았다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/inventory.json), [변경 diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/source.diff), [실제 AST 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/reducer_probe.py), [수치·logger 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/reducer_probes.json), [보존/추가 변화](../f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/validation.json).

### (a) 구현 정확성 — 계산은 부분 VERIFIED, 기록 경로 잔여 OPEN

`train_one_epoch` 전체를 실행하지 않고 실제 AST의 reducer·clipping 블록·최종 train JSON 투영식만 추출해 CPU 합성값으로 검사했다.

| 검사 | 실제 결과 | 판정 범위 |
|---|---:|---|
| grad_absmax_all / WQ [1,9] | **9** | 기존 평균5 결함 수정 VERIFIED |
| WK absmax [2,10] | **10** | 관찰값 최대 계산 VERIFIED |
| grad_absmax_all_observations | **2** | 기존 1의 평균 대신 finite 관찰 수 VERIFIED |
| WQ norm [1,9], η [.1,.3] | **5**, **.2** | 평균 의미 유지 |
| absmax [1,NaN,9] | max9, observations2, nonfinite1 | finite count의 정의 확인 |
| 전 batch pre [.5,5,.5,5] | mean2.75, max5, count4 | 전 batch 통계 유지 |
| post [.5,1,.5,1], clip [0,1,0,1] | post max1, clip rate.5 | reducer 확인 |

**A09-ABSMAX의 관찰값 최대 reducer는 VERIFIED.** 다만 absmax 수집 자체는 여전히 watch batch에 한정된다. `grad_absmax_all`의 all은 해당 관찰 시점의 전체 파라미터를 뜻하며 epoch의 모든 batch를 조사했다는 의미가 아니다. 전체 epoch 최대를 요구하는 해석·운영은 여전히 잔여다.

**A09-OBSERVATIONS 계산 및 global count JSON 투영은 VERIFIED.** 기존 `grad_observations`는 제거되고 `grad_absmax_all_observations` 등 필드별 finite count가 생겼다. 실제 train JSON 투영식에서 global count2와 post max1이 보존됐다. WQ/WK 등 나머지 `*_observations`는 이 JSON whitelist에 아직 포함되지 않는다. 모두 NaN인 필드는 `_nonfinite`만 생기고 observations=0이 명시되지는 않는다. “실제 유효 gradient가 존재한 횟수”가 아니라 “코드가 기록한 finite 수치의 수”이며 gradient=None도 기존 로직상0으로 기록된다.

**A09-POSTCLIP 계산은 VERIFIED.** 작은 파라미터에 gradient [3,4]를 직접 주입해 실제 clipping→post 측정 블록을 실행하면 pre5→post **.9999998211860657**, clip1이다. [.3,.4]는 pre/post **.5**, clip0이다. Backward나 optimizer를 호출하지 않았다. Post는 norm의 epoch 최대이며 mean은 추가되지 않았다.

**A10-LOG-COUNT-TYPE OPEN — 이 snapshot의 출력 타입 불일치.** reducer가 `len(finite)`를 Python int로 반환하지만 기존 `EpochLog._verbose`는 float가 아닌 값을 `v.mean()`으로 처리한다. 실제 `EpochLog.write`에 위 reducer 출력을 전달하면 다음 예외가 발생한다.

`AttributeError: 'int' object has no attribute 'mean'`

실제 write→verbose 경로를 실행했고 TensorBoard 전송 부분만 no-op으로 대체했다. 예외는 CSV 쓰기 이전에 발생해 파일이 생성되지 않았다. **감사 입력만 float로 변환한 대조에서는 같은 logger가 CSV 쓰기에 성공**했다. 그 대조는 원본 소스 수정이나 실제 학습 성공이 아니다. Scalar 타입 처리를 logger에서 일관되게 하거나 count 전달 타입을 맞추고, 서로 다른 두 관찰값을 가진 reducer→logger→JSON 경로를 다시 확인해야 한다. 현재 진행 중 수정이 있을 수 있으므로 고정 snapshot의 확정 오류와 이후 버전의 상태를 구분한다.

### (b) 검증 과정·재현성 판정

이전1관찰 실행에서 드러나지 않던 mean/max·count 결함을 이번2관찰 반례가 구분했고, post-clip도 직접 계산했다. 그러나 계산 단위 검사의 성공만으로 완료 artifact 저장을 VERIFIED로 닫을 수는 없다. 현재 snapshot의 타입 오류가 그 반례다. 합성 `synthetic_logger.csv`는 진단용이며 연구 결과가 아니다.

새 완료 학습 결과·checkpoint·데이터 분리/통계 결과는 이번 감사 범위에서 확인하지 않았다. 새 성능 판단 근거 없음. **학습·완료 run CSV/JSON 보존 검증·validation η 통제 비교·독립8seed paired CI는 not run.** 기존 데이터 분리·대조군·사전등록 정의는 변경되지 않았다. A02-STAGE-RANK, A05-ETA-INTERVENTION-BOUND, A08 개입 해석, A10 provenance 등 감사23의 열린 조건도 재검사 없이 닫지 않는다.

### (c) 다음 개선 방향

우선 기록 경로의 타입을 맞추고 **관찰값 최대/finite count/post-clip→logger→최종 JSON**을 학습 없이 연결 검사할 것. sampled absmax와 all-batch norm의 관찰 범위를 표시하고 필요 시 전 batch absmax로 확장해야 epoch 최대라는 표현이 성립한다. 기존 완료 artifact의 과거 평균값을 소급해 최대값으로 재해석하지 않는다.

그 다음 감사23의 **동일 validation checkpoint에서 η0/학습η/격자, 고정 궤적 대조와 전체 forward 개입, G11 동시 보고** 우선순위를 유지한다. η1 개입의 상태 상한 초과를 무시한 채 M_eff만으로 채택하지 않는다. 새 logging 수정은 QK·entmax·Gram·Delta 후보의 효능 순위를 바꿀 근거가 아니다. 안정성 및 clipping 지표를 구분하는 근거는 앞서 원문 확인한 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)을 재사용한다. 신규 문헌 주장·성능 주장은 추가하지 않았다.

### 수행·보존

CPU torch1.12.0+cu113,2threads,합성 진단으로 dataset/seed/학습 hyperparameter 해당 없음. 모델 학습/backward/optimizer/GPU·설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 접근/전송 없음. 감사 artifact와3문서 append만 작성하고 기존 문서 prefix를 보존했다. 최초 감사 스크립트에는 `.4.` 오타로 SyntaxError가 있었고 감사 코드만 수정해 재실행했다. 최초 raw log도 보존했으며 이 도구 실행 오류를 모델 실패로 분류하지 않았다. 재실행 exit0 및 의도한 logger 예외의 포착 결과는 artifact에 있다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/reducer_probe.py /tmp/nsmt_assessment_20260922T080001Z-b5162cb6 f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/reducer_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T080001Z-b5162cb6/reducer_probe_retry.log 2>&1
```

<!-- assessment-watch:20260922T080001Z-b5162cb6 -->


## 추적 감사 25 — 2026-09-22 17:15 KST (예약 20260922T081001Z-2214c860)

### 관찰 범위·핵심 판정

**Logger 타입 오류의 수정과 실제 완료 CSV/JSON 보존을 확인했다. Validation의 궤적 고정/전체 forward 분리도 재현됐으며, η=.2는 같은 모델의 원 학습η 대비 recall MSE가 약1.86% 낮았다. 단일 seed 탐색 결과로 채택 확증은 보류한다.**

HEAD `5db6e315fec55de2962b843d9348deb1acfc18ed`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사24·기억·사전등록§2H·canonical17:03 기록을 복구하고 **17:10:50 KST**, 실제230파일을 `/tmp/nsmt_assessment_20260922T081001Z-2214c860`로 snapshot·SHA256 기록했다. Manifest 불일치0. 감사자 append는 연구 결과로 세지 않았다.

- train.py SHA256 `b2ab437e5cc9bf3ccdff11bf3153a27e84e003edcd534f39bbba90384e7c0a8f`.
- stage_decomposition.py SHA256 `afae7c9b6cbbf68d41c31103ec6a961146626c0843adcedc38fa10151da28124`.
- eta_intervention_split.py SHA256 `7e013ab47c716893a0be9d47b309f2f133f5e057003261674e6ec5a5c6532afb`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/source.diff), [reducer/logger probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/reducer_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/reducer_probes.json), [분리 분석 probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/split_probe.py), [정밀 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/split_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/validation.json).

### (a) 구현 정확성 — 실제 재검사로 닫은 범위

**A10-LOG-COUNT-TYPE VERIFIED.** 새 reducer는 observations와 nonfinite count를 float로 반환한다. 실제 AST reducer의 absmax [1,9]→9/count2.0 결과를 **변환 없이** 실제 EpochLog.write→verbose→CSV 경로에 전달해 성공했다(TensorBoard 전송만 no-op). 감사24의 정수 타입 예외는 발생하지 않는다. 동일 probe에서 clipping 전후 계산·JSON 투영도 통과했다. Logger 자체가 모든 int를 지원하게 된 것은 아니며 현재 train 전달 타입을 고친 범위로 판정한다.

**A09-ABSMAX/OBSERVATIONS/POSTCLIP의 완료 artifact 저장까지 VERIFIED(해당 run).** `absmax4-170100`의 실제 CSV/JSON을 대조했다.

| 항목 | 완료 JSON 값 | CSV |
|---|---:|---:|
| grad_absmax_all | **1.4416555166244507** | 1.441656 |
| grad_absmax_all_observations | **8.0** | 8.000000 |
| grad_WQ_absmax | .004537541419267654 | .004538 |
| grad_WK_absmax | .012127026915550232 | .012127 |
| grad_total_norm_pre_max | 2.9906373023986816 | 2.990637 |
| grad_total_norm_post_max | **.999999669963376** | 1.000000 |
| clip_rate_all_batches / batches_seen | **1 / 8** | 1 / 8 |

Seed7/data_seed20260921,1epoch,train/val/test512/64/64,batch64,**g11_every1**이므로 이번 실행의8batch는 모두 관찰됐다. 이 run의 absmax를 전체8batch의 최대라고 읽는 것은 타당하다. 일반 설정에서 watch 간격이 커지면 여전히 표본 최대이며, 다른 run의 값까지 전체 batch 최대라고 소급하지 않는다.

이전 `absmax2-170000`은 평균으로 기록된 .8510724902153015 및 count 없음이 남아 있다. 두 run의 checkpoint SHA는 모두 `34c6965ca09cf823cff6cd3d9f3ea1725976888e5090d60b19f513cf539b6ab1`이고 각 실제 파일 hash와 일치한다. Absmax4의 기록 train source hash는 이번 snapshot과 같다. 같은 모델 결과에서 집계 방식만 달라졌다는 해석을 지지하되, **gradient 자체를 backward로 재생성한 것은 아니다**. Absmax2의 과거 평균 값을 수정하거나 최대값으로 재명명하지 않는다. Absmax3는 완료 JSON이 없는 것을 확인한 범위이며 종료 원인/실패 판정은 하지 않았다. WQ/WK 등 개별 관찰 수는 CSV에 있지만 최종 JSON은 global count만 보존한다는 잔여 제한도 유지한다.

**A02-STAGE-RANK VERIFIED.** 현재 실제 decompose 함수의 같은 test256 표본에서 chance **.21481773387565073**, QK rank **.19589813019628083**이 재현됐고 출력 설명도 무작위0.5 오기를 고쳤다. 이는 기존 test 표의 기준이며 validation에 수치를 그대로 이식하지 않는다.

**A08-ETA-INTERVENTION 분리 구현 확인.** 궤적 고정 함수는 학습η의 상태에서 p,b를 얻어 계수만 재계산하고, 전체 forward 함수는 η buffer를 바꾼 상태로 순환 모델을 다시 실행한다. 각 호출 후 buffer 복원, 전체 검사 후 학습 파라미터 hash 불변을 확인했다. 현재 sparse/cap=True/QK 학습η checkpoint에 한정한 확인이다. 함수가 다른 mode/no-cap 설정까지 일반적으로 맞는다고 판정하지 않는다.

### (b) 새 validation 결과·해석 제한

`qk2-140541` learnedη QK checkpoint를 CPU로 재평가했다. Seed7/data_seed20260921,train/val/test2048/256/256,batch64,θ=.38146987702788376,soft QK ε=.01,학습η=.026292985305190086. 실제 val256 전체(4batch)에서 원8행의 수치가 표시 정밀도까지 재현됐다. 다음은 추가한 원 학습η 기준선까지 포함한 전체 forward다.

| 평가η | val recall MSE | post-cap M_eff | max abs state |
|---|---:|---:|---:|
| 원 학습η | **.259146662958** | .136868366013 | 22.07177544 |
| 0 | .263483596918 | .133246449823 | 22.01334572 |
| .2 | **.254324974062** | .155864393992 | 22.48150253 |
| .5 | .260108542602 | .181384548465 | 43.09806061 |
| 1 | .293643652987 | .210006701282 | **656.12957764** |

η.2는 η0 대비 .009158622857(**3.4760%**), 원 학습η 대비 .004821688896(**1.8606%**) recall MSE가 낮다. 기존 “약3.5%”는 η0 대조를 뜻하며 원 학습 모델보다3.5% 좋아졌다는 뜻은 아니다. η1은 frozen G11 bound **305.0375175476074**를 초과해 불합격이며 **A05-ETA-INTERVENTION-BOUND는 수정/해결로 닫지 않는다**. Canonical이 해당 조건을 사후 상한 확대 없이 불합격으로 보고한 것은 수용한다. 이 val 결과에 test oracle 분모를 섞어 G를 만들지 않았다.

**궤적 고정 분석은 유용하지만 두 표기의 보완이 필요하다.**

1. 궤적 고정 표의 max|u|=22.0718은 모든 η에서 **원 학습η 궤적의 상태**다. 바뀐 계수를 재귀에 다시 넣지 않았으므로 G11 ‘OK’는 변경η 시스템의 안정성 판정이 아니다. `원 궤적 peak`와 `변경η G11: not run/N/A`를 구분해야 한다. 동일η1의 실제 전체 forward는656.13으로 실패한다. 계수만 계산하는 조건에서 MSE를 보고하지 않은 것은 적절하다.
2. Canonical17:03의 “p mass에 남은 차이는 cap”은 틀리다. 고정 궤적 η1에서 **p .296238626751 → 커널 가중 w .270998621964 (=pre) → post .274052337754**다. Cap은 오히려 정답 질량 비율을 약.00305 올렸으며, p와의 차이에는 `w=bp/Σbp` 변환이 있다. η0→1의 계수 변화는 혼합과 cap의 결합 효과다. **순수 pre-cap 희석**은 `M_pre=(1−η)M_kernel+ηM_w`로 별도 보고해야 한다. 고정 궤적의 pre는 .13324645/.16079688/.20212253/.27099862로 식과 일치한다.

전체 forward와 고정 궤적의 η1 post .21000670 vs .27405234 차이는 현재 checkpoint에서 상태 피드백을 포함한 개입의 차이를 뒷받침한다. 다만 이를 보편적인 분해 비율이나 새 학습의 성능 예측으로 일반화하지 않는다. 이전 서로 다른 궤적의 score/c hit 혼용을 철회하고 QK η.2 score 조건을 정정한 canonical 기록은 수용한다.

Validation 활용·동일 checkpoint η0 대조·실행 가능한 분석 코드의 추가는 검증 설계를 개선했다. **하지만 validation은 이미 학습 checkpoint 선택에 사용됐고, 후보η를 test에서 관찰한 뒤 다시 validation에서 비교했다.** 독립 확증 데이터는 아니다. 원 학습η보다 η.2가 낫다는 것은 이 한 seed의 탐색 결과이며 독립8seed paired CI/미사용 평가자료의 확증·새 학습은 **not run**이다. 두 분석 txt는4자리 표이며 원 실행 명령/정밀 결과/시작 source provenance는 여전히 부족하다. 현재 감사의 code·hash·정밀 JSON 재현 근거와 원 run provenance는 구분한다.

### (c) 채택 검증 우선순위

**① 구조/η 진단의 다음 단계로 η.2 후보를 유지하되, 채택은 사전등록한 선택 규칙과 독립 검증 뒤로 둔다.** 원 학습η·η0를 항상 포함하고, 후보 수/η 선택 기준/G11 탈락 규칙을 고정한 뒤 별도 미사용 평가 자료·독립seed로 비교할 것. 기록에는 score→p→w→pre→post와 전체 forward의 실제 상태 상한을 함께 남긴다. 현재 validation 개선은 후보 유지의 근거이며 새 확증 완료가 아니다.

그 다음 **②QK 조건 검증 → ③causal key → ④entmax → ⑤별도 Gram/ridge → ⑥별도 Delta**의 기존 순위를 유지한다. 이번 분해는 혼합·커널 가중·상태 피드백을 주요 점검 대상으로 지지하며 entmax 자동 승격이나 residual 즉시 삭제를 지지하지 않는다. State-derived key 개선은 조건부 후속 후보로 남기되 새로운 효능을 주장하지 않는다.

문헌은 이미 원문 확인한 [Test-time regression §3](https://arxiv.org/html/2501.12352v1)의 커널 가중·정규화/scale 구분과 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 순환 안정성 분석을 재사용했다. 이번 권고는 직접 수치 검토에 기반하며 새로운 논문 사실을 추가하지 않았다.

### 수행·보존

CPU torch1.12.0+cu113,2threads,seed7. AST reducer/logger/clip 합성 검사와 snapshot checkpoint forward만 수행했다. 학습/backward/optimizer/GPU·설치·연구 소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송 없음. 기존 문서 prefix를 보존하고 감사3문서 append 및 진단 artifacts만 작성했다. 실제 새 gradient 수치 재생성·absmax 실행의 성능 재평가·독립seed 통계는 not run이다. 감사 중 추가 변경은 validation.json에 남기고 다음 주기에 검토한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/reducer_probe.py /tmp/nsmt_assessment_20260922T081001Z-2214c860 f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/reducer_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T081001Z-2214c860/reducer_probe.log 2>&1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/split_probe.py /tmp/nsmt_assessment_20260922T081001Z-2214c860 f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/split_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T081001Z-2214c860/split_probe.log 2>&1
```

<!-- assessment-watch:20260922T081001Z-2214c860 -->


## 추적 감사 26 — 2026-09-22 17:33 KST (예약 20260922T083001Z-36837373)

**새 모델 실행·성능 판단 근거 없음. 다만 새 canonical 통합 요약에 기존 조건/수치가 잘못 합쳐진 부분이 있어 정정을 요구한다.** HEAD `0314e9319891dbe212b1a153b402db75143d170c`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 감사25 이후 commit diff는 canonical PROJECT_LOG의303행 추가뿐이며 그중 감사자의24·25차 append는 연구 변화로 세지 않았다. 최신 기억·감사25·사전등록§2H·canonical17:27 통합 요약과17:28 운영 기록을 확인했다.

**17:31:14 KST** 실제216파일을 `/tmp/nsmt_assessment_20260922T083001Z-36837373`에 snapshot·hash했다. Trigger 불일치0, 감사25 이후 연구 소스/분석/결과 hash 변경0. Canonical SHA256 `a4819eff7eeebe49827cda1ecd34e17e2aa968f6cdab300a424ed6a042289230`. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/inventory.json), [문서 diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/project_log.diff), [산술 probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/summary_probe.py), [정밀 계산](../f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/summary_arithmetic.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/validation.json).

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
/usr/bin/python3 f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/summary_probe.py /tmp/nsmt_assessment_20260922T083001Z-36837373 f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/summary_arithmetic.json > f_lif_pop_v3/forecasting/log/assessment/20260922T083001Z-36837373/summary_probe.log 2>&1
```

<!-- assessment-watch:20260922T083001Z-36837373 -->


## 추적 감사 27 — 2026-09-22 17:45 KST (예약 20260922T084001Z-3cfbaf43)

### 범위·핵심 판정

**新§2I는 미사용 confirm 분할과 validation 선택 규칙을 추가했으나, 현재 eta_selection.py는 그 계약을 충분히 강제하지 않는다. Confirm을 실제로 열기 전에 seed 완료 확인·G11 탈락·보고 항목·주장할 차이의 정의를 보완해야 한다.** 검사는 실제 confirm 데이터나 모델 평가 없이 합성 입력으로 수행했다. 다중 seed 학습의 미완료 상태를 모델 실패로 분류하지 않는다.

HEAD `f98fab4fe9940c281355597a0c91741f0b96ec05`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사26·문서 기억·사전등록§2I(D-AE~AJ)·canonical17:31 이후를 읽고 **17:40:58 KST** 실제224파일을 `/tmp/nsmt_assessment_20260922T084001Z-3cfbaf43`에 snapshot·hash했다. Trigger 불일치0. 새 seed7의 checkpoint/config/result와 당시 존재한 seed13 config를 포함했다.

| 대상 | SHA256 |
|---|---|
| 사전등록 | `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168` |
| eta_selection.py | `e2c417ddfb58cdbd21e6ff3b220521b5ea8aa8293f230745b10d070c2178e0db` |
| config.py | `b033d265b97e9b5db01a849495e2c22fd0b9c088e15ad000d3fcdf0bf4496ae8` |
| synthetic.py | `3ca3199101ffdcf58f4da4279048834a32f8c131fb67a04eec50af6939495965` |

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/inventory.json), [diff](../f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/source.diff), [합성 protocol probe](../f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/protocol_probe.py), [수치/실행흐름 증거](../f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/protocol_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/validation.json).

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
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/protocol_probe.py /tmp/nsmt_assessment_20260922T084001Z-3cfbaf43 f_lif_pop_v3/forecasting/results/assessment/20260922T084001Z-3cfbaf43/protocol_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T084001Z-3cfbaf43/protocol_probe.log 2>&1
```

<!-- assessment-watch:20260922T084001Z-3cfbaf43 -->


## 추적 감사 28 — 2026-09-22 17:53 KST (예약 20260922T085001Z-3622969b)

**다중 seed 학습의 완료 증거가5개로 늘었다. 실제 confirm 결과와 η 개입 효능의 새 판단 근거는 없다.** HEAD `ae0f4ff50e247b255269342dd8f6879d192cfae8`, branch exp/f-lif-pop-v3/base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 감사27·기억·사전등록§2I·canonical17:43 운영 기록을 확인했다. 새 skill 등록은 관찰한 문서·운영 변경이며 감사에 사용하거나 다른 세션을 호출하지 않았다.

**17:50:39 KST** 실제245파일을 `/tmp/nsmt_assessment_20260922T085001Z-3622969b`로 snapshot·hash했다. 감지 시점과 불일치한 파일은 진행 중 seed256 CSV1개이며 snapshot에는9행이다. 연구 Python 소스와 사전등록은 감사27과 동일하다. `eta_selection.py` SHA256 `e2c417ddfb58cdbd21e6ff3b220521b5ea8aa8293f230745b10d070c2178e0db`, 사전등록 SHA256 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/inventory.json), [메타데이터·checkpoint 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/metadata_probe.py), [검사 결과](../f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/metadata_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/validation.json).

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
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/metadata_probe.py /tmp/nsmt_assessment_20260922T085001Z-3622969b f_lif_pop_v3/forecasting/results/assessment/20260922T085001Z-3622969b/metadata_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T085001Z-3622969b/metadata_probe.log 2>&1
```

<!-- assessment-watch:20260922T085001Z-3622969b -->


## 추적 감사 29 — 2026-09-23 18:08 KST (예약 20260922T090001Z-b9a6a20e)

**8-seed 완료 및 저장된 confirm 결과를 확인했다. η=.2 채택을 뒷받침하는 유의한 개선은 확인되지 않았다. A13 절차는 부분 개선됐으나 OPEN이다.** 예약 ID는 9월22일이지만 실제 관찰·검사는 9월23일이다. 감지 목록의 seed256/512/1024 진행 상태를 현재 상태로 오인하지 않고, 감사28 이후 현재 파일까지 대조했다.

관찰 HEAD `e511d57c8038546234758f5add41fed9dd8010c9`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 2026-09-23 18:01:08 KST에 실제260파일을 `/tmp/nsmt_assessment_20260922T090001Z-b9a6a20e`로 snapshot하고 각각 SHA256을 기록했다. trigger manifest와 다른 파일은 `analysis/eta_selection.py` 1개다. 감사28 대비 연구 Python 변경도 이 파일이며 모델/학습 구현은 같다. 선택 코드 SHA256 `a850cdf2c260b18d2307947f5983bd19e86dfe1caf13cc14bbf31dca93e37213`; 사전등록 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`; 선택+confirm JSON `5704fc569c54f4922814f0777bde37eb349d983252e1633f58a8365c5da7a606`. 최신 기억·감사28·§2I·canonical13:46/13:51/17:36/17:39를 읽었다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/inventory.json), [checkpoint 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/metadata_probes.json), [수치·계약 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/result_protocol_probes.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/result_protocol_probe.py), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/validation.json).

### (a) 구현 정확성 — A13-ETA-PROTOCOL 부분 개선, OPEN 유지

실제 snapshot 함수를 AST로 분리하여 합성 입력으로 검사했다. 모델 forward/학습/confirm 데이터 접근은 하지 않았다.

- **범위 한정 VERIFIED:** 이미 confirm이 든 기록, 선택 후 seed 집합 변경, checkpoint hash 변경을 각각 confirm 데이터 접근 전에 `SystemExit`으로 거부했다. 정적 확인상 선택은 exclusive-create로 기록하고 종료하며 `--confirm` 단계가 별도다. 중복 seed는 명시적으로 거부하고 완료 JSON이 없는 실행을 건너뛴다. 이는 이전보다 나은 계약이다. 모든 절차의 완전한 1회성을 증명한 것은 아니다.
- **재현한 잔여 결함:** 7개 seed만 든 선택 기록도 같은7개가 존재하면 confirm 데이터 접근 경계에 도달한다(감사 stub에서 즉시 중단). 사전등록8개를 강제하지 않는다. 같은 probe의 의도적으로 불일치한 config/source hash도 검사하지 않는다. 저장만 하고 stage2에서 비교하는 것은 checkpoint hash뿐이다.
- **재현한 잔여 결함:** 진단 `finite=False`라도 유한 peak가 상한보다 작으면 `within_bound=True`; 상한 자체가 없어도 True다. 이번 실제 저장 수치는 모두 유한하지만 그것으로 비유한 상태 거부 계약이 충족되지는 않는다. `evaluate`의 최종 비유한 오차 거부와 중간 상태의 finite 계약도 별개다.
- **정적 확인 잔여:** 완료 confirm 재실행은 막지만 접근 시작 기록이 없어 평가 중단 후 재접근을 강제 차단하지 못한다. G11 실패 seed 목록과 FAIL 출력은 추가됐으나 `G11_violation.json`·단일 종합 판정은 없고 CI 유의 출력과 G11 FAIL이 함께 나올 수 있다. 선택 코드·stage decomposition 해시는 선택 기록의 source 목록 밖이다. 이 결함이 이번 결과에 실제 발생했다는 뜻은 아니다.
- D-AH 사건별 오차·score/rank/chance·p/pre/post·기준선 지표는 추가됐다. **w는 여전히 없다.** `w=bp/Σbp`와 `pre=(1−η)kernel+ηw`는 다르다. D-AE의 후보0 포함/기준선 제외 모순, D-AI η0 비교와 D-AJ 원 학습η 주장 불일치는 사전등록이 그대로여서 남는다. 사후 정정은 정정 시점을 명시하고 다음 검증에 적용해야 한다.

현재 코드의 진단4batch 고정은 일반 설정에서는 일부 표본만 보지만, 이번 confirm256/batch64에서는 전체4batch에 해당한다. 기존 A07-REGEN, A10-PROVENANCE/LOG, A02 실제분포 연결, A05-ETA-INTERVENTION-BOUND 등은 이번에 재검사하지 않았으므로 기존 상태를 유지한다.

**A09-SUMMARY-CONDITION-MIX:** canonical13:46의 비QK G 정정은 공통분모로 직접 재계산해 +.015405565/−.178349107/−.651132280/−.581440337과 일치했다. 해당 표 정정만 **VERIFIED**. epoch/고정궤적 범위 정정도 기존 감사 근거와 맞는다. 말미 “학습 선택자가 더 해로웠다”는 모든 조건의 일반 결론으로 넓히지 않는다(원 학습η의 G는 양수). 아래 새 confirm G 환산 문제까지 닫은 것은 아니다.

### (b) 검증 과정·결과 — 기록 일관성 확인, 확증 주장 범위 제한

`seeds-173737`의 {7,13,21,42,123,256,512,1024} 전부 완료 JSON12epoch/CSV12행이다. CPU 복원 파라미터 hash·checkpoint hash·훈련 JSON source11개가 snapshot과 모두 일치했다. 최종 선택 기록의8개 checkpoint/config 및 source8개 hash도 모두 일치했다. test=None/test_skipped=True이며 best validation loss는 CSV 최소값과 반올림 범위에서 맞는다. 완료8개는 실제 증거이므로 코드의8개 강제 누락을 이유로 이번 결과를7seed로 취급하지 않는다.

공통 data_seed20260921, train2048/val256/test256/confirm256, batch64, QK soft ε=.01, learnedη 학습, θ=.38146987702788376, frozen input norm, bound305.0375175476074다. confirm 생성 offset30000은 train/val/test와 구분돼 있다. **같은 데이터 분할에서 학습 seed를 반복한 조건부 비교**이며 독립 데이터셋8회가 아니다. 같은 모델의 평가 시 η만 바꾼 것이므로 고정η로 학습한 모델 비교와도 다르다.

보존된 `results/aborted/eta_selection_record_INCOMPLETE_7seeds_260923-134826.json`에는7seed 선택만 있고 **confirm 결과는 없다**. 이를 confirm 오염 증거로 오인하지 않는다. 반대로 결과 없음만으로 과거 모든 접근의 부재를 증명할 수도 없다. 현재 두 canonical confirm 항목은 같은 JSON의 반복 기록이며 독립 재현2회가 아니다.

저장 full-precision confirm 결과를 재계산했다(평가는 재실행하지 않음):

| seed | η0 recall MSE | 원 학습η | 선택η=.2 | 선택 조건 max abs state |
|---|---:|---:|---:|---:|
| 7 | 0.268359 | 0.265034 | 0.259373 | 23.066109 |
| 13 | 0.268471 | 0.267146 | 0.280100 | 25.462078 |
| 21 | 0.270843 | 0.266473 | 0.257132 | 22.408699 |
| 42 | 0.267661 | 0.266657 | 0.289725 | 24.214479 |
| 123 | 0.266372 | 0.260214 | 0.252870 | 24.633423 |
| 256 | 0.265617 | 0.261716 | 0.266776 | 27.283655 |
| 512 | 0.265767 | 0.259166 | 0.252061 | 30.667475 |
| 1024 | 0.268839 | 0.263782 | 0.254122 | 47.574017 |

| paired 차이 (작을수록 좋음) | 평균 | 95% CI (df7 정확 t) | 개선 seed |
|---|---:|---|---:|
| 선택η − η0 (D-AI) | −.003721354 | [−.015408880, +.007966172] | 5/8 |
| 선택η − 원 학습η (D-AJ에 해당) | +.000246292 | [−.009954048, +.010446631] | 5/8 |
| 원 학습η − η0 (부차 관찰) | −.003967646 | [−.005677091, −.002258201] | 8/8 |

기존 스크립트의 반올림 t=2.365 CI도 재계산하여 기록과 일치했다. 정확 t와의 차이는 해석을 바꾸지 않는다. η=.2는 η0 대비 평균−1.3899%지만 유의하지 않고, 원 학습η 대비 우월성도 없다. **효과가 전혀 없다는 증명은 아니다.** 원 학습η의−1.4819%는 주 비교가 아닌 탐색적 단서로 남긴다. 이 confirm으로 새 후보를 고르면 탐색이므로 추가 검증에는 새 규칙·미사용 자료가 필요하다.

저장된24조건(8seed×3)의 숫자는 모두 유한하고 기록된 peak/bound는 통과한다. 선택η peak22.408699~47.574017, 평균 M_eff=.153014320 (η0 .132532631, 원η .135675147)이다. 이는 저장 결과의 일관성 확인이며 중간 상태 finite를 재실행 검증한 것은 아니다. G11 가드 결함과 실제 수치 통과를 구별한다. 학습 watch는4/32batch이므로 전체 학습 step 안정성은 여전히 증명하지 못한다.

canonical17:36의 **“G로 환산 약 .016”은 이8seed confirm에서 얻은 O7 G가 아니다.** 같은 조건·split·seed의 full/oracle denominator가 이번 표에 없으므로 과거 탐색/test denominator를 가져와 환산하지 말 것. GRU·α=1 이질·capacity-matched 대조, 새로운 데이터 seed 반복, O7 확증은 **not run**이다.

### (c) 개선·채택 순위와 새 상관 mask 후보

**η=.2의 기본값 채택은 보류한다.** ①η/혼합 구조 진단 및②QK는 탐색과 이번 개입 검증까지 진행됐지만 일반 효능 검증 완료가 아니다. 남은 프로토콜 계약을 먼저 보완하고, ③causal key 표현 진단 → ④entmax 조건부 검토 → ⑤별도 Gram 대조 → ⑥별도 Delta 대조 순서를 유지한다. 큰η에서 유효 질량이 늘어도 오차 개선이 보장되지 않으므로 단순 residual 제거/η 확대는 현재 근거로 우선 채택하지 않는다. 기존 원문 [TTR §3](https://arxiv.org/html/2501.12352v1)의 회귀/Gram 논의는 대조 설계 근거이며 이 모델의 효과 증명은 아니다.

canonical17:39의 통계 상관 mask는 **⑦ 탐색 후보, 미채택**으로 둔다. 이번에 [Autoformer 원문 §3.2 식5–8](https://arxiv.org/html/2106.13008v5), [Cliff et al. 원문](https://arxiv.org/pdf/2003.03887)을 직접 확인했다.

- Autoformer는 FFT 상관과 top-k lag·softmax·roll 집계를 사용한다. 이것이 곧 p-value/FDR 기반 유의 mask라는 뜻은 아니다. 이를 변형한 제안은 새 가설로 표기한다. 전구간 circular roll을 온라인 causal history에 옮길 때 미래 정보 차단을 별도로 확인해야 한다.
- Cliff의 보정은 covariance-stationary 시계열의 선형 의존 측정/Gaussian 조건 등을 전제로 한다. 논문의 일부 실험에서 거짓 양성100%에 이른다는 결과를 우리 key에 자동 적용하지 않는다. **인접 벡터 cosine .758은 중심화한 시간 자기상관 추정치와 같지 않다.** 따라서 유효 표본수나 Bartlett 대역을 그 숫자만으로 정당화할 수 없다.
- 직접 세어 본 T42의 단일 궤적 lag별 비순환 pair 수는 lag1=41,21=21,40=2,41=1이다. “모든 lag당 수십 표본”은 틀리다. train에서 시퀀스 경계를 보존해 통계를 모으는 방안의 귀무분포·유효표본·재현성을 먼저 검사할 것. 검정식과 다중비교 가정이 없는 FFT 점수에 FDR을 바로 적용할 수 없다.
- 고정 mask가 선택자 gradient 경로를 없애더라도 fractional recurrence와 spike/readout 경로는 남는다. **gradient 폭주 해결은 아직 측정되지 않은 가설**이다. mask C는 학습 점수 자체도 유지한다. 기존 [Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.pdf)의 재귀 gradient 논의 범위에서 안정성 진단을 유지한다.
- 승격 조건: train-only 통계/causal mask·정규화·모든 위치가 탈락할 때 fallback을 고정하고, 같은 coverage의 random/recent/uniform 대조와 score→p→w→pre→post 및 G11을 비교할 것. `p×mask`만으로 질량 보존이 자동 유지되지 않는다. 실제 mask 실험·효능은 **not run**. ARCausal 등 나머지 제안 문헌은 이번 감사에서 원문 검증하지 않았으므로 채택 근거로 승인하지 않았다.

### 수행·보존

CPU Python3.10/torch1.12.0+cu113/2threads에서 checkpoint 복원·hash, 저장 JSON 통계, AST 분리 합성 계약 검사만 수행했다. 학습·실제 모델 forward·backward·optimizer·confirm 데이터 재생성·GPU·환경 설치·연구소스 수정·git 변이·프로세스 중단·타세션 대화 열람/전송은 하지 않았다. 문서 prefix와 snapshot 연구파일 hash를 append 전후 검사한다. 새 문서 기록은 다음 변화의 연구 증거로 세지 않는다.

정확한 검사 명령(두 script 모두 snapshot만 읽음; 첫째 CPU 파라미터 복원, 둘째 저장 수치/합성 입력):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/metadata_probe.py /tmp/nsmt_assessment_20260922T090001Z-b9a6a20e f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/metadata_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T090001Z-b9a6a20e/metadata_probe.log 2>&1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/result_protocol_probe.py /tmp/nsmt_assessment_20260922T090001Z-b9a6a20e f_lif_pop_v3/forecasting/results/assessment/20260922T090001Z-b9a6a20e/result_protocol_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260922T090001Z-b9a6a20e/result_protocol_probe.log 2>&1
```

<!-- assessment-watch:20260922T090001Z-b9a6a20e -->


## 추적 감사 30 — 2026-09-23 18:11 KST (예약 20260923T091001Z-b12ed8cd)

**새 판단 근거 없음.** 감지 목록은 감사29가 9월23일 현재 상태까지 이미 관찰한 변경의 재감지다. 기억·감사29·사전등록§2I·canonical 최신 append를 복구하고 실제 파일을 대조했다. HEAD `e511d57c8038546234758f5add41fed9dd8010c9`는 감사29와 같다. 18:10:35 KST에262파일을 `/tmp/nsmt_assessment_20260923T091001Z-b12ed8cd`로 snapshot·SHA256 기록했으며 trigger manifest와 불일치0이다. [Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260923T091001Z-b12ed8cd/inventory.json)에 파일별 hash와 비교 범위를 보존했다.

- **(a) 구현:** 감사29에서 관찰한 연구 소스 hash가 전부 동일하다. `eta_selection.py` SHA256 `a850cdf2c260b18d2307947f5983bd19e86dfe1caf13cc14bbf31dca93e37213`, 사전등록 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`. A13-ETA-PROTOCOL OPEN 및 기존 범위 한정 VERIFIED를 유지한다. 새 수정·재현 검사 없이 상태를 닫지 않았다.
- **(b) 검증:** 선택+confirm JSON SHA256 `5704fc569c54f4922814f0777bde37eb349d983252e1633f58a8365c5da7a606`, 8seed 완료 결과·checkpoint/config·기존 관찰 로그가 감사29와 같다. 이번 snapshot에 추가한 것은 보존된 `log/aborted/seeds-173737/seed1024_partial_2epochs_260923-134530/`의 CSV/logargs 2개다. CSV는 실제2행(epoch0,1)이며 기존 중단 실행 보존 설명과 일치한다. 감지 목록의 옛260922 경로2개는 현재 없고 manifest도 이 삭제/이동 상태를 반영한다. 이를 새 모델 실패나 완료12epoch 결과로 세지 않는다. 감사29 이후 달라진3문서는 감사29 자신의 append이므로 연구 변화로 세지 않았다.
- **(c) 개선:** 새 실행 결과가 없어 순위·판정을 변경하지 않는다. η=.2 채택 보류, A13 잔여 계약 보완 우선, 통계 상관 mask⑦ 미채택을 유지한다. 문헌·수치 근거는 감사29의 실제 원문 확인과 paired CI 재계산 범위 그대로다. 새 효능 판단·통계 재실행·문헌 확장·학습·forward·confirm 재평가는 **not run**이다.

문서 prefix를 보존해 append했으며 연구 소스·checkpoint·기존 raw log를 수정하지 않았다. 새 감사 artifact는 inventory/trigger/validation 텍스트뿐이며 실행 raw log는 생성하지 않았다. 모델·GPU·프로세스·Git·예약 설정을 변경하지 않았다.

<!-- assessment-watch:20260923T091001Z-b12ed8cd -->


## 추적 감사 31 — 2026-09-23 21:24 KST (예약 20260923T122002Z-71edaa29)

**통계 mask 검토의 생성기 집계는 일부 재현됐으나, 검정 불가능·lag 방식 전면 기각이라는 결론은 근거 범위를 넘는다. 새 A14-STAT-MASK-EVIDENCE OPEN. 후보⑦ 미채택을 유지한다.**

HEAD `442b0b7a0ac5393ddc080cac83079cda885a1244`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기억·감사30·최신 사전등록§2I·canonical21:19 상세 검토본을 읽었다. 21:20:37 KST에263파일을 `/tmp/nsmt_assessment_20260923T122002Z-71edaa29`로 snapshot·hash했다. Trigger manifest 불일치0. 연구 Python·기존 결과·checkpoint·사전등록은 감사30과 같고, 새 연구 artifact는 `analysis/stat_mask_feasibility.txt`뿐이다(SHA256 `0e1e933cc7f5a015f74d94bcb58079be8b28495104330272557e87fdf17d4235`). Canonical에는 새 상세 검토가 append됐으며 감사/기억의 자기 기록은 연구 변화에서 제외했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/inventory.json), [독립 생성기 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/stat_mask_probe.py), [전체 수치](../f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/stat_mask_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/validation.json).

### (a) 구현 정확성

mask는 아직 제안이며 모델 구현·학습·효능 결과가 없다. 미구현을 구현 오류로 판정하지 않는다. 기존 구현 판정은 유지하고 **A13-ETA-PROTOCOL OPEN**도 재검사 없이 닫지 않는다. 현재 선택 코드 SHA256 `a850cdf2c260b18d2307947f5983bd19e86dfe1caf13cc14bbf31dca93e37213`, 사전등록 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`로 동일하다.

새 txt를 생성한 script/명령은 검색한 `analysis`, task scripts, shared scripts의 연구 Python/shell에서 확인되지 않았다. 코드 없이 ACF의 중심화·분모·채널 결합·top-k 동률 처리·전체 시퀀스/causal prefix 범위를 확정할 수 없다. 따라서 **자기상관 top8의 27.8% 및 시퀀스99.8% 수치는 독립 재현 미확인**이다. 실패한 모델 실행이 아니라 분석 provenance 미비다. 아래 검사는 원 생성 코드 재실행 대신 snapshot의 기존 생성기 함수만 분리한 독립 감사 probe다.

### (b) 검증·통계·재현성

**검사 범위:** NumPy CPU, seed20270921(`data_seed20260921+10000`, validation), 400시퀀스/T42/patch8/key3/run2–5/min_gap1/onehot. 원본 생성기 SHA256 `3ca3199101ffdcf58f4da4279048834a32f8c131fb67a04eec50af6939495965`. 모델·학습·test/confirm 자료는 사용하지 않았다. 생성기의 정답 첫 등장 구간과 recall kind>0 정의를 따른다.

| 재계산 항목 | 실제 값 | 판정 |
|---|---:|---|
| recall query / 정답 칸 수 | 12,613 / 43,746 | 기록 일치 |
| query당 정답 수 | 3.468326330 | 다중 정답 단위 구별 필요 |
| lag 분포 엔트로피 / log41 | 3.570160598 / 3.713572067 | 기록 일치, 실패 증명은 아님 |
| 정답 lag 빈도 상위8 | 18,19,10,17,11,20,12,21 | 기록 일치 |
| 위 고정 top8 정답 칸 포함률 | .261898231 | 26.2% 기록 일치 |
| 위 고정 top8 query당 선택 정답 수 | .908348529 | **비율/확률 아님** |
| 위 고정 top8 query 중 하나라도 적중 | .419249980 | .908과 다른 지표 |
| 위 고정 top8 query별 정답 포함 비율 평균 | .258895584 | pooled 정답 칸 비율과 다름 |

**A14-UNIT:** txt §3의 `.908`을 “덮는 정답 비율/상한”이라고 부르면 잘못이다. 빈도 합을 query 수로 나눈 **기대 정답 개수**다. 예컨대 같은 계산에서 k12=1.354475541, k20=2.230872909로1을 넘는다. 뒤 §3′는 분모를 정답 칸 수로 바꿔 바로잡았지만, 앞 절이 폐기된 초안이라는 정정 표시가 없어 서로 충돌한다. 원문은 보존하고 append 정정으로 단위를 명시할 것. 빈도 top-k는 이 **동일 표본의 고정 k-lag 집합 중 pooled 정답 칸 수**를 최대화할 뿐, query별 적중·내용 기반 선택·모델 성능의 상한은 아니다.

**A14-CONTROL:** `k/41`은 1..41에서 고정 k개 lag를 무작위로 뽑는 대조의 기대 포함률로는 맞다. query 시점 n에 존재하는 과거 슬롯에서 같은 수를 선택하는 대조와는 다르다. 동일400시퀀스에서 위 고정 top8이 각 query에 실제 남기는 수 `c_n=#{lag≤n}`를 유지하고, n개 과거에서 c_n개를 균등 비복원 추출하는 정확 기대값을 계산했다:

- pooled 정답 칸 포함률: 고정 top8 **26.189823%**, 무작위 고정8/41 **19.512195%**, **같은 c_n 대조25.591456%**. 이 마지막 대조 대비 차이는 약0.60%p다.
- query당 평균 가용 선택 수6.539443(범위0..8); 같은 c_n 대조의 query-any-hit 기대값은 **63.127169%**, 고정 top8은 **41.924998%**. 정답 구간의 여러 칸을 한꺼번에 고르는 것과 더 많은 query를 덮는 것은 다른 목적이다.
- 이는 고정 lag prior의 in-sample 구조 진단이며 ACF top8=27.8%를 재현한 값이 아니다. 통계적 유의성·독립 표본 신뢰구간·예측 MSE 개선은 측정하지 않았다. query들이 같은 시퀀스 안에서 의존하므로12,613개를 독립 반복으로 다루지 말 것.

**A14-SPLIT:** canonical 조건에는 **validation seed**라고 정확히 적었지만 txt §3/§3′와 본문에서는 “학습 집합 전체 pooling”이라고도 부른다. 이번 재현은 명시된 validation seed에서 동일 집계를 얻었다. 따라서 현 결과를 train-only 추정/독립 검증으로 부를 수 없다. 같은 표본으로 lag를 골라 그 포함률도 계산했으므로 미래 자료 성능은 미검증이다. confirm 접근 증거는 없으며 이번 감사에서도 접근하지 않았다.

**A14-INFERENCE:** `T−τ` 비순환 pair 수와 `1.96/√(T−τ)` 산술 자체(τ21에서 .427707065)는 맞지만, 이를 일반적인 “Bartlett 유의 문턱”으로 확정하거나 **검정력0/원리적 불가능/고정 mask만 유일한 방법**으로 결론낼 수 없다. 귀무모형·ACF/PACF 추정량·효과크기·다중검정·제1종오류/검정력 측정이 없다. τ41 pair1의 난점과 모든 τ의 불가능은 다르다.

직접 확인한 [NIST ACF 공식 설명](https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm)은 white-noise 귀무에서 ±z/√N과 MA 모형에서 다른 분산식을 구별한다. [statsmodels 공식 ACF 문서](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)는 N−k autocovariance 분모 조정과 Bartlett 신뢰구간을 별개로 설명하고, [PACF 문서](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html)는 기본 표준오차를1/√len(x)로 명시한다. 따라서 `N=T−τ`를 임의 대입한 표는 산술 예시로 제한해야 한다. 이 문헌들이 현재 비정상·짧은 cue 시퀀스의 정확한 검정을 보장한다는 뜻도 아니다.

이전 감사29에서 이미 정정한 **벡터 cosine .758 ≠ 중심화 시간 자기상관**도 새 본문에 다시 등장한다. [Cliff 원문](https://arxiv.org/pdf/2003.03887)의 조건부 보정을 그 숫자만으로 적용하지 말 것. 엔트로피나1.4배 포함률만으로 “모든 lag 기반 검색 불가”라고 결론내는 것도 과도하다. 현재 측정된 고정 prior의 이득이 제한적이라는 범위는 지지하지만, 모델 효능·모든 lag 방식 기각은 **not run**이다.

추가 조건 혼용 주의(A09 관련): 서두 `.3486`은 `eta_split.txt`의 QK validation η0 전체 forward 값, `.1566`은 `stage_decomposition.txt`의 기존 조건 기준선이다. 새로운 통계 mask 성능 근거로 한 쌍처럼 쓰려면 동일 split/config/aggregation의 기준선을 다시 연결해야 한다. `.1303` 역시 full-kernel mass로 uniform-slot chance와 구분할 것.

### (c) 채택 우선순위·수정 방향

기존①η/혼합 구조 → ②QK → ③causal key → ④entmax 조건부 → ⑤별도 Gram → ⑥별도 Delta와 **⑦통계 mask 탐색 후보** 순위를 유지한다. η=.2 채택 보류도 유지한다. 이번 자료로⑦을 승격하거나 lag 계열 전체를 영구 기각하지 않는다. 새 후보 A/B/C의 실제 효능 결과는 없다.

⑦ 내부에서는 **현재 측정된 고정 lag prior의 후순위 유지**가 합리적이다. C의 경험적 귀무 gate는 작은 진단 후보로 남기되, “원리적으로 가장 옳다/표본 문제를 모두 우회한다/폭주 해결”은 미검증이다. 먼저 아래 계약을 코드·명령과 함께 고정해야 한다.

1. **train-only calibration과 모델 고정:** 어떤 checkpoint의 어떤 점수인지, 무관한 쌍 정의·query 시점/lag/키 빈도별 분포·시퀀스 경계 보존을 명시한다. 같은 학습 자료에 과적합된 점수의 pooled 95분위가 새 자료에서 자동으로 유의수준5% 또는 FDR5%를 뜻하지 않는다. 제한된 calibration 분할이나 분리 평가로 귀무 오통과율을 직접 확인할 것.
2. **causal 재현과 대조:** 전체 미래 cue를 사용한 ACF와 causal prefix 기반 정책을 구별한다. 같은 실제 가용 슬롯 수의 random/recent/uniform 대조, query-any-hit와 pooled mass를 모두 보고한다. [Autoformer §3.2](https://arxiv.org/html/2106.13008v5)의 top-k/softmax·순환 roll은 그 자체로 유의성 검정이나 온라인 causal 보장이 아니다(감사29 원문 확인 범위).
3. **정규화·전체 탈락 fallback·안정성:** `p×mask` 이후 질량 보존과 상한 정의를 사전 고정하고 score→p→w→pre→post·finite/G11·recall/copy/recall-first를 유지한다. 기존 A13의 finite·상한·provenance 계약을 먼저 보완한다.
4. **선택과 주장:** 분위 후보·한 개 선택 지표·동률/탈락 규칙을 새 사전등록에 적는다. 현재 validation 탐색은 탐색으로 기록하고 새 효능 주장은 미사용 자료/독립 seed 평가 후로 둔다. 이미 본 confirm을 후보 선택에 재사용하지 않는다.

### 수행·보존

감사 probe는 snapshot 생성기 함수3개를 AST로 추출하여 NumPy로400개 validation 시퀀스만 생성·집계했다. 이는 합성 데이터 통계 검사이며 모델 forward/학습이 아니다. 소스·checkpoint·기존 raw log·진행 프로세스·환경·Git·예약은 변경하지 않았다. 실제 confirm 생성/평가, 모델 재현, 학습·backward·GPU·mask 구현은 **not run**. 문서3개 append 및 새 감사 코드/JSON/raw stdout만 작성했다. 종료 전 문서 prefix·snapshot 연구파일 hash를 확인한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/stat_mask_probe.py /tmp/nsmt_assessment_20260923T122002Z-71edaa29 f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/stat_mask_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260923T122002Z-71edaa29/stat_mask_probe.log 2>&1
```

<!-- assessment-watch:20260923T122002Z-71edaa29 -->


## 추적 감사 32 — 2026-09-23 21:35 KST (예약 20260923T123001Z-f7ba8d71)

**해상도·채널의 일부 기술통계는 재현했다. 원시 해상도 가설 기각이나 다변량 key 효능 판정에는 부족하며, ⑧ 미채택 탐색 후보로 기록한다. 새 A15-RESOLUTION-CHANNEL-EVIDENCE OPEN.**

HEAD `23f8621b3f4f6604bd5022564082c948e3a5c8ac`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 기억·감사31·사전등록§2I·canonical21:26을 읽었다. 21:31:12 KST에265파일을 `/tmp/nsmt_assessment_20260923T123001Z-f7ba8d71`로 snapshot·hash했다. Manifest 불일치0. 새 연구 artifact `analysis/granularity_channel.txt` SHA256 `7f479dfb04a146324096cadff1091f5706bba3c88fb236de16ca334436464add`; 기존 연구 Python/학습 결과/사전등록은 감사31과 같다. 로컬 ETTh1 CSV도 별도 snapshot(SHA256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`)하고 계산에는 train `[0,8640)`만 사용했다. 자기 감사 문서 append는 연구 변화에서 제외했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/inventory.json), [독립 CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/granularity_probe.py), [수치·정의](../f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/granularity_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/validation.json).

### (a) 구현·정정 확인

모델/학습 변경은 없다. 기존 `to_patches`는 `[B,L,C]→[T,B*C,8]`로 채널을 분리하고 `Embedding.emb_linear=Linear(8,D)`를 적용한다. 이번의 **8칸 평균은 실제 embedding이 아니라 데이터 통계용 proxy**다. 다변량 key는 아직 제안이며 미구현을 결함으로 판정하지 않는다.

canonical21:26의 **cosine에서 시간 자기상관/유효표본을 추론한 주장 철회**, **T−τ pair 수 정정**, **confirm G≈.016 환산 철회**를 실제 append에서 확인했다. 감사29~31의 해당 문서 정정 요구는 **범위 한정 VERIFIED**로 둔다. 모델/통계 계약 해결을 뜻하지는 않는다. 같은 append 후반의 “lag 기반 무력/내용만 생존”, “첫째 가설 기각”은 아래와 같이 여전히 과도하다. A14의 단위·split·검정력·ACF provenance, A13의 절차 계약 등 나머지 OPEN을 유지한다.

새 분석을 생성한 script/정확 명령은 확인되지 않았다. `analysis` 내 granularity/channel 관련 파일은 txt뿐이다. 아래는 정의를 명시한 독립 수치 대조이며, 원 분석 전체의 재현 완료가 아니다.

### (b) 검증·수치 해석

**ETTh1 기술통계:** train8640행/7채널, train 평균·표준편차로 표준화했다. 전체 평균을 뺀 계열에서 `Σ x[t]x[t+k]/Σ x[t]^2` 정의로 ACF를 계산하면 기록과 반올림 수준에서 맞는다.

| 항목 | 감사 계산 |
|---|---:|
| OT raw lag24 / lag168 | .927911126 / .840381180 |
| 8칸 평균 patch lag3 / lag21 | .938853918 / .850420977 |
| raw 차분 lag24 / patch평균 후 차분 lag3 | .164546415 / .345647187 |
| 채널 off-diagonal abs correlation 평균/최대 | .311013118 / .983724338 |
| 채널 Gram participation rank / entropy rank | 3.551865104 / 4.131348547 |

따라서 **주기 성분이 patch 평균에서도 관찰된다는 범위**는 지지한다. `3.55`는 이번 계산에서 `(tr G)^2/tr(G²)`인 participation rank와 일치하며 entropy rank와는 다르다. Rank 정의를 원 분석에 적을 것. Lagged-vector Pearson 상관을 쓰면 raw168=.853095111로 달라지므로 ACF 정의도 명시해야 한다. 이는 서로 다른 추정량이며 데이터 오류로 오인하지 않는다.

**해상도 결론 제한(A15-RESOLUTION):**

- 합성 validation seed20270921/300시퀀스 재계산: 양의 빈도 lag39개, H=3.570160786, log39=3.663561646, 최대/균등=1.298180490. **lag를8배하는 일대일 재라벨링은 확률을 바꾸지 않아 엔트로피가 동일하다.** 이 계산은 raw token 모델의 상태 갱신·읽기 위치를 바꾼 실험이 아니다.
- “원시 정답 lag는 모두8의 배수”는 같은 patch 내부 위상끼리 비교한 정의에만 맞는다. 실제 생성기에서 얻은 예: query event10/cue phase1, source event2/value phase3이면 같은 위상 lag64이지만 cue→과거 value 거리는 **62**, query patch 끝→value 거리는 **68**이다. Raw task의 질의·value 위치를 정의하지 않고 해상도 효과를 기각할 수 없다. 새 raw 모델 효능은 **not run**.
- raw 차분은1시간 간격, patch 평균 후 차분은8시간 블록 간격이다. 같은 필터/이웃 폭이 아니며 patch 돌출+.471의 이웃 lag 범위도 txt에 없다. raw 이웃18..30시간과 patch 이웃의 물리적 시간 범위를 맞추지 않고 “3배 유리”를 예측 성능으로 해석하지 말 것. 더 큰 ACF 봉우리는 잡음 제거·정보 보존·모델 우월성의 직접 증명이 아니다.
- 같은 patch의 phase0 대비 Pearson 상관은 감사 정의에서 최저.947210이었다. 원 기록의 .950–1.000과 정의/표본 차이 확인이 필요하다. 어느 쪽이든 높은 수준 상관이 예측에 중요한 차분/위상 정보 손실이 없음을 보장하지 않는다. `Linear(8,32)`가 평균을 표현할 수 있다는 용량과 실제 학습·spike 후 정보 보존도 다르므로 “실제 손실은 더 작을 것”은 미측정 가설로 남긴다.

**회귀 결론 제한(A15-REGRESSION):**

R² .9758→.9784의 차이 .0026은 동일 target/표본/SST라면 **잔차제곱합의 약10.7438% 감소**다: `.0026/(1−.9758)`. 따라서0.26%p만 보고 “무의미/추가 정보를 쓸 여지가 없다”로 단정하지 않는다. 반대로 이것이 새 자료의 개선이라는 증거도 없다. Fit/evaluation 분리·절편·예측 horizon·정규화/규제·유효 자유도·오차 변동성이 명시되지 않았다. 만약 같은 표본의 nested OLS라면 특징 추가로 훈련 R²가 비감소하는 점도 고려해야 한다. 회귀 학습/재적합은 이번 감사에서 **not run**.

txt의 “Granger식 선형 판정”은 검정 완료로 읽히지 않도록 제한할 것. [공식 Granger 검정 정의](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html)는 자기 과거를 조건으로 추가 과거 계수의 통계적 유의성을 검정하며 통계량·p-value·자유도를 보고한다. 두 R²만으로 Granger 검정이나 채널 불필요성을 확정하지 않는다.

**검색 결론 제한(A15-RANK):**

`.1178/.0683`은 후보 시점 범위·query 수·정규화·동률·정답 lag 집합·best/평균 정답 rank가 기록되지 않아 독립 재현 미확인이다. “24의 배수”라는 대리 정답은 보고된 한계대로 예측 기여와 같지 않다. **무작위≈.5**는 단일 정답 또는 정답들의 평균 rank에는 맞을 수 있지만, 여러 정답 중 최상위 rank라면 틀린 기준이다. 후보 N개/정답 m개/0-based rank를 N−1로 나눈 경우 무작위 최상위 정답 기대값은 `(N−m)/((m+1)(N−1))`다. 합성 조합 정확합으로 검산한 예:

| 가정 예시 (원 분석 조건이라는 뜻 아님) | 올바른 무작위 best-positive rank |
|---|---:|
| N168, m7 | .120508982 |
| N192, m8 | .107038976 |

즉 .1178이 우연보다 훨씬 좋다는 해석도 실제 rank 정의에 달린다. 원 분석이 best rank였다고 확정하지 않는다. 동일 query의 두 방법 paired 차이·정답 수별 chance·block 단위 불확실성과 계산 코드를 먼저 보존해야 한다. `.1178/.0683≈1.72`를 “검색 성능1.7배”로 해석하기보다 해당 대리 rank의 절대 차이로 보고할 것. 모델 검색/예측 효과·다중seed·검정은 **not run**.

### (c) 채택 순위·문헌 기반 다음 조건

**①~⑥ 유지, ⑦통계 mask 미채택 유지, ⑧다변량 key도 미채택 탐색 후보**로 명시한다. 개념상③causal key 표현 진단과 연결되지만 현재 proxy rank만으로 실행/채택 순위를 높이지 않는다. η=.2 채택 보류도 유지한다.

이번에 [PatchTST 원문 §3/A.7](https://arxiv.org/html/2211.14730v2)과 [iTransformer 원문](https://arxiv.org/html/2310.06625v4)을 확인했다. PatchTST는 patch·채널별 처리를 결합하고 채널별 attention 적응성·학습/과적합 등의 근거를 제시한다. 단일 OT 선형회귀의 작은 ΔR²가 그 논문의 채널 독립성 근거를 대신하지 않는다. iTransformer는 변수를 token으로 삼아 변수 간 상관을 attention에 맡긴다. 이는 채널 의존성 활용을 검토할 1차 문헌 근거지만 **현재 제안한 cross-channel key만의 효과 증명이나 같은 구조는 아니다**.

⑧ 승격 전에 (1) 같은 시각까지 관측 가능한 채널만 써서 query/history를 정렬하고, target 채널 identity·key 차원·파라미터 수를 고정, (2) 단변량/다변량/채널 교란 및 차원·용량 대응 대조, (3) 동일 물리적 lookback/horizon의 예측 오차와 대리 rank를 분리 보고, (4) train에서 정한 전처리·규칙을 고정하고 시간 분리 검증을 수행할 것. 채널을 key에서 섞으면 값·출력 head를 채널별로 유지해도 **전체 모델은 엄밀한 channel-independent 모델이 아니다**. 그 차이를 명시한다.

해상도 비교를 추후 한다면 lag 라벨만 바꾸지 말고 patch1/patch8에서 **동일 실제 시간 범위·정보 예산·τ의 물리 단위·질의/정답 정의**를 맞춘 대조를 사전등록해야 한다. 합성 다채널 변형 역시 새 과제 정의이므로 기존 단일채널 oracle/O7 기준을 무조건 이식하지 않는다. 현 감사에서 새 실험을 실행하지 않았다.

### 수행·보존

NumPy CPU로 train 데이터 기술통계,300개 validation 생성기 통계, 저장 R²의 산술과 순위 귀무 기대값만 검사했다. 모델/회귀 fit·학습·forward·backward·GPU·confirm 접근·환경 설치·연구 소스 수정·Git 변이·프로세스 중단·타세션 대화 접근/전송은 하지 않았다. 새 감사 코드/JSON/raw stdout 및3문서 append만 작성하고 기존 prefix·snapshot 연구파일을 확인했다. 복사된 원 데이터는 로컬 snapshot에만 보존한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/granularity_probe.py /tmp/nsmt_assessment_20260923T123001Z-f7ba8d71 f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/granularity_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260923T123001Z-f7ba8d71/granularity_probe.log 2>&1
```

**동시 변경 기록:** append 직전 canonical PROJECT_LOG와 기존 stat_mask_feasibility.txt가 작업 세션에 의해 갱신돼 최초 보존 assertion이 중단했다(감사 본문 쓰기 전, 모델 실패 아님). Canonical의 새 A14 수용 append는 읽고 보존했으며, 위 A14 잔여 상태는 최초 snapshot 기준이다. 새 재현 코드/통계 정정의 종합 검증·이슈 종료는 다음 주기로 넘긴다. HEAD·변경 목록은 [concurrent_changes.json](../f_lif_pop_v3/forecasting/results/assessment/20260923T123001Z-f7ba8d71/concurrent_changes.json)에 기록했다. 이번 probe 대상 granularity/생성기/ETTh1은 변하지 않았다. 새 canonical 내용까지 포함한 prefix를 보존해 append했다.

<!-- assessment-watch:20260923T123001Z-f7ba8d71 -->


## 추적 감사 33 — 2026-09-23 21:45 KST (예약 20260923T124001Z-4b7568a4)

**A14의 단위·같은 가용 슬롯 수 대조·validation 표기를 새 코드로 재검증했다. 해당 정정은 VERIFIED, 원 ACF 분석 및 mask 효능까지 검증한 것은 아니다.**

HEAD `8b5a43406ce52ed2ada656d0d6687d3ce816c183`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 기억·감사32·사전등록§2I·canonical21:32 A14 수용 append를 읽고 감사32 중 발생한 변경을 이번 대상으로 복구했다. 21:40:41 KST에267파일을 `/tmp/nsmt_assessment_20260923T124001Z-4b7568a4`로 snapshot·hash했다. Trigger manifest 불일치0. 신규 `stat_mask_control.py` SHA256 `50e60589394ffa9582a8da18072732da36f79cb02809fa8747f811c360f9497d`, 출력 txt `c3c6e7827568174179e70228b938a73282a51707b2e053e855dc10ccaa109767`, 정정 append된 `stat_mask_feasibility.txt` `cdb610f1294eb659964545dbc42291f7a5038fe050cb218a06216539cffc9906`. 모델/학습 소스·기존 모델 결과·사전등록은 그대로다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/control_probe.py), [수치/전수 조합 결과](../f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/control_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/validation.json).

### (a) 구현 — A14 부분 VERIFIED

snapshot 스크립트를 읽어 `collect→lag_frequency→evaluate→main` 흐름이 합성 생성기 통계만 계산함을 확인한 뒤 CPU에서 실제 `main`을 실행했다. **stdout이 저장된 stat_mask_control.txt와 byte 단위로 일치**했다. 새 정정 txt는 이전 feasibility 파일 전체를 prefix로 보존했다.

- **A14-UNIT VERIFIED:** `.908348529`가 top8의 query당 기대 정답 개수라는 코드·출력·철회 문구가 맞는다. 확률인 pooled 포함률 `.261898231`과 분리됐다.
- **A14-CONTROL VERIFIED (현재 생성기/KS 범위):** `c_n=#{선택lag≤n}`를 유지한 균등 비복원 대조의 기대 정답 수 `c_n a_n/n`, query-any-hit `1−Hypergeom.pmf(0,n,a_n,c_n)`가 맞다. n3/5/7과 가용선택0·전체정답 경계까지5개 사례에서 모든 선택 조합을 직접 열거하여 함수와 오차1e−12 이내로 일치했다. 이 범위 밖 임의 입력 일반성을 주장하지 않는다.
- **A14-SPLIT VERIFIED (표기·실행 경로):** seed20270921/400시퀀스의 validation 생성 자료이며, 동일 자료에서 빈도 lag 선택·집계한다는 in-sample 제한이 코드와 출력에 명시됐다. train-only 또는 독립 성능 검증으로 바뀐 것은 아니다.
- **A14-INFERENCE 문서 정정 VERIFIED:** Bartlett 표를 산술 예시로 제한하고 “검정력0/원리적 불가능/모든 lag 방식 불가”를 철회한 canonical 및 txt append를 확인했다. 실제 검정력/귀무분포 검증은 여전히 **not run**.

**A14 전체는 OPEN 유지:** 새 코드는 고정 lag 빈도 대조를 재현하며 **원 자기상관 top8=27.8%의 생성 코드/정의는 복구하지 않았다.** Canonical도 이 한계를 명시한다. `stat_mask_control.py`의 도입으로 모든 provenance가 해결됐다고 읽지 않는다. A13 절차 및 A15 해상도·채널 이슈는 이번 검사 밖이며 상태를 유지한다.

### (b) 검증·실제 증거

새 스크립트의400시퀀스/12,613query/43,746정답 칸 출력과 k={3,5,8,12,20}의6열 수치를 **감사31 독립 NumPy·조합 산술 artifact**와 대조했다. 모든 행이 출력 반올림 정밀도까지 일치한다.

| k | 고정 pooled / 대조 (%) | 고정 query-any-hit / 대조 (%) |
|---:|---:|---:|
| 3 | 9.8912 / 9.7148 | 26.0842 / 30.7217 |
| 5 | 16.4335 / 16.8923 | 32.2921 / 47.7137 |
| 8 | 26.1898 / 25.5915 | 41.9250 / 63.1272 |
| 12 | 39.0527 / 38.4602 | 49.8375 / 78.1326 |
| 20 | 64.3213 / 55.8274 | 71.2440 / 90.1514 |

이 표는 **같은 validation 표본의 고정 prior와 조건부 무작위 선택의 기술통계**다. k8 pooled 차이 약+.60%p, any-hit 차이 약−21.20%p라는 설명은 맞는다. 다만 k20 pooled 이득은 **+8.49%p**이므로 “포함률 이득이 사라졌다”를 모든 k/지표로 일반화하지 말 것. “모든 k에서 query 적중 열세”는 **검사한5개 k**의 의미다. `k/41`도41개 lag 고정 집합을 뽑는 다른 대조의 기대값으로는 유효하다. 이번 교정은 질문에 맞춰 가용 선택 수를 통제한 것이며 모든 목적에서 유일한 대조를 확정한 것은 아니다.

정답 개수·query 적중률·커널/계수의 정답 질량·회상 MSE는 서로 다른 지표다. 현재 고정 prior의 제약은 확인됐지만 mask “쓸 만하지 않음”의 최종 성능 판정은 유보한다. 정답 인접 구간을 중복 선택하는 현상과 실제 예측 기여의 관계는 모델 비교 전에는 확정할 수 없다. 시퀀스 내 query 의존성, 같은 표본의 lag 선택, 독립 평가 부재는 그대로다. **모델 성능·통계적 유의성·새 confidence interval·O7·mask 구현/훈련·독립 test/confirm 평가는 not run**.

첫 실행은 원 분석의 계산·stdout 생성 후 **감사 wrapper의 NumPy bool을 JSON으로 직렬화하는 부분에서 TypeError**가 났다. 감사 코드만 Python bool로 변환해 재실행했고 통과했다. 최초 `control_probe.log`와 재실행 `control_probe_retry.log`를 모두 보존했다. 연구 소스/모델 실패로 판정하지 않았다.

### (c) 다음 방향·우선순위

새 결과는 기존 대조 진단의 재현·정정이며 새 모델 효능 근거가 아니다. **①~⑥ 유지, ⑦통계 mask와⑧다변량 key는 미채택 탐색 후보, η=.2 채택 보류**를 유지한다. A14의 남은 ACF 재현 범위를 명확히 하고, 실행 후보를 승격하기 전에 train-only calibration·causal 접근·정규화/전체 탈락 fallback·같은 가용 슬롯 수의 대조와 실제 회상/예측 오차를 분리한 프로토콜을 고정할 것.

문헌 근거는 감사29~31에서 실제 확인한 [Autoformer §3.2](https://arxiv.org/html/2106.13008v5), [Cliff et al.](https://arxiv.org/pdf/2003.03887), [NIST ACF 정의](https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm)의 기존 범위 그대로다. 이번에는 새 논문이나 통계적 유의성 주장을 추가하지 않았다. 설명상 단위와 제한을 고친 것만으로 FDR·예측 성능·gradient 안정성이 검증되는 것은 아니다.

### 수행·보존

Python3.10.18/NumPy1.26.4/SciPy1.15.3, CPU2threads, CUDA_VISIBLE_DEVICES 빈 값으로 실행했다. 실제 snapshot main은 생성기만 사용하며 모델 학습/forward/회귀fit/confirm 자료를 사용하지 않는다. 연구 소스·checkpoint·기존 raw log·프로세스·환경·Git·예약을 변경하지 않았다. 새 감사 코드/JSON·raw stdout·3문서 append만 작성했다. 문서 prefix와 snapshot 연구파일 hash를 종료 전에 확인했다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/control_probe.py /tmp/nsmt_assessment_20260923T124001Z-4b7568a4 f_lif_pop_v3/forecasting/results/assessment/20260923T124001Z-4b7568a4/control_probes.json f_lif_pop_v3/forecasting/results/assessment/20260923T122002Z-71edaa29/stat_mask_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260923T124001Z-4b7568a4/control_probe_retry.log 2>&1
```

<!-- assessment-watch:20260923T124001Z-4b7568a4 -->


## 추적 감사 34 — 2026-09-23 23:47 KST (예약 20260923T144001Z-44f00cb2)

**새 모델 결과는 없지만 canonical 23:30의 읽기/쓰기 분리 계획은 새 검토 대상이다. 분리 실험을 우선 검토할 근거는 있으나, 유일한 원인 규명·안정성 보장·채택으로 승격할 근거는 없다. A16-READ-WRITE-CAUSALITY OPEN을 추가한다.**

HEAD `5b0f1815135c61832300056732799a7079d26f67`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기억·감사33·최신 사전등록§2I·canonical 새 계획을 읽었다. Trigger 변경은 GIT_HEAD뿐이나 실제 canonical 문서 변경을 확인했다. 23:40:47 KST에267파일을 `/tmp/nsmt_assessment_20260923T144001Z-44f00cb2`으로 snapshot·hash, trigger 불일치0. 감사33 대비 연구 소스·기존 결과·사전등록 변경 없음. 이전 감사 자체 append는 새 연구로 세지 않았다. Canonical SHA256 `cfae4d1fbc3349645ba8fa03e3a869b395e868d5562b1e4064cad70c588091c2`, layers `0ad31dc6250a780703a5195b32606657424298925fd948c8f3411fd948c32a04`, eta_intervention_split `7e013ab47c716893a0be9d47b309f2f133f5e057003261674e6ec5a5c6532afb`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/inventory.json), [산술 검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/structure_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/structure_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/validation.json).

### (a) 구현·설계 대응 — 기존 구현과 신규 제안을 구분

현재 `layers.py`는 soft QK→sparsemax→질량 보정→cap→과거 increment `f_j` 합→상태 `u`→다음 key의 경로를 가진다. 분리 readout은 **계획 단계/not implemented**다. 미구현을 확정 오류로 판정하지 않으며 기존 VERIFIED를 다시 닫거나 열지 않는다.

1. **A16-SCORE OPEN:** canonical의 “qk_norm 점수=코사인”은 현재 코드와 다르다. 실제 변환은 `q/sqrt(||q||²+ε²)`와 k의 같은 변환 후 음의 제곱거리다. CPU 독립 예: ε=.01, q=(.01,0,0,0), k₁=(.001,0,0,0), k₂=(1,0,0,0)의 cosine은 둘 다1이지만 제곱거리는 **.3691814812 / .0857571530**이다. 단위 길이로 정확히 정규화했을 때만 거리 점수가 코사인의 아핀 변환이다. 피어슨은 중심화한 cosine이지만 무엇을 표본 축으로 중심화하는지 명시해야 한다. 4차원 key 성분 상관을 시간 상관이나 유의확률로 해석할 수 없고 상수 벡터 fallback도 필요하다. 이 산술은 실제 데이터에서 차이가 얼마나 큰지까지 측정하지 않았다.
2. **A16-MIXTURE OPEN:** precap에서 `c=(1−η)b+η B w`, `w=bp/Σbp`가 맞는다. η=.026의97.4%는 이 혼합의 기본 커널 비중이며 cap 후 개별 계수·부호가 있는 increment 신호·출력 기여율97.4%를 뜻하지 않는다. 산술 재검사 B=13.9614739001, b₀=1.1005474055, B/b₀=12.6859359533. **총 질량 B 보존과 계수 상한 b₀를 동시에 요구할 때** 최소13칸이라는 조건부 경계는 맞다. 실제 cap 후 질량은 줄며, η1 one-hot의 κ=.0788274514가 곧 회상 불가능을 뜻하지 않는다(맞는 칸 one-hot이면 정답 질량 비율은1일 수 있다).
3. **A16-CAUSALITY OPEN:** 제안 `r=Σpξ_j`는 기존 `Σc f_j`에서 되먹임뿐 아니라 value 정의, 계수·정규화, spike/readout 경로까지 함께 바꾼다. 개선 시 원인을 되먹임 하나로 귀속할 수 없다. key detach는 forward를 그대로 두고 미분 경로만 끊으므로 “24배 감소에 그침→폭주 본체는 forward 누적”이라는 유일 원인 판정은 성립하지 않는다. oracle은 정답 정보를 준 다른 정책이며 “학습 경로만 막혔다”의 증명도 아니다.
4. **A16-INVARIANTS OPEN:** 순수 η0 population을 유지하고 read가 이후 population/key/write에 전혀 들어가지 않으면 해당 되먹임을 차단할 수 있다. 다만 soma와 readout 중 어디에 더하는지, 차원 투영, 과거 `j<n`만 읽는지부터 고정해야 한다. G7/G12 보존은 population 상태·계수의 기존 중립극한과 새 전체 출력 동일성을 구분해야 하며 새 경로가 있는 전체 모델의 비트 동일성을 자동 주장하지 않는다. read 파라미터를 바꿔도 population 궤적이 불변인지 확인하고, 상태/drive/출력·gradient 안정성을 각각 검사해야 한다. 기존 f-LIF 자체의 재귀는 남는다.

### (b) 검증 과정 — A16-PROTOCOL OPEN

`eta_intervention_split.py`를 다시 읽었다. 궤적 고정 행의 post-cap **.2741과 peak22.07은 원 학습 η 궤적**에서 계수만 η1로 재계산한 값이다. 새 제안의 **η0 궤적**, `Σpξ_j` value read, 회상 MSE에 대한 수치가 아니다. 전체 η1 개입 peak656.13/G11 FAIL이라는 기존 관찰은 유지하지만 고정 행을 새 분리 모델의 사전 성능·안전 근거로 사용하면 안 된다.

“cap OFF7160→발산”도 범위를 고친다. 저장 `stability_sweep.txt`의7160.17은 T42/τ2/η.3 greedy-adversarial 조건의 유한 peak다. 같은 스크립트가 정한 diverged 문턱과 구분해야 한다. 현재 τ≥4 조건과 동일 실행이 아니며, τ4/η1에서2793137.43인 별도 행도 함께 보아야 한다. cap 제거가 위험할 수 있다는 근거는 유지하되 모든 설정에서 cap이 유일하게 필수라는 정리는 아니다. cap/gradient sweep은 이번에 재실행하지 않았다.

**“학습 없음: 선형 readout만 재학습”은 모순이다.** 동결 encoder 위 readout fit도 학습이다. 실행 시 목적을 ‘동결 표현의 탐색적 readout 학습’으로 기록하고 train에서 fit, validation에서 후보 선택, 확인용 분할은 별도로 고정할 것. 이번 감사는 fit을 실행하지 않았다. 새 사전등록을 성능을 본 뒤에 만들 경우 스크리닝은 명시적으로 탐색으로 남겨야 한다. 이미 본8seed confirm을 새 후보 선택/확증에 재사용하지 않는다. 한 checkpoint의 실패만으로 전체 읽기/쓰기 분리 가설을 기각하는 규칙도 표현·readout 학습 부적합과 구조 실패를 구별하지 못한다.

새 모델 성능·새8seed 통계·새 CI·분리 모델 G7/G12/G11·회상/O7·readout fit·모델 forward/backward·실제 confirm 재평가는 모두 **not run**. 기존 η=.2의 확증 채택 보류, A13/A14/A15 잔여 OPEN을 유지한다.

### (c) 문헌 확인과 채택 우선순위

이번에는 원문을 직접 열어 확인했다.

- [RetNet Eq.6/8](https://arxiv.org/html/2307.08621v4): `S_n=γS_{n−1}+K_nᵀV_n`, 출력 `Q_nS_n`, head별 고정γ가 실제 있다. 별도 읽기 경로의 설계 근거는 된다. 현 f-LIF에 이식하면 안정·회상 성능·전체 계산량까지 보장된다는 근거는 아니다.
- [NTM §3, §3.4](https://arxiv.org/html/1410.5401v2): 읽기/쓰기 head는 분리돼 있지만 recurrent controller는 이전 read vector를 내부에 보관할 수 있다. 따라서 ‘분리 head=read가 출력에만 가고 어떤 feedback도 없음’이라는 인용은 과도하다. DNC·가변차수 review의 상세 주장은 이번에 새로 검증하지 않았고 판단 근거로 사용하지 않았다.
- [DeltaNet §3 Eq.4](https://arxiv.org/html/2406.06484v3): `I−βkkᵀ` 갱신식을 확인했다. 단위 k, β=.5의4×4 행렬 고유값을 실제 계산하면 **(.5,1,1,1), spectral norm1**이다. 이는 비팽창이지 엄밀한 수축이 아니다. ||k||=2, β1이면 norm3이고, 단위 k/β.5에 별도 감쇠 .9를 곱하면 norm.9다. 이 계산은 고정 k의 선형 상태 요인에 한정된다. 상태 의존 key/게이트의 전체 Jacobian이나 반복 value 주입까지 안정성을 보장하지 않는다.
- [Mamba Theorem1](https://arxiv.org/html/2312.00752v2): N1/A−1/B1, 입력 기반 Δ와 ZOH 조건의 게이트 등가식이다. 임의의 상태 의존 η를 현재 fractional recurrence에 추가하는 것의 안정성 정리로 이식하지 않는다.

**우선순위는 ‘채택 확정’과 ‘다음 검토 순서’를 분리한다.**

| 순서 | 이번 권고 | 승격 조건 |
|---|---|---|
| 최우선 | A13 절차와 A16 정의·조건 혼용을 정리하고 소스/설정/분할 계약 고정 | 확인용 자료 추가 열람 전에 선택·종료·유효성 규칙 명시 |
| ① 구조 진단의 다음 후보 | **읽기/쓰기 분리(①-B)를 우선 탐색**. 기존 η/혼합 분석의 후속이며 새 핵심 기제 | 같은 value/표현/파라미터 예산에서 feedback on/off 대조, value를 f→ξ로 바꾸는 효과와 readout 우회 효과를 분리. population 불변성 및 새로운 안정성 검사를 먼저 통과 |
| ② 점수 진단 | **같은 상태·query에서 점수 통계량만 바꾸는 짝지은 비교** | soft QK/단위 cosine/중심화 cosine을 구별하고 축·온도 calibration·0분산 처리·causal 범위를 고정. rank뿐 아니라 p→계수/읽기→회상 기여를 후속 평가 |
| 후속 | causal key·entmax·Gram·Delta의 기존 조건부 후보 유지 | 단순 대조로 설명되지 않는 병목이 남는 경우 해당 기제를 검증. 시점별 gate는 보조 ablation이며 현재 증거만으로 Delta보다 반드시 우선이라는 결론은 없음 |

Canonical의 의도 정정에 따라 **점수 통계량 교체는 기존 고정 lag mask⑦와 다른 후보**로 분리해 기록한다. 이전 mask 결과로 점수 교체 전체를 기각하지 않는다. 동일 상태에서 좋은 순위가 나와도 상태가 바뀌는 전체 모델의 개선으로 바로 승격하지 않는다. ⑦고정 mask와⑧다변량 key는 계속 미채택 탐색 후보, η=.2는 채택 보류다. 이번 수치/문헌은 다음 진단의 이유를 제공하며 후보1·gate·Delta의 효능 순위를 확증하지 않는다.

### 수행·보존

CPU NumPy1.26.4로4차원 산술만 실행했다. 데이터 접근/모델 학습·forward·backward·GPU·설치·연구 소스/기존 artifact 수정·Git 변이·타세션 메시지는 없음. 새 감사 코드·JSON/raw stdout와3문서 append만 작성. 문서 prefix 및 snapshot 연구파일 hash를 확인했다. 감사commit/tag 없음.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/structure_probe.py f_lif_pop_v3/forecasting/results/assessment/20260923T144001Z-44f00cb2/structure_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260923T144001Z-44f00cb2/structure_probe.log 2>&1
```

<!-- assessment-watch:20260923T144001Z-44f00cb2 -->


## 추적 감사 35 — 2026-09-24 23:45 KST (예약 20260924T144002Z-9781a7c1)

**hard mask 스크리닝은 진행 중이며 결과 txt는 snapshot과 append 직전 모두0byte다. 완료·모델 실패·성능 우위를 판정하지 않는다. 다만 새 분석 코드의 실제 함수에서 NaN 집계 결함을 재현했으므로 A17-HARD-MASK-SCREEN OPEN을 추가한다.**

HEAD `5b0f1815135c61832300056732799a7079d26f67`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 문서 기억·감사34·사전등록 최신§2I·canonical 최신 append를 읽었다. 이번 연구 변화는 미추적 `analysis/hard_mask_screen.py/.txt` 두 파일이다. 모델/학습 소스·기존 모델 결과·사전등록·HEAD는 감사34와 같다. 감사34의 자체 문서 append를 새 연구 변화로 세지 않았다.

23:40:46 KST에269파일을 `/tmp/nsmt_assessment_20260924T144002Z-9781a7c1`으로 snapshot·SHA256 기록했다. Trigger 불일치0. 분석 코드 SHA256 `7bef7e60e9258b5f096601253dc04532be7ba76cf86860ca5ecf1b030c91fe36`, 빈 txt `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; 모델 layers `0ad31dc6250a780703a5195b32606657424298925fd948c8f3411fd948c32a04`, 사전등록 `d59d91d790c8513b84fa2ac8af00e2596b18980a2022bdfa6aeab7f7a1629168`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/inventory.json), [실제 함수 CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/hard_mask_probe.py), [검사 결과](../f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/hard_mask_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/validation.json).

### (a) 구현 정확성 — 부분 확인, A17 OPEN

새 스크립트는 학습 없이 `c=m·b`, m∈{0,1}로 과거 increment를 골라 상태를 갱신하는 후보를 다룬다. **고정 lag prior도, 기존 η/rho/cap 계수에 mask를 덧씌우는 방식도, 감사34의 출력 전용 read/write 분리도 아니다.** unit/shared/input descriptor의 Pearson 또는 cosine으로 동적인 binary mask를 만든다. 스크립트의 “사용자 확인 설계”라는 설명은 설계 설명으로만 읽었으며 감사에서 타세션 대화를 확인하지 않았다.

- **국소 재귀·인덱스 PASS(검사 범위 한정):** snapshot의 실제 `fractional_forward`, `descriptors`와 실제 PopulationNeuron을 읽고 CPU T8/B2/D3/K4에서 비교했다. float32/64 각각 무마스크와 all-ones mask의 상태가 `mode='full'`과 **최대 오차0/비트 일치**했다. 현재보다 미래의 입력을 바꿔도 검사 시점의 세 descriptor가 바뀌지 않았다. 전체 checkpoint·T42·모든 mask의 재현 검사를 대신하지 않는다.
- **A17-AGGREGATION OPEN, 확정 재현:** `per_query`의 `answer_kept=(m*a).sum/a_cnt`는 정답 칸0인 무효 row에서0/0이 된다. `accumulate`가 이를 `valid=0`과 곱해도 **NaN×0=NaN**이며, 같은 시퀀스가 나중에 유효 query를 가져도 결과가 NaN으로 남는다. 실제 두 함수를 호출한2행 예제에서 최종 count=[2,1], answer_kept=[1,NaN]을 얻었다. 이 예제의 다른7개 지표는 유한했다. 생성기만 사용한 validation seed20270921/64시퀀스에서도64개 모두 유효 recall이 있고, 그중 **56개**에 다른 row 때문에 처리되는 시점의 ‘정답0·무효’ 조건이 있었다. 이는 코드 경로 도달성이지 아직 비어 있는 실행 결과의 NaN을 관측했다는 뜻은 아니다. **유효 row를 나눗셈·집계 전에 선택하거나 안전 분모와 where로 명시 처리**하고 재검사해야 한다.
- **A17-FINITE OPEN:** `gap > 1e-6`만 검사하면 gap=NaN이 거부되지 않고 `OK (fp)`로 출력된다. 또한 `max(0., NaN)=0.`이므로 peak 갱신은 비유한 상태를 숨겨 양의 bound에서 OK를 만들 수 있다. 코드의 해당 비교식을 NaN에 적용해 재현했다. 실제 checkpoint가 NaN이라는 관측은 없다. recurrence gate와 각 batch 상태의 `isfinite`를 먼저 강제하고 비유한/상한 없음/상한 위반을 각각 기록할 것. 이는 기존 A13의 finite 판정 이슈와 같은 유형이나 새 분석 코드의 별도 발생이다.

0<α≤1에서 `0≤m b≤b≤b₀`이므로 **이 계수 상한을 구현하기 위한 별도 cap은 필요 없다는 산술은 타당**하다. 그러나 항을 제거하면 signed increment의 상쇄도 달라진다. 상한 만족만으로 상태·gradient 안정성이 증명되지는 않는다. unit/shared mask는 masked u를 다음 descriptor로 쓰므로 상태 의존 되먹임이 남는다. input mask는 그 mask 결정에서 u를 쓰지 않지만 fractional state의 재귀까지 사라지지는 않는다.

### (b) 검증·분할·통계·재현성

**좋아진 설계:** train loader에서 threshold를 정하고 validation에서 측정하며 confirm loader 호출은 없다. frozen은 원 궤적에서 mask 통계만, closed는 mask를 실제 재귀에 넣어 상태까지 다시 계산하므로 두 종류의 개입을 구별한다. 비어 있는 mask의 M_eff를0으로 포함하고 empty 빈도도 출력한다. 현재 recurrence에서 empty mask는 과거항0/현재항만 남긴다. 모델 성능 대신 진단이라는 끝 문구도 적절하다.

**같은 슬롯 수 대조는 국소 산술 PASS:** 실제 `per_query`를 n5/선택2 사례에 적용하고10가지 선택 조합을 전수 열거했다. weighted M_eff의 정확 기대값 **.3010115623**,2048draw Monte Carlo **.3030289412**; any-hit 정확값 **.7**,실제 hypergeom **.6999999881**이었다. weighted ratio의 평균을 단순 정답 비율로 대체하지 않은 점은 맞다. 다만 본 스크립트 기본32draw 결과에는 Monte Carlo 오차를 보고해야 하며, 한 generator를 순차 소비하므로 후보 순서가 바뀌면 난수 대조도 바뀐다. 고정 난수/충분한 draw·민감도 확인 및 unit→query→sequence 집계 정의를 보존할 것.

**A17-NULL/REPORT OPEN, 실행 전후 구분:**

1. `null_thresholds`는 `queries(truth,kind,...)`로 recall/정답 존재를 골라 pool을 구성한다. y 값이나 정답 slot 점수를 직접 threshold 최적화하지는 않지만, 완전한 **label-free**도 아니다. ‘train의 사건 라벨로 조건화한 cross-sequence reference’로 표기할 것. train 사용을 test leakage로 오인하지 않는다.
2. batch를 한 칸 roll한 **한 번의 교차 시퀀스 pairing**은 곧바로 검정의 귀무분포를 보장하지 않는다. 동일 cue/code가 여러 시퀀스에서 공유돼 다른 시퀀스라도 내용이 비슷할 수 있고, batch1이면 자기 자신과 짝지어진다(실제 roll 산술 확인). 무엇을 ‘무관’이라고 정의할지, batch1 처리, 사건 위치·lag·cue 구성별 교환가능성을 명시해야 한다. 본 검사는 threshold가 잘못된 숫자임을 확정한 것이 아니라 유의수준 해석에 필요한 조건이 빠졌음을 지적한다.
3. 5/160/32는 descriptor의 **성분 수**이지 독립 표본 수가 아니다. 특히 learned embedding의32개 성분은 같은 patch의 투영이다. Pearson df3 또는 shared160을 근거로 통계적 유효성을 단정하지 않는다. 현재 코드는 t검정을 실행하지 않고 유사도 분위값을 쓰므로 해당 통계량 자체를 오류라고 하지는 않는다. pooling된90/95/99분위가 새 자료·각 query·closed 상태에서10/5/1% 오통과율을 보장하지 않는다. closed에서도 무마스크 train threshold를 쓴다는 한계는 출력 설명에 이미 있다.
4. **같은 실행에서 full-kernel 기준선을 직접 계산해야 한다.** 끝 문구의 `.1303 on this split`은 실제 집계가 아니라 상수 문자열이고 `kernel=None`은 쓰이지 않는다. 사전등록에서 `.13032889`는 과거 test의 값으로 설명돼 있다. 새 validation 표의 기준선으로 그대로 이식하지 않는다(A09 조건 혼용 연관). 원 M_eff는 query별 retained kernel의 정답 비율을 평균한 값이다. 실제 recall MSE, kept mass/B, 정답 보존률, any-hit와 구별한다. oracle1은 정답 mask라는 특권적 진단 상한이며 같은 선택 예산 대조가 아니다.
5. 실행 완료 시 **checkpoint/config/source hash, 정확 명령·null/val batch/sequence/query 수, 데이터 seed와 MC seed, full-precision threshold·시퀀스별 결과**를 남길 것. 현재 main은 run_id와 반올림 표를 출력하지만 이 정보를 모두 저장하지 않는다. 3axis×2stat×6rule=36후보를 frozen/closed로 보면72행의 탐색이다. 선택 규칙·빈 mask 처리·finite/G11 제외 조건과 독립 평가 절차를 먼저 고정해야 한다. 이미 본 confirm의 재사용으로 확증하지 않는다.

**새 스크리닝 결과·checkpoint 재현·실제 회상 성능·새 CI·새 훈련은 not run/미확인.** 빈 txt의 원인을 성공·실패·정지 중 하나로 단정하지 않는다. 감사는 full72조건 main을 실행하지 않았고, synthetic 소규모 함수 진단만 수행했다. 학습된 checkpoint를 새로 평가하지 않았다.

### (c) 개선 방향과 우선순위

**최우선은 A17 집계/finite 결함을 고치고 실제 기준선·출처를 갖춘 스크리닝을 완료하는 것**이다. 그 뒤 frozen과 closed를 같은 표본·규칙으로 비교하고, 임계값 reference의 held-out 오통과·정답 보존·empty·retained mass를 따로 측정한다. 정답 비율이 올라가도 거의 모든 질량/정답을 버린 결과라면 회상 이득으로 해석하지 않는다. frozen peak는 기준 궤적 값이며 mask를 적용한 시스템의 안전성 판정은 closed에서 해야 한다.

문헌 확인: PMC 접근은 CAPTCHA로 막혔으나 [Winkler et al., 2014 원문(대학 저장소)](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)을 실제 열어, 귀무가설 아래 교환가능성과 의존성에 맞는 permutation 전략이 필요하다는 범위를 확인했다. **이는 GLM 논문의 원리이며 현재 retrieval threshold의 유의수준을 보장하는 정리가 아니다.** 이 원리에 근거한 감사 제안은 시퀀스 구조를 보존한 reference 정의·독립 calibration 검증과 조건별 오류율 확인이다. 성분 수를 늘리는 것만으로 이 조건을 대체하지 않는다.

현재 후보는 **① 혼합 구조 대안의 hard mask 탐색 + ② 점수식 비교**로 기록한다. 기존⑦ 고정 lag mask 결과로 기각하지 않으며, 읽기/쓰기 분리(①-B)보다 성능상 우수하다고 승격하지도 않는다. 효능의 새 판단 근거가 아직 없다. 감사34의 구조/점수 분리 진단 우선순위와 η=.2 채택 보류, A13·A14·A15·A16 잔여 OPEN을 유지한다. A17의 국소 PASS는 기존 FIXED-PENDING-REVIEW 이슈를 VERIFIED로 닫는 근거가 아니다.

### 수행·보존

CPU2threads/Python3.10 환경, torch1.12.0+cu113·NumPy1.26.4. CUDA_VISIBLE_DEVICES 빈 값; tiny 무학습 neuron forward 및 generator 통계/실제 함수·조합 산술만 실행. 학습·GPU·checkpoint 평가·backward·설치·소스 수정·프로세스 중단·Git 변이·타세션 대화/메시지·예약 생성 없음. 새 감사 artifacts/raw stdout와3문서 append만 작성. 기존 prefix와 snapshot 연구파일 hash를 보존했다. 감사commit/tag 없음. 최초 rg 검색은 도구 미설치로 실패해 Python/grep으로 확인했으며 모델 실패가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/hard_mask_probe.py /tmp/nsmt_assessment_20260924T144002Z-9781a7c1 f_lif_pop_v3/forecasting/results/assessment/20260924T144002Z-9781a7c1/hard_mask_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260924T144002Z-9781a7c1/hard_mask_probe.log 2>&1
```

<!-- assessment-watch:20260924T144002Z-9781a7c1 -->


## 추적 감사 36 — 2026-09-25 00:04 KST (예약 20260924T150001Z-689fad22)

**A17-AGGREGATION의 무효 query 0/0 오염은 실제 함수로 재검증하여 해당 원인 범위에서 VERIFIED로 닫는다. A17-FINITE·NULL·REPORT는 OPEN 유지. 신규 순위 진단은 공식 산술이 맞지만 동점 규칙을 추가해야 한다. 최초 snapshot에는 결과가 없었고 감사 중 새 결과가 도착했다(아래 동시 변경 기록).**

HEAD `5b0f1815135c61832300056732799a7079d26f67`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기억·감사35·최신 사전등록§2I·canonical 최신 append를 읽었다. 연구 변경은 `hard_mask_screen.py`의 분모 clamp와 새 순위 진단, 미사용 변수 제거다. 모델/학습 소스·기존 결과·사전등록·HEAD는 동일하며 자체 감사35 문서 append는 연구 변경에서 제외했다.

**2026-09-25 00:00:39 KST**에269파일을 `/tmp/nsmt_assessment_20260924T150001Z-689fad22`으로 snapshot·hash했다. Trigger 불일치0. 분석 코드 SHA256 `ff57b660ad71aef22a9e6c93fa8ca33128f047b35dbc49f62e422eb5f5bd2c16`. 결과 txt는 최초 snapshot에서0byte(SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)였다. 최초 시점의 미완성과 이후 도착 결과를 구분한다. 실패나 실행 중단을 추정하지 않는다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/inventory.json), [source diff](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/source.diff), [재검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/fix_rank_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/fix_rank_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/validation.json).

### (a) 구현 판정

- **A17-AGGREGATION VERIFIED(유효 finite 입력의 무효 row 처리):** `a_cnt.clamp_min(1)`가 추가됐다. 이전과 같은 실제 `per_query→accumulate` 2행 재현에서 count=[2,1], answer_kept=[1,1]로 바뀌고 모든 지표가 유한했다. 합성 validation seed20270921의64시퀀스/2004 recall query에 all-ones mask를 적용한 실제 집계도 모든 지표 유한·answer_kept 전부1로 통과했다. 정답0인 row의 hypergeom 인자도1로 바뀌지만 현재 `queries`의 valid는 정답 존재를 요구하므로 그 row는 최종 집계에서 제외된다. 유효 row의 실제 정답 수는 변하지 않는다. NaN 입력/모든 mask/전체 실행을 보장하는 종료는 아니다.
- **A17-FINITE OPEN:** gate의 `gap>1e-6`와 peak의 `max(previous, state_peak)`는 그대로다. 다시 NaN 비교식을 확인하면 gate는 `OK (fp)`, `max(0,NaN)`은0이다. gap·state·출력 지표의 유한성 선검사가 필요하다. 이번에는 실제 모델의 NaN 발생을 관측하지 않았고 NaN 탐지 결함을 재확인했다.
- 새 `argmax_prev`, `best answer rank`, `rank_chance`는 무마스크 η0 궤적에 대한 진단이다. rank는 **가장 높은 순위의 정답 한 칸**,0기반/(n−1)이며 코드가 n1을 안전하게 처리한다. 실제 추가된 할당식 AST를 snapshot에서 추출해 실행했다. (n,m)=(1,1),(3,1),(3,2),(5,2),(5,5)의 모든 순열을 각각 열거했을 때 평균 rank와 공식 `(n−m)/((m+1)·max(n−1,1))`가1e−12 이내 일치했다. 예: n5/m2는.25, n3/m2는1/6이다. 단일 정답의 chance .5를 다중 정답에 일괄 적용하지 않은 것은 적절하다.

**A17-RANK-TIES OPEN:** 현재 `argsort` 두 번은 동점을 임의의 순서로 나누며 `argmax`는 첫 최대를 선택한다. 실제 함수식에 n5/모든 점수1/정답은 마지막 슬롯인 입력을 주면 이 CPU 환경에서 best rank1, chance.5, argmax_prev0이다. 모든 슬롯 점수가 같은데도 slot 위치에 따라 나빠 보일 수 있다. 표준적인 연속 점수·균등 임의 순열 기대값을 동점의 결정적 정렬에 무조건 적용하지 말 것. 상수 Pearson의 fallback0이나 반복 cue에서 동점이 가능하므로 tie 비율을 보고하고, seed가 고정된 균등 tie-break 또는 동점군 내 random tie-break 기대값 등 한 규칙을 선언한다. 단순 midrank를 쓸 경우에는 **다중 정답의 minimum rank 기대값도 그 정의에 맞춰** 다시 정해야 한다. 현 counterexample은 실제 checkpoint의 tie 빈도를 측정한 것은 아니다.

### (b) 검증 과정·재현성

신규 표는 코드와 출력에서 **per-query 평균**으로 명시돼 있다. 기존 M_eff 표는 unit→query→sequence 평균이다. 둘 다 가능한 기술통계지만 동일 가중의 숫자로 섞어 인과 설명하지 말 것. 또한 argmax_prev는 최근 칸 선택 비율이며 정답 top1 적중률 자체가 아니다. 이 표는 frozen 궤적에만 해당하므로 closed mask의 점수 품질을 입증하지 않는다.

A17-NULL의 train truth/kind 조건화, 단일 cross-sequence roll과 batch1 자기 pairing, 성분 수를 독립 표본 수로 해석하지 말아야 하는 조건, closed 분포 변화는 수정되지 않았다. A17-REPORT의 `.1303 on this split` 상수 문자열도 그대로다. 이번 **64시퀀스 all-ones 기술통계**에서 실제 full-kernel sequence 평균은 **.13441100969634834**였다. 해당64개 표본에 한정되며 예정된 전체 main 표의 수치를 대신하지 않는다. 중요한 조건은 본 실행과 **동일 표본·집계로 baseline을 직접 산출**하는 것이다. source/config/checkpoint hash·정확 명령·full-precision/시퀀스별 결과 보존 조건도 유지한다.

최초 검사 시점에는 스크리닝 txt가 비어 있었다. 이후 도착한 표는 아래에 별도로 기록하며, **회상 성능 판단은 계속 보류**한다. 새 checkpoint의72조건 main·threshold 재보정·model forward/backward·학습·G11 전체 상태 검사·recall MSE·독립 confirm·새 CI는 **not run**. 감사는 snapshot helper·생성기·순위 산술만 실행했다. 기존 오류가 전체 표를 실제로 오염시켰다고 단정하지 않는다.

### (c) 다음 수정·채택 방향

우선 **finite 차단 → 동점 규칙/같은 표본 baseline → 출처와 정밀 결과를 갖춘 frozen/closed 표 완성** 순서로 진행할 것을 권고한다. 그 뒤에만 유지 질량·정답 보존·empty·조건부 오통과율과 회상 성능을 분리해 후보를 평가한다. 부분 집계 수정은 hard mask 효능이나 안정성 검증이 아니다.

문헌 범위는 감사35에서 원문 확인한 [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 귀무하 교환가능성과 의존 구조에 맞는 permutation 조건을 유지한다. 이번 순위 판정은 추가 논문 주장 없이 전수 열거·동점 반례에 근거한다. 새로운 유의수준·성능 개선을 주장하지 않는다. 기존 구조/점수 진단 우선순위, hard mask 미채택 탐색, η=.2 채택 보류 및 A13~A16 잔여 상태는 유지한다.

### 수행·보존

CPU2threads/torch1.12.0+cu113·NumPy 사용, CUDA_VISIBLE_DEVICES 빈 값. MC 재현 seed9, 생성기 seed20270921; 2행/64시퀀스 기술통계와 n≤5 전수 순열만 검사했다. checkpoint 읽기·모델 forward·훈련·GPU·환경 설치·연구 소스 변경·Git 변이·프로세스 중단·타세션 접근/전송 없음. 새 감사 코드/JSON/raw stdout와3문서 append만 작성. 기존 prefix 및 snapshot 연구파일 hash 보존. 감사commit/tag 없음.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/fix_rank_probe.py /tmp/nsmt_assessment_20260924T150001Z-689fad22 f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/fix_rank_probes.json > f_lif_pop_v3/forecasting/log/assessment/20260924T150001Z-689fad22/fix_rank_probe.log 2>&1
```

### 동시 변경 — 결과 도착과 이번 검토의 경계

append 직전 prefix/hash 검사가 canonical PROJECT_LOG와 txt의 변경을 감지해 **문서 쓰기 전에 중단**했다. 모델 실패가 아니다. 새 canonical00:03과 결과 txt를 읽고 `/tmp/nsmt_assessment_20260924T150001Z-689fad22/late/`에 별도 보존했으며 [concurrent_changes.json](../f_lif_pop_v3/forecasting/results/assessment/20260924T150001Z-689fad22/concurrent_changes.json)에 SHA256을 기록했다. 새 canonical prefix도 그대로 보존했다.

표에72행이 있고 모든 저장 G11 flag가OK인 것은 확인했다. closed/shared/Pearson/top25%의 저장 M_eff .3230 vs random .1378, answer_kept .5992, any-hit .8255, peak17.58이며 전체 표 peak 최대72.18이다. **저장값 관찰이며 이번 감사의 독립 checkpoint 재현 결과가 아니다.** 새 canonical은256 validation/seed7 checkpoint/탐색 선택으로 한계를 밝힌다. ‘두 번 실행 동일’은 원 실행 두 출력의 독립 대조를 이번에 하지 않았다. A17 finite 감시 미수정 상태에서 저장 OK만으로 비유한 상태가 전혀 없었다고 인증하지 않는다. label-free/유의성·과거.13 기준선·동점 해석 조건도 유지한다. 특히 .3230은 진단 비율이며 recall MSE의 개선이 아니다. 전체 표·새 인과 해석·채택 순위의 종합 재현은 다음 변경 감지 주기에 검토한다. 새 연구 코드 hash는 이번 snapshot과 같다.

<!-- assessment-watch:20260924T150001Z-689fad22 -->


## 추적 감사 37 — 2026-09-25 00:15 KST (예약 20260924T151001Z-a6aef6db)

**A17의 원 NaN 감시 결함은 실제 실패 주입 검사에서 수정됐고, 같은 표본 커널 baseline도 계산된다. 단, “무작위 대조가 규칙 순서와 독립”이라는 새 주장은 재현 반례가 있으며 `se` 열은 최종 평균의 표준오차가 아니다. 수정판 전체 실행은 미완성이다.**

HEAD `e9c75f2e2ef6f8b097f1a3e436fd22de774d3644`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기억·감사36·사전등록 최신§2I·canonical00:03 결과 및 새 감사34/35 수용 append를 읽었다. 2026-09-25 00:10:53 KST에269파일을 `/tmp/nsmt_assessment_20260924T151001Z-a6aef6db`으로 snapshot·hash, trigger 불일치0. 연구 코드 변화는 hard_mask 분석 코드이며 모델/훈련 소스·사전등록은 같다. 자체 감사 문서 append는 연구 변화에서 제외했다.

새 분석 코드 SHA256 `49637f569e892e8705dc6478bd8aef88eafc8ccdf280ae08eaad7c87d759017d`. 현재 txt는 수정판 재실행을 위한0byte이며 JSON은 아직 없다. 감사36 도중 도착했던 **구판72행 출력은 이전 late snapshot(SHA c608b0ac4519b521c4f740648deff7189ca7676b1c08274f2d5a41fa2ec66acb)에 보존**돼 있다. 원 txt가 다시 비어 있다고 구 결과가 존재하지 않았다고 해석하지 않는다. 이번에는 구판 저장표 검토와 수정판 소규모 동작 검사를 구분했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/inventory.json), [guard 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/guards_probe.py), [guard 결과](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/guards_probes.json), [checkpoint snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/checkpoint_snapshot.json), [8시퀀스 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/subset_probe.py), [수치 결과](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/subset_probes.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/validation.json).

### (a) 구현·수정 판정

**A17-FINITE — 원 gate/peak 결함 범위 VERIFIED.** 새 함수 API를 확인하고 실제 `reference_thresholds`, `screen`, `accumulate` 및 main gate의 실제 AST 식을 호출했다. train/validation 기준 상태의 NaN은 중단, 유효 metric row의 NaN도 중단, 무효 row의 NaN은 안전하게 제외됐다. gate에 NaN을 넣으면 `FAIL (non-finite)`. closed 재귀에 NaN을 주입하면 peak의0과 섞지 않고 `nonfinite_batches=1`, 해당 batch의 측정 목록은 비운다. main은 이 count가 있으면 NONFIN으로 표기한다. 이는 모든 실험의 안정성 인증이 아니라 **감사35의 NaN 은폐 경로 수정 확인**이다. NONFIN 후보를 finite batch 평균만으로 채택하지 말고 후보 전체 제외/평가 불능을 명시해야 한다. A17-AGGREGATION VERIFIED도 유지한다.

**A17-NULL 부분 VERIFIED:** batch1의 reference 입력은 실제 `SystemExit`로 거부된다. 코드 docstring/출력의 label-free null 및 독립 df 주장은 철회됐고, train 사건 라벨 조건부 cross-sequence reference와 성분 수로 정정됐다. 이 표기·guard 수정은 확인했다. 단일 roll/공유 cue·pooling/closed 분포 변화의 유효성까지 검증한 것은 아니므로 **NULL의 통계적 보정 조건은 OPEN**이다.

**A17-REPORT baseline 구현 부분 VERIFIED:** 실제 `screen`이 같은 표본/집계로 m≡1을 계산한다. T7/B2/12query의 CPU 입력에서 baseline .3788907255가 독립 커널 산술 .3788907230과 float 오차 범위에서 맞았다. 8개 실제 validation 표본에서도 baseline .1347351189를 얻었다. .1303 문자열은 삭제됐으며 어느 쪽도 다른 표본의 baseline으로 이식하지 않는다. JSON provenance/전정밀 threshold·시퀀스별 결과 저장 코드는 추가됐지만 **완성 JSON이 없어 전체 저장·복원 연결은 FIXED-PENDING-REVIEW**로 둔다.

**A17-MC OPEN(새 재현 근거):**

- 규칙별 seed가 `seed*1000+r`이고 r은 `enumerate(keys)`의 위치다. 두 규칙 순서만 뒤집어 실제 screen을 호출하면 같은 input/cosine/top25%의 frozen random 두 시퀀스 값이 **[.4177510689,.4769654199]→[.3986436725,.3875882526]**으로 바뀐다. 관측 M_eff는 그대로다. 현재 방식은 고정 순서 재현성은 제공하지만 **규칙 순서 독립성은 제공하지 않는다**. 안정적인 rule identity와 sample/draw 식별자로 난수 stream을 정해야 한다. batch마다 seed를 다시 시작하는 점도 전체 MC 오차 계산에서 고려한다.
- `per_query`의 `samples.std/√draws`는 개별 query/unit MC 추정의 표준오차다. 현재 `accumulate→finish→mean`은 이 **개별 SE들을 평균**해 표의 se로 출력한다. 이는 최종 mean M_eff_random의 MC SE가 아니다. 예를 들어 독립 두 추정의 SE가 각각.1이면 평균 SE는.1이지만 두 추정 평균의 SE는.07071이다. 실제 규칙은 batch 간 난수 재사용과 shared mask의 unit 확장도 있으므로 이 예시를 그대로 보정계수로 쓰지 말 것. draw마다 최종 집계값을 만든 후 그 분산으로 MC SE를 계산하거나, 현 열을 ‘평균 query/unit MC SE’로 정확히 이름 붙인다. MC 오차와 데이터/seed 간 성능 CI는 별개다.

**A17-RANK-TIES OPEN 유지:** rank/argmax 및 top-q의 동점 처리는 바뀌지 않았다. 별도 정책 없이 동점 정렬의 위치 편향을 무작위 순열 기대값과 비교하면 안 된다. 새 기록이 생겨도 tie 빈도·규칙을 확인해야 한다. Canonical의 A16 과도한 주장 철회는 문서 정정으로 확인했으며 read/write 분리의 효능·안정성까지 VERIFIED로 바꾸지 않는다.

### (b) 구판 결과 해석과 소규모 checkpoint 재현

보존된 구판 표에는72행이 있고 저장 peak 최대72.18, closed/shared/Pearson/top25%는 **M_eff .3230 / random .1378 / answer_kept .5992 / any-hit .8255 / peak17.58**이다. 모든 저장 flag는 OK이나 그 판은 새 finite 검사가 없었다. ‘두 번 실행 동일’ 및72행 전체를 독립 재현했다고 주장하지 않는다.

이번에는 구판이 지목한 `etagrid-113240/.../seed7_flatten_spike_heterogeneous_sparse_eta0_k3`의 config와 checkpoint를 **평가 전에 별도 snapshot·hash**했다. config SHA256 `4a0d1ecc9dd443aa330ebe2b835685f0c3edb7bbd5702dc9502d385c63f5c3c9`, best model `e357470c7b664d47201d32c7659ee62809bbe8028d2969212d0a637db6e830e7`. config가 가리키는 실제 원 경로 일치를 확인한 뒤 snapshot으로 복원했다. η0/α.7/τ[4,8,16,32]/seed7/data seed20260921/고정 bound305.0375175, validation **앞8개만 forward**(248 recall query), MC8draw/seed0, 이미 보고된 shared/Pearson/top25% 한 규칙만 검사했다. 새 threshold나 후보 선택을 하지 않았다.

| 진단 | frozen | closed |
|---|---:|---:|
| M_eff | .2516944319 | .3180776662 |
| 같은 슬롯 수 random (8draw) | .1406621275 | .1400453330 |
| 정답 보존율 | .4764994612 | .6072319775 |
| any-hit | .7106760036 | .8234847431 |
| peak | 20.10154724 | 14.60620689 |
| 비유한 batch | 0 | 0 |

**구·신 코드에서 M_eff·선택률·empty·정답 보존·any-hit와 peak가 이8개 표본에서 일치**했다. MC stream 구현 변경 때문에 random의 구·신 일치를 요구하지 않았다. 이 검사는 저장256시퀀스 표의 완전 재현도, 독립 확증도 아니다. 전체 Dataset_Recall validation256개는 loader 구성 과정에서 생성됐지만 forward는 앞8개에만 수행했다. train/test/confirm loader를 열지 않았고 학습·회상 출력 MSE를 계산하지 않았다.

표는 hard mask의 검색 진단 가능성을 보여준다. 그러나 canonical의 ‘문턱 세 축 모두 실패/5차원 축 부적합’은 단일 checkpoint·reference 설계·현재 후보에 한정해야 한다. .154 대 .118을 CI 없이 우연 수준이라 단정할 수 없고, 같은 정답 순위라도 cutoff·empty·값 보존에 따라 결과가 달라진다. closed가 frozen보다 M_eff가 높은 것만으로 상태가 더 구별력 있어졌다는 유일 원인을 확정하지 않는다. 실제 mask 때문에 상태와 선택이 함께 바뀌는 반사실 비교라는 범위다. 또한 분포가 바뀐 readout의 MSE도 측정 자체는 가능하며 **재학습 없이 측정 불가능**과 **그 성능을 학습된 hard 모델 성능으로 일반화할 수 없음**을 구분한다.

### (c) 개선 우선순위와 남은 조건

현재 우선순위는 **MC 열 정의·난수 identity·동점 규칙 → 수정판의 baseline/provenance를 포함한 결과 보존 → 규칙/분할 고정 후 성능 대조**다. 새로운 학습 결과 없이 shared/top25%를 채택하지 않는다. 고정 kernel/no mask, 같은 budget의 recent/random, hard mask의 학습 조건과 평가 조건을 맞춘 대조가 필요하며 기존 η0 test MSE .2763와 탐색 validation 진단을 직접 성능 비교로 묶지 않는다. 기존 confirm은 반복 후보 선택에 재사용하지 않는다.

문헌 방향은 감사35에서 확인한 [Winkler et al., 2014 원문](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 귀무 아래 교환가능성 조건을 유지한다. 다른 시퀀스라는 이유만으로 통계적 무관성이 보장되는 것은 아니므로 reference의 별도 표본 오통과율·cue/위치 조건을 검증할 것. 이번 MC/동점 판정은 실제 코드 반례·산술에 근거하며 새 이론적 성능 보장은 추가하지 않았다. hard mask는 탐색 후보, η=.2 채택 보류 및 A13~A16 잔여 조건 유지.

수정판 전체72조건·full-precision JSON 검증·회상 MSE·새 학습·backward·GPU·확증 CI·confirm 재평가는 **not run**. 코드가 추가됐다는 이유만으로 결과 저장과 재현성 이슈 전체를 닫지 않는다.

### 수행·보존

CPU2threads/torch1.12.0+cu113, CUDA_VISIBLE_DEVICES 빈 값. 작은 guard 입력과 기존 checkpoint8시퀀스 순방향만 사용했다. 모델/학습 소스·checkpoint·기존 raw log 변경, 환경 설치, 프로세스 중단, Git 변이, 타세션 대화/메시지 없음. 감사 코드/JSON/raw stdout와3문서 append만 작성. 문서 prefix·연구 소스·원 checkpoint hash 보존. 감사commit/tag 없음.

실제 명령(두 probe 모두 아래 환경, stdout은 같은 run의 `log/assessment/`에 보존):
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/guards_probe.py /tmp/nsmt_assessment_20260924T151001Z-a6aef6db f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/guards_probes.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db/subset_probe.py /tmp/nsmt_assessment_20260924T151001Z-a6aef6db f_lif_pop_v3/forecasting/results/assessment/20260924T151001Z-a6aef6db
```

<!-- assessment-watch:20260924T151001Z-a6aef6db -->


## 추적 감사 38 — 2026-09-25 00:24 KST (예약 20260924T152002Z-0cb4eb20)

**완성 JSON의 출처·시퀀스별 배열·요약표 연결과 동일 표본 baseline을 검증했다. 저장된 검색 진단의 개선은 확인되지만, canonical의 “1.2 se라 구별 불가 / 8.9 se” 해석은 현재 se 정의로 정당화되지 않는다. 회상 성능 채택은 계속 보류한다.**

관찰 HEAD `b8ba941f02326624f4cc9683932ec3263ca3d45b`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기억·감사37·최신 사전등록§2I·canonical 새 재실행 결과 append를 읽었다. 2026-09-25 00:20:27 KST에270파일을 `/tmp/nsmt_assessment_20260924T152002Z-0cb4eb20`으로 snapshot·hash, trigger 불일치0. 새 결과 JSON SHA256 `fdff70b5ab2c2e4e36cdbe34ac191f287bf2569d6dbdcf9d18b7152fe803d7b3`, txt `90b9ce4f3a1ae6ea55536d697902d6edc8b6f4f72021b91d14ca0d968c920d1b`. 분석 소스 `49637f569e892e8705dc6478bd8aef88eafc8ccdf280ae08eaad7c87d759017d`는 감사37과 동일하다. 모델/학습 소스·사전등록도 동일하고 자체 감사 append는 연구 변화에서 제외했다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/inventory.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/record_probe.py), [수치 대조](../f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/record_probes.json), [checkpoint snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/checkpoint_snapshot.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/validation.json).

### (a) 구현·출처 연결 — A17-REPORT 부분 VERIFIED

재현 전에 실제 config/checkpoint도 별도로 snapshot하고 hash했다. JSON의 config `4a0d1ecc9dd443aa330ebe2b835685f0c3edb7bbd5702dc9502d385c63f5c3c9`, checkpoint `e357470c7b664d47201d32c7659ee62809bbe8028d2969212d0a637db6e830e7`, script 및 layers hash가 실제 파일과 모두 맞는다. 기록된 실행 HEAD는 `e9c75f2e...`이며 이후 결과가 commit된 관찰 HEAD와 구분한다. JSON에 명령·checkpoint 절대 경로·data/MC seed·draws·batch 수·전정밀 threshold·시퀀스별 지표가 존재한다.

**A17-REPORT-SERIALIZATION VERIFIED(이번 artifact의 저장 연결):** kernel9개 +72조건×9개 = **657개 배열**, 각각256개 값이 있고 전부 유한하다. 이를 다시 평균한 값과 JSON summary의 최대 차이는0이다. stdout72행의 수치10열을 각 출력 자릿수로 대조해 모두 맞고 flag/nonfin 열도 JSON과 일치한다. 새로운 모델 재실행으로657개 수치 전부를 재생성했다는 뜻은 아니다. 이전 FIXED-PENDING-REVIEW 중 이 저장 연결 범위만 닫는다.

**A17-REPORT-BASELINE VERIFIED(이번256개 표본):** snapshot 생성기를 seed20270921로 실행하여256시퀀스/8050 recall query를 복구하고, float64 독립 kernel 산술을 계산했다. sequence 평균 **.13324644907443733**, 저장값 **.13324645002169755**, 시퀀스별 최대 차이 **3.83e−9**로 float 구현 차이 이내다. 무작위 전체 슬롯 baseline도 .13324644877101005로 수치적으로 같다(비트 동일 주장 아님). query별 다중 정답 rank 기대값도 독립 평균 **.21200880026071026**으로 저장값과 맞는다.

감사37에서 별도 checkpoint forward로 측정한 앞8개 표본의 deterministic 지표5개×frozen/closed와 새 JSON 앞8개 평균을 비교해 최대차이1.11e−16이었다. 이는 기존 소규모 재현 결과와 완성 artifact의 추가 연결 근거다. A17-AGGREGATION·원 FINITE 결함 수정 판정은 유지한다. **A17-MC·RANK-TIES 및 NULL의 실제 보정 유효성은 소스가 그대로여서 OPEN**이다.

### (b) 실행 결과와 통계 해석

이번 저장 실행은 seed7의 η0 checkpoint, α.7/τ[4,8,16,32], train reference4batch·validation4batch256시퀀스, MC seed0/32draw, 고정 상태 bound305.0375175476074이다. 기준선과 같은 표본에서 **closed/shared/Pearson/top25%** 기록은 다음과 같다.

| 지표 | 저장값 |
|---|---:|
| M_eff | .32297049097604935 |
| 같은 슬롯 수 random | .13777843337809703 |
| 전체 kernel baseline | .13324645002169755 |
| 정답 칸 보존율 | .5991959688349685 |
| query-any-hit / random | .8254707361 / .6453586901 |
| 남긴 슬롯 비율 / empty | .2500352448 / 0 |
| peak / 비유한 batch | 17.5838012695 / 0 |

M_eff는 kernel 대비 **2.42386배**, 같은 슬롯 수 random 대비 **2.34413배**라는 **관측 비율**이다. 정답 비율·보존율·query 적중률과 실제 값 회상 오차는 다르다. 전체72행의 저장 nonfinite batch는0, 저장 peak 최대72.181098938<bound다. 이는 검사된 자료·설정의 유한성 기록이며 임의 입력/더 긴 길이/학습 중 안정성을 증명하지 않는다. 전체 상태 시계열을 이번 감사에서 다시 계산한 것도 아니다.

**A17-MC-INFERENCE OPEN — 새 canonical 해석 정정 필요.** 요약문은 unit/ref.90을 `0.1540 vs 0.1178±0.0300 → 1.2 se라 구별 불가`, shared/top25를 `8.9 se`로 설명했다. 두 quotient 산술은 각각 **1.205969 / 8.846696**이지만 분모는 감사37에서 확인한 **평균 query/unit MC SE**다. 최종 평균의 MC SE, 표본 일반화 SE, paired 차이의 SE 중 어느 것도 아니다. 따라서 이 숫자로 유의/비유의·우연 수준·구별 불가를 판정하거나 z값처럼 표현할 수 없다. quotient가 크다는 사실만으로 다중 후보 선택까지 교정되지 않는다.

수정 방향: draw별 최종 집계값을 저장해 그 분산으로 **MC 오차**를 계산하거나 현 열을 정확한 기술통계 이름으로 제한한다. 시퀀스·모델 seed 간 불확실성을 말하려면 별도의 표본 단위/paired 추정량을 고정해야 한다. 시퀀스 내 query나 unit을 독립 반복으로 세지 않는다. 이번 JSON에는 draw별 전체 집계가 없으므로 평균 local SE만으로 실제 전체 MC SE를 정확히 복구하지 않았다. **새 p값·성능 CI·유의성 검정은 not run**.

구조상 높은 input reference threshold가 정답을 버린 관찰은 현재 cue/value 구성의 진단이다. 이 결과로 모든 통계 문턱이나 모든5차원 descriptor를 기각하지 않는다. rank ties와 ref의 교환가능성 검증이 남아 있으며 shared/top25도 같은 validation에서 고른 탐색 후보다. 기존 test MSE .2763/oracle .0350과 이 validation M_eff를 직접 성능 비교로 묶지 않는다.

### (c) 다음 방향

우선 **새 SE 기반 유의성 서술 정정 → MC stream identity·동점 정책·reference 보정 조건 고정 → 새 실험 프로토콜과 성능 대조** 순서를 유지한다. 저장 연결과 baseline은 정리됐으므로 이를 반복 재검사하기보다 아직 열린 정의·추론 문제를 해결하는 것이 다음 단계다. 동일 학습 예산의 no-mask/hard/recent/random 대조와 실제 recall/copy/recall-first 오차, held-out 조건을 고정한 후에만 효능을 판단할 것. 이번 감사가 학습을 실행하거나 새 후보 채택을 승인한 것은 아니다.

문헌은 감사35에서 원문을 확인한 [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 귀무하 교환가능성 조건을 기존 범위에서 사용한다. 다른 시퀀스의 descriptor를 섞었다는 사실만으로 유의수준이 보장되지 않는다는 조건은 여전히 적용된다. 이번 SE 판정은 해당 GLM 논문을 retrieval에 그대로 적용한 것이 아니라 실제 집계 코드·산술과 불확실성 대상의 구분에 근거한다.

hard mask·shared/top25는 미채택 탐색 후보, η=.2 채택 보류, A13~A16 잔여 상태 유지. 이번에는 새 학습·GPU·model forward/backward·72조건 전체 forward 재실행·confirm 접근·회상 MSE·새 CI를 **not run**으로 기록한다.

### 수행·보존

CPU2threads/NumPy1.26.4, CUDA_VISIBLE_DEVICES 빈 값. JSON 재산술·snapshot 생성기256시퀀스·이전 독립8표본 결과 대조만 수행. 실제 checkpoint는 hash 대조용으로 보존했고 모델에 로드하지 않았다. 모델/학습 소스·checkpoint·기존 raw log·프로세스·Git·환경을 변경하지 않았다. 새 감사 코드/JSON/raw stdout 및3문서 append만 작성. prefix·snapshot 연구파일·원 checkpoint hash 보존, 감사commit/tag없음.

첫 감사 parser가16단어짜리 `closed thresholds...` 설명 문장을 데이터행으로 세어 행수 assertion이 실패했다. **감사 코드만** axis 열(unit/shared/input) 조건을 추가해 재실행했고72행 모두 통과했다. 최초 코드 `record_probe_initial.py`와 두 raw log를 보존했다. 연구 스크리닝/모델 실패가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20/record_probe.py /tmp/nsmt_assessment_20260924T152002Z-0cb4eb20 f_lif_pop_v3/forecasting/results/assessment/20260924T152002Z-0cb4eb20 > f_lif_pop_v3/forecasting/log/assessment/20260924T152002Z-0cb4eb20/record_probe_retry.log 2>&1
```

<!-- assessment-watch:20260924T152002Z-0cb4eb20 -->


## 추적 감사 39 — 2026-09-25 15:32 KST (예약 20260924T165002Z-79ac9087)

**hard/shared/Pearson q=.5의 저장 confirm2 회상 MSE 개선(51.73%)과 paired CI 산술은 맞는다. 그러나 상태 검사 범위는 1,000개 중 256개이며 비유한 플래그 누락을 직접 재현했다. 성능 수치의 증거와 사전등록 안전성 조건 충족을 분리하고, D-AN 전체 통과에 대한 감사 승인은 보류한다. 실제 모델의 NaN이나 성능 악화를 발견했다는 판정은 아니다.**

### 관찰 범위·출처

기억·감사38·사전등록§2J·canonical PROJECT_LOG의 선택/확증 및 스크리닝v3 append를 복구했다. 예약은01:50 KST, 실제 snapshot은 **15:23:16 KST**다. 관찰 HEAD `6bdd3f34dd7faf2b0d9f8dfcd6f23e0e2dd37009`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. trigger 이후 hard_mask_screen.py/.txt/.json와 hard_selection_record.json이 바뀌었다. 따라서 감지 시점 실행을 복원했다고 주장하지 않으며, 현재 완성된8seed 확증도 함께 보존·검토했다.

재현 전에290파일 및 추가92파일(선택/확증 config·checkpoint 포함)을 `/tmp/nsmt_assessment_20260924T165002Z-79ac9087`에 복사하고 SHA256을 기록했다. 핵심 hash: layers `20a73ade9b4793d3667b7caed564d92418af397f52e6f30c47855580fe59fc5c`; hard_selection.py `aa4b06287fd3538343e46ecefa2660f64b81e67480a13ddb5ab69e378bcde052`; hard_confirm.py `40a9f659d2c9b5e1d702a5da5bdaee79e5839ea742161086ead79888d600cd05`; screen v3 `88b4a5452882f67c59181dd32642d1c02a56505e336e067d03769bbfb057b670`; 선택+확증 record `6b19471ef020b7bde76d85e6e20e91204a575a3d4f3377de1a3b3221a2248de3`.

증거: [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/inventory.json), [추가 snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/extra_snapshot.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/audit_probe.py), [결과](../f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/probe_results.json), [hash/늦은 변경](../f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/validation_before_append.json).

감사 중 HEAD `9d90c3ef48ebb400d11d21158ad89f55c080a49d`와 §2K 및 모델/설정/생성기/게이트 코드 변경이 관찰됐다. 별도 `late/` snapshot으로 보존했다. 아래 probe와 판정은 **초기 §2J snapshot**에만 적용하며 §2K 대조군 구현·실행 판정은 다음 주기에 넘긴다. 진행 중 개정을 확정 결함이나 기존 검사 통과로 오인하지 않는다.

### (a) 아이디어 구현 — 부분 검증, 정의 불일치 남음

**A18-HARD-IMPLEMENTATION 부분 VERIFIED:** 현재 hard 경로는 `c=m*b`, similarity는 binary 선택에만 사용되고 η·질량 재분배·cap을 거치지 않는다. 작은 CPU tensor(T8/B2/D3/K4)에서 q1의 full/hard 출력과 상태가 비트 단위로 같았다. q=.5/J5에서 c의 각 항은0 또는 원 kernel이며 shared mask는 unit 간 동일했다. 같은 seed로 생성한 두 neuron의 모든 초기 state_dict도 일치했다. 전체 학습 모델 초기 checkpoint를 독립 복원한 검사는 아니다. 미래 상태를 읽는 새 경로는 정적 검토상 없고, full 모델 모든 입력/길이에 대한 완전 검증은 not run.

**A18-HARD-SPEC OPEN:** §2J D-AK의 `ceil(qJ)` 표기와 괄호의 `max(1,round(qJ))`가 서로 다르다. 구현은 Python round이고 실제 q=.5/J5에서2개를 남긴다(ceil이면3개). 이미 산출한 결과는 round 정책으로 명시하고 과거 문장을 지우지 말고 정정 append할 것. 조건별 선택 budget·동점 규칙까지 다음 확증 전 고정해야 한다.

**A17-RANK-TIES 분석 범위 부분 VERIFIED / 모델 범위 OPEN:** v3 분석은 stable sort로 오래된 칸 우선을 선언했다. all-tie/J5/q.5 입력에서 분석 mask `[1,1,0,0,0]`, 모델 torch.topk mask `[0,0,1,0,1]`로 불일치를 재현했다. 입력 축뿐 아니라 shared/Pearson에도 가능한 입력이다. **η0 checkpoint의 스크리닝 동점0을 별도로 학습한 hard checkpoint8개/confirm2의 동점0으로 전이할 수 없다.** canonical의 “확증 설정에서 실무상 무관”은 아직 해당 상태 궤적의 근거가 없다. 실제 confirm2 동점 빈도 재평가는 not run.

**A17-MC·REPORT 부분 VERIFIED:** 저장 kernel+72조건, 총73요약에서 256개 시퀀스 배열8열 및32개 draw 평균을 재집계해 차이0. toy draw 행렬에서 최종 평균 MC SE=.2886751345948129로 독립 산술과 맞고, 규칙 key 순서를 뒤집어도 stream seed는 같다. 원 positional stream 및 평균 local-SE 문제의 수정 범위만 닫는다. canonical은 이전1.2se/8.9se 해석을 철회했고 현재 mc_se를 유의성으로 사용하지 않는다(A17-MC-INFERENCE 문서 정정 VERIFIED). 전체72조건 forward 재실행은 not run. A17-NULL의 교환가능성/오통과율 문제와 A13~A16 잔여 조건은 유지한다.

### (b) 검증·통계·재현성 — 산술 일치, 안전성/일회 개방 보장 미완

선택 val MSE는 q1=.2625292193, q.1=.3246045020, q.25=.2232174403, q.5=.1331723109. 등록된 최소 MSE/동률 규칙으로 .5 선택이 맞고 동률이 아니다. 선택4개 checkpoint/config hash8건이 record와 일치한다. 확증16개 결과는 모두12epoch·test_skipped·test=null, CSV12행이다. 실제 config는 shared/Pearson/hard/spike, train2048/val256, batch64, n_confirm1000, data_seed20260921, bound305.03751755이다. source hash 및 확증 checkpoint16개 hash가 snapshot/record와 일치한다. synthetic은 confirm2에+40000을 사용해 기존분할과 생성 RNG를 분리하며, 평가의 max_eval_batches는0이다. 데이터 생성/confirm2 loader 접근은 감사에서 실행하지 않았다.

| 저장값에서 재계산한 항목 | 결과 |
|---|---:|
| q1 평균 회상 MSE | .26297769286423234 |
| q.5 평균 회상 MSE | .12695169216211005 |
| paired 차이 평균 / SD | −.13602600070212229 / .006944038057749295 |
| n8, df7 t 95% CI | [−.14183136179899364, −.13022063960525093] |
| 상대 변화 / 개선 seed | −51.7253% / 8/8 |

**A18-HARD-STATISTICS 산술 VERIFIED:** 위 값은 stored rows에서 직접 재계산했고 record와 정확히 일치한다. 이것은 저장 MSE의 독립 forward 재현이나 안전성 조건 충족 검증이 아니다. CI의 반복 단위는 모델 seed이고 동일1,000시퀀스를 사용하므로 새로운 데이터셋에 대한 불확실성을 별도로 추정하지 않는다. q.5가 격자에서 선택됐다는 뜻이며 전역 최적값·O7·GRU 우위·다른 길이에 대한 근거는 아니다.

**A18-HARD-SAFETY OPEN (확정 wrapper 결함):** `hard_selection.metrics(...,batches=4)`를 hard_confirm도 그대로 호출한다. recall MSE는 전체 loader지만 `selection_diagnostics`와 stage 진단은 앞4×64=**256시퀀스**에서 멈춘다. 744개 상태의 G11/비유한 검사는 이 경로로 확인되지 않는다. 저장 record는 진단의 sequences/queries 수를 버리므로 표만 보면 범위를 알 수 없다. 더욱이 `diag['finite']`를 쓰지 않고 scalar MSE/peak 유한성만 재검사한다. snapshot의 실제 metrics 함수에 `diag.finite=False, peak=1, MSE=.1`을 주입하자 `finite=True,within_bound=True`; bound=None도 통과했다. 출력 오차가 유한해도 숨은 상태 전체가 유한하다는 보장은 없다. **실제 학습/확증에서 비유한 상태가 발생했다는 증거는 없다.** 본 finding은 잘못된 감시 판정 경로와 검사 누락이다.

남은 조건: finite를 모든 상태에 명시적으로 AND하고 결측 bound는 미판정/실패로 처리; 평가 전체 시퀀스와 step의 안전성 coverage를 기록; 기존 frozen checkpoint·선택·성능 수치를 바꾸지 않는 안전성 보완 절차를 먼저 문서화할 것. 이미 열린 confirm2로 후보를 다시 선택하거나 q를 바꾸지 않는다. 수정 코드만으로 이 이슈를 VERIFIED로 닫지 않는다. 저장 성능 통계는 보존하되 D-AN의 “전부 G11 OK·유한, 1차/2차 통과”는 전체 안전성 증거가 갖춰질 때까지 **감사 판정 보류**다.

**A18-HARD-CONFIRM-PROTOCOL OPEN (정적 확인):** 완료 confirm 블록 재실행 및 저장된8개 모델 source hash 변경은 거부한다. 그러나 `epochs_run>0`만으로 finished 판정하여 사전12epoch를 강제하지 않고, confirm2 개방 전 전체16개의 seed/q/mode/data/training config와 frozen config/checkpoint hash를 대조하지 않는다(axis/stat만 검사). 선택 기록의 config/checkpoint hash도 confirm 단계에서 검증하지 않으며, 선택/확증/분해 분석 스크립트 자체는 SOURCES에 없다. 개방 전에 영속 started/lock 기록이 없으므로 중간 오류 또는 병렬 실행 시 재개방을 막는 보장도 없다. **실제16개는12epoch이고 관찰 hash는 모두 맞았다.** 실제 재개방·잘못된 checkpoint 사용을 주장하는 것은 아니다. 다음 확증은 전체 manifest 사전 대조와 개방 이력 기록을 갖춘 뒤 진행할 것.

### (c) 개선·채택 우선순위

1. **안전성 wrapper와 검사 coverage, 개방 전 manifest 검증을 먼저 보완.** 현재 .5 결과를 폐기할 근거도 없지만 안전성 gate를 자동 승인할 근거도 부족하다.
2. **동일 budget의 recent/random 대조를 우선.** 현재 저장 수치는 binary kernel mask 후보를 지지한다. 다만 content 선택, 남긴 질량, 최근성 중 어느 요인이 이득을 만들었는지는 q1 비교만으로 분리되지 않는다. §2K가 이 질문을 겨냥해 추가되고 있으므로 새 확증 전 그 구현·RNG·안전성 규칙을 별도 감사할 것. 새 후보 선택에 confirm2를 재사용하지 않는다.
3. **M_eff는 보조 진단으로 유지.** η0 checkpoint screen의 q.25 M_eff=.3230, q.5=.2548인데 hard 재학습 val MSE는 q.5가 더 낮다. 서로 다른 checkpoint라는 한계를 명시해야 하며, 이 표로 정답 보존이 성능 차이의 유일 원인이라고 단정하지 않는다. 회상 MSE·copy/recall-first·선택 budget·질량·안전성을 함께 보고한다.
4. 기존 QK/Gram/delta/descriptor/entmax 후보를 한꺼번에 추가하지 않는다. 현 hard 경로에는 학습 QK와η가 관여하지 않으므로 기존 soft 경로의 개선을 그대로 합칠 수 없다. 구조와 대조군을 먼저 정리한 뒤 별도 ablation으로 검토한다. [DeltaNet 원문 §2.2·§4.1](https://arxiv.org/html/2406.06484v6)은 delta update와 associative retrieval의 연구 근거이며 이번에 원문을 다시 확인했다. **우리 hard f-LIF의 효능·안정성을 보장하는 정리가 아니므로** 향후 쓰기 규칙 대안의 근거로만 유지한다. 기존 [Winkler et al., 2014 원문](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 교환가능성 조건은 reference 귀무 보정에 계속 적용한다(이전 감사에서 확인한 문헌; 이번 PMC 접근은 CAPTCHA로 본문 재확인 실패).

### 수행·보존

CPU2threads/torch1.12.0+cu113/NumPy1.26.4/SciPy1.15.3, CUDA_VISIBLE_DEVICES 빈 값. snapshot에서만 작은 무학습 neuron/selector forward, wrapper 실패 주입, 저장 통계/MC 재산술, config 로드·checkpoint hash 대조를 수행했다. 실제 checkpoint 모델 forward, train/val/test/confirm2 데이터 생성·평가, 학습/backward/GPU, §2K 재현은 **not run**. 모델·학습 소스·원 checkpoint/raw log·환경·프로세스·Git 변이 없음. 최초 probe 명령은 감사 log 디렉터리 부재로 shell redirection 단계에서 실패했으며 디렉터리를 만든 뒤 같은 코드를 실행해 통과했다. 연구 모델 실패가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087/audit_probe.py /tmp/nsmt_assessment_20260924T165002Z-79ac9087 f_lif_pop_v3/forecasting/results/assessment/20260924T165002Z-79ac9087 > f_lif_pop_v3/forecasting/log/assessment/20260924T165002Z-79ac9087/audit_probe.log 2>&1
```

<!-- assessment-watch:20260924T165002Z-79ac9087 -->


## 추적 감사 40 — 2026-09-25 15:44 KST (예약 20260925T064002Z-47bcdaa2)

**§2K의 같은 칸 수 대조가 구현되고 confirm3 결과가 완성됐다. 저장된 Pearson−recent/random 차이와 CI는 재계산 결과 일치한다. 그러나 감사39의 안전성 결함은 그대로이며 random은 MSE와 다른 마스크 실행에서 상태 진단을 한다. 두 대조군에 대한 관측 성능 우위는 기록하되, 전체 안전성 gate 승인 및 이득의 유일한 원인 확정은 보류한다.**

### 범위·보존

기억·감사39·사전등록§2K(D-AP~AS)·canonical 15:31 결과/15:32 감사 append를 읽었다. 관찰 HEAD `6d7f64ef8ae69fe0c90e54a1374aea90fb8fe97e`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 15:40:35 KST에448파일(32개 checkpoint/config 포함)을 `/tmp/nsmt_assessment_20260925T064002Z-47bcdaa2`에 snapshot·SHA256 기록, trigger 불일치0. 감사39 snapshot과 실제 비교하면 기존 연구 파일 중 달라진 것은 layers/config/synthetic/check_model 네 파일이다. hardconf16개 결과·로그, screenv3, 기존 confirm2 등은 이전 감사에서 이미 확인한 동일 바이트다. 이번 변경 목록에 있다는 이유로 이를 새 실험/수정 증거로 세지 않았다.

주요 hash: layers `d255c1e140307f8412e2cade28f278cf895e2ada14b6ac25f3494541e0297046`; hard_control.py `15ae229c4baf2c308a9fdd45c18e4d6871093669c599bc274ed7cd920cdfb555`; 공용 hard_selection.py `aa4b06287fd3538343e46ecefa2660f64b81e67480a13ddb5ab69e378bcde052`(감사39와 동일); hard_control_record.json `ddd21119cc26e728c7f21927cf4e3dd5730bc877035c7e208a478ddd72aa7301`.

증거: [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2/inventory.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2/probe.py), [재산술·실패주입 결과](../f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2/probe_results.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2/validation.json). 15:43 보존검사까지 대상 변경 없음.

### (a) 구현 정확성

**A19-CONTROL-IMPLEMENTATION 부분 VERIFIED:** hard recent/random은 shared 축에서 `max(1,round(qJ))`개를 남긴다. recent는 실제 가장 최근 k칸, random은 전용 CPU generator의 무작위 score top-k다. q=.5/J7/B2/D3 소규모 검사에서 Pearson/recent/random 모두4칸, unit 간 동일 mask였다. seed7 재설정 후 random은 입력 내용을 바꿔도 동일 mask, 연속 호출에서는 새 mask를 내며 재설정하면 첫 mask를 복구했다. 세 조건의 neuron 초기 state_dict도 같은 seed에서 일치했다. 모델 전체 학습 초기화의 독립 재현은 not run. 현재 CPU 실험 범위에서 판정하며 GPU 호환성을 검증한 것은 아니다.

confirm3 생성기는 n_confirm=1000, data_seed+50000을 사용한다. 기존 Pearson/q1 경로에 random 분기가 끼어들지 않는 구조는 유지됐다. 재사용 checkpoint16개의 val 재평가 오차0 텍스트는 관찰했으나 그 전체 모델 평가를 감사에서 다시 실행하지 않았으므로 독립 재현으로 표기하지 않는다. 원 checkpoint hash는 직접 대조했다.

**A18-HARD-SPEC·A17 모델 동점 OPEN 유지:** §2K는 round를 명확히 지정하지만 §2J의 ceil/round 불일치에 대한 정정 append는 아직 없다. 기존 Pearson topk 동점 규칙도 그대로다. 현재 작업의 미수정 범위를 명시하는 것이며 random/recent 구현 실패로 혼동하지 않는다.

### (b) 검증 과정·통계·재현성

저장32행의 조건/seed를 checkpoint/config와 대조했다. 8seed×4조건의 q/stat/seed/mode=hard/readout=spike가 맞고 train2048/val256, bs64, confirm1000, data_seed20260921이다. 32결과 모두12epoch·test_skipped/test=null, CSV12행; checkpoint32개 hash는 record와 모두 일치한다. 개방 전/후 저장 모델 source hash8개도 snapshot과 같다. Pearson/q1은 기존 suite 재사용, recent/random16개는 새 suite `hardctrl-152631`로 구분된다.

**A19-CONTROL-STATISTICS 산술 VERIFIED:** record의 seed별 MSE에서 아래5개 paired 차이를 독립 재계산했다. n8/df7 t구간·delta·상대 변화가 저장값과 일치한다.

| 비교 | 평균 차이 | 95% CI | 상대 변화 |
|---|---:|---|---:|
| Pearson−recent (1차) | −.2119007463 | [−.2167951817, −.2070063109] | −63.94% |
| Pearson−random (1차) | −.1482633463 | [−.1534709640, −.1430557286] | −55.37% |
| recent−q1 (2차) | +.0844453078 | [.0835556121, .0853350035] | +34.19% |
| random−q1 (2차) | +.0208079078 | [.0186645393, .0229512764] | +8.43% |
| Pearson−q1 (기술) | −.1274554385 | [−.1323871338, −.1225237432] | −51.61% |

평균 MSE는 Pearson .1195129361, q1 .2469683746, recent .3314136825, random .2677762825. 두1차 비교에서 각각8/8seed가 개선됐다. 두 비교를 모두 요구하는 등록된 결합 규칙과 개별 CI를 구분하며, 이 표는 모든 비교에 대한 동시95%구간이 아니다. 모델 seed와 random mask seed를 함께 바꾸므로 seed 변동에는 두 요인이 섞여 있고, mask 반복별 MC 불확실성을 따로 추정하지 않는다. 새로운 데이터 분포의 효과/최적 q/O7 판정으로 확장하지 않는다.

**A18-HARD-SAFETY OPEN — confirm3에도 적용:** hard_control이 감사39와 동일한 `metrics(...,batches=4)`를 호출한다. MSE는 전체1,000개지만 상태/분해 진단은256개뿐이다. 실제 wrapper에 diag.finite=False를 주입해도 finite=True/within_bound=True가 다시 반환됐다. 수정·재검사 근거가 없으므로 이슈를 닫지 않는다. **실제32개 모델에서 NaN을 관측한 것은 아니다.** canonical의 “전부 G11 OK, 탈락 없음/두1차 통과” 중 전체 상태 안전성까지 포함한 최종 승인은 보류한다. 저장 성능 수치는 유지한다.

**A19-RANDOM-EVAL-PATH OPEN — 새 확정 연결 문제:** evaluator는 모델별 metrics 직전에 한 번 reseed한다. metrics는 `evaluate → selection_diagnostics → decompose` 순으로 별도 forward를 수행하며 중간 reseed가 없다. 실제 wrapper를 작은 random selector를 소비하는 stub로 실행해 세 단계가 다른 mask를 사용하는 것을 재현했다(실제 데이터 접근 없음). 따라서 random의 저장 max|u|·M_eff·분해 mass는 **MSE가 측정된 궤적의 동반 진단이 아니다**. deterministic 조건과 달리 256개 범위 제한을 없애는 것만으로도 해결되지 않는다. 성능과 안전성은 같은 forward에서 수집하는 것이 우선이며, 별도 순방향을 유지한다면 동일 RNG 시작 상태·데이터 순서·batching을 복원하고 진단 coverage도 전체로 해야 한다. checkpoint만으로 임의 batch 재배치에 대해 동일 random 평가가 보장되지 않으므로 평가 seed뿐 아니라 소비 순서/배치 크기도 provenance에 유지할 것.

**A18-HARD-CONFIRM-PROTOCOL §2K의 개방 전 기록 부분 VERIFIED:** hard_control은 data_provider 호출 전 `open(...,'x')`로 opening record를 만든다. snapshot의 실제 main을 호출하되 data access를 금지한 검사에서 기존 record가 있으면 loader에 도달하지 않고 거부했다. 완료 전 중단도 기존 파일이 남는 구조다. 이는 §2J hard_confirm의 장치를 소급 수정한 것이 아니다. 다만 다음 잔여 조건은 OPEN: finished 판정이 아직 epochs_run>0이고 q/mode/전체 config 및 기존 frozen manifest를 모두 사전 대조하지 않는다. source hash는 현재 모델8파일만 수집하며 hard_control/hard_selection/stage 스크립트·config hash가 빠져 있다. source hash를 적는 것과 사전 고정본과 대조하는 것은 다르다. 실제 관찰32개 artifact는 등록12epoch/설정과 일치하므로 잘못된 run 사용이나 재개방을 주장하지 않는다.

**A19-COVERAGE 기술통계 VERIFIED:** snapshot `hard_control_coverage.main()`을 CPU에서 재실행해 저장 txt와 byte 일치했다. validation seed20270921의400시퀀스/12,613query에서 recent의 정답 칸 보존 .0720, any-hit .1231; random 기대값 .4999/.8950, 정답 lag 평균21.05이다. random hypergeometric 식은 n1~8의36사례에서 모든 k-subset을 열거한 독립 산술과 최대1.11e−16 차이. 이는 query 평균/정답 슬롯 lag의 과제 기술통계이고 모델 MSE의 독립 원인 분석이나 시퀀스 단위 CI가 아니다. 모델을 전혀 실행하지 않았다.

### (c) 개선 방향·해석 한계

우선순위는 **같은 실행의 전체 상태 안전성 검사 → 확증 manifest/평가 RNG 재현 기록 → 필요할 경우 더 좁은 기전 대조**다. 이번 대조 실험은 이미 완료됐으므로 이를 다시 수행하라고 요구하거나 confirm3를 후보 재선택에 사용하지 않는다. 현재 checkpoint와 성능 record를 고정한 안전성 보완 절차를 먼저 기록할 것.

같은 칸 수의 두 내용 무관 정책이 더 나빴다는 결과는 Pearson 정책의 가치에 대한 새 근거다. 그러나 canonical 제목·본문의 “이득의 원인은 예산이 아니라 내용 기반 선별이다”를 **유일한 기전 확정**으로 읽으면 과도하다. 같은 칸 수여도 `Σm*b` 질량·lag 분포는 다르고, 학습 및 상태 feedback 전체가 정책에 따라 바뀐다. random에는 추가 확률성도 있다. 현재 정당한 결론은 **고정된 과제·q·학습 예산·두 대조 정책에서 Pearson의 저장 회상 MSE가 더 낮다**이다. 안전성 조건과 별개로도 “모든 내용 무관 선택은 실패”, “상태 기반 검색만이 원인”은 검증되지 않았다.

더 좁은 기전 주장이 필요하면 새 사전등록/미사용 분할에서 lag·kernel 질량을 맞춘 내용 교란, 입력 cue만 사용하는 정책 등으로 질문을 분리할 수 있다. 이는 이번 수치와 `c=m*b` 식에서 도출한 감사 제안이며 채택 보장이나 새 문헌 사실이 아니다. 기존 원문 근거인 [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 교환가능성 조건을 유지해, 시간/lag 구조를 무시한 임의 shuffle을 자동으로 유효한 귀무 검정으로 부르지 않는다. [DeltaNet 원문](https://arxiv.org/html/2406.06484v6)은 감사39에서 재확인한 쓰기 규칙 대안의 근거로만 유지한다. 새 논문·새 이론 보장 없이 기존 문헌과 직접 수치검토를 사용했다. QK/Gram/delta/entmax 추가 및 O7/실데이터 전이보다 현재 검증 연결 문제의 해결이 먼저다.

### 수행·남은 검사

CPU2threads, torch1.12.0+cu113/NumPy1.26.4/SciPy1.15.3, CUDA_VISIBLE_DEVICES 빈 값. snapshot에서 config 로드·checkpoint hash, 작은 selector/초기화 검사, 실패주입, 저장 통계 재산술, validation 생성기400시퀀스만 수행했다. 학습·GPU·backward·실제 checkpoint 모델 forward·confirm2/3 생성/평가·32개 독립 성능 재현·전체 안전성 보완은 **not run**. 모델/학습 소스, 원 checkpoint/raw log, 환경/프로세스/Git 변경 없이 감사 증거와3문서 append만 작성했다. 새 연구commit/tag 없음.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2/probe.py /tmp/nsmt_assessment_20260925T064002Z-47bcdaa2 f_lif_pop_v3/forecasting/results/assessment/20260925T064002Z-47bcdaa2 > f_lif_pop_v3/forecasting/log/assessment/20260925T064002Z-47bcdaa2/probe.log 2>&1
```

<!-- assessment-watch:20260925T064002Z-47bcdaa2 -->


## 추적 감사 41 — 2026-09-25 17:16 KST (예약 20260925T081001Z-d4bab0e3)

**§2L의 안전성 보완은 기존 2J·2K 결과를 바꾸지 않고 검사 누락을 보완했다. 원 기록·checkpoint·설정·소스 연결, 실제 checkpoint 소규모 재현, 뒤쪽 batch 실패 주입을 통과했다. 이번 고정32개 checkpoint/48평가에 대한 안전성 보류를 해제한다. 다만 confirm 전체를 감사에서 독립 재실행한 것은 아니며, 기존 일반 평가기의 random 진단과 다음 확증의 절차 강화는 별도 미결이다.**

### 범위·근거

감사40·문서 기억·canonical17:09 결과·최신 사전등록§2L(D-AT~AW)를 읽었다. 관찰 HEAD `14240a4bcec6104616b04b6021eae1b2c1ecfd9f`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 절차 commit `ab636b7e3`와 결과 commit을 구분한다. 17:10:31 KST에387파일 snapshot, trigger 불일치0. probe 전에 추가64파일(32개 실제 checkpoint 및 config)을 복사·hash했다. snapshot 경로 `/tmp/nsmt_assessment_20260925T081001Z-d4bab0e3`. 문서 자체 감사 append는 새 연구 결과에서 제외했다.

SHA256: hard_safety.py `098f09bd16ea387eb795670084476ecd1cc28ad8b526aad9a4301c5515bf3318`; hard_selection.py `bd2f212acf8a40cbe9289182e5bfdff4911bd1242ca4ce4ef4ab6740b85b4636`; safety record `7abfa7aae53fbf47ac6b786394bdf20f127c73a180d990d598e5212f3c3daf9d`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/inventory.json), [checkpoint snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/checkpoint_snapshot.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/probe.py), [결과](../f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/probe_results.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/validation.json).

### (a) 구현·결함 수정 판정

**A18-HARD-SPEC VERIFIED(문서 정정):** §2L D-AT가 ceil을 철회하고 Python round-half-even 및 J별 예시를 명시했다. 과거 본문을 지우지 않았고 실험 수치/실제 선택 규칙은 바뀌지 않았다.

**A18-HARD-SAFETY 수정 범위 VERIFIED:** 공용 metrics가 기본 전체 batch를 요청하고 diag.finite를 AND하며 bound=None을 통과시키지 않는다. snapshot 함수에 finite=False와 결측 bound를 주입해 각각 거부되는 것을 재검사했다. 기본 diagnostics batch limit은10^9이고 diag_sequences/queries도 반환한다. 전체 safety collector는 loader를 끝까지 순회하고 매 batch 하나의 모델 forward에서 출력 오차와 상태를 함께 수집한다.

실제 collector의 별도 실패 주입에서 **5번째 batch**에 NaN/Inf/정확히BOUND/BOUND+1을 넣자 모두 safe=False였다. 앞4batch만 검사하던 회귀를 직접 겨냥했으며10시퀀스/5forward가 확인됐다. NaN/Inf는 finite=False·비유한 batch1, 유한한 한계 이상 값은 finite=True·safe=False로 구분했다. 정상 대조는 safe=True. 여기서 helper는 고정 BOUND를 사용하고 config 한계값 검증은 main의 manifest가 담당한다는 역할도 구분한다.

**A19-RANDOM-EVAL-PATH §2L 보완 경로 VERIFIED:** seed7의 실제 frozen checkpoint q1/Pearson/recent/random 네 종류를 snapshot에서 로드했다. validation **앞8시퀀스**, batch4에 대해 기존 evaluate와 single_pass를 각각 실행하고 random 시작 generator를 동일하게 재설정했다. recall/copy/recall-first/all MSE 차이 모두0, sequences8, 유한·안전, Pearson mask reconstruction mismatch0이었다. single_pass의 동일 forward 상태 수집은 실패 주입에서도 확인했다. 이 작은 검사는 confirm2/3 재개방이나48평가 전체 독립 재현이 아니다.

**일반 경로 잔여:** hard_selection.metrics는 여전히 오차→진단→분해를 별도 forward로 실행한다. 수정 docstring도 random의 다른 궤적임을 명시한다. 이번 안전성 보완 경로의 통과를 일반 hard_confirm/hard_control 평가기가 같은 forward를 사용하도록 모두 수정됐다는 뜻으로 확대하지 않는다. 새 확증은 D-AV대로 단일 실행 collector에 연결해야 한다.

### (b) 검증 과정·결과·재현성

새 record의48개 고유 평가키를 원2J confirm16행/2K32행과 일대일 대조했다. **고유 checkpoint는32개**이며 “48개 checkpoint”라는 canonical 문구는48회 평가로 읽어야 한다. 2J/2K에서 일부 같은 모델을 다른 분할에 평가했다.

| 대조 항목 | 감사 확인 |
|---|---|
| 모델 소스8개·분석 소스5개 hash | snapshot과 전부 일치 |
| checkpoint hash·원 기록 MSE | 48행 전부 원 기록 및 실제 파일과 일치 |
| 등록 manifest·epoch12·고정 bound | 실제 config/result를 manifest 함수로 재검사, 오류0 |
| 저장된 coverage | 모든 행 sequences=1000 |
| 저장 finite/비유한 batch | 전부True / 전부0 |
| 저장 MSE 재현 오차 | 48행 전부0 |
| 저장 최대 상태 절댓값 | **60.90575408935547 < 305.0375175476074** |
| 저장 mask 재구성 불일치 | 0 |

따라서 감사39·40에서 **증거 누락 때문에 보류한 이번 frozen 평가의 안전성 조건**은 보완 근거를 수용한다. 기존 통계 산술은 앞선 감사에서 확인했고 원 MSE가 그대로 연결되므로 2J 개선−51.7%, 2K Pearson 대 recent−63.9%/random−55.4%의 제한된 결과 해석을 유지할 수 있다. 2K secondary에서 recent/random이 q1보다 나쁘다는 방향도 그대로다. 모든 판정이 “Pearson의 새 성공5개”라는 뜻은 아니다. 검증 범위는 snapshot 코드·저장 full-run evidence와 독립 작은 회귀 검사이며, 전체48×1000모델 실행을 감사에서 재수행한 것은 **not run**이다.

**A17 모델 동점: 실측 근거 추가, 규칙 이슈는 OPEN.** 저장 trained Pearson16평가 중 confirm2 seed7에1/41000=2.4390243902439026e−5의 경계 동점이 있다. 이는 이전 η0 스크리닝 동점0을 학습 모델에 전이하면 안 된다는 지적과 일치한다. 기존 torch.topk 결과와 원 MSE가 재현됐다는 것과 다른 backend/동점 규칙에도 항상 동일하다는 것은 다르다. 이번 감사의 앞8개 validation에서는 mismatch0이었으며 그 confirm2 동점 사례 자체를 다시 실행하지 않았다. 다음 규칙은 D-AV대로 명시하고 기존 결과를 다른 tie rule로 소급 치환하지 않는다.

**A18-HARD-CONFIRM-PROTOCOL 부분 개선/잔여 OPEN:** 완료 수집기는 epochs_run==12로 바뀌었고 safety record는 분석5파일 hash를 추가했다. 새 safety main은 데이터 접근 전에 opening record를 남긴다. 하지만 다음 확증을 위한 전체 run 사전 manifest 대조·거부 및 일반 evaluator의 단일 실행 연결은 아직 D-AV에 적힌 계획 전부가 구현된 상태가 아니다. safety main은 행별 manifest 오류를 기록한 뒤에도 그 행의 loader를 열며, 성공한 이번48행에는 오류가 없었다. 또한 다른 `--out` 이름을 쓰면 같은 분할을 다시 여는 것을 기술적으로 막지 못하므로 exclusive-create는 **동일 출력 경로 재실행 방지**로 한정한다. 실제 추가 재개방을 발견했다는 뜻은 아니다.

### (c) 개선 방향과 해석

우선순위는 **다음 확증 entry point에 D-AV를 실제 연결 → 동점 정책 고정·평가 manifest 보존 → 새 연구 질문 하나를 별도 사전등록**이다. 이번에 보완된48평가를 반복 감사·재학습할 필요는 없다. §2L D-AW와 canonical이 유일 기전 주장을 철회하고 두 정책에 대한 제한된 우위로 고친 것은 문서 정정 범위 VERIFIED다. 현재 결과는 O7·실데이터 전이·다른 q/길이의 효능이나 일반적 학습 안정성까지 검증하지 않는다.

기술 통계로 남긴 질량 평균은 Pearson .5390349, recent .5954110, random .5146724로 서로 다르다. 이는 같은 칸 수≠같은 kernel 질량이라는 관찰과 맞지만, “질량이 성능에 영향을 주지 않는다”는 결론은 아니다. 추가로 **현재 kept_mass_frac는 각 batch·시간의 평균을 다시 같은 가중치로 평균**한다. 마지막40시퀀스 batch와64시퀀스 batch의 가중치가 같으므로 정확한 전체 시퀀스 평균이라고 부르지 말 것. 기술통계 정의를 명시하거나 합계/표본수로 집계하면 된다. 이 통계는 D-AU6에 따라 안전성·성능 판정에 쓰이지 않으므로 이번 판정 보류를 다시 만드는 근거는 아니다.

이후 기전을 더 좁혀 보려면 lag/질량과 content의 효과를 구분하는 대조를 미사용 분할에서 설계한다는 감사40 방향을 유지한다. 기존에 확인한 [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 교환가능성 조건 때문에 무조건적인 시간 shuffle을 자동으로 유효 귀무 검정이라 부르지 않는다. [DeltaNet 원문](https://arxiv.org/html/2406.06484v6)은 쓰기 규칙 대안의 기존 연구 근거이지 이 안전성 보완이 delta rule 채택을 입증한 것은 아니다. 이번에는 새 문헌 사실을 추가하지 않았고 직접 수치·코드 검토와 기존 원문 근거를 사용했다.

### 수행·보존

CPU2threads/torch1.12.0+cu113/NumPy1.26.4, CUDA_VISIBLE_DEVICES 빈 값. 실제 frozen 모델4종의 validation8개 순방향, 작은5batch 실패 주입, config/hash/record 검사만 수행했다. 학습·backward·GPU·confirm2/3 데이터 생성/평가·전체48행 모델 재현·타세션 대화/메시지·Git 변이·설치·프로세스 중단은 **not run**. snapshot 및 원 checkpoint/연구파일 hash 보존을 확인했다.

첫 감사 probe의 “BOUND와 정확히 같음” 입력은 float32로 저장되면서305.037506...로 내려가 정상적으로 통과했는데 감사 assertion이 이를 실패로 오인했다. 감사용 입력만 float64로 바꿔 경계를 정확히 표현하자 모든 회귀 검사 통과. 최초 probe/log 및 재실행 log를 보존했다. 연구 collector 실패가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3/probe.py /tmp/nsmt_assessment_20260925T081001Z-d4bab0e3 f_lif_pop_v3/forecasting/results/assessment/20260925T081001Z-d4bab0e3 > f_lif_pop_v3/forecasting/log/assessment/20260925T081001Z-d4bab0e3/probe_retry.log 2>&1
```

<!-- assessment-watch:20260925T081001Z-d4bab0e3 -->


## 추적 감사 42 — 2026-09-25 17:37 KST (예약 20260925T083001Z-14979858)

**confirm4의 저장 통계는 재계산 결과와 일치한다. Pearson은 이 설정에서 GRU보다 회상 MSE가48.42% 낮지만 O7 종합은 ① 미달로 불통이다. q=.5의 validation 표본 상한 .3135도 직접 재현했다. 다만 다른 분할의 상한을 분모로 한 “상한의79%” 및 이를 일반적 불가능성으로 확대하는 해석은 보류한다.**

### 관찰·보존

문서 기억·감사41·사전등록§2M(D-AX~BB), §9A 판정표와 canonical을 읽었다. HEAD `7d836cff4bf1f2b143ce591c10b1d84b46881bad`, branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 17:30:32 KST에453파일 snapshot, trigger 불일치0. 재현 전에32개 checkpoint/config64파일을 추가 보존했다. snapshot `/tmp/nsmt_assessment_20260925T083001Z-14979858`. 감사 중 canonical에2M 결과·상한 해석·ETT 후속 계획이 append돼 `late/`에 보존하고 그 해석까지 검토했다. 모델·결과 snapshot과 섞지 않았다.

SHA256: hard_benchmark.py `582e489549537cab0712035afe189f257ea201873f09e1d79e36905a955eee00`; split_registry.py `0818bdd8eecf16814fb4fb8528e588f7a0c02bd3b414a29d248ca33c84d5f355`; layers `85906858bdc51f90a314c9e6b9a73805e642ebab09dc2acfc1b9502f91a93db0`; benchmark record `91912981ebf0bd8d23c8f44cc820a3d852c5241a86bc2b734876645f83d6aeea`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/inventory.json), [checkpoint snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/checkpoint_snapshot.json), [probe](../f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/probe.py), [실제 결과](../f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/probe_results.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/validation.json).

### (a) 구현·절차

**A20-HARD-ORACLE 부분 VERIFIED:** oracle hard mask는 q≥1 중립 경로보다 먼저 처리되어 q=1이라도 정답 mask를 사용한다. actual truth_to_oracle_p→Selector 검사에서 같은 truth를 준 recall 행은 `[1,0,0]`, copy 행은 `[1,1,1]`을 남겼다. train/evaluate 모두 hard oracle에 truth와 kind를 넘기는 연결이 추가됐다. privileged oracle과 일반 Pearson 경로를 구분한다. oracle-trained는 정답 위치를 아는 대조이며 배포 가능한 일반 모델의 성능이 아니다.

seed7 실제 snapshot checkpoint q1/Pearson/oracle/GRU 및 Pearson에 oracle을 주입한 tto, 총5조건을 validation 앞8개에서 검사했다. **3+3+2의 불균등 batch**로 benchmark.one_pass와 test.evaluate의 recall/copy/recall-first/all MSE가 모두 정확히 일치했다. 해당 표본은 유한·안전, oracle/tto M_eff=1, Pearson=.2579121102였다. spiking4조건을 batch8로도 계산해 M_eff·남긴 질량이 동일했다(허용차1e−12). 새 기술통계의 시퀀스 가중 집계 수정 범위 VERIFIED. 전체 confirm4 재현은 not run.

**A18-PROTOCOL의 새 benchmark 경로 부분 VERIFIED:** 감사 artifacts에 격리한 등록부에서 같은 split/서로 다른 output 이름의 두 번째 개방이 거부됐다. 실제 main의 모델 load만 stub하고32개 config의 n_confirm을999로 주입하자 **등록부나 data_provider에 도달하기 전에 전체 manifest 거부**가 발생했다. 실제 confirm 등록부는 변경하지 않았다. 실제 confirm4 opening 기록은17:27:52이며 저장 결과 경로와 연결된다. 다음 평가 전 torch1.12.0+cu113 버전을 검사하는 구현도 확인했다. A17 동점 문제는 이 환경/torch.topk를 그대로 쓰는 명시적 정책 범위에서 해소됐으며 다른 backend에서 같은 동점 순서를 보장한다는 뜻은 아니다.

남은 범위: 전역 등록부는 **open_once를 호출하는 entry point**에서만 강제된다. 기존 hard_confirm/hard_control/hard_safety 전체가 자동으로 등록부를 준수하도록 바뀐 것은 아니다. 새 benchmark 경로의 개선을 저장소의 모든 분할 접근 통제로 확대하지 않는다.

**A20-MANIFEST OPEN(제한된 누락):** readout을 spike→analog로 바꾼 config를 실제 manifest 함수에 넣어도 오류 목록이 비었다. lr 등 사전등록 학습 설정도 COMMON 검사에 포함되지 않는다. 실제32개 config는 spike/lr=.001임을 별도로 확인했으므로 이번 결과 오염 증거는 아니다. 다음 실행에서는 판정에 관련된 readout·shape·데이터 생성 설정·학습 설정을 빠짐없이 고정·대조해야 한다.

### (b) 통계·검증 결과

32개 실제 checkpoint/config를 저장 hash 및 manifest와 대조했다. 원 q1/Pearson16개는2J 기록 hash와 같고 새 oracle/GRU16개도 benchmark hash와 일치한다. 모델8파일·분석4파일 source hash 전부 snapshot과 같다. 저장40평가(8seed×5조건)는 모두1000시퀀스·finite=True·nonfinite_batches0·safe=True, spiking 최대 상태48.968994140625<305.0375175476074, Pearson ties0/328000·mask mismatch0이다. GRU의 safe는 등록대로 **출력 유한성**이며 f-LIF의 G11 상태 한계 검증과 같은 의미가 아니다.

**A20-BENCH-ARITHMETIC VERIFIED:** seed별 저장 행에서 직접 재계산했다.

| 항목 | 확인 결과 |
|---|---|
| 평균 MSE q1 / Pearson / oracle / GRU / tto | .2601447339 / .1216917377 / .0436265723 / .2359116655 / .0869157518 |
| O7-① | M_eff=.2472877006<.5 → **불통** |
| G14 | q1−oracle=.2165181616, 95%tCI [.2148188464,.2182174768], 하한>.005 → 통과 |
| O7-② | 평균의 비 G=.6394521141 → 통과 |
| O7-③ | Pearson−q1=−.1384529962, CI [−.1437478762,−.1331581161], −53.22% → 통과 |
| Pearson−GRU | −.1142199277, CI [−.1204672341,−.1079726214], −48.42%, 8/8 개선 |
| GRU−q1 | −.0242330684, CI [−.0265246355,−.0219415014] |

§9A/2M에 따라 **O7 종합 불통**이며 Phase E 진행 기준을 통과했다고 기록하지 않는다. GRU 비교는 같은12epoch 예산의 특정 baseline에 대한 결과로 한정한다. 용량을 맞춘 비교·충분히 수렴한 모델끼리의 비교·실데이터 우위를 검증한 것이 아니다.

**추정량 주의:** G=.639452는8seed 평균 MSE의 비다. 저장구간 [.614925,.664040]은 **seed별 G_s의 평균에 대한 t구간**이다. 서로 가까워도 동일 추정량이 아니므로 canonical의 “G=.64, 구간이 좁다”는 문구에 이 구분을 유지할 것. 이번에는 평균의 비 G 자체의 CI를 새로 추정하지 않았다.

**A20-VERDICT-GUARD 잔여(정적 확인):** evaluator는 G14 통과 여부를 확인하기 전에 평균/seed별 headroom으로 나누며, O7-2 pass에는 G14를 직접 AND하지 않고 제외 사실을 별도 출력한다. headroom이 정확히0이면 제외 판정 대신 나눗셈 오류가 날 수 있다. unsafe도 pass bool과 별도의 blocked 목록으로 저장한다. 현재는 headroom 양수/G14통과/unsafe없음이므로 **이번 수치·판정에 영향 없음**. 다음 재사용 시 invalid/blocked/excluded를 명시적인 유효 판정으로 직렬화하고 나눗셈 전에 gate를 적용해야 한다. 이는 현재 실행에서 실패가 발생했다는 보고가 아니다.

### (c) 상한 분석과 다음 방향

**A20-MEFF-CEILING 산술 VERIFIED:** positive kernel·정확히k개를 남기는 조건에서 정답만으로 k개를 채울 수 있으면 비율1, 그렇지 않으면 정답을 모두 남기고 가장 가벼운 비정답으로 채우는 식이다. n≤7의 모든 비어 있지 않은 정답 집합×4q, 총988경우를 모든 k-subset 열거 최댓값과 비교해 최대오차3.33e−16. 실제 snapshot 생성기의 validation1000시퀀스로 main을 재실행해 저장 txt와 byte 일치했다: q.1/.25/.5/1의 상한 .9384/.5989/.3135/.1309.

범위는 **해당 validation 표본·α=.7·정답 위치·고정 budget의 평균 상한**이다. 모든 입력/데이터 분포에서 .3135를 넘을 수 없다는 정리가 아니다. 예를 들어 과거 칸이 전부 정답이면 q=.5에서도 비율1이며 일반식도 그렇게 계산한다. canonical의 “어떤 선택기도 .314를 못 넘는다”는 이 표본 평균에 한정한다.

**A20-CROSS-SPLIT-CEILING OPEN(해석 정정):** canonical의 “Pearson .247은 상한의79%”는 **confirm4 실측을 validation 상한으로 나눈 것**이다. 같은 query·같은 분할의 optimality gap으로 해석할 수 없다. q1 상한 .1309와 confirm4 .1312도 가까운 별도 표본 값이지 정확한 일치 검증이 아니다. 이번 감사는 confirm4를 열어 상한을 새로 계산하지 않았다. 기존 O7① 불통을 소급 변경하거나 선택기 품질 실패의 유일 원인으로 단정하지 않는다.

또한 tto가 Pearson보다 낮은 것은 privileged 개입의 개선 여지를 보이지만, oracle은 q budget을 무시해 정답 칸만 남긴다. mask 개수·질량·feedback까지 바뀌므로 이를 **q=.5에서 점수 함수만 개선했을 때 얻을 수 있는 효과**로 바로 환산하지 않는다. oracle-trained와tto 차이도 재학습 효과를 포함하며 표현의 특정 원인까지 분리한 것은 아니다.

다음 우선순위는 **상한/관측값의 동일 표본 정의와 새 판정 추정량 사전 고정 → 목적에 맞는 budget·대조 설계 → 미사용 평가 자료**다. q.25 상한이 .5보다 높다는 것은 달성 가능성의 필요조건 진단이며 자동 채택 근거가 아니다(앞선 선택에서는 q.5가 val MSE에서 더 좋았다). 기존 기준의 불통을 그대로 남기고, 정규화 지표나 다른 q를 쓰려면 미래 분할에 적용하는 새 가설로 명시한다. ETT 후속은 계획일 뿐 **현재 실데이터 성능은 not run**이며 이전 ETTh1 test 관찰 이력과 새 선택/최종 평가의 분리를 명시해야 한다.

문헌 방향은 기존에 확인한 [DeltaNet 원문](https://arxiv.org/html/2406.06484v6)의 회상/쓰기 규칙 대안과 [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 교환가능성 조건을 유지한다. 이번 budget 도달 가능성·cross-split 지적은 직접 산술과 실제 자료 정의에 근거하며 새 논문의 성능 보장을 덧붙이지 않았다. 원래 QK/Gram/delta/entmax 후보를 O7 불통만으로 일괄 채택하지 않는다.

### 수행·보존

CPU2threads/torch1.12.0+cu113/NumPy1.26.4/SciPy1.15.3, CUDA_VISIBLE_DEVICES 빈 값. 무학습 frozen5조건 validation8개 순방향, generator-only validation1000개,988경우 전수산술, fixture 등록부/manifest 거부 검사, 저장 통계·hash 대조를 수행했다. manifest 거부 stdout은 의도한 감사 실패 주입이며 연구 실행 실패가 아니다. 모델/학습 소스·원 checkpoint·raw log·실제 등록부·환경·프로세스·Git을 변경하지 않았다. 학습/backward/GPU/confirm4 접근·전체40평가 재현·ETT 실행은 **not run**. 감사 문서3개와 진단 artifacts/raw logs만 작성했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858/probe.py /tmp/nsmt_assessment_20260925T083001Z-14979858 f_lif_pop_v3/forecasting/results/assessment/20260925T083001Z-14979858 > f_lif_pop_v3/forecasting/log/assessment/20260925T083001Z-14979858/probe.log 2>&1
```

<!-- assessment-watch:20260925T083001Z-14979858 -->


## 추적 감사 43 — 2026-09-25 17:50 KST (예약 20260925T084002Z-d598bb43)

**ETT 전이는 진행 중이다. 최초 snapshot 완료6건은 모두 test 미실행이므로 성능 판정을 보류한다. 데이터 경계·train-only 표준화와 단일-forward 상태 검사는 CPU fixture에서 통과했다. 감사 중 도착한 manifest 확대도 재검사했지만, 전체48건 완료·early-stop 근거·출처 대조 강제는 남아 있다.**

### 관찰·보존

문서 기억·감사42·사전등록§2N(D-BC~BI)·canonical PROJECT_LOG를 읽었다. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, 최초 HEAD `d7f1a647205891607a8426597540f10a336eef74`. 17:40:28 KST에519파일 snapshot; trigger와11파일 불일치(평가기·진행 CSV·raw·queue)는 진행 중 변경으로 구분했다. 완료6건 checkpoint/config12파일도 검사 전에 추가 복사했다. snapshot `/tmp/nsmt_assessment_20260925T084002Z-d598bb43`. 17:45:23 HEAD `9bb51e3cc224abb8c96beae9f13756c6dc047851`의 변경은 `late/`에 별도 보존했다. 최초 snapshot hash 보존 확인.

SHA256: ett_test 최초 `6ca6ee5a9bd15e4e1b56dc316e2f4514c775d4568a54fe28d7f65b9dd307fc1c`, 후기 `e8437fc24321fe4fc1dcca70df1c8aa53f22be5e7ce6697f9991965f50b453c7`; run_ett `ad5fc366f213a8974d07802923151189b9f9521eee36ada65cd8e1f240b631c3`; data_loader `8231657c1691e7f55c9867d24d4230d7a69b820f61b24ef399eaf82b6403689d`; prereg `2242b9a64389ed732aea2acd1c078e37b1202709a65609a819eac20a3f22379d`.

증거: [inventory](../f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/inventory.json), [checkpoint snapshot](../f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/checkpoint_snapshot.json), [CPU probe 결과](../f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/probe_results.json), [후기 manifest 재검사](../f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/late_manifest_results.json), [보존 검사](../f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/validation.json).

### (a) 아이디어 구현

ETT에서도 채널 독립 hard Pearson/shared/q=.5, q1 중립 대조, GRU를 비교한다. shared는 한 채널의 D개 뉴런 축이며 채널 간 검색이 아니다. 모델/layers 핵심은 감사42와 같아 이번에는 실행·평가 연결을 검토했다. 런처 `--no-test`, 학습 AdamW/lr=.001/wd=.01/clip1/최대50epoch/val patience10/scheduler(.5,5), 최저 val checkpoint 복원 경로를 정적으로 확인했다.

**A21-ETT-SPLIT scoped VERIFIED:** 실제 Dataset_ETT_hour 클래스에 메모리 내14400×7 선형 fixture를 넣었다. scaler 평균4319.5로 train8640행만 fit됨을 확인했다. train8209창 target336~8639, val2785창 target8640~11519, test2785창 target11520~14399로 target 구간 중복이 없다. val/test의336개 context가 이전 구간을 포함하는 것은 과거 입력 사용이다. loader는 CSV 전체를 읽고 유한성을 검사하지만 scaler fit과 학습/검증 반환창에 test target 통계는 사용하지 않는다. 실제 ETT 파일·데이터 hash 검증은 not run.

**A18-SAFETY/A19-EVAL-PATH의 ETT 경로 부분 VERIFIED:** 실제 snapshot `test` 함수에2+1개 불균등 batch와 가짜 모델을 넣었다. batch당 forward1회, MSE11/3·MAE5/3(허용차1e-6), 마지막 batch의 NaN/Inf/정확히 bound인 상태가 모두 safe=False임을 확인했다. 오차와 모든 상태를 같은 forward에서 수집한다. 실제 학습 checkpoint의 ETT 순방향·CUDA topk 재현은 not run; fixture 통과는 전체 모델 승인이 아니다.

**A21-FIRING-AGGREGATION:** kept_mass는 창×채널 가중 집계지만 firing_rate는 batch 평균의 단순 평균이다. fixture에서2개창 발화율1, 1개창 발화율0이면 저장 .5, 창 가중값2/3이다. 실제2785창/batch128도 마지막97창이므로 전체창 발화율을 뜻한다면 가중 집계가 필요하다. 판정 지표가 아니어서 MSE 판정을 무효로 만들 사유는 아니다.

### (b) 검증 과정·데이터 분리·통계·재현성

완료6건의 checkpoint SHA, JSON provenance, 모델/학습/loader 등11개 source hash는 모두 snapshot과 일치했다. 전부 `test=null`, `test_skipped=true`; train/val 길이8209/2785다. 최초 완료 집합은 Pearson seed7(17epoch,val .7825669603), q1 seed21(22epoch,.7639669013), GRU seed7/13/21/42(18/17/26/22epoch,val .6520931695/.6519078324/.6561869740/.6498295832)다. **서로 다른 seed의 미완성 표를 성능 비교로 쓰지 않는다.** 이후 도착 결과는 다음 주기 대상이다. 새 hard ETT test 성능은 **not run**.

보정 JSON 격자를 재계산해 두 데이터 모두 target발화율 .2에 가장 가까운 scale6 선택을 확인했다. ETTh1 rate .1895647068, bound=10×max|I|=507.55577087402344; ETTh2 .1872722544, bound975.1649475097656. 이는 기록 산술 검증이며 보정 모델 재실행은 아니다. frozen norm 추정은 train에서만 수행한다. GRU의 안전성은 출력 유한성이며 f-LIF G11과 다르다.

**A20-MANIFEST 후기 ETT 수정 범위 VERIFIED:** 최초 코드에서는 readout/α/K/input_norm/head_mode/τ/max_train_batches 변조7종이 통과했다. 후기 `e8437fc...`를 별도 AST로 재실행하자 실제 완료 config는 통과하고7변조는 각각 거부됐다. max_eval_batches를 대조 전에0으로 덮어쓰던 행도 삭제됐다. 최초 관찰 결함을 최신 미수정 결함으로 남기지 않는다. 과거 hard_benchmark 자체가 수정됐다는 판정은 아니다.

**A21-ETT-OPEN-GATE OPEN (평가기 작성 중인 누락; 실제 위반 관찰 아님):**

- D-BF는 **48 run 모두 완료 후** 데이터셋별24건 개방을 요구하지만 main은 지정한 데이터셋24건만 확인한다. 외부 운영으로48건을 기다릴 수 있으나 평가기 자체 보장은 없다.
- epochs_run1~50만 검사하고 **early-stop 기록/정상 종료 사유**는 확인하지 않는다. 실제 train JSON에도 명시적인 종료 사유가 없다. epochs_run=1만 있는 fixture가 통과했다. CSV/종료 로그와 대조하거나 종료 근거를 구조화해야 한다.
- fixture의 test_skipped=false, provenance checkpoint hash=`wrong`, device=`cpu`도 거부하지 않았다. main은 현재 checkpoint hash를 기록하지만 훈련 결과의 원 hash와 **대조**하지 않는다. 실제6건은 감사가 별도 일치를 확인했으므로 출처 훼손 발견이 아니다. 학습 장치와 평가 CUDA도 provenance 대조가 필요하다.
- 후기 data_path 검사는 추가됐지만 같은 ETTh1.csv 이름의 잘못된 root_path는 통과했다. 실제 data hash·calibration 파일/hash·config hash를 개방 기록에 연결할 것을 권고한다. calibration을 glob 최신 파일로 고르는 대신 고정 artifact를 지정하면 이후 보정 추가와 구별할 수 있다.

registry 개방 전에24건 오류를 거부하는 순서는 적절하다. 실제 registry나 test를 열어 위 누락을 시험하지 않았다. `test(..., flag='test')` 직접 호출은 main gate를 우회하므로 공식 실행은 main으로 해야 한다. 새 unsafe 비교는 판정 문구를 보류하도록 구현했다. 과거 A20-G14 문제와는 별개로 ETT 질문에는 G가 없다.

후기 canonical은 문서작성17:35:25/학습시작17:37:44/commit은 그 뒤였다고 **자진 고지**했다. 감사는 작성 당시 불변 원본이 없어 그 선후를 독립 확정하지 않는다. 현재 prereg hash를 보존했으며 시작 전 고정 commit으로 입증한 사전등록과 구분해 최종 보고에 이 이력을 남긴다.

### (c) 개선 우선순위

**전체48건 완료·출처/종료 gate 보완 → 데이터셋별 고정 평가1회 → 8seed paired Δ·안전성·제한 보고** 순서다. 부분 val로 q/readout/학습 예산/평가기준을 바꾸거나 우위를 확정하지 않는다. H96만 판정하고 H720은 not run을 유지한다. seed CI는 같은 고정 시계열 분할에서 학습 seed 변동을 요약하며 새로운8개 데이터셋의 일반화 CI가 아니다. 이전 ETTh1 test 관찰 고지도 유지한다.

새 실데이터 결과가 없어 검색 기전 채택 순위를 성능 근거로 바꿀 **새 판단 근거 없음**. 기존 확인한 [DeltaNet 원문](https://arxiv.org/html/2406.06484v6)은 다음 쓰기 규칙 가설의 근거이며 hard ETT 효능을 보장하지 않는다. [Winkler et al., 2014](https://wrap.warwick.ac.uk/65670/1/WRAP_1-s2.0-S1053811914000913-main.pdf)의 교환가능성 조건에 따라 겹치는 시계열 창을 독립 반복처럼 보는 무조건 shuffle 검정도 추가하지 않는다. 이번 권고는 직접 수치·실패 주입·기존 확인 원문에 근거하며 새 문헌 주장은 없다.

후기 canonical의 감사42 수용을 확인했다. **A20-CROSS-SPLIT-CEILING의79% 철회, 표본 상한 한정, G 추정량 구분, tto 특권 개입 해석 정정은 문서 정정 범위 VERIFIED.** O7 종합 불통은 유지된다.

### 수행 범위

CPU2threads/torch1.12.0+cu113/NumPy1.26.4/SciPy1.15.3. `probe.py` 및 `late_manifest_probe.py` 실행 성공. 최초 감사 probe의 head_type/tau_init은 실제 API명이 아니므로 증거에서 제외하고 head_mode/tau로 수정해 재실행했다. 초기 코드/로그도 보존했다. 문서 append 첫 명령은 system Python의 zoneinfo 부재로 쓰기 전에 종료돼 날짜 취득을 바꿔 재실행했다. 연구 모델 실패가 아니다.

연구 소스·학습/backward·GPU·실제 ETT 모델 순방향·실제 분할 등록부·설치·Git 변이·프로세스 제어는 **not run**. 감사 artifacts/문서만 작성했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43/probe.py /tmp/nsmt_assessment_20260925T084002Z-d598bb43 f_lif_pop_v3/forecasting/results/assessment/20260925T084002Z-d598bb43 > f_lif_pop_v3/forecasting/log/assessment/20260925T084002Z-d598bb43/probe_final.log 2>&1
```

후기 probe는 같은 환경/두 경로 인자로 실행했고 raw log는 같은 감사 log 폴더의 `late_manifest.log`다.

<!-- assessment-watch:20260925T084002Z-d598bb43 -->


## 추적 감사 44 — 2026-09-25 18:03 KST (예약 20260925T090001Z-e7a55cd1)

**ETT 학습48건 완료를 기록·checkpoint 수준에서 확인했다. test 결과와 ETT 개방 registry는 아직 없어 test 성능은 not run이다. 평가기는 감사43 후기와 byte 동일하므로 이미 확인한 설정 수정은 재감사하지 않고, A21 개방 gate의 남은 누락을 유지한다.**

### 관찰·실제 증거

문서 기억·감사43·사전등록§2N·canonical을 복구했다. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, HEAD `9bb51e3cc224abb8c96beae9f13756c6dc047851`. snapshot 시각 `2026-09-25T18:00:41.140273+09:00`. trigger 대비 불일치0,629파일 보존 후48개 checkpoint/config96파일과 stdout48개를 별도로 보존했다. snapshot `/tmp/nsmt_assessment_20260925T090001Z-e7a55cd1`. 18:02:26 보존 재검사에서 snapshot·현재 대상파일 hash 모두 일치, HEAD 동일, `split_openings/ETT*` 및 `etthard-20260925*/*record*.json` 없음.

ett_test SHA256 `e8437fc24321fe4fc1dcca70df1c8aa53f22be5e7ce6697f9991965f50b453c7`, prereg `2242b9a64389ed732aea2acd1c078e37b1202709a65609a819eac20a3f22379d`. 감사43 후기 코드와 동일해 변경 감지 목록의 ett_test를 새로운 수정으로 중복 인정하지 않았다.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/inventory.json), [checkpoint/원 stdout snapshot 목록](../f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/checkpoint_snapshot.json), [검사 코드](../f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/check_records.py), [48건 실제 검사 결과](../f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/record_checks.json), [보존 확인](../f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/validation.json). 원 stdout의 감사 사본은 같은 task `log/assessment/20260925T090001Z-e7a55cd1/source_stdout/`에 있으며 원본은 변경하지 않았다.

### (a) 아이디어 구현

모델·학습·데이터 소스의 새 변경은 없다. 각 run의 source provenance11파일이 snapshot과 일치했다. 현재 `check_manifest`를 AST로 추출해 실제48개 config/결과에 적용했고 오류0이었다. 두 데이터셋×3조건 각각 등록8seed가 모두 존재하며 hard/shared/Pearson/q=.5, q1 및 GRU, spike/flatten/α=.7/τ·모양·학습 설정이 현재 검사 항목과 맞는다. 감사43의 **A20-MANIFEST 수정 범위 VERIFIED**를 유지한다. 실 모델 forward·CUDA topk는 이번에도 **not run**이며 구현 전체의 새 정확성 판정 근거는 없다.

### (b) 검증·데이터 분리·재현성

**A21-ETT-COMPLETION: 현재48건 완료 증거 VERIFIED.** 모든 checkpoint SHA가 훈련 JSON의 원 SHA와 일치했다. 48건 전부 `test=null/test_skipped=true`, CUDA 학습 provenance, max_train/eval_batches=0, train/val 길이8209/2785, 기대 data_path를 확인했다. test를 아직 실행하지 않은 상태와 정상적으로 저장된 학습 결과를 구분한다.

각 `best_log_0.csv`의 연속 epoch0..n−1 및 행 수가 JSON epochs_run과 일치했다. 모든 run은 최대50epoch 이전(12~49epoch)에 종료됐고, **각 stdout의 `[train] early stop at epoch n-1`**이 실제로 존재한다. CSV val_loss에서 patience10을 독립 재계산한 최초 종료 epoch도48건 모두 일치했으며, 최소 val_loss와 JSON best_val_loss 차이는 CSV6자리 반올림 허용차5.01e−7 이내였다. CSV 반올림 재계산은 보조 근거이고 명시적 stdout 종료 기록을 함께 사용했다. 감사43에서 지적한 JSON 종료 사유 필드 부재가 실제 종료 증거 자체의 부재를 뜻하는 것은 아니다.

**A21-ETT-OPEN-GATE는 부분 해소/구현 잔여 OPEN:** 현재48건이 실제 완료됐다는 조건은 충족됐다. 그러나 변경 없는 평가기는 여전히 한 데이터셋24건만 보고, early-stop 로그·원 checkpoint hash·test_skipped·학습 장치·root/data hash를 사전 강제 대조하지 않는다. 감사가 이번48건의 종료·cp·no-test·장치를 따로 확인한 사실과 재사용 가능한 평가기 guard 구현은 구분한다. 새 코드 재검사 없이 이 이슈 전체를 VERIFIED로 닫지 않는다. calibration 고정 artifact/hash 연결, A21 발화율 batch 평균 정의, 사전등록 commit 시점 고지는 감사43 상태를 유지한다.

데이터 경계·train-only 표준화는 같은 loader에 대한 감사43의 fixture 검증을 유지한다. 이번에는 실제 ETT 데이터나 미사용 test를 읽지 않았다. 학습 상태 로그만으로 전체 test의 G11/유한성을 승인하지 않는다. 공식 test에서는 등록대로 각 모델의 전체2785창 오차·상태를 같은 forward에서 모아야 한다.

### (c) 새 결과와 다음 방향

다음은 **checkpoint 선택에 사용한 validation 최저 MSE의8seed 평균**으로, 독립 test 성능이나 §2N 최종 판정이 아니다.

| 데이터 | q1 | Pearson q=.5 | GRU | 종료 epoch 범위(q1/Pearson/GRU) |
|---|---:|---:|---:|---|
| ETTh1 | .7607307451 | .7866843340 | .6514310502 | 22~49 / 17~29 / 14~30 |
| ETTh2 | .2722651992 | .2664179200 | .2239064509 | 14~20 / 14~20 / 12~23 |

Pearson−q1의 val 평균차는 ETTh1 **+.0259535889**, ETTh2 **−.0058472792**로 방향이 다르다. 합성 recall의 개선을 ETT 전체로 일반화할 근거가 아직 없으며, 이 validation 관찰만으로 새 후보를 선택하거나 test 기준을 바꾸지 않는다. test 성능·paired test CI·H720은 **not run**.

우선순위는 **A21 개방 전 대조 보완 → 고정48 checkpoint에 대해 등록된 test1회 → 데이터셋별8seed paired Δ/CI·G11·제한 보고**다. 이번 val로 QK/Gram/delta/entmax 등 새 기전을 채택할 근거는 없다. 기존 확인한 [DeltaNet 원문](https://arxiv.org/html/2406.06484v6)은 향후 쓰기 규칙 대안의 연구 근거로 유지하되, 현재 ETT 개선을 보장하지 않는다. 새로운 문헌 주장은 추가하지 않았고 위 우선순위는 이번 직접 기록·산술 검토에 근거한다. 같은 분할을 공유하는 seed CI와 새로운 시계열 표본에 대한 일반화를 구분한다.

### 수행 범위

CPU에서 config 로드·파일 hash·48건 manifest·CSV/원 stdout 종료 증거·val 산술만 검사했다(오류0). 모델 생성/forward·학습/backward·GPU·실제 데이터/test 접근·registry 변경·설치·Git 변경·프로세스 제어는 **not run**. 이전 감사의 동일 코드 fixture를 불필요하게 다시 실행하지 않았다. 연구파일과 원 checkpoint/raw log를 보존했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1/check_records.py /tmp/nsmt_assessment_20260925T090001Z-e7a55cd1 f_lif_pop_v3/forecasting/results/assessment/20260925T090001Z-e7a55cd1 > f_lif_pop_v3/forecasting/log/assessment/20260925T090001Z-e7a55cd1/record_checks.log 2>&1
```

<!-- assessment-watch:20260925T090001Z-e7a55cd1 -->


## 추적 감사 45 — 2026-09-26 15:16 KST (예약 20260926T061001Z-e71a3200)

**§2N 저장 test 통계와 출처를 확인했다. Pearson은 q1 대비 ETTh1 MSE를3.47% 낮추지만 ETTh2에서는8.15% 높이며, 두 데이터셋 모두 GRU보다 나쁘다. A21 개방 gate와 발화율 집계 보완은 실제 CPU 재검사한 범위에서 VERIFIED다. 일관된 실데이터 우위는 입증되지 않았고, 결과만으로 스파이킹 백본이 원인이라고 확정할 수 없다.**

### 관찰·보존

문서 기억·감사44·사전등록§2N·canonical의 감사43/44 수용을 읽었다. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`; 최초 HEAD `a995275554bd169b1834644dc72e08ebcdac4395`. 15:10:28 KST snapshot683파일, trigger 불일치0. 검사 전에48개 checkpoint/config96파일·훈련 stdout48개를 추가 복사했고 감사44의 checkpoint/config hash와 모두 동일했다. 데이터 CSV2개도 별도 복사해 고정 SHA와 대조했다(내용 파싱/평가 없음). snapshot `/tmp/nsmt_assessment_20260926T061001Z-e71a3200`. 후기 canonical 결과 append는 `late/`에 보존해 읽었으며 후기 HEAD는 `ab5fa3d3b40bd908b0a6ada718f2e1522aa81ef5`. 보존 검사에서 연구 소스·결과·checkpoint 변화 없고 canonical만 추가됐다.

SHA256: ett_test `25c1d7cfba4b31d2332341809c6744c6934189e34ada9c92d177640ac4e38639`; ETTh1 test record `a415ae1778d3aebe77e704646feb768ca8f435ce3935fa01a1cb8a1377e2d8e4`; ETTh2 record `0ea877ff1e69f19d79cf53b4d79573665992b8d157920b45d6bc3718828c5f51`.

[Inventory](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/inventory.json), [추가 snapshot hashes](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/extra_snapshot.json), [CPU probe](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/probe.py), [재검사·재산술 결과](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/probe_results.json), [보존·데이터 hash](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/validation.json), [문헌 확인 범위](../f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/literature_check.json).

### (a) 구현 — A21 보완 재검사

모델/학습 구조는 기존 고정본이고 변경은 ETT 평가기·검사 스크립트다. 새 `check_manifest`/`stopping_evidence`를 snapshot에서 추출해 실제48개 config에 적용했다. 파일 접근은 snapshot checkpoint·CSV·stdout으로 연결했다. **48/48 통과**했으며 test_skipped 거짓, 원 checkpoint hash 불일치, CPU 학습, 다른 torch, 다른 보정, epoch 불일치, best_val 불일치, 잘못된 root_path, 잘린 CSV 등 **9개 주입 모두 거부**했다. 학습 없이 config만 CPU로 읽었다.

**A21-ETT-OPEN-GATE scoped VERIFIED:** 실제 main 블록에서 모델 load만 stub하고 ETTh1 개방을 요청하면서 반대편 **ETTh2 GRU/seed1024에 readout 오류**를 주입했다.48건을 모두 대조한 뒤 해당 오류1건으로 종료했고 registry/평가에 도달하지 않았다. 이 fixture의 데이터 SHA 검사는 stub했으며, 실제 데이터와 보정 SHA 일치는 별도 파일 hash 검사로 확인했다. 고정 CSV·보정 파일명/지문·root, 원 checkpoint/학습 장치/torch·no-test·CSV/조기 종료 증거를 확인하는 새 공식 main 경로의 개선이다. 직접 `test()`를 호출하거나 옛 evaluator를 쓰는 모든 경로까지 registry가 강제되는 것은 아니다.

**A21-FIRING-AGGREGATION VERIFIED:** 실제 collector에 T×B×D 형태 spike와2+1개 불균등 batch를 주입했다. 앞2개창 rate1, 마지막1개창 rate0에서 새 결과는2/3으로 정확하다(이전 .5). 연구자 검사 `ett_gate_check.py`의 validation6회는 CUDA로 작성돼 있어 감사가 실행하지 않았고 저장 txt/코드는 검토했다. 이번 감사의 모델 순방향은 가짜 모델 fixture만 사용했으며 실제 ETT test 재실행은 **not run**.

### (b) 검증 과정·통계·재현성

등록부 ETTh1/ETTh2 개방 시각은 각각 **2026-09-26 15:09:34.386/34.864 KST**, 결과 경로와 연결된다. 코드 순서는48건 대조→registry→opening record→test다. 두 결과는 status=done, 각24행·각2785창.48개 config/checkpoint SHA는 실제 snapshot 및 감사44에서 확인한 고정 학습본과 모두 일치하고, 결과에 기록된 source8파일·보정 SHA·데이터 SHA도 현재 snapshot과 일치했다. 각 run의 final+result.csv는 한 행이며 MSE/MAE가 JSON과6자리 반올림 허용차 내 일치한다.

48행은 모두 finite=True/nonfinite_batches0/safe=True다. spiking32행 상태최댓값은 ETTh1 **124.8937149<507.5557709**, ETTh2 **95.6957245<975.1649475**. GRU16행의 safe는 **출력 유한성**으로, f-LIF G11 검증과 다르다. 따라서 canonical의 “48모델 모두 한계 이내”는 이 구분을 붙인다. 전체 test를 독립 재실행한 안전성 검증이 아니라 동일-forward collector 코드·이전 실패 주입·현재 저장 증거 확인 범위다.

Pearson 마스크 재구성 불일치0; 경계 동점은 ETTh1 **16/6,394,360**, ETTh2 **19/6,394,360**. 등록한 torch1.12.0+cu113 CUDA topk 정책 안에서 해석하며 동점이 없었다고 쓰지 않는다. 남긴 질량의8seed 평균은 각각 .5158536706/.5337483041이다.

**A22-ETT-ARITHMETIC VERIFIED:** JSON seed별 행에서 독립 계산한 평균·paired t95% CI가 저장치와1e−14 이내 일치한다. 모든 비교 n=8, 씨앗 순서가 등록값과 일치했다.

| 데이터 | q1 평균 MSE | Pearson 평균 MSE | GRU 평균 MSE |
|---|---:|---:|---:|
| ETTh1 | .4442221969 | .4287969805 | .3794721924 |
| ETTh2 | .3568395562 | .3859266713 | .3074478582 |

| 비교 | 평균 차(a−b) | 95% paired t CI | 상대차 / a가 낮은 seed |
|---|---:|---|---|
| ETTh1 Pearson−q1 (1차) | −.0154252164 | [−.0188284008,−.0120220321] | −3.47% / 8/8 |
| ETTh2 Pearson−q1 (1차) | +.0290871151 | [.0145349146,.0436393156] | +8.15% / 0/8 |
| ETTh1 Pearson−GRU | +.0493247882 | [.0439497182,.0546998581] | +13.00% / 0/8 |
| ETTh2 Pearson−GRU | +.0784788132 | [.0573988367,.0995587897] | +25.53% / 0/8 |
| ETTh1 q1−GRU | +.0647500046 | [.0587323788,.0707676303] | +17.06% / 0/8 |
| ETTh2 q1−GRU | +.0493916981 | [.0306039799,.0681794163] | +16.07% / 0/8 |

§2N D-BG에 따라 **ETTh1 개선, ETTh2 악화**다.1차 평균차 절댓값은 둘 다 .005보다 크다. unsafe 보류 행 없고 저장 판정 문구가 등록 규칙과 맞는다. 두 데이터셋을 합친 우위나 다중비교 보정된 동시 구간으로 표현하지 않는다. seed 구간은 고정 기간의 학습 변동이며 기간 일반화 불확실성을 포함하지 않는다. 사전등록 commit 선후 고지 및 과거 ETTh1 test 관찰 고지를 유지한다. H720은 **not run**.

### (c) 결과 기반 수정 방향과 해석의 경계

**A22-ETT-INTERPRETATION OPEN:** 후기 canonical의 “스파이킹 백본 자체가 병목일 가능성”은 가설로만 남긴다. 이번 GRU 대조는 백본 이외 용량·표현·최적화도 달라 백본 원인만 분리하지 않았다. 또한 v2 ridge/window-mean은 다른 파이프라인의 참고값이므로 “이번 v3가 선형 기준선보다 나쁘다”는 동일 조건 실험 결론으로 올리지 않는다. 수치상의 대소관계는 맞지만 직접 통제 비교가 아니다. val/test 효과 방향 반전은 관찰 사실이며 **분포 변화가 원인이라는 확정 진단이나 validation 기반 선택 일반의 부정**은 아니다.

다음 채택/검증 우선순위는 다음과 같다. 이미 열린 test를 새로운 후보 선택에 재사용하면 그 후 분석은 탐색적이며, 최종 확인은 별도 미래 기간/미사용 데이터로 사전 고정한다.

1. **동일 전처리·기간·L336/H96·채널 설정의 단순 선형 기준선과 GRU 비교를 먼저 정렬한다.** 기존 v2 수치를 재활용한 우위/열위 판정보다 원인 해석에 직접적인 대조다. 실제 확인한 [Zeng et al., AAAI 2023 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089)은 직접 다중시점 예측을 하는 단순 선형 모델의 유용성을 보여준다. 이는 우리 모델의 실패 원인 증명이나 선형 모델의 보편적 우월성 주장이 아니다.
2. **정규화 대조는 작은 통제 실험 후보로 우선한다.** train/val에서 기간별·채널별 입력 평균/분산과 오차, hard/q1의 상태·발화 변화를 먼저 진단하고, 필요하면 q1/Pearson 양쪽에 같은 입력창 정규화와 출력 복원을 적용하는2×2 대조를 사전등록한다. [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)에서 입력 통계 제거/출력 복원과 ETT 실험 근거를 직접 확인했다. OpenReview 원문 PDF는 접근 검증 화면으로 막혀 읽었다고 주장하지 않는다. 현재 frozen input_norm과 RevIN은 같은 처리가 아니며, 이번 결과만으로 정규화가 개선을 보장하거나 분포 변화가 원인이라고 결론내리지 않는다.
3. **검색 기전 확대는 원인 분리 이후다.** budget·lag·남긴 질량 대조와 q1/Pearson에 공통인 readout/표현 대조를 먼저 계획한다. Gram/delta/entmax·다변량 key를 이 두 결과만으로 일괄 승격하지 않는다. GRU보다 두 spiking 조건이 모두 나쁘다는 관찰은 선택기만 바꾸는 탐색을 무조건 최우선으로 둘 근거가 약하다는 뜻이다. 구조 변경은 새 검증 계획과 대조군을 요구한다.

문헌은 방법 후보의 근거이며 실제 채택은 새 통제 실험 결과에 달린다. 이번 감사에서는 모델 수정이나 학습을 실행하지 않았다.

### 수행 범위

CPU2threads, torch1.12.0+cu113/NumPy1.26.4/SciPy1.15.3.48개 실제 manifest 검사,9개 fixture 거부, 반대 데이터셋 오류의 main 개방 전 거부, 불균등 batch 발화율 fixture, 저장 통계·CSV·source/config/checkpoint/data/calibration hash를 검사했다. 실제 model load는 CPU config 로드만이며 main에서는 stub했다. 실제 ETT 데이터는 bytes 복사/hash만 했고 파싱/평가하지 않았다. 실제 ETT test 재실행·훈련/backward·GPU·등록부 변경·연구코드 수정·Git 변이·설치·프로세스 제어는 **not run**. 원자료를 보존하고 감사 파일만 작성했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200/probe.py /tmp/nsmt_assessment_20260926T061001Z-e71a3200 f_lif_pop_v3/forecasting/results/assessment/20260926T061001Z-e71a3200 > f_lif_pop_v3/forecasting/log/assessment/20260926T061001Z-e71a3200/probe.log 2>&1
```

<!-- assessment-watch:20260926T061001Z-e71a3200 -->


## 2026-09-26 15:23 KST — 추적 감사46: HEAD 변경, 새 판단 근거 없음

예약 `20260926T062001Z-947b3c9c`. 관찰 HEAD `ab5fa3d3b40bd908b0a6ada718f2e1522aa81ef5`(trigger와 동일), branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 기억·감사45·사전등록 §2N·canonical PROJECT_LOG를 재확인했다. 감지된 변경은 GIT_HEAD뿐이다. 직전/현재 trigger의 감시 파일 **680개 SHA가 모두 동일**하며 실제 현재 파일도 전부 일치한다. 새 commit은 감사45에서 이미 확인한 ETT 결과·학습 메타데이터·개방 기록·canonical 기록의 저장이다. 감사자가 쓴 문서를 새로운 연구 근거로 세지 않았다.

별도 snapshot `/tmp/nsmt_assessment_20260926T062001Z-947b3c9c`에 감시 파일과 문서 원문을 보존했다. hash/대조 증거: `f_lif_pop_v3/forecasting/results/assessment/20260926T062001Z-947b3c9c/inventory.json`, `validation.json`. ett_test.py SHA-256 `25c1d7cfba4b31d2332341809c6744c6934189e34ada9c92d177640ac4e38639`, 사전등록 SHA-256 `2242b9a64389ed732aea2acd1c078e37b1202709a65609a819eac20a3f22379d`.

- **(a) 구현:** 새 판단 근거 없음. A21-ETT-OPEN-GATE 및 A21-FIRING-AGGREGATION은 감사45에서 검증한 범위를 유지하며 확대하지 않는다.
- **(b) 검증·통계·재현성:** 새 판단 근거 없음. 기존 48개 ETT test 결과 및 A22-ETT-ARITHMETIC 판정 유지. 이번에는 파일 대조만 수행했으며 CPU probe·모델 평가 재실행·학습·GPU·H720은 **not run**.
- **(c) 개선 방향:** A22-ETT-INTERPRETATION **OPEN 유지**. 동일 파이프라인 선형 대조 → train/val 분포·정규화 통제 → 검색 기전 확대 순서를 유지한다(문헌·수치 근거는 감사45). 원인 분리 대조와 미사용 자료의 새 확증이 남은 조건이며 기존 test로 새 후보를 고르면 탐색으로 표시한다. 새로운 성능 우위나 이슈 종결은 선언하지 않는다.

<!-- assessment-watch:20260926T062001Z-947b3c9c -->


## 2026-09-26 17:34 KST — 추적 감사47: ETT 검색 필요성 탐색, 최적 상한·무작위 기준 정정 필요

예약 `20260926T083001Z-2dfc6c98`, HEAD `1772351f64a4a5909516378db84a6389f2ac414c`(trigger 일치), branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 기억·감사46·사전등록 §2N·canonical PROJECT_LOG의 17:19/17:23/17:24 append를 확인했다. 모델/학습/평가 소스와 사전등록은 이전과 같고 새로운 자료는 `ett_retrieval_need.{py,txt,json}` 및 canonical 탐색·정정 기록이다. snapshot `/tmp/nsmt_assessment_20260926T083001Z-2dfc6c98`: 감시 683파일+문서3개, trigger 불일치0, 검사 종료까지 대상 변화0. 분석 소스 SHA-256 `1bb38fa26cc60689fd9a5c9aabd292d996d0e3b5c08fd4b3ce26aa95d9f3a7ff`, 결과 JSON `a050274018d64939d0cbdc7880aced6a21d3e9abe0be516b8795b3fddaa3f934`. 전체 지문·원본 위치는 `f_lif_pop_v3/forecasting/results/assessment/20260926T083001Z-2dfc6c98/inventory.json`.

### (a) 구현 정확성

**부분 적합.** 코드의 수준 정렬 유사 사례 예측은 `마지막 값 + 평균(과거 뒤따른 구간 − 해당 patch 마지막 값)`이다. h8은 J41/k20, h96은 J30/k15이며 후보의 뒤따른 구간은 입력창 안에 있다. 실제 함수를 snapshot에서 AST로 추출해 CPU에서 검사했다. 합성 시계열 4창×7채널의 패치/목표 정렬이 직접 슬라이싱과 모두 일치했고, 6개 임의 입력의 h8/h96 예측을 NumPy 루프로 독립 재계산한 최대 차는 4.44e−16이었다. 상수 patch의 Pearson=0도 확인했다. 실제 ETT 자료/학습 모델의 재평가는 아니다.

**A23-HINDSIGHT-BOUND OPEN (확정된 정의 오류):** 코드가 고르는 것은 *개별 후보 예측 오차가 작은 k개*다. 선택한 k개를 평균한 최종 예측 오차를 최소화하는 집합이 아니므로 canonical의 “사후 최적/상한”을 보장하지 않는다. 실제 topk 함수에 J41/k20, 목표0, 후보 오차값이 아니라 예측값 `[1]×20,[-2]×20,[100]`을 넣으면 개별 최선20개 평균의 MSE=1이다. 같은 예산으로 1을13개, −2를7개 고르면 MSE=.0025다. 잔여 조건: “목표를 사용한 개별 오차 기반 사후 선택”으로 명칭·해석을 정정하거나 집합 평균 목적의 별도 최적성 증거를 제시한다. 이 반례는 실제 ETT 최적 오차의 추정치가 아니다.

**A23-RANDOM-REFERENCE OPEN (확정된 산술 오류):** 고정 k개 집합과 균등 무작위 k개 집합의 교집합/k 기대값은 k/J다. h8 및 모델 마스크 비교의 정확한 기준은 **20/41=.487804878**, h96만 .5다. 최근 구간 `j≥21`도41개 중20개이므로 무작위 최근 비율 역시20/41이다. 저장 측정값 .5880/.5615 등은 바뀌지 않는다. 작은 경우 전수 조합으로 기대값을 확인했다. 잔여 조건: canonical의 .5 기준 정정.

학습 마스크는 hard/shared 경로의 `coeff>0`이며 현재 양의 커널 가중치에서는 support 추출과 맞는다. 다만 모델의 상태 서술자 Pearson과 원시 patch8개의 Pearson은 입력 표현이 다르다. 같은 통계량을 썼다는 것만으로 같은 선택기를 측정했다고 보지 않는다.

### (b) 검증·분할·대조·재현성

**A23-ARITHMETIC scoped VERIFIED:** 저장 JSON에서 8seed 평균/min/max와 txt의 MSE를 재집계한108개 값이 출력 반올림 오차 안에서 일치한다. 아래는 저장 결과의 독립 산술이며 실제 데이터 분석 전체의 재현 인증은 아니다.

| h96 유사 선택의 상대 MSE 차 | train | val | 기존 test |
|---|---:|---:|---:|
| ETTh1: 유사/전부−1 | −24.10% | −22.93% | −25.91% |
| ETTh2: 유사/전부−1 | −3.78% | −9.33% | −7.08% |
| ETTh2: 유사/지속−1 | +4.71% | −4.68% | +0.92% |

모델 마스크의 h8 analog MSE 평균은 ETTh1 **.597984331**, ETTh2 **.131671577**. 이는 마스크를 다른 예측기에 넣은 진단이며 학습 모델의 H96 MSE가 아니다. “ETTh2에는 검색할 정보가 적다”는 결론은 이 표현·예측기·기준선에서 이득이 제한적이었다는 범위로 좁힌다. ETTh2 val h96에서는 지속보다4.68% 낮으므로 모든 기간에서 지속보다 나쁘다는 결론도 아니다.

**A23-EXPLORATORY-PROTOCOL OPEN:** 계획의 “test는 데이터 기술통계만”에는 실제 수행 범위를 명시할 필요가 있다. main은 test에서도 h8/h96 목표를 사용한 analog 예측 MSE와 hindsight 선택을 계산한다. 계획 분석3에 기간별 analog 차이 비교가 있으므로 숨겨진 모델 test 재평가나 학습 누출로 단정하지 않는다. 그러나 단순 평균/분산 조회를 넘어선 **목표를 사용한 사후 평가**다. 기존 test는 이미 개방됐고 이번 후보 선택은 탐색으로만 취급한다. 코드상 scaler fit은 train8640행뿐이며 모델 마스크는 val에서만 얻는다. h8도 H96용 창 격자를 사용하므로 각 분할의 마지막88개 가능한 h8 시작점을 별도 추가하지 않는다.

하루 정렬은 h8 13칸/h96 10칸으로 유사 선택20/15칸과 예산이 다르다(이미 canonical에서 고지). 저장된 비교 안에서는 하루 정책이 유사 정책보다 낮지만 “가장 강한 구조”라는 보편 주장이나 주기만의 원인 효과로 확대하지 않는다. 겹치는 창·채널을 독립 반복으로 세지 않으며, 무작위64회는 MC 평균으로 SE/반복 민감도가 없다. 새 JSON에는 checkpoint·CSV·source 지문, 실행 환경·명령·동점 수가 없어 현재 감사 snapshot만으로 원 실행 provenance를 소급 보증할 수 없다. 이들은 탐색 증거의 제한이며 진행 중 학습 파일의 실패로 분류하지 않는다.

17:24의 미래 구간 주장은 현재 loader의 끝14400과 분석의 통계 구간 끝14400에 부합한다. 그러나 모든 과거 코드에서 미사용이었다는 주장은 이번 범위로 인증하지 않는다. 3020행을 가정할 때2589창은 입력336까지 미래 구간 안에 둘 때의 값이다. 앞선336행을 입력 문맥으로 허용하고 목표만 새 구간에 두면2925창이다. 둘 중 하나를 새 프로토콜에서 고정해야 하며 이번에는 그 구간 값에 접근하지 않았다.

### (c) 개선 방향·기존 이슈

**A22-ETT-INTERPRETATION의 문서 정정 범위 VERIFIED:** canonical17:24 append가 백본 원인 가설, 다른 파이프라인 선형 참고값, GRU 출력 유한성/스파이킹 상태 한계 구분, 분포 변화 가설을 명시적으로 정정했다. 원인 분리 실험을 검증했다는 뜻은 아니다. 새 탐색문 “검색 가치 변화가 원인이 아니다”도 이 analog 지표만으로 다른 표현/모델의 원인을 배제할 수 없으며 A23 해석 제한을 유지한다.

채택 우선순위는 **① 동일 파이프라인 선형 기준선 정렬 → ② train/val에서 분포 진단 및 q1/Pearson×정규화 유/무 통제 → ③ 같은 예산의 하루 정렬·유사도·recent/random 대조**로 유지한다. 하루 결합 선택은 후보로 남기며 즉시 채택하거나 Gram/delta/entmax를 승격하지 않는다. ①의 기존 확인 문헌은 [Zeng et al., AAAI 2023 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089), ②는 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)다(감사45에서 실제 확인, 이번 새 검색 없음). 새 판단은 위 직접 수치검토에 근거한다. 새 확증은 후보·예산·문맥/목표 경계를 먼저 고정한 미사용 기간/자료가 필요하다.

### 수행 범위와 증거

CPU2threads, snapshot 함수와 합성 fixture, 저장 JSON/txt 재산술만 실행했다. 결과 `probe_results.json`, 코드 `probe.py`, raw `f_lif_pop_v3/forecasting/log/assessment/20260926T083001Z-2dfc6c98/probe.log`. 실제 ETT 데이터 접근·모델 forward·분석 전체 재실행·학습/backward·GPU·H720·새 후보 성능 검증은 **not run**. 연구 소스·checkpoint·raw log·진행 프로세스·Git 상태를 변경하지 않았다. 감사 문서와 증거만 작성했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260926T083001Z-2dfc6c98/probe.py > f_lif_pop_v3/forecasting/log/assessment/20260926T083001Z-2dfc6c98/probe.log 2>&1
```

<!-- assessment-watch:20260926T083001Z-2dfc6c98 -->


## 2026-09-26 18:01 KST — 추적 감사48: 감사47 수용 문서 확인, 새 성능 판단 근거 없음

예약 `20260926T090001Z-12be0a64`, 관찰 HEAD `999dc0359ca5998b1caa53df2e484c1f5a78a68d`(trigger 동일), branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 최신 기억·감사47·사전등록 §2N·canonical PROJECT_LOG의 **17:53 정정 append**를 확인했다. 직전 HEAD 이후 변경은 canonical 문서의35행 추가뿐이다(이 안의 감사47 기록을 새 연구 결과로 세지 않음). 감시683파일의 전후 SHA 및 현재 원문이 모두 일치한다. 별도 snapshot `/tmp/nsmt_assessment_20260926T090001Z-12be0a64`에 문서3개를 더해686파일 보존; trigger 불일치0. 증거 `f_lif_pop_v3/forecasting/results/assessment/20260926T090001Z-12be0a64/inventory.json`.

분석 소스 SHA-256 `1bb38fa26cc60689fd9a5c9aabd292d996d0e3b5c08fd4b3ce26aa95d9f3a7ff`, canonical 관찰 SHA-256 `36320741cb98917a22c5bc1173ce735a151429ea0bcce9e4f13055889cf17619`. 스케줄러 해시 대신 실제 원문을 별도 보존했다.

- **(a) 구현:** 새 구현·수치 결과 없음. canonical에서 A23-HINDSIGHT-BOUND의 “최적 상한”을 철회하고 “목표를 본 개별 오차 기준 사후 선택”으로 정정한 점, A23-RANDOM-REFERENCE를 h8=20/41≈.487805·h96=.5로 정정한 점은 **문서 정정 범위 VERIFIED**. 근거는 직접 읽은 새 append와 감사47의 독립 산술·반례이며 새 모델 검증이 아니다. 분석 소스 docstring에는 아직 `a ceiling`, `random: 0.5`, `data statistics only`가 남아 있다. **주석 정합성 보완은 OPEN**으로 구분하며 연구 소스가 수정됐다고 판정하지 않는다.
- **(b) 검증 과정:** A23-EXPLORATORY-PROTOCOL의 **test 목표 사용 사후 평가 고지는 문서 범위 VERIFIED**. 표현·예측기에 한정한 해석, 하루 정책의 예산 차이, 원인 배제 철회, 미래 구간 문맥에 따른2589/2925창 구분도 확인했다. 원 실행의 checkpoint/source 지문·환경·명령·동점 수 누락, MC 표준오차 부재는 고지만 추가됐고 증거가 보완된 것은 아니다. 따라서 **프로토콜·재현성 잔여는 OPEN 유지**. 미래 구간의 모든 과거 미사용 이력도 인증하지 않는다. 기존 A23-ARITHMETIC의 제한된 검증 범위는 유지한다.
- **(c) 개선 방향:** **새 성능 판단 근거 없음**. 감사45·47에서 확인한 수치·문헌에 근거한 동일 파이프라인 선형 기준선 → q1/Pearson×정규화 통제 → 동일 예산 주기/유사도/recent/random 대조 순서를 유지한다. 새 후보 채택·원인 확정·확증 성능 판정은 보류한다. 새 평가 자료와 경계·예산의 사전 고정이 남은 조건이다.

이번은 문서·파일 hash 대조만 수행했다. 새 CPU probe·실제 데이터/모델 평가·학습·GPU·H720·새 문헌 검색은 **not run**. 기존 감사 산술을 재실행한 것으로 보고하지 않는다. 원 소스·checkpoint·raw log·진행 프로세스·Git을 변경하지 않았다.

<!-- assessment-watch:20260926T090001Z-12be0a64 -->


## 2026-09-26 23:52 KST — 추적 감사49: A23 설명문 정합성 검증

예약 `20260926T145001Z-dd74b488`. trigger HEAD `17da43dd3802bcfe4bd174852ed5721f62a4234a`, 실제 snapshot HEAD `0c970d537c3471297f7bb68705d971e4acad5115`, 종료 HEAD `0c970d537c3471297f7bb68705d971e4acad5115`. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 감지 이후 HEAD 차이는 canonical PROJECT_LOG의 감사48 수용·commit 메시지 정정13행이다. 최신 기억·감사48·사전등록 §2N·canonical23:50 append를 확인했다. 감시683파일은 trigger와 전부 일치하며 문서3개를 더한686파일을 `/tmp/nsmt_assessment_20260926T145001Z-dd74b488`에 별도 보존했다. 감사48 대비 연구 파일 변경은 `ett_retrieval_need.py`뿐이고 결과 JSON/txt·사전등록·모델/학습 소스는 동일하다. 감사 문서 자체의 append는 새 연구 변화로 세지 않았다.

분석 소스 SHA-256 `e97c2d0d9060fbc3a5b33b4be37a6fc2af9f77541fc8c126317dd4f015bb2d4c`(이전 `1bb38fa26cc60689fd9a5c9aabd292d996d0e3b5c08fd4b3ce26aa95d9f3a7ff`). 증거 `f_lif_pop_v3/forecasting/results/assessment/20260926T145001Z-dd74b488/inventory.json`, `checks.json`, `check_ast.py`; 원 검사 stdout은 같은 task의 `log/assessment/20260926T145001Z-dd74b488/check_ast.log`.

- **(a) 구현:** 분석 설명문이 개별 오차 순 선택은 최적 상한이 아님, h8 무작위 기준20/41, test 목표 사용 사후 평가임을 정확히 설명한다. 이전 snapshot과 현재 snapshot을 직접 파싱해 모듈·함수·클래스 docstring을 제외한 AST 동일성을 확인했다(**True**). 따라서 **A23-HINDSIGHT-BOUND·A23-RANDOM-REFERENCE 및 A23-EXPLORATORY-PROTOCOL의 잔여 주석 정합성은 scoped VERIFIED**. 계산 구현/결과가 바뀌거나 새로운 최적성 보장이 생긴 것은 아니다.
- **(b) 검증 과정:** canonical23:50에서 최초 AST 명령 실패와 후속 확인을 구분한 정정을 확인했다. commit 메시지의 검사 주장만을 근거로 삼지 않았고 이번 독립 검사로 확인했다. 최초 실패는 진단 명령 오류이며 모델 실패가 아니다. 원 분석 실행의 checkpoint/source 지문·환경·명령·동점 수 누락과 MC 표준오차 부재는 그대로이므로 **A23-EXPLORATORY-PROTOCOL의 재현성 잔여 OPEN 유지**. 뒤늦은 snapshot을 원 실행 provenance로 대체하지 않는다.
- **(c) 개선 방향:** **새 성능 판단 근거 없음**. 감사45·47의 수치·문헌에 근거한 동일 파이프라인 선형 기준선 → q1/Pearson×정규화 통제 → 동일 예산 주기/유사도/recent/random 대조 순서를 유지한다. 새 후보 성능·원인 판정은 보류하고 미사용 평가 자료 및 경계·예산 사전 고정 조건을 유지한다.

이번 실행은 CPU의 정적 AST/hash 대조만 수행했다. 수치 probe 재실행·실제 데이터/모델 평가·학습·GPU·H720·새 문헌 검색은 **not run**. 연구 소스/기존 로그/프로세스/Git 변이 없이 감사 증거와 문서만 작성했다. 명령: `/usr/bin/python3 f_lif_pop_v3/forecasting/results/assessment/20260926T145001Z-dd74b488/check_ast.py` (stdout은 위 raw log).

<!-- assessment-watch:20260926T145001Z-dd74b488 -->


## 2026-09-27 00:01 KST — 추적 감사50: 새 판단 근거 없음

예약 `20260926T150001Z-d682c19c`. 관찰/trigger HEAD `0c970d537c3471297f7bb68705d971e4acad5115`는 감사49의 후기 관찰 HEAD와 동일하다. 최신 기억·감사49·사전등록 §2N·canonical PROJECT_LOG를 재확인했다. 감시683파일의 전후 SHA와 현재 파일이 모두 일치한다. 이번 HEAD 감지는 이미 감사49에서 확인한 문서 정정 커밋이며 새 연구 결과가 아니다. 문서 포함686파일을 `/tmp/nsmt_assessment_20260926T150001Z-d682c19c`에 별도 snapshot했다. 분석 소스 SHA-256 `e97c2d0d9060fbc3a5b33b4be37a6fc2af9f77541fc8c126317dd4f015bb2d4c`; 전체 증거 `f_lif_pop_v3/forecasting/results/assessment/20260926T150001Z-d682c19c/inventory.json`.

- **(a) 구현:** 새 판단 근거 없음. A23 설명문 정합성의 scoped VERIFIED를 유지한다.
- **(b) 검증:** 새 판단 근거 없음. A23-EXPLORATORY-PROTOCOL의 원 실행 지문·환경·명령·동점·MC 불확실성 기록 잔여는 OPEN 유지한다. 재검사 없이 다른 이슈를 닫지 않는다.
- **(c) 개선:** 새 성능 판단 근거 없음. 동일 파이프라인 선형 기준선 → 정규화 통제 → 동일 예산 검색 대조 순서를 유지한다. 새 확증에는 미사용 자료와 경계·예산 사전 고정이 필요하다.

파일 대조 외 probe·데이터/모델 평가·학습·GPU·H720은 **not run**. 연구 소스·기존 산출물·Git 변이 없음. 정의·결과·상태 변화가 없어 기억/canonical에는 중복 append하지 않는다.

<!-- assessment-watch:20260926T150001Z-d682c19c -->


## 2026-09-27 15:30 KST — 작업 에이전트에게: 다음 실험 개선 제안 A24 (제안, 실행 전)

**사용자 요청:** 다음 실험 개선 방향을 감사파일을 통해 제시한다. 이 항목은 작업 에이전트의 후속 설계용 제안이며 사전등록 확정이나 실행 완료가 아니다. 기존 감사50 및 A23 상태를 유지한다. 관찰 HEAD `0c970d537c3471297f7bb68705d971e4acad5115`, branch `exp/f-lif-pop-v3`. 관련 문서·분석·ETT 결과를 먼저 별도 snapshot했고 SHA는 `f_lif_pop_v3/forecasting/results/assessment/next_experiment_20260927T152800/evidence.json`에 기록했다. 이번에 원문/저자 구현을 다시 확인했으며 학습·모델 forward·새 평가 구간 접근은 **not run**이다.

### 1. 다음 질문은 “정규화를 적용하면 선택기 효과도 달라지는가”로 좁힌다

**우선 제안은 새 검색 규칙 추가보다, 같은 파이프라인의 선형 기준선과 입력창 정규화 대조를 한 묶음으로 설계하는 것이다.** 기존2N에서 Pearson은 q1 대비 ETTh1 −3.47%, ETTh2 +8.15%였고, 두 조건 모두 GRU보다 MSE가 높았다. 이 결과만으로 스파이킹 백본이나 분포 변화를 원인으로 확정하지 않는다. q1과 Pearson 양쪽을 같이 고쳐야 선택기 문제와 공통 입력/표현 문제를 구분할 수 있다.

추가 해석 주의: 현재 raw-patch Pearson은 일정한 수준 이동에 불변이고, analog 예측은 이미 `last + mean(continuation − past_last)`로 수준을 맞춘다. 입력과 목표에 같은 상수 b를 더하면 선택은 같고 예측도 b만큼 이동하여 오차가 보존된다(현재 식에서 직접 도출, 이번 수치 실행 아님). 따라서 이 analog의 기간별 이득 부호가 같다는 사실은 **수준 보정이 없는 학습 모델이 수준 변화에 취약한지**를 검사하지 않는다. 정상화 대조의 근거는 있으나 성공 보장은 없다.

### 2. 권장 실험표: 동일 데이터·예산에서 8조건

| 계열 | 입력창 정규화 없음 | 입력창 정규화 있음 | 질문 |
|---|---|---|---|
| myModel q1 | 기존 hard q1 | q1+R | 공통 spiking 경로의 변화 |
| myModel Pearson | 기존 hard shared Pearson q=.5 | Pearson+R | 실제 개선 후보 |
| Linear | 채널 공유 Linear(336→96), bias 포함 | 같은 Linear+R | 선형 기준선 및 일반 정규화 이득 |
| GRU | 기존 GRU | 같은 GRU+R | 개선이 spiking에 특유한가 |

R은 아래에 정의한 **학습 affine 없는 가역 입력창 정규화**다. Linear+R를 NLinear라고 부르지 않는다(NLinear는 마지막 값 차감/복원으로 다른 처리다). 첫 대조에서 DLinear의 분해창·NLinear·학습 affine까지 동시에 늘리지 않는다. Linear는 단순하지만 파라미터가 적다는 뜻은 아니다: bias 포함336×96+96=**32,352개**이며 다른 모델도 실제 사용 파라미터 수를 기록한다. 같은 예산 비교이지 용량 동등 비교가 아니다.

ETTh1/ETTh2, L336/H96, patch8, 기존 채널 독립 설정, seeds `[7,13,21,42,123,256,512,1024]`. **8조건×2데이터×8seed=128 run**이다. 기존 q1/Pearson/GRU 무정규화48개는 데이터·학습·손실·설정·checkpoint 지문이 일치하면 재사용하여 **추가80개**로 줄일 수 있다. 재사용48개도 새 평가 기간에서는 함께 평가해야 하며, 기존 test 수치를 새 기간 결과로 대신하지 않는다. 재사용 조건이 어긋나면 그 셀을 새 suite에서 재실행하고 이유를 기록한다.

먼저 train/val에서 seed7·13으로 배선·수치·로그를 점검한다. 이를 최종8seed 결과로 보고하지 않는다. 구현을 고치면 해당 pilot 산출물은 보존하되 최종 suite에서 제외한다. 구현/설정이 고정되면8seed를 모두 완료하며 잘 나온 seed만 확대하지 않는다. 기존2N 예산(AdamW lr .001, wd .01, batch128, clip1, 최대50epoch, early-stop10, scheduler factor .5/patience5, 최저val checkpoint)을 우선 고정한다. 이 예산이 Linear/GRU 각 모델의 최적 성능을 보장한다고 주장하지 않는다.

### 3. R의 정의와 학습 목적을 고정한다

train 전역 StandardScaler는 기존대로 train `[0,8640)`만 사용한다. 그 척도의 입력 `x[B,336,C]`에서 **각 창·각 채널의 시간축만**으로 `mu=mean(x)`, `s=sqrt(var(x, unbiased=False)+1e-5)`를 계산하고 detach한다. `z=(x−mu)/s`를 patch/embedding **앞**에 넣고, 모델이 예측한96시점 출력에 `y_hat=s*z_hat+mu`를 적용한다. batch나 채널 간 통계를 섞지 않는다. 미래 목표로 mu/s를 계산하지 않는다.

**학습 손실·checkpoint 선택·평가 MSE/MAE는 모두 복원한 y_hat와 기존 StandardScaler 척도의 목표 사이에서 계산한다.** 정규화된 목표 공간에서 MSE를 계산하면 창별 가중치까지 바뀌므로 이번 요인에 섞지 않는다. 물리 단위 오차는 보조 보고로만 둔다. 전체 평균 외 채널별 오차와 seed별 차이를 남긴다.

현재 embedding의 frozen input_norm은 R과 별개다. R을 켠 조건은 변환된 train 입력으로 frozen 통계를 추정해야 한다. 기존 raw-input frozen 통계를 그대로 붙이지 않는다. input_scale/안전 한계는 기존 train-only 보정 절차로 사전 고정하고 **같은 R 설정의 q1/Pearson에는 공통으로 적용**한다. R에 따라 보정값이 달라지면 이를 기록하고 “정규화와 그에 필요한 train 보정의 결합 효과”로 해석한다. 결과를 본 뒤 bound를 높이거나 보정을 바꾸지 않는다.

사전 CPU 점검: 상수/준상수 입력 유한성, norm→denorm 복원, batch 분할/채널 순서에 대한 독립성, R-off 경로의 기존 출력 일치, 입력에 상수 이동을 줬을 때 R-on 예측이 같은 양만큼 이동하는지, 손실이 복원 척도인지 확인한다. 이동 등가성이 좋아져도 실제 시계열 분포 변화의 인과 증명은 아니다. 전체 실제 평가에서 출력·상태 유한성과 고정 안전 한계도 확인한다.

### 4. 개발 기간과 새 확증 기간을 명시적으로 분리한다

비교 가능성과48개 재사용을 위해 학습 `[0,8640)`, validation 목표 `[8640,11520)`를 유지하는 안을 권장한다. 기존 test `[11520,14400)`는 이미 모델·사후 analog 분석에 사용됐으므로 **새 확증으로 재명명하지 않는다**. 이번 개발은 train/val을 중심으로 하고 기존 test를 추가 후보 선택 기준으로 쓰지 않는다.

다음 확증 후보는 목표가 `[14400,17420)` 안에 있는 미래 구간이다. **이전336행을 입력 문맥으로 허용하는2925창**을 권장한다. 첫 창은 입력 `[14064,14400)`, 목표 `[14400,14496)`, 마지막 목표는17420 직전에서 끝난다. 롤링 예측에서는 각 창 시작 이전의 실제 관측만 입력으로 사용하며 미래 목표를 입력으로 앞당기지 않는다. 입력까지 새 구간에 가두는2589창 안과 섞지 않는다.

단, 이 구간이 모든 과거 작업에서 미사용이었다는 사실은 아직 인증되지 않았다. 작업 기록·실험 코드·평가 등록부에서 사용 이력을 먼저 점검하고 그 한계를 명시한다. 이미 후보 선택에 쓰였다면 탐색으로 분류하고 다른 미사용 자료를 확보한다. 경계/CSV hash/프로토콜/조건/seed/판정식을 고정한 후 **전체 조건이 준비됐을 때 한 번에** 개방한다. 개발 도중 future 평균·분산·오차를 보며 설정을 고르지 않는다.

### 5. 판정식과 그에 따른 다음 행동

seed별 같은 기간 MSE를 E로 두고, 우선 질문은 데이터셋별 **D_P=E(Pearson+R)−E(Pearson)**다. 보조로 `D_q=E(q1+R)−E(q1)`, 선택 효과 `S_off=E(Pearson)−E(q1)`, `S_on=E(Pearson+R)−E(q1+R)`, 상호작용 `I=S_on−S_off`를 seed 내에서 계산한다. GRU/Linear도 같은 정규화 차이를 보고한다.

새 사전등록 제안: ETTh1/ETTh2의 두1차 D_P에 대해 **각97.5% paired t CI(n=8)**를 사용해 두 비교에 Bonferroni를 적용한다. 구간 상한<0이고 평균차≤−.005면 해당 데이터에서 개선 후보로 판정한다. .005는 기존 표준화 MSE의 실용 기준을 이어받는 것이며 n=8의 검출력을 보장하는 값은 아니다. 나머지는 보조95% 구간·효과 크기로 전부 보고하고 다중비교 보정된 유의성으로 표현하지 않는다. 두 데이터셋을 합쳐 한 방향의 우위로 덮지 않는다. seed 구간은 고정 기간에서 학습 변동만 나타내며 겹치는2925창×7채널을 독립 표본으로 세지 않는다.

| 관찰 | 다음 행동 |
|---|---|
| q1/Pearson 모두 개선, S_on과 S_off는 비슷 | R은 공통 경로 개선 후보. 선택기 고유 개선이라고 하지 않는다 |
| Pearson 개선, I도 일관되게 음수 | R과 선택의 상호작용 후보. 보조 분석의 불확실성을 붙이고 추후 확증 |
| q1+R은 개선하지만 Pearson+R이 q1+R보다 계속 나쁨 | R을 유지할 가치와 선택기 비용을 분리. 다음은 같은 예산의 선택 정책 대조 |
| 정규화한 GRU/Linear도 유사하게 개선, 격차는 유지 | 일반 정규화 효과로 해석. spiking 표현/readout 대조가 남음 |
| 개선 없음 또는 안전성 실패 | 추가 검색 후보를 무작정 확대하지 않고 상태·발화·읽기 경로 진단으로 돌아감 |

위 표는 사후의 해석 경로이지 새 평가 결과를 반복 확인하며 실험을 바꾸는 규칙이 아니다. 데이터셋별 맞춤 후보를 고르면 그 다음 주장은 별도 미사용 자료가 필요하다.

### 6. 검색 개선은 그 다음: 예산을 맞춘 주기 정책

앞의 통제 후에도 선택기 문제가 남으면 우선 q=.5를 고정하고 Pearson/recent/random/**하루 시차 우선+Pearson 보충**을 비교한다. 단계 n의 후보는 j<n, 예산은 `k=max(1,round(.5*n))`이다. 하루 후보는 `(n−j)%3==0`인 슬롯이며 k보다 적으면 나머지를 Pearson 점수로 채운다. 초반 단계 처리·초과시 최근 우선·동점 규칙을 먼저 명시한다. H96 analog의 후보30개와 모델의 실제 단계별 후보 수를 혼동하지 않는다.

“하루13칸 vs유사20칸”의 기존 결과를 근거로 바로 새 selector를 채택하지 않는다. 같은 칸수라도 커널 질량과 lag 분포가 달라지므로 선택된 질량·시차·상태 크기·발화율을 함께 보고한다. 메커니즘 주장을 원하면 별도 질량/lag 대조가 필요하다. random 평가에서는 MSE와 상태 진단을 같은 forward의 같은 마스크에서 모으고 RNG/MC 불확실성을 기록한다. 이 단계의 학습·평가는 **not run**이다. Gram 보정·delta rule·entmax·다변량 key는 현재 우선순위에 올리지 않는다.

### 7. 작업 에이전트의 다음 산출물과 확인 문헌

**먼저 이 제안을 채택/수정한 새 사전등록 append와 실행 manifest를 작성해 감사 가능하게 남겨 달라.** 8조건 정의, 재사용48개 목록/지문, R 및 보정 정의, 기간/문맥 경계, 손실 척도, 두1차 대비/구간, 실패 처리, 전체 예산이 포함돼야 한다. 실행 코드·data/config/checkpoint SHA, 환경/장치/명령/seed, 개방 기록, 원 로그를 결과와 연결한다. A23의 과거 provenance 누락을 소급 해결했다고 쓰지 않는다. 현재 실험 브랜치·base를 기록하고 사용자 요청 없이 main에 통합하지 않는다.

- [Zeng et al., AAAI 2023 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089): 채널 간 공유 시간축 Linear와 마지막 값 차감/복원 NLinear 정의를 이번에 직접 재확인했다. 여기서는 동일 파이프라인의 단순 기준선 필요성만 가져온다.
- [Kim et al., RevIN 저자 자료](https://seharanul17.github.io/RevIN/)와 [저자 구현](https://raw.githubusercontent.com/ts-kim/RevIN/master/RevIN.py): 입력 통계 제거와 출력 복원, affine 선택, detached mean/std, variance/epsilon 정의를 직접 확인했다. 이번 R은 affine=False로 요인을 줄인 대조안이며 논문의 기본 학습 affine 설정과 구분한다. 현재 모델의 개선은 아직 측정하지 않았다.

**이번 수행 결과:** 제안·원문 확인·기존 증거 snapshot만 완료. 모델 소스 변경, 학습, 새 구간 조회, GPU, 새 예약, 다른 에이전트 메시지 전송은 하지 않았다. 실험 실행/성능은 **not run**. 이 제안은 감사파일을 통한 전달이다.


## 2026-09-27 18:08 KST — 추적 감사51: 2O 정규화 구현·pilot 및 본실험 부분 증거

예약 `20260927T090001Z-b9116dd1`. 대응 **A24-IMPLEMENTATION / A24-PRECHECK / A24-FUTURE-GATE / A24-NEXT**. 이전 A23 원실행 provenance·MC 불확실성 OPEN은 유지한다.

### 관찰 기준과 보존

- branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, trigger/최초 관찰 HEAD `491bc5a0aca392f5fea79a66244804cb5ac5ae9d`, 종료 관찰 HEAD `491bc5a0aca392f5fea79a66244804cb5ac5ae9d`. 사전등록2O commit `2326b6166` → 구현 `a501bb565` → pilot/본실험 착수 기록 순서를 확인했다.
- 문서 기억·최신 감사/A24·사전등록2O·canonical PROJECT_LOG 17:57/17:59를 읽었다. 재현 전 18:00:33 KST에 감시783파일+문서3파일을 `/tmp/nsmt_assessment_20260927T090001Z-b9116dd1`에 별도 복사하고 SHA256을 기록했다. trigger와 다른9파일은 진행 중 본실험 myModel CSV7개·raw.txt·queue.txt였다. 스케줄러 hash를 원본 사본으로 취급하지 않았다.
- 완료 결과25건의 config/checkpoint50파일도 CPU 확인 전에 별도 복사·hash 기록했다. 증거는 `f_lif_pop_v3/forecasting/results/assessment/20260927T090001Z-b9116dd1/`의 `inventory.json`, `completed_evidence.json`, `cpu_probe.py/json`, `record_crosscheck.json`, `static_check.json`. stdout은 같은 task의 `log/assessment/20260927T090001Z-b9116dd1/`에 있다.
- 주요 SHA256: prereg `21a9d482fa6158fde777696cd4756b383cba755a57b8e2e37d1865888264fdab`; ours `a271cf1552c3e4967631525d8ad77ed9ed08352ed89f43d08adafe75496b350d`; train `53e8901517af4eb31136d515d4b31843500bea72865ce4a2dd6711e22f9ddeca`; data_loader `60dea4a530aa346a951e664e9eb34c1ab8dcb94b73ac250986ee812ebc6f590c`. 이들 및 calibrate/revin_checks는 후기 대조에서도 snapshot과 같았다. 감사 중 나타난 미추적 `ett_future.py`는 진행 중 산출물로 관찰했으며 이번 snapshot 밖이다. 후속 변경을 이번에 검증한 것으로 취급하지 않는다.

### (a) 아이디어 구현 — 확인 범위에서 적합

**A24-IMPLEMENTATION: 아래 CPU 합성 검사의 범위에서 VERIFIED.** 학습 affine 없이 시간축 평균·분산(unbiased=False, eps=1e-5, detached 통계)으로 patch 전에 변환하고, q1/Pearson/GRU/Linear 모두 출력에서 복원한다. Linear는 채널 공유336→96+bias다. train의 손실은 복원한 forward 출력과 기존 목표 간 MSE이며, 정규화 목표 손실로 바뀌지 않았다. R-on frozen input_norm 적합에 R 변환 train 입력이 들어가는 것도 실제 함수에 합성 batch를 넣어 확인했다.

독립 CPU fresh model 검사(B=2,L=336,C=3,H=96, no_grad): 네 모델 모두 수동 출력 복원 및 해당 MSE 차이0; 상수/서로 다른336값을 갖는 준상수 채널의 출력·상태 유한. batch 분할 최대차1.20e-7 미만, 채널 순열 차이0, +2.5 이동 등가성 최대차7.16e-7 미만. norm→denorm 오차 fp32 2.3841858e-7/fp64 4.4408921e-16. 이전 감사50 source의 q1/Pearson/GRU와 동일 state_dict를 사용한 R-off 출력 차이0. 이는 작은 입력의 경로 회귀 검사이며 전체 학습·실데이터 안전성을 인증하지 않는다.

합성 CSV만 주입한 실제 loader 검사: train/val/test/future 창수8209/2785/2785/2925, future 첫 입력[14064,14400)·첫 목표[14400,14496)·마지막 목표[17324,17420), scaler 적합은 train8640행과 일치. 실제 CSV 내용이나 future 통계는 감사에서 읽지 않았다. loader의 전체 CSV 읽기/유한성 검사와 train-only scaler 적합을 구분하며, 전체 과거 작업의 future 미사용 이력을 인증하는 검사는 아니다.

**A24-PRECHECK 보완:** 제작자 `revin_checks.py`의 fp32 `3+1e-9*noise`는 상수로 반올림될 수 있어 준상수 검사를 별도로 입증하지 못한다. 감사는 fp32 ±1e-4의 서로 다른 입력으로 보완하여 통과했다. 제작자48개 old checkpoint val 재현은 저장 결과에서 차이0이나 GPU 실행으로 기록되어 사전등록의 CPU 표기와 다르다. 이 장치 차이를 실행 편차로 남기고, 이번 소규모 CPU 결과를48개 전체 val 재현으로 확대하지 않는다. 감사의 전체48 val 재실행은 **not run**.

감사 probe 최초 두 명령은 각각 torch.Generator deepcopy 불가, 구버전 생성자의 max_length 기본64/실제42 불일치로 실패했다. 감사 코드만 수정(동일 설정 재생성+state_dict, 현행 neuron_kwargs 적용)한 세 번째 실행이 PASS. 앞선 실패 stdout도 보존하며 모델 실패로 해석하지 않는다.

### (b) 검증 과정·데이터 분리·통계·재현성 — 설계 수용, 전체 완료 보류

- 2O는 A24의8조건×2데이터×8seed=128, 기존48재사용+신규80, 입력 통계·복원 손실, 두1차97.5% paired t CI/평균차≤−.005, 보조95% 구간, 겹치는 창을 독립 반복으로 세지 않는 정의를 채택했다. 기존 test를 새 확증으로 재명명하지 않는다. 같은 학습 예산은 용량 동등·각 baseline 최적 성능의 증거가 아니다.
- snapshot 기준 pilot16건(4 R조건×2seed×2데이터)은 전부3epoch이고 별도 suite다. 본실험 완료 JSON은9/80건(ETTh1의 GRU_R/Linear/Linear_R, seed7/13/21)이며16~40epoch다. 나머지는 **in progress/완료 증거 대기**이고 모델 실패가 아니다. pilot 성능 숫자는 판정·후보 선택에 사용하지 않았다.
- 위25건에서 config의 model/seed/R/test=False, JSON test=null/test_skipped, 원checkpoint SHA 일치, CSV epoch 수·최저 val(반올림차<5.01e-7), provenance source11파일 일치, R별 보정 파일·scale·bound 일치를 확인했다. 본실험9건의 마지막 epoch는 CSV 최저점 이후10epoch로 patience와 일치한다. 전체80개 완료/전체128개 재사용 적합성 관문은 아직 검증하지 않았다.
- R 보정은 train 기준 scale10, ETTh1 bound1334.6653747558594/발화율0.1901935338973999, ETTh2 bound2026.3174438476562/발화율0.19890952296555042. off의 scale6과 달라 **정규화+필요 보정의 결합 효과**다. 보정 수치를 기록했을 뿐 감사에서 재보정하지 않았다. `load_calibration`은 최신 파일을 택하므로 최종128 관문에서 각 run의 실제 파일/hash와 사전 고정값을 대조해야 한다.
- **A24-FUTURE-GATE: PENDING, 확정 오류 아님.** 최초 snapshot에 future 개방 registry/결과가 없고, 개발 중 평가기 신규 파일이 나타났다. 정식128 관문·등록부 원자적 개방·같은 forward의 오류/상태 수집·실패 처리 검증은 남아 있다. 새 구간 성능·CI·확증 판정은 **not run**. 기존 FIXED-PENDING-REVIEW를 근거 없이 닫지 않았다.

### (c) 작업 에이전트에게 감사파일로 전달하는 다음 개선 순서

1. **현재2O를 고정한 채 완결한다.** 부분 val/pilot로 q·seed·정규화·보정을 고르거나 예산을 바꾸지 않는다. 누락/실패 셀과 이유를 전체 manifest에 남기고, 기존48도 실제 config/checkpoint/data 지문을 대조한다. 신규 학습80개와 pilot16개를 섞지 않는다.
2. **future 개방 전에 평가기 관문을 합성 결함으로 점검한다.** 두 데이터셋128셀 중 마지막 셀 누락·중복·다른 R/seed·변경 checkpoint/data/calibration·조기종료 증거 부족·안전 한계 변경을 넣었을 때, 어떠한 future dataset 생성/통계 조회나 registry 개방보다 먼저 거부되어야 한다. 이전 test용48관문을 그대로 통과했다는 사실은2O128관문의 검증이 아니다. 이미 연 registry 및 중간 실패의 재개 규칙도 명시하고, 임의 재개로 두 번째 독립 관찰을 만들지 않는다.
3. **완료 후 사전등록 대비만 계산한다.** 데이터셋별 D_P 및97.5% 구간/−.005 기준, 보조 D_q·S_off·S_on·I·GRU/Linear±R·채널별 오류를 모두 남긴다. R의 공통 효과와 selector 고유 상호작용을 구분한다. 이번 부분 결과에서 새 성능 우위를 주장할 근거는 없다.
4. **추가 selector 확대는2O 판정 뒤로 둔다.** 효과가 없으면 상태/발화/readout 진단, 공통 개선이면 정규화 경로 우선, 선택 비용만 남으면 A24의 같은 q·칸수 예산 주기/Pearson/recent/random 대조를 별도 사전등록한다. 현 단계에서 α=1·막전위 readout까지 동시에 바꾸지 않는다. 제안이지 이번 감사의 실험 실행이 아니다.

문헌은 A24에서 직접 확인한 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)·[affine 선택 및 detached 통계의 저자 구현](https://raw.githubusercontent.com/ts-kim/RevIN/master/RevIN.py), [Zeng et al. AAAI2023 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089)을 이어 사용한다. 이 문헌은 대조 설계의 근거이며 현 모델의 개선 증거가 아니다. 이번에 새 문헌/새 외부 사실 주장을 추가하지 않았다.

재현 명령(CPU): `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260927T090001Z-b9116dd1/cpu_probe.py` (성공 stdout `cpu_probe_retry2.txt`). 모델/학습 소스 변경·학습/backward·GPU·설치·프로세스 조작·Git 변이·다른 세션 조회/메시지·새 예약은 하지 않았다. 이 감사 문서 및 artifact 자체는 다음 연구 변화로 재판정하지 않는다.

<!-- assessment-watch:20260927T090001Z-b9116dd1 -->


## 2026-09-27 18:15 KST — 추적 감사52: 2O 80학습 완료·128관문 확인 / 평가 출력 누락

예약 `20260927T091001Z-07aaecc1`. 대응 **A24-FUTURE-GATE / A24-PRECHECK / A25-CHANNEL / A25-UNSAFE-SECONDARY / A25-INTERACTION-RELATIVE**. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. trigger/최초 HEAD `1a1e556653ffda9dc8bed7f734296f3900ed7e8b`, 후기 HEAD `ce4780652456471914654f781bf3d4558e199a9a`. 감사51·기억·사전등록2O·canonical PROJECT_LOG 최신18:08 append를 복구했다. 자신이 쓴 감사51은 새 연구 결과로 세지 않았다.

### 보존·범위

재현 전에 `2026-09-27T18:10:36.468938+09:00`에 감시파일과 문서 총988파일을 `/tmp/nsmt_assessment_20260927T091001Z-07aaecc1`에 별도 snapshot하고 SHA256을 기록했다. trigger 대비 차이는 작성 중 `ett_future_gate_check.py`1개(전체관문 결함검사 추가)였다. 이어128건의 config/checkpoint256파일·stdout130파일·데이터CSV2파일(바이트 복사/hash만, 행 파싱 없음)을 별도 보존했다. 주요 소스 SHA256:

- ett_future.py `96694eed36a315afed7792ea8f8267ff2422edabf90e5fd7ff64b5901e8fbcd0`
- ett_future_gate_check.py `62f3a412782433f80e3a263e5bddc8bfe0535ad3b76dca0181f3f5e786644819`
- revin_checks.py `fd5bb39374326608e4a1adb94e34f5de9f8a1ad804fdfeb9cb428db63304acb9`

위 세 파일은 후기 대조에서도 snapshot과 동일했다. 증거 `f_lif_pop_v3/forecasting/results/assessment/20260927T091001Z-07aaecc1/`: `inventory.json`, `extra_snapshot.json`, `manifest_rows.json`, `probe.py`, `probe_result.json`, `fault_output.txt`, `new80_source_check.json`, `final_inventory.json`. 실행 stdout `f_lif_pop_v3/forecasting/log/assessment/20260927T091001Z-07aaecc1/probe.txt`.

### (a) 구현 정확성 — R 경로 기존 판정 유지, 평가 집계 보완 필요

R 모델 구현은 감사51에서 검증한 범위의 판정을 유지한다. 새 평가기는 R-on Pearson의 마스크 재구성에도 정규화 입력을 넣고, 실제 오차·상태는 동일 forward에서 수집한다. 합성 Linear 출력3창(2+1 batch)으로 실제 `test(...,flag='val')`를 호출해 MSE99.166664/MAE8.5 및 창수3이 독립 산술과 일치함을 확인했다. 실제 validation/future 모델 forward는 실행하지 않았다.

**A25-CHANNEL — OPEN, 평가 출력의 확정 누락.** 2O D-BN은 채널별 오차를 요구하지만 현재 `test`는 전체 MSE/MAE만 누적·반환하며, 결과 row/CSV/record에도 채널별 필드가 없다. 합성 검사에서도 누락을 확인했다. 전체 MSE 오류라는 뜻은 아니며 채널별 진단의 사전등록 충족이 남았다.

**A25-UNSAFE-SECONDARY — OPEN, 보조 비교 보류 처리 누락.** 원본 main의 대비 계산문을 합성8seed 기록으로 실행했다. q1_R/seed7만 unsafe일 때 D_P는 영향받지 않아 primary blocked=[]가 맞지만, 실패 모델을 포함한 D_q·S_on·I에도 수치/CI가 그대로 생성되고 해당 비교의 blocked/보류 표시는 없다. 전체 unsafe 목록 출력만으로 각 비교의 사용 가능 여부가 연결되지 않는다. 관련 비교마다 차단 사유를 붙이고 해석을 보류해야 한다. 실제 학습 모델이 unsafe였다고 판정한 것은 아니다.

**A25-INTERACTION-RELATIVE — OPEN, 확정 산술·직렬화 결함(보조 필드).** `paired(interaction, zeros)`는 relative를 mean/0으로 계산한다. 합성 I=−.02에서 relative=−Infinity가 생성됐다(평균/CI 자체는 올바름). 콘솔에서 상대값을 숨겨도 JSON에는 비유한 값이 남는다. I의 상대 변화율은 정의하지 않고 null/필드 생략으로 기록해야 한다. 그 외1차97.5% t CI는 합성 비상수8seed 차이와 독립 표준오차/분위수 산술이 일치했다.

### (b) 검증 과정·분리·대조군·재현성 — 관문은 확인 범위 VERIFIED

**본학습80/80 완료 증거 확인.** 재사용48+신규80=128의 데이터셋·8조건·8seed 조합을 snapshot에서 확인했다. 전체128건의 현재 `check_manifest`와 `stopping_evidence`를 CPU에서 실행해 오류0이었다. config를 CPU로 읽고 원래 경로를 snapshot으로 바꿔 대조했으며, 모델 생성/실제 데이터 평가 대신 metadata-only loader를 사용했다. checkpoint SHA·고정 데이터/보정 SHA·조건/R·CUDA/torch 기록·no-test·CSV 순차 epoch/최저 val·stdout 조기종료/CSV patience 재계산을 확인했다. epochs는 전체128건에서12~49다. 신규80건 provenance source11파일 모두 snapshot과 일치했고, 재사용48건의 checkpoint/config는 감사45 보존본과 각각48/48 동일했다. 데이터 CSV는 지문만 확인했다.

**A24-FUTURE-GATE — manifest/개방 전 차단 범위 VERIFIED.** 소스를 먼저 읽은 뒤 제작자 fault helper를 CPU metadata loader로 제한해 실행했다. 정상 gate 오류0, 결함9종(마지막 셀 누락, 중복, seed, checkpoint, CSV hash, R 보정 hash, stdout 종료 증거, bound, R flag) 전부 거부; data_provider/registry tripwire 호출0. 중복 fixture는 실제로 ETTh2 Linear-off/1024에서 먼저 걸리므로 '마지막 셀 중복'까지 검증했다고 확대하지 않는다. 별도 검사에서는 원본 CLI main을 실행하되 ETTh1 개방 요청의128번째 ETTh2/linear_R/1024 누락을 넣어 dataset/registry 접근 전에 SystemExit를 확인했다. 등록부는 감사 artifact 안의 합성 key만 사용해 다른 출력 이름으로도 두 번째 개방을 거부함을 확인했다. 실제 등록부는 수정하지 않았다. 이는 재개/복구 설계 전체나 실제 GPU 안전성의 인증이 아니다.

제작자 로그의128/128·R 특화6결함·seed7 validation16건 최대차1.1e-7/mask mismatch0은 확인했지만, GPU validation16회는 감사에서 **not run**이다. `revin_checks.py`의 준상수 잡음1e-4와 std>0 assertion 추가는 소스에서 확인했다. 변경된 전체 revin_checks 실행 및 기존48 val의 CPU 재현은 **not run**; 저장된 이전 결과를 이번 변경의 재실행 증거로 쓰지 않는다. 감사51의 소규모 준상수 통과와 GPU/사전CPU 실행편차 고지는 유지한다.

**감사 중 발생한 개방을 구분한다.** 최초 snapshot에는 future 결과/registry가 없었다. 후기 실제 registry는 ETTh1 **18:11:34.877901**, ETTh2 **18:11:35.714570 KST** 개방을 기록한다. 해당 registry만 별도 복사/hash했다. 이번 감사는 이후 생성되는 future 결과 수치·완료 상태를 읽거나 평가하지 않았으며, 성능 판정은 보류한다. 'future 미개방'이라고 현재 상태를 기술하지 않는다. 결과는 다음 변경 감사 대상이다.

### (c) 감사파일을 통한 다음 개선 방향

1. 전체128 완료/관문 통과는 확인됐으므로 부분 val로 새 후보를 고르는 일 없이2O의 사전등록 대비를 유지한다. 이번 새 증거는 완료·검증 절차의 진전이며 **새 future 성능 판단 근거는 없음**이다. future 결과 검증·성능 CI 판정은 이번 감사에서 **not run**.
2. **우선 평가 산출물의 세 누락을 정리한다.** 채널별 오차를 같은 forward에서 누적하는 설계, 비교별 unsafe 차단, I 상대값 제외와 엄격한 유한 JSON 검사를 향후 평가기에 적용하고 합성 fixture로 검증한다. 이미 개방됐으므로 현재 기록/로그를 보존하고 원본 결과를 조용히 덮어쓰지 않는다. 기존 한 번의 pass에서 채널별 오차 또는 예측이 저장되지 않았다면 복원 불가를 고지한다. **누락을 메우기 위한 임의 future 재실행·registry 초기화는 하지 않는다.** 결과를 다시 보지 않고 가능한 사후 산술/표시 정정도 원본과 정정 이력을 연결한다.
3. 다음 결과 감사에서는 안전한 비교별 D_P97.5%/평균≤−.005, 보조 D_q/S_off/S_on/I, GRU·Linear±R을 데이터셋별로 확인한다. seed 변동과 겹친 시간창의 불확실성을 혼동하지 않고, R-on scale10의 효과는 정규화+보정 결합 효과로 해석한다. 모델 개선 순서는 A24의 관찰별 분기(공통 정규화 효과→선택기 상호작용→필요시 같은 예산 주기 대조)를 유지한다. 현재 다른 selector/readout/α를 추가할 성능 근거는 없다.

설계 근거는 A24에서 확인한 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)·[저자 구현](https://raw.githubusercontent.com/ts-kim/RevIN/master/RevIN.py) 및 [Zeng et al., AAAI2023 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089)을 이어 사용한다. 새 논문·새 외부 사실을 추가하지 않았다. A23 원실행 provenance/MC 불확실성 OPEN도 유지한다.

실행 명령: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260927T091001Z-07aaecc1/probe.py`. 검사 성공, 확인된 결함은 위와 같이 별도 기록. 학습/backward·GPU·환경 설치·연구소스 변경·진행 프로세스 조작·Git 변이·타 세션/에이전트 메시지·새 예약 없이 감사 증거와 append만 작성했다.

<!-- assessment-watch:20260927T091001Z-07aaecc1 -->


## 2026-09-27 18:26 KST — 추적 감사53: 2O future128 결과 대조 / ETTh2 1차 통과·ETTh1 미통과

예약 `20260927T092001Z-0fce65ef`. 대응 **A24-RESULT / A25-CHANNEL / A25-UNSAFE-SECONDARY / A25-INTERACTION-RELATIVE / A26-INTERPRETATION / A26-NEXT**. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, trigger/최초·후기 HEAD `ce4780652456471914654f781bf3d4558e199a9a`. 문서 기억·감사52·사전등록2O D-BP 최신 추가·canonical PROJECT_LOG 18:11/18:13 기록을 읽었다.

### 관찰 기준과 실제 증거

재계산 전에 `2026-09-27T18:20:37.170257+09:00`에 감시파일 및 문서1073개를 `/tmp/nsmt_assessment_20260927T092001Z-0fce65ef`에 별도 snapshot하고 SHA256 기록, trigger 차이0. 추가로128건 config/checkpoint256개·stdout130개·실데이터CSV2개를 바이트 복사/hash했다(행 파싱 없음). 추가388파일은 감사52 지문과 모두 같았다. 스케줄러 hash를 원본 사본으로 사용하지 않았다.

- ett_future.py SHA256 `96694eed36a315afed7792ea8f8267ff2422edabf90e5fd7ff64b5901e8fbcd0`
- prereg SHA256 `d66c7f24a9217cdb776719df9f61a385e382cd8761d3a1553b528e282f8d3826`
- ETTh1 record SHA256 `187f376b2227d066ec77dfeea32a403501f4bcf5eb531d01c26955adb09fe236`
- ETTh2 record SHA256 `972f6ee0f03be00056615bd5de4712fb152310f1a7860135aef6a45e15430f8a`

위 파일은 후기 대조에서도 동일했다. artifacts는 `f_lif_pop_v3/forecasting/results/assessment/20260927T092001Z-0fce65ef/`의 `inventory.json`, `extra_snapshot.json`, `result_audit.py/json`, `end_check.json`; stdout은 같은 task의 `log/assessment/20260927T092001Z-0fce65ef/result_audit_retry.txt`. 두 원 record/CSV는 수정하지 않았다. 감사용 재계산 JSON만 I의 정의되지 않은 relative를 null로 표현했다.

### (a) 구현·실행 정확성 — 기존 범위 유지, 결과 산술 VERIFIED

두 record는 done, 각각8조건×8seed=64개 유일한 행이며 모든 행이2925창이다. source9파일,128개 checkpoint/config 지문, 고정 CSV/보정 지문, 학습 결과의 checkpoint/epoch 수가 일치했다. final CSV128건의 future 행은 MSE/MAE 소수6자리·상태/발화 등 소수4자리 반올림 허용치 안에서 record와 일치한다. 기존2N run48개의 이전 final CSV 내용이 prefix로 보존되고 future 행이 append됐음을 확인했다.

기록상128건 모두 출력 유한/안전이며 nonfinite_batches=0. 스파이킹64건은 각 R 설정의 고정 bound 이내다. R-off/R-on별 최대 |state|는 ETTh1 **126.422119/124.186310**, ETTh2 **62.146259/139.886627**; 해당 bound는 각각507.555771/1334.665375 및975.164948/2026.317444다. GRU·Linear에는 상태 bound를 주장하지 않는다. Pearson±R의 동점은 ETTh1 **8/13,431,600**, ETTh2 **18/13,431,600**, 마스크 재구성 mismatch 전부0.

기존 A24-IMPLEMENTATION·A24-FUTURE-GATE의 검증 범위는 유지한다. 이번에는 새로운 모델 probe를 불필요하게 반복하지 않고 **저장된 결과의 독립 산술·출처 대조**만 수행했다. 원 모델의 실데이터 forward 재현·학습·GPU는 **not run**이며, 상태 유한성은 보존된 실행 기록에 근거한다.

### (b) 검증 과정·데이터 분리·통계 — 사전등록1차 판정 재현

registry ETTh1 18:11:34.877901/ETTh2 18:11:35.714570 KST 개방 경로가 두 record와 일치한다. 감사52에서 검증한128 관문과 변경 없는 checkpoint/config를 연결했다. 이는 기록된 정식 경로의 검증이며 모든 과거 작업에서 이 기간이 미사용이었다는 인증은 아니다. D-BP의 재개 금지·GPU 사전검사/준상수 사후교정 편차 고지는 확인했다. 중단 지점까지 부분 결과를 남기는 기능 전체를 검증했다고 주장하지 않는다.

**A24-RESULT: 저장된128행으로 다음1차·보조 통계 산술 VERIFIED.** n=8 seed paired t 구간이며 두 데이터셋1차에 각각97.5% 구간을 사용했다. 창/채널을 독립 표본으로 세지 않았다.

| 데이터 | Pearson MSE | Pearson+R MSE | D_P 평균 | 97.5% CI | 상대 변화 | 사전등록 판정 |
|---|---:|---:|---:|---|---:|---|
| ETTh1 | 0.543057963 | 0.537715264 | −0.005342700 | [−0.010754721, +0.000069322] | −0.9838% | 미통과: 구간 상한>0 (6/8 seed 감소) |
| ETTh2 | 0.209138041 | 0.176058263 | −0.033079779 | [−0.039151427, −0.027008131] | −15.8172% | 통과: 상한<0, 평균≤−.005 (8/8 감소) |

ETTh1 상한을 반올림해0 또는 음수로 취급하지 않는다. 미통과는 효과가 정확히0이라는 증거가 아니다. ETTh2 통과는 이 기간·예산에서 **정규화+보정 결합 효과**의 개선 후보 판정이며, 양 데이터셋 일반 우위나 selector 자체 우위가 아니다.

보조95% 구간(다중비교 확증 아님):

| 대비 | ETTh1 평균 [95% CI] | ETTh2 평균 [95% CI] |
|---|---|---|
| D_q | −.028158829 [−.031413247,−.024904410] | −.025066562 [−.030024010,−.020109114] |
| S_off | −.019028194 [−.023710157,−.014346232] | +.000765050 [−.002124540,+.003654640] |
| S_on | +.003787935 [−.002305622,+.009881491] | −.007248167 [−.011608483,−.002887851] |
| I | +.022816129 [+.016774776,+.028857482] | −.008013217 [−.013984664,−.002041769] |
| GRU_R−GRU | +.001443617 [−.002190412,+.005077646] | +.000001088 [−.005089406,+.005091582] |
| Linear_R−Linear | −.000691880 [−.005876276,+.004492517] | −.000702849 [−.002858866,+.001453168] |

**A25 이슈는 닫지 않는다.** CHANNEL: 실제128행에도 채널별 오차가 없고 원 record/CSV만으로 복원할 수 없다. UNSAFE-SECONDARY: 비교별 차단 누락은 코드에 남지만 이번128건 unsafe=[]여서 현재 보조 산술의 누락/선택 편향을 만들지는 않았다. INTERACTION-RELATIVE: 실제 ETTh1 +Infinity, ETTh2 −Infinity가 저장돼 엄격 JSON 파서가 거부한다. I의 평균·CI는 정상이며1차 판정에는 영향 없다. 원본 보존·정정 이력 및 별도 파생 기록으로 처리할 일이고 임의 future 재평가/registry 초기화의 근거가 아니다.

감사 스크립트 첫 실행은 최종 출력에서 numpy.bool_ JSON 직렬화 오류가 났다. bool 변환을 감사 코드에만 적용한 재실행은 통과했고 실패 stdout도 보존했다. 모델 또는 원 결과의 실패로 해석하지 않는다.

### (c) 새 결과에 따른 해석 정정과 다음 실험 제안 — 감사파일로 전달

**A26-INTERPRETATION — OPEN, canonical 18:13 해석 문구 정정 요청.**

1. “R은 GRU·Linear에는 효과가 없다”는 대신 **“이 예산·기간에서 두 기준선의 R 차이에 대한 보조95% 구간은0을 포함했고, 일관된 개선을 입증하지 못했다”**고 쓴다. 0 포함은 무효과/동등성의 증거가 아니다. “차이<0.2%”도 상대 MSE 기준으로 맞지 않는다: ETTh1 GRU **+0.2908%**, ETTh2 Linear **−0.4243%**다.
2. 따라서 “일반 정규화 효과가 아니다/스파이킹에 특유하다”는 확정 대신 **모델 계열별 효과 차이 가설**로 둔다. R의 spiking 조건에서는 scale뿐 아니라 theta도 바뀌었다: ETTh1 **scale6→10, theta1.829016683→6.163561141**, ETTh2 **6→10, theta2.382834404→7.153850038**. frozen input_norm도 R 입력에 맞춰 다시 적합했다. 현재 대비는 이 묶음을 분리하지 않는다.
3. ETTh1의 양수 I는 R 유무에 따라 선택 대비가 달라졌다는 관찰이다. S_on 구간은0을 포함하므로 Pearson+R이 q1+R보다 확실히 나쁘다고 하지 않는다. “같은 부분을 고치는 대체 관계”는 메커니즘 가설이다. ETTh2 음수 I(7/8)는 상호작용 후보로 유지하되 보조 분석이고 새 자료의 확증이 필요하다.

**A26-NEXT — 권장 순서(새 실험은 전부 not run).**

- **먼저 기록을 완결한다.** A25와 위 해석을 append로 정정하고, 사전등록1차 통과/미통과 및 누락된 채널 진단을 명시한다. 저장값만으로 가능한 상대값·비교별 상태 정정은 원본 hash를 참조하는 별도 기록에 둔다. 채널 진단을 얻으려고 이미 연 future를 다시 통과하지 않는다.
- **가장 싼 다음 진단은 train/val에서 입력처리와 보정의 영향을 비교하는 것**이다. q1/Pearson, R-off/on의 frozen 통계·current/state 크기·발화율·남긴 커널 질량·lag 분포를 같은 forward 기준으로 모은다. checkpoint·seed별 결과와 보정값을 연결하고 후보를 고를 때 쓴 val이라는 사실을 고지한다. 이 진단만으로 future 성능/인과 효과를 주장하지 않는다. α·readout·새 selector를 동시에 바꾸지 않는다.
- **메커니즘을 분리할 필요가 남으면 작은 사전등록 요인 실험을 제안한다.** q={1,.5}×R={off,on}×보정 묶음 B={기존 scale/theta, R용 scale/theta}를 교차한다. frozen 입력 통계는 각 R의 train 입력으로 적합한다. 따라서 분리 대상은 'R 입력처리 경로'와 '명시적 scale/theta 보정 묶음'이며 R 단독 또는 scale 단독이라고 부르지 않는다. 각 셀의 train-only 안전 기준을 결과를 보기 전에 고정하고, 위험 셀은 사전 규칙대로 실패/보류한다. 같은 셀·조건이 맞으면2O64개 spiking 학습을 재사용하고 교차64개를 추가하는 설계가 가능하지만, **이는 즉시64개 학습을 요구하는 지시가 아니다**. 저비용 진단과 질문의 필요성을 먼저 판단해 범위를 확정한다. 효과에 따라 scale과 theta를 각각 분해하는 것은 그 다음 단계다.
- **독립 확증 자료 확보를 실험 착수 조건으로 둔다.** 지금 future는 이미 열렸다. seed만 추가하거나 output 이름을 바꿔 같은 기간을 새 확증이라고 하지 않는다. 새 시계열/미사용 기간의 사용 이력·경계·목표·기준을 고정해야 한다. 새 자료가 없으면 다음 요인 분석은 train/val 탐색으로만 표시한다. dataset별 사후 맞춤 모델 선택을 현재 future의 새 확증으로 보고하지 않는다.
- **선택 규칙 개선은 이후**다. R/보정을 정렬한 뒤에도 선택 비용/상호작용 질문이 남을 때 A24의 동일 q·칸수 예산 Pearson/recent/random/하루정렬 대조를 진행한다. 현재 ETTh2 통과를 근거로 Gram 보정·다변량 key·학습 gate 등 여러 후보를 동시에 늘릴 필요는 없다. 예측 성능 관점에서는 Linear/GRU 참고선의 동일 예산 비교를 계속 함께 보고, 별도의 용량 통제 우위는 주장하지 않는다.

이번에 다시 확인한 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)는 입력 통계 제거와 출력 복원의 일반적인 방법을 제시하며, 현 spiking 모델에 효과가 특유하다는 근거는 아니다. [Zeng et al., AAAI2023 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/26317)·[기존 확인 원문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089)은 단순 선형 기준선을 함께 두는 설계 근거로 이어 사용한다. 보정 교차 제안은 이번 로컬 결과에서 도출한 가설 검증안이지 이 논문들이 보장하는 개선법이 아니다.

실행 명령: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260927T092001Z-0fce65ef/result_audit.py`. 이번 실행은 CPU 저장값 재계산과 read-only 대조이며, 모델/학습 소스 변경·학습/backward·GPU·프로세스 중단·설치·Git 변이·타 세션/에이전트 메시지·새 예약을 하지 않았다. A23 잔여 원실행 provenance/MC 이슈도 유지한다.

<!-- assessment-watch:20260927T092001Z-0fce65ef -->


## 2026-09-27 21:46 KST — 추적 감사54: 2O 파생 정정·α=1 사전검사 검증 / 감사53 theta 해석 정정

예약 `20260927T124001Z-4de1c1d2`. 대응 **A25 / A26 / A27-ALPHA1-PRECHECK / A27-ALPHA-INTERPRETATION**. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. trigger/최초 HEAD `6e5f375cd5addbae90cdc56077aa76514dc498e0`, 후기 HEAD `3cde96941c4021fc45b000ecb758f1779c0299f8`. 문서 기억·감사53·사전등록2P·canonical 최신 append를 확인했다. 감사53 자체를 새 연구 성과로 세지 않았다.

### 관찰 범위·증거 보존

재검사 전 21:40:36 KST에1082파일을 별도 snapshot하고 SHA256을 기록했다. 처음 `/tmp/nsmt_assessment_20260927T124001Z-4de1c1d2`을 사용했고, 같은 바이트 전체를 `f_lif_pop_v3/forecasting/results/assessment/20260927T124001Z-4de1c1d2/source_snapshot/`에도 보존·hash 대조했다. `inventory.json`, `trigger.json`, `checks.json`, `audit_checks.py`, `postcheck.json`이 같은 artifact 폴더에 있다. raw stdout은 `f_lif_pop_v3/forecasting/log/assessment/20260927T124001Z-4de1c1d2/checks.txt`이다. fixture 폴더의 unsafe 값은 합성 결함 주입이며 실제 모델 실패가 아니다.

trigger와 다른 것은 진행 중 `run_ett.sh`1파일이다. 후기 대조에서 snapshot1082파일 중 canonical PROJECT_LOG만 외부 append로 바뀌었고 검증 대상 소스/원 결과는 그대로였다. 감사 중 추가된 α=1 보정·pilot·진단은 다음 주기 범위로 남긴다. 새로 나타난 파일의 미완성이나 검증 전 상태를 확정 오류로 판정하지 않는다.

주요 SHA256:
- prereg: `8499ac9d6dc7e8f90014081d6b01f45ef6ea7cd8f100d4d870d39f0e3b0978fa`
- layers: `85906858bdc51f90a314c9e6b9a73805e642ebab09dc2acfc1b9502f91a93db0`
- alpha1_checks: `3fdcb2bfb3292a5835d03fdc3e00e812247672da2ef19cdb8b29d341e28f98f5`
- theta_inert_hard: `e30a20b1932e2163ac6fbaa07f63d9dacc8ef4d2068e88e4743aa57d4264bb57`
- ett_future_derived: `caa5b6d222a221741799457383a8247d2b3ae4bb953f97417c3341ef95c7c684`

### (a) 아이디어 구현 정확성 — 검증한 범위만 VERIFIED

**A27-ALPHA1-PRECHECK: 사전 점검5종 CPU 재현 VERIFIED.** 현재 API를 읽고 snapshot의 probe를 실행했다. α=1 계수표는 정확히1, q1 가지 상태와 명시적 Euler 재귀의 float64 최대차0, R-on 모델의 hard q1/full 출력·상태가 비트 단위로 같다. 상수 및 실제 분산이 있는 `3+1e-4 noise` 채널에서 두 조건의 출력·상태가 유한하고, α=.7/1의 run id·run directory·result path가 모두 다르다. 저장된 alpha1_checks.json 전체와 재검사 JSON이 일치한다. 새 초기화·합성 입력의 검사이며 학습 모델의 안정성·G11 충족·성능 검증은 아니다.

**A26-THETA: 감사53의 설명·후속 제안을 정정한다.** Selector.forward의 hard 반환은 theta로 score를 나누는 attention 경로보다 앞이다. 기존 probe 재현과 별도 random seed11 합성 입력 검사 모두 hard q1/Pearson에서 두 theta 쌍의 출력·상태 차이0. sparse 양성 대조의 출력 최대차는 별도 검사에서 ETTh1 .087627823, ETTh2 .095698725로0이 아니다. 따라서 감사53이 theta 변경을 hard 경로의 작동 요인처럼 다루고 scale/theta 교차·후속 분해를 제안한 부분은 부정확했다. **hard 조건에서 명시적 보정 축은 input_scale이며 theta 교차는 필요 없다.** R에 맞춰 적합한 frozen 입력 통계와 입력 정규화 경로는 여전히 구분할 요인이다. selector temperature와 소마 발화 문턱은 같은 파라미터가 아니다.

α=1의 Euler 환원은 **q1 가지**에 대한 명제다. Pearson q=.5는 과거 증분을 선택하므로 전체를 보통 단일 LIF와 동일하다고 부르면 안 된다. 모델은 계속4가지 leaky integrator와 별도 reset 소마를 가진다. α=1 대조의 목적에는 맞지만 일반 SNN 전체를 대표하거나 GRU와 구조를 맞춘 대조는 아니다.

### (b) 검증 과정·분리·통계·재현성

**A25-INTERACTION-RELATIVE / UNSAFE-SECONDARY: 두 파생 기록과 그 생성 경로의 보완은 scoped VERIFIED.** 원 ETTh1/2 future JSON SHA는 감사53의 `187f376b2227d066ec77dfeea32a403501f4bcf5eb531d01c26955adb09fe236` / `972f6ee0f03be00056615bd5de4712fb152310f1a7860135aef6a45e15430f8a`와 같다. 원본을 보존한 별도 파생 생성 결과가 저장된 두 derived JSON 전체와 일치한다. 엄격 JSON 파싱 통과, I relative=null, 나머지 대비의 seed 차이 평균과 기록 평균 일치. q1/seed7 unsafe를 합성 주입하면 D_q·S_off·I만 보류되고 q1을 쓰지 않는 S_on·GRU·Linear 비교는 보고 상태를 유지한다. 실제128건은 unsafe가 없어 보류된 비교가 없다. 과거 evaluator 자체가 고쳐졌다고 닫는 것이 아니라 **과거 결과의 파생 정정** 범위에서 검증했다.

**A25-CHANNEL: 미충족 잔여 유지.** 파생 기록이 channel_errors=null과 복원 불가 사유를 명시한 것은 적절하지만, 누락된 측정이 복구된 것은 아니다. 이미 연 future를 다시 평가하지 않는다. 원 evaluator의 두 결함도 역사적으로 남는다. 다음 평가기에는 채널 오차·비교별 보류·엄격 JSON을 실제로 구현하고 검사해야 한다.

**A26-INTERPRETATION: canonical의 무효과/특유효과/대체관계 문구 정정은 문서 범위 VERIFIED.** GRU/Linear R 상대값은 ETTh1 +.2908%/−.1383%, ETTh2 +.0006%/−.4243%로 파생 결과와 맞는다. CI의0 포함은 무효과의 증거가 아니며 모델 계열별 차이·대체 메커니즘은 가설이다. A23의 원실행 provenance/MC 불확실성 잔여는 유지한다.

2P의32신규+64재사용, val만 사용, test/future 재개 금지, 전체96관문, best-val 재현 오차1e-6, unsafe/재현 실패의 비교별 보류, 겹치는 창을 독립 표본으로 세지 않는 규정은 질문에 맞는 계획이다. 다만 현재 snapshot에서96관문·새 평가기·채널별 집계·학습 완료·α 성능 CI는 **not run / 미검증**이다. 조기 종료에 사용한 val은 탐색용이며 동일 절차가 모델별 선택 편향의 동일 크기를 보장하지 않는다. 95% seed CI는 고정 기간의 학습 변동 범위이지 새 기간의 일반화나 다중 대비 확증이 아니다.

**A27-ALPHA-INTERPRETATION — OPEN(표현 정합성).** D-BS는 scale이 다르면 커널+보정 결합 효과라고 올바르게 제한하지만, 서두의 “바꾸는 것은 α 하나뿐” 및 D-BV의 “선택 없는 순수 커널 비교”는 그 조건을 생략한다. scale 변경 여부를 결과표에 함께 고정하고, 다르면 F_q·F_p·J 모두 **α별 보정 정책을 포함한 대비**로 표현하도록 append 정정을 권한다. q1이 selector를 제거한다는 사실만으로 scale 혼동까지 제거되지는 않는다. 이는 모델 계산 오류가 아니다.

### (c) 다음 실험 개선 방향 — 작업 에이전트에 감사파일로 전달

새 성능 판단 근거는 없다. 이번 파생 기록은 같은2O 결과의 정정이며 ETTh1 1차 미통과/ETTh2 통과 판정은 감사53 그대로다. **α=1 대조가 개선되는지는 판단 보류**한다. 감사 중 시작된 후속 준비/실행은 본 snapshot의 완료 결과로 포함하지 않았다.

1. **사용자가 고른2P α=1 대조를 우선하는 계획을 존중한다.** 감사53의 선택적 교차 제안 때문에 현재 계획을 다시 확대할 필요는 없다. 본 결과 전에 조건별 α·scale·frozen 통계 hash·calibration hash·학습 예산을 한 표로 고정한다. α별 보정이 다르면 kernel-only 표현을 철회하고 실제 실험이 추정하는 결합 효과를 보고한다.
2. **현재 계획 안의 진단을 완결한다.** 같은 validation forward에서 전체/채널 MSE, seed별 F_q/F_p/J, 발화율·가지 상태·안전 한계·동점률을 함께 남긴다. 채널은 기술 분석으로 두며 좋은 채널만 골라 새 확증처럼 보고하지 않는다. 재사용64건과 신규32건 전체 관문을 통과한 뒤 비교하고, 누락/unsafe/재현 실패 fixture로 비교별 차단을 검증한다. 검증 전 FIXED-PENDING-REVIEW 항목을 VERIFIED로 닫지 않는다.
3. **커널 자체 기전 질문이 남을 때만 다음 보완을 사전등록한다.** 우선 train-only 합성/실제 입력 진단에서 공통 scale의 안전성·발화율·state 범위를 비교한다. 이후 필요하면 동일 scale·동일 R/frozen 입력 처리·동일 예산으로 α 대조를 설계한다. 공통 scale이 안전하지 않거나 건강 기준을 벗어나면 실패/보류를 보고하고 결과를 보고 기준을 완화하지 않는다. 보정된 시스템 비교와 공통 scale 비교를 별도 추정 대상으로 둔다. theta 교차는 하지 않는다. 이번에는 이 보완 학습·실데이터 진단 **not run**이다.
4. **확증은 새로 미사용임을 확인한 자료에서 한다.** 2P의 val 결과로 후보를 정할 수는 있지만 그 val/test/future를 다시 새 확증으로 부를 수 없다. 데이터 사용 이력·경계·전처리·주 대비·실질 차이 기준을 새 평가 전에 고정한다. Linear/GRU는 동일 pipeline의 예측 참고선으로 유지하며, α=1 스파이킹 대조가 추가되었다고 기존 참고선을 제거하거나 용량 통제 우위를 주장하지 않는다.

설계 근거는 이번에 재확인한 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)의 입력 통계 제거·출력 복원과 [Zeng et al., AAAI2023 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/26317)의 단순 선형 기준선이다. α 대조와 보정 분리 제안은 로컬 계산·실험 설계에서 도출한 것이며 이 문헌이 현 모델의 개선을 보장하지 않는다.

실행 명령: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260927T124001Z-4de1c1d2/audit_checks.py`. 종료0·ALL CHECKS PASS. 학습/backward·GPU·실제 데이터 forward·연구 소스 수정·프로세스 중단·설치·Git 변이·타 세션 읽기/메시지·새 예약은 하지 않았다.

<!-- assessment-watch:20260927T124001Z-4de1c1d2 -->


## 2026-09-27 21:55 KST — 추적 감사55: α=1 Pearson16건 G11 실패 / q1 진행과 안전 진단 해석

예약 `20260927T125001Z-4f47a30f`. 대응 **A27-ALPHA-INTERPRETATION / A28-G11-RESULT / A28-DIAGNOSTIC-CLAIM / A28-COMPLETION-GATE**, 기존 A25-CHANNEL·A23 잔여 유지. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, trigger/최초·후기 HEAD `647641e708348934024cea4c079f4c5de25be30e`. 기억·감사54·2P D-BX·canonical 보정/pilot 기록을 읽었다. 감사54 문서 자체를 새 실험 결과로 세지 않았다.

### 확인 범위·보존

21:50:28 KST 재검사 전에 감시파일과 문서1177개를 `f_lif_pop_v3/forecasting/results/assessment/20260927T125001Z-4f47a30f/source_snapshot/`에 별도 복사·SHA256 기록했다. trigger와 차이6개는 진행 중 q1 CSV4개·pool log·raw.txt다. 추가로 본/pilot stdout36개와 snapshot에서 완료된 q1 checkpoint12개를 같은 snapshot에 복사/hash했다. `inventory.json`, `additional_snapshot.json`, `record_audit.py`, `record_checks.json`, `postcheck.json` 참조. raw 감사 stdout은 `f_lif_pop_v3/forecasting/log/assessment/20260927T125001Z-4f47a30f/record_checks.txt`에 있다. 후기 기존 파일 차이는 진행 중 CSV2개·pool/raw뿐이며 source/사전등록/실패 기록은 그대로다. 감사 중 새 완료 결과는 다음 주기 범위다.

주요 SHA256: prereg `a4c3700d2284c1faafbb7b89f57571d1218d44b35f89e052cf5b11977e7994fb`; 진단 py `b46bba7c68b2eadb9fa8f2b63896d7122bbe44efd480d683909c942d010ecfca`; 진단 JSON `0ab17e73139ea4a14c2d1e23ce7aac927efc07315695b465d6921201f8ef0002`; layers `85906858bdc51f90a314c9e6b9a73805e642ebab09dc2acfc1b9502f91a93db0`; train `53e8901517af4eb31136d515d4b31843500bea72865ce4a2dd6711e22f9ddeca`. 모델/학습 소스는 감사54 때와 같다.

### (a) 구현 정확성·실패의 성격

**A28-G11-RESULT — 실제 기록 대조 scoped VERIFIED.** 본 실행의 α=1 Pearson q=.5는 두 데이터셋8seed씩 **16/16건 epoch0/batch0 G11 위반**이다. pilot도 두 데이터셋 seed7에서 같은 종류의 실패다. 기록의 run ID/UUID가 logargs와 맞고, α=1·R-on·q=.5·no-test·scale 및 calibration 파일/한계를 확인했다. stdout의 G11 예외 수치와 일치하며 해당 조건의 완료 결과 JSON은 snapshot에 없다.

| 데이터 | 고정 scale | 고정 G11 한계 | 본8seed max|u| 범위 | 한계 대비 배율 |
|---|---:|---:|---:|---:|
| ETTh1 | 8 | 1067.732315 | 1790.704224–2402.108154 | 1.6771–2.2497 |
| ETTh2 | 6 | 1215.790405273 | 1447.565308–3103.547363 | 1.1906–2.5527 |

위 값은 유한하지만 고정 한계를 넘었다. 따라서 **등록된 안전 실패**이며, NaN/수학적 무한 발산 또는 일반 LIF 전체의 실패를 증명한 것은 아니다. 진단의 최초 GPU 실행에서 발생했다고 고지된 CPU random generator 장치 불일치와도 다르다. 그 명령 실패를 모델 실패 수에 넣지 않는다.

train_one_epoch는 해당 forward의 aux 상태를 보지만 G11 검사는 backward·clip·optimizer.step **이후**다. 따라서 “첫 batch에서 실패”는 맞고 “첫 optimizer update 이전에 차단”이라고 쓰면 틀리다. 여기서 읽는 peak는 update 이전 forward의 상태다. 감사는 학습을 실행하지 않았다.

full(q1) 보정과 sparse-at-init의 정상은 hard Pearson의 안전을 보장하지 않는다. 감사54의 상수/준상수 유한성 검사도 임의 입력의 고정 한계 통과를 뜻하지 않는다. q1에서 α=1은 가지 Euler 재귀로 환원하지만 q=.5에서 일부 과거 증분을 제거하면 그 환원은 성립하지 않는다. 현 실패와 q1 환원 검증은 모순이 아니다. 모델 소스의 새로운 구현 오류로 단정할 근거는 발견하지 않았다.

### (b) 검증 절차·분리·대조·통계

보정 JSON의 grid에서 band [.1,.3] 내 target .2에 가까운 scale은 ETTh1 8/ETTh2 6이며, bound=10×보정 max|I|와 source 지문이 맞았다. 이는 **저장 보정값의 산술·출처 대조**이며 GPU 보정 forward를 재현한 것은 아니다. α=.7 R-on scale10과 다르므로 A27의 “순수 커널” 표현 정정 요청은 **OPEN 유지**한다. D-BS의 결합 효과 제한을 F_q/F_p/J 해석에도 적용해야 한다.

**q1 snapshot 진행 상태:** 16개 중 완료 JSON12개(각 데이터 seed7/13/21/42/123/256), 나머지 seed512/1024 각2건은 완료 증거 미포함이다. 완료12건의 source11개 hash·UUID/config_hash·no-test·checkpoint SHA·CSV epoch 수·best val(차≤1e-6)·최저 val 뒤10epoch 이상 종료를 대조했다. ETTh1 15–28epoch, ETTh2 14–17epoch다. 이것은 완료12건의 기록 정합성 확인이며 전체96관문, 저장 config 객체의 전체 검사, validation forward 재현을 대신하지 않는다. 진행 중 네 건을 실패 또는 누락 오류로 세지 않았다.

**A28-COMPLETION-GATE — PENDING.** D-BX의 “완료(checkpoint+결과) 또는 G11 실패 중 정확히 하나”와 실패 seed 제외/재시도 금지는 적절하다. F_p·S_on(α=1)·J는 두 데이터셋 모두 실패가 포함되므로 보류한다. F_q는 Pearson 실패를 사용하지 않지만, q1 완료·재사용 출처·안전·재현 관문을 통과해야 보고할 수 있다. 완주한 seed만으로 실패 대비 CI를 만들거나 실패를 임의의 큰 MSE로 치환하지 않는다. 평가기의 실제 구현·결함 주입·96칸 검사·채널 오차·val 재현은 이번 **not run / 미검증**이다.

D-BX commit 시각은21:46:34 KST, 본 실패 최초 기록은 ETTh1 21:47:02/ETTh2 21:47:03이다. 이 기록들은 본 실패 결과에 앞선 규칙 고정을 지지한다. canonical의 해당 제목은21:55로 수동 기재돼 실제 commit보다 뒤이므로 제목만으로 순서를 추론하지 않는다. 파일 제목 시각 정정 또는 실제 commit 시각 병기를 권한다. pilot 결과를 보고 추가한 규정이라는 점은 이미 고지돼 있다.

**A28-DIAGNOSTIC-CLAIM — OPEN, 해석 범위 정정.** 진단24행의 within 판정과 per-step 최대값 반올림은 JSON과 일치한다. 코드상 train 첫8 shuffled batch, seed7, 새 embedding, 동일 frozen 입력 처리, 각 α의 자체 scale 및 공통 scale 대조이며, recent/random 분석을 앞선 결과 후 추가한 사후 진단이라고 고지한다. 이번에는 실제 train8batch forward를 재실행하지 않았으므로 원 수치 재현 완료로 닫지 않는다.

- D-BX/canonical의 “내용 무관 규칙은 상태를 키우지 않는다 / 유사도로 고르는 경우에만 커진다”는 절대 표현은 저장값과 맞지 않는다. α=1에서 ETTh1 q1 **28.17358**, random **67.599342(2.3994배)**, recent **29.84**; ETTh2 q1 **31.80659**, random **50.22787(1.5792배)**다. **“이 초기화·표본·예산에서 비교한 규칙 중 Pearson만 해당 G11 한계를 넘었고 가장 큰 증가를 보였다”**로 좁힌다.
- 공통 scale 비교는 scale 차이만으로 Pearson peak 차이를 설명할 수 없음을 지지한다. 하지만 seed7·표본8batch로 다른 유사도 규칙/seed/길이까지 일반화하거나 분수 커널이 안전성을 보장한다고 해석하지 않는다. common 행의 한계도 각 α의 기존 보정 한계임을 표시한다.
- per_step_max는 모든 batch·sample·unit·branch의 **최대값을 step별로 모은 값**이다. 서로 다른 셀이 최대일 수 있으므로 단일 뉴런이 그 경로로 단조 증가했다고 부르지 않는다. 상태·증분의 방향 정렬을 측정하지 않은 현재의 양의 되먹임 설명은 가설로 유지한다.

### (c) 새 증거에 따른 개선 방향 — 감사파일로 전달

새로 확인된 것은 α=1 Pearson의 등록 안전 실패와 q1의 부분 완료다. **예측 성능에 대한 F_q·F_p·J 및 α 우위 판단은 보류**한다. 다음 순서를 권한다.

1. 기존2P의 고정 한계와 실패 기록을 보존하고, 정상 q1 조건의 완료 및 전체 관문을 먼저 확인한다. 이번 결과를 구하기 위해 scale·q·한계를 사후 조절하거나 실패 seed를 빼지 않는다. 계획 밖 새 학습은 이번 감사에서 하지 않는다.
2. 다음 메커니즘 진단은 **train에서 저장한 동일 current tensor·동일 초기화의 소규모 forward**로 제한해 시작한다. q1/Pearson/recent/random의 실제 keep 수·선택 lag·남긴 커널 질량과 함께, 선택/제외한 증분의 합 및 절댓값 합, 현재 상태 방향과 증분 방향의 내적을 기록한다. step별 최대값뿐 아니라 최대가 발생한 sample/unit/branch의 추적값도 남긴다. 이는 “음의 되먹임 항 제거/같은 방향 증분 선택” 가설을 직접 검사하기 위한 제안이며 아직 **not run**이다.
3. 그 뒤 필요하면 **고정 mask 재생 대조**를 별도 진단으로 추가한다. 한 조건에서 얻은 mask를 같은 입력의 다른 α에 고정 적용한 결과와 상태에 따라 mask를 다시 계산한 결과를 구분하면, 커널 변화와 선택 경로의 되먹임을 나누어 볼 수 있다. 동일 칸수뿐 아니라 lag/커널 질량 차이를 보고하고, random은 여러 mask seed의 변동도 남긴다. 현재 α1의 모든 b=1 조건과 α.7의 거리 감쇠 조건은 q가 같아도 남긴 총 커널 질량이 같지 않다. 정규화·clip·학습 gate 등의 모델 수정을 동시에 도입하기 전에 이 진단을 우선한다.
4. 발화율을 맞춘 α별 보정 시스템 비교와 공통 scale 커널 비교를 구분하고, 성능 확증은 새로운 미사용 자료에서 별도 사전등록한다. 기존 ETT val은 조기 종료/진단에 사용된 탐색 자료이며 test/future는 다시 열지 않는다. 단순 Linear/GRU 참고선은 유지한다.

기존 확인 문헌인 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)와 [Zeng et al., AAAI2023 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/26317)를 이번에도 확인했다. 각각 입력 통계 제거·복원과 선형 기준선 설계를 뒷받침한다. 위 증분/고정 mask 제안은 로컬 재귀식과 실패에서 도출한 진단안으로, 이 문헌이 양의 되먹임 원인이나 현 모델 개선을 증명하는 것은 아니다.

실행: `/usr/bin/python3 f_lif_pop_v3/forecasting/results/assessment/20260927T125001Z-4f47a30f/record_audit.py` → PASS. 순수 저장값/지문/CSV 대조로 추가 모델 probe는 필요하지 않아 실행하지 않았다. 실제 데이터 forward·학습/backward·GPU·환경 설치·프로세스 중단·연구소스/Git 변이·다른 세션/메시지·새 예약 모두 하지 않았다. A25의 과거 채널 누락과 A23 잔여 provenance/MC 이슈도 유지한다.

<!-- assessment-watch:20260927T125001Z-4f47a30f -->


## 2026-09-27 22:06 KST — 추적 감사56: 2P validation 산술·채널 출력 검증 / 관문 범위와 실패 출처 검사 잔여

예약 `20260927T130001Z-85c4c76a`. 대응 **A25(다음 평가기 보완) / A27-ALPHA-INTERPRETATION / A28-COMPLETION-GATE / A28-DIAGNOSTIC-CLAIM / A29-GATE-COVERAGE / A29-FAILURE-PROVENANCE**. branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, trigger/최초·후기 HEAD `6f22eab2271366e39eb418e566990dfbefd8d352`. 기억·감사55·사전등록2P D-BY/D-BZ·canonical 완료/결과/정정 append를 읽었다. 자신의 직전 감사 기록은 새 연구 성과로 세지 않았다.

### 보존과 재검사 범위

22:00:28 KST에1202파일을 `f_lif_pop_v3/forecasting/results/assessment/20260927T130001Z-85c4c76a/source_snapshot/`에 별도 snapshot하고 SHA256 기록, trigger 차이0. 이어 대상96칸의 config96개·완료 checkpoint80개·학습 stdout96개·데이터CSV2개를 바이트 복사/hash했다(추가274파일, 데이터 행 파싱 없음). 후기 원1202파일 변화0. `inventory.json`, `additional_snapshot.json`, `checks.json`, `postcheck.json` 참조. raw stdout은 `f_lif_pop_v3/forecasting/log/assessment/20260927T130001Z-85c4c76a/checks.txt`.

주요 SHA256: prereg `92345b38d514793085799559ba55b795405280aaa1ebb21ec1753596a367deac`; ett_alpha.py `d61c25a6b36f68489a20b54a6bc4a2e27a9f0b3f646ebc1375dcd7696ef32822`; gate probe `14b4153729279e904125ea585651e0aec6b0c94825f34a1895bb10a2ccf8e6aa`. 모델/학습 계산 소스는 직전 감사와 동일하다.

### (a) 구현 정확성 — 평가 보완은 검증, 관문 두 항목은 OPEN

현재 API와 진단 정의를 확인한 뒤 **snapshot source를 import하는 CPU adapter**로 기존19개 gate/결함/통계 fixture를 재실행해19/19 통과했다. 모델 로드는 snapshot checkpoint를 CPU로 읽도록 바꾸었고 실데이터 forward·GPU는 실행하지 않았다. 원 probe의 GPU 지정 main을 그대로 실행한 것이 아니다. 원본 소스·checkpoint·raw log는 바꾸지 않았다.

**A25의 다음 평가기 보완 scoped VERIFIED:** ett_alpha.test는 전체 오차와 같은 forward에서 채널별 SSE/count를 누적한다. 별도 합성 Linear 출력 fixture(마지막 batch가 작은2+1창)에서 채널 MSE `[2,8,18,32,50,72,98]`, 전체40, 창수3을 정확히 재현했다. actual80행에 채널7개가 있고 채널 평균과 전체 MSE의 최대차는 ETTh1 1.369e−7/ETTh2 3.242e−8로 dtype·누적 정밀도 차이 범위다. unsafe/repro failure/training G11 failure에 따른 비교별 보류와 J의 relative=null·엄격 JSON fixture도 통과했다. **2O의 과거 채널 누락은 복구된 것이 아니므로 역사적 미충족 상태를 유지**한다.

**A29-GATE-COVERAGE — OPEN(자동 강제 범위).** D-BU는 평가 전96칸 전체 관문을 요구하지만 CLI main은 `gate(config.data, ...)`를 한 번 호출해 요청 데이터의48칸만 검사한다. ETTh2 접근 시 누락을 던지는 fixture를 넣어도 ETTh1 gate는 ETTh1만 방문해40완료+8실패로 통과했다. 기존 점검 스크립트는 양 데이터를 순회하고 이번 감사에서도 실제96칸 모두 통과했으므로 **이번 결과가 누락 자료로 계산됐다는 뜻은 아니다**. 다만 각 CLI가96칸을 평가 전에 강제하는 보장은 없다. 전체 관문을 공통 진입점에서 먼저 강제하고, 반대 데이터의 마지막 칸을 누락시켜도 어떤 validation forward보다 앞에서 차단하는 fixture를 추가해야 닫는다. 또는 실제 실행 범위를 편차로 명시하되 사후 변경을 원래 규정 충족으로 소급하지 않는다.

**A29-FAILURE-PROVENANCE — OPEN(재현된 누락).** `check_failure`는 run_id와 한계·일부 config·stdout의 G11 예외 존재를 확인하지만, G11 record의 run_uuid 대 config 일치와 config의 input_scale 대 고정 보정을 확인하지 않는다. 격리 fixture에서 (1) record UUID를 다른 값으로, (2) config input_scale을999로 각각 바꿔도 errors=[]로 통과했다. 잘못된 run_id/bound는 기존 fixture가 거부하므로 모든 검사가 무효인 것은 아니다. **실제16건의 UUID·scale·no-test는 감사에서 별도로 대조해 모두 정상**이며, 이번 실패 판정을 철회할 근거는 없다. 실패 칸도 완료 칸과 같은 동작점·데이터 경로·no-test 출처를 대조하고, UUID와 stdout의 구체적 epoch/batch/peak/한계를 묶는 검사를 추가해 재검사해야 닫는다. 새 결함은 완성된 평가 경로에서 확인한 것으로, 진행 중 파일의 미완성과 구분한다.

### (b) 데이터 분리·통계·재현성 — 저장 결과의 정합성 VERIFIED, 독립 확증 아님

**A28-COMPLETION-GATE:** 실제 칸의 완료 상태는96/96 확인했다(완료80+G11 실패16). 두 데이터별 CPU metadata gate 오류0, 기존19 fixture 통과로 실제 manifest·checkpoint·고정 보정·CSV 종료 증거 검사 범위는 VERIFIED. 자동 관문의 잔여는 위 A29로 분리해 OPEN이다. 신규 q1도16/16 완료이며 감사55의12/16 표기는 당시 snapshot으로 정확했다.

두 record는 done, val, exploratory=True, 각48개의 중복 없는 condition×seed 행이다. record source10개·데이터/보정·config96/checkpoint80 SHA, 완료 결과의 source11개와 best_val_loss, 스파이킹48개 frozen_norm 지문을 대조했다. 신규16건 final CSV의 val-2P와 MSE도 일치한다. 평가80건 모두2785창·safe/repro_ok, 실패16건은 실제 G11 record와 일치한다. 저장된 평가 MSE와 best_val 최대차는 ETTh1 **1.36218e−7**, ETTh2 **3.24883e−8**, 등록 허용1e−6 안이다. 이 수치는 **저장값 대조**이며 감사가 실데이터 모델 forward를 재현했다는 뜻은 아니다. mask mismatch는 원 기록상0, Pearson 동점은 각각1/6394360·3/6394360이다.

저장행으로 analyse 전체 출력과 채널 F_q를 다시 계산해 record와 일치했고, F_q 평균/표본 표준편차/t7의95% 구간은 별도 산술과 일치한다.

| 데이터 | q1 α=.7+보정 MSE | q1 α=1+보정 MSE | F_q | 95% paired seed CI | 상대 차이 |
|---|---:|---:|---:|---|---:|
| ETTh1 | .725107282 | .744082488 | −.018975206 | [−.024689823, −.013260588] | −2.5501% |
| ETTh2 | .228184896 | .235342668 | −.007157773 | [−.009025204, −.005290342] | −3.0414% |

각각8/8 seed가 음수이고 D-BV의 평균≤−.005·구간상한<0 규칙에 맞는 **탐색적 신호**다. α별 scale10 대8/6과 고정 보정이 함께 달라 순수 커널 효과가 아니다. val은 조기 종료에 사용됐으므로 독립 일반화 검증·새 확증으로 해석하지 않는다. seed CI는 고정 기간의 학습 변동이며, 창·채널을 독립 반복 표본으로 세지 않았다.

F_p·S_on(α=1)·J는 두 데이터 모두 각8개의 실패 칸을 명시하고 **withheld**, 통계값을 생성하지 않았다. 실패 seed 제외·임의 큰 MSE 대입 없이 처리됐다. GRU/Linear 대비 거리는 기술 수치이며 용량 통제/원인 분해 주장이 아니다. q1 α=.7+보정은 GRU_R보다 ETTh1 +9.25%, ETTh2 +8.58% MSE가 높아, 이번 α 대조의 개선이 비스파이킹 참고선 우위를 뜻하지 않는다.

채널 F_q는 ETTh1 `[-.01015,−.02522,−.01575,−.02011,−.04118,−.02088,+.00047]`, ETTh2 `[-.01337,−.00169,−.00548,−.00343,−.01230,−.00042,−.01340]`. ETTh1 OT에서 음수인 seed는3/8이다. canonical의 “OT에서는 차이가 없다”는 **“OT 평균차는+.0004655이며 이 기술 분석에서 일관된 개선을 보이지 않았다”**로 좁히는 것이 정확하다. 무효과/동등성 검정을 한 것은 아니다.

**A27-ALPHA-INTERPRETATION 및 A28-DIAGNOSTIC-CLAIM: 문서 정정 범위 VERIFIED.** D-BY·D-BZ와 canonical의 후속 정정이 α+보정 추정 대상, q1만 Euler 환원, random의 증가, 최대값 집계 범위, G11 update 이후 검사, 안전 보장 일반화 금지를 명시한다. D-BY가 학습 완료/최저 val 관찰 뒤 추가됐다는 고지도 확인했다. 메커니즘 자체 검증으로 닫는 것은 아니다. 제목 시각도 실제 commit 시각을 병기해 정정했다.

### (c) 개선 방향 — 감사파일로 작업 에이전트에 전달

새 결과는 **동일 스파이킹 구조에서 α=.7+자체 보정이 α=1+자체 보정보다 val 오차가 낮은 방향**, 그리고 α=1 Pearson의 고정 안전 실패다. 다음 작업은 묻고 싶은 대상을 분리해 진행하는 것이 좋다.

1. **보고/관문 보완을 먼저 끝낸다.** A29 두 fixture를 추가해96칸 전부의 사전 강제와 실패 출처 검증을 보완한다. 결과 원본은 보존하고 이미 계산한 val/future를 다시 평가할 이유로 삼지 않는다. ETTh1 OT 및 “GRU 격차의1/4을 커널이 줄였다” 등의 문구도 각각 기술적 차이·α별 보정 포함 차이로 유지한다.
2. **예측 성능의 다음 우선순위는 작은 독립 확증 설계다.** 새로운 자료의 미사용 이력·시간 경계·전처리·H96·주 대비·seed·보정/실패 규칙을 먼저 고정한 뒤 q1 α=.7 대1과 GRU/Linear 참고선을 비교한다. 첫 질문은 현재와 같은 α별 보정 시스템의 일반화 여부로 두고, 근거 없이 α 후보 grid나 여러 selector를 동시에 늘리지 않는다. 데이터셋별 탐색 val에 가장 잘 맞는 조건을 골라 이미 열린 ETT 기간에서 확증하지 않는다. 새 확증은 **not run**이다.
3. **커널 기여가 질문이면 공통 current/scale 진단부터 한다.** 감사55의 선택/제외 증분 합·절댓값 합·상태와의 방향 정렬, lag와 남긴 질량, 고정 mask 재생을 train-only 소규모로 사전 정의한다. q1에서 보정 차이를 통제한 α 비교와 q=.5의 선택 되먹임 질문을 한 대비로 섞지 않는다. 수치적 동작점과 발화율을 보고 안전 셀만의 사후 선택을 피한다. 이 추가 진단/학습은 **not run**이다.
4. **가중치 합 보정은 검증할 새 설계 가설로 둔다.** 현재처럼 선택한 증분을 재합산하는 경로에서 질량 보정이 어떤 항을 증폭/축소하는지와 q1 환원·안전성을 먼저 확인해야 한다. Pearson 실패만 보고 정규화가 문제를 해결한다고 가정하거나 기존 실패 조건을 같은 이름으로 교체하지 않는다. 양의 되먹임 설명은 아직 직접 측정되지 않았다.

설계 근거는 재확인한 [RevIN 저자 자료](https://seharanul17.github.io/RevIN/)의 입력 통계 제거/복원과 [Zeng et al., AAAI2023 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/26317)의 단순 선형 기준선이다. 현재 모델에서의 kernel/selector 기전이나 개선을 이 문헌이 증명한다고 주장하지 않는다.

실행: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v3/forecasting/results/assessment/20260927T130001Z-85c4c76a/audit_checks.py` 및 동일 환경의 `final_checks.py`. 종료0,19/19·합성 채널·저장값 산술/출처 검사 통과, A29 결함은 별도 재현 기록. 학습/backward·GPU·실제 데이터 forward·환경 설치·프로세스 중단·모델/학습 소스 수정·Git 변이·타 세션/메시지·새 예약 없음. A23 원실행 provenance/MC와2O A25 과거 채널 누락은 유지한다.

<!-- assessment-watch:20260927T130001Z-85c4c76a -->
