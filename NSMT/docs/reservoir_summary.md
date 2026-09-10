# NSMT Neo를 Stateful Spiking Reservoir로 교체하기 위한 조사·설계 문서

- 작성일: 2026-08-11 (Asia/Seoul)
- 조사 기준일: 2026-08-11
- 대상 저장소: `/home/yschoi/NSMT`
- 기준 구현: 최초 공개 코드 스냅샷 `79ec94c`와 실험 시작 직전 기준점 `191f366`
- 대상 논문: [NSMT manuscript](</home/yschoi/NSMT/papers/3724_Fast_Spikes_Slow_Trends_N (3).pdf>), [Zipser et al. 1993](/home/yschoi/NSMT/papers/A_Spiking_Network_Model_of_Short-Term_Active_Memory.pdf)
- 문서 목적: 다른 세션이 이 문서만 읽고도 현재 NSMT의 실제 동작, reservoir/SNN 연구 근거, Neo 교체 설계, 학습·평가 순서를 바로 이어서 구현할 수 있게 하는 것

> 핵심 결론: **고정 recurrent spiking reservoir만 넣는 것으로 “진짜 장기 메모리”가 생기지는 않는다.** 안정적인 reservoir computing은 원칙적으로 과거 영향이 감쇠하는 `fading memory`를 사용한다. NSMT에 필요한 것은 (1) 시계열별로 window 경계를 넘어 유지되는 reservoir state, (2) 지연 탭과 다중 시정수 또는 compact slow state, (3) reset 뒤에도 남아야 한다면 명시적 episodic store나 online readout weight, (4) 이 세 층을 검증하는 causal ablation이다.

**권장 읽기 순서:** 결론과 구현 우선순위는 0장, 이론과 최신 연구는 2–3장 및 부록 A, NSMT 논문·전체 코드 감사는 4–5장, 실제 교체 설계와 학습·평가는 7–17장, 다음 세션이 바로 실행할 결정 사항은 부록 B에 있다.

---

## 0. 이 문서에서 내리는 최종 판단

### 0.1 현재 NSMT에 대한 판단

1. 논문과 최초 코드의 Neo는 reservoir가 아니다. 각 look-back window를 받을 때마다 새로 계산되는 feed-forward spiking MLP이며, patch 순서 `N`을 인공적인 spiking time으로 사용한다.
2. 현재 이름이 `Memory Replay`인 모듈은 과거 샘플이나 상태를 재생하지 않는다. 같은 forward 안의 Hippo feature와 Neo feature를 곱하고 cross-attention/IAND로 융합하는 연산이다.
3. 모든 학습·평가 batch에서 `functional.reset_net(model)`을 호출한다. 최초 Neo의 membrane state는 다음 window로 넘어가지 않는다.
4. 현행 recurrent Neo 역시 forward 시작 시 자체 LIF를 reset하고 window 내부 recurrent loop를 surrogate-gradient BPTT로 학습한다. 이것도 cross-window memory가 아니다.
5. `forecasting/neo_bank.py`의 EMA만 batch 사이에 남지만, series/sample ID가 없는 전역 firing-rate prior다. shuffled batch와 채널을 모두 평균하므로 episodic memory로 해석할 수 없다.
6. DCT 저주파 복원은 current window 안의 느린 추세를 학습시키는 좋은 auxiliary bias일 수 있으나, window 밖 사건의 저장·회상을 증명하지 않는다.

### 0.2 권장 목표 모델

Neo를 다음 세 층으로 나눈다.

| 층 | 역할 | 지속 범위 | 학습 |
|---|---|---|---|
| Fast spiking liquid | 현재 입력의 비선형 temporal expansion | patch와 연속 window | recurrent/input weights 고정 |
| Slow state / delayed traces | reservoir의 짧은 fading horizon을 확장 | 여러 window | 고정 delay taps, heterogeneous traces 또는 고정 LMU/SSM |
| Durable memory | reset·긴 gap 뒤에도 선택된 episode/regime 보존 | episode/session/checkpoint | 명시적 write/read 정책, ridge/RLS 또는 별도 replay |

첫 구현은 `StatefulDelayedSpikingReservoir`로 시작한다. liquid의 spike뿐 아니라 membrane/current/smoothed rate와 지연 탭을 readout에 사용하고, `stream_id`별 state를 보존한다. “진짜 장기 메모리”라는 표현은 **state-carry·reset·shuffle·delayed-recall 대조를 통과한 뒤에만** 사용한다.

### 0.3 구현 우선순위

1. 최초 모델 parity를 별도 실험 디렉터리나 branch에서 재현한다.
2. Neo encoder만 고정 LSM adapter로 교체하고 gate/replay는 우선 제거한다.
3. chronological stream state와 split reset을 먼저 완성한다.
4. ridge readout으로 stateless/stateful/delayed LSM을 비교한다.
5. additive zero-init gate를 추가한다.
6. 마지막에만 episodic buffer와 online RLS/FORCE를 추가한다.

이 순서를 지켜야 어떤 성능 변화가 liquid, 상태 유지, delay, fusion, replay 중 무엇 때문인지 분리할 수 있다.

---

## 1. 조사 범위와 감사 방법

### 1.1 웹 문헌 조사 범위

검색은 2026-08-11 기준으로 다음 범주를 나눠 수행했다.

- reservoir computing, echo state property, fading memory, memory capacity의 원전·이론
- liquid state machine과 spiking reservoir의 대표 review
- 2022–2026 time-series forecasting/classification에 직접 적용된 spiking reservoir 연구
- 2025–2026 criticality, E/I balance, connectivity, delay-readout 연구
- pure reservoir의 장기기억 한계를 보완하는 external memory와 fast–slow spiking memory 연구
- 각 논문의 DOI/출판사 원문, 출판 이력, 공개 코드와 공개된 peer-review/indexing 정보

“최근 연구를 전부”라는 요구는 데이터베이스의 색인 지연, paywall, 2026년 preprint의 지속적 추가 때문에 수학적 완전성을 보장할 수 없다. 따라서 이 문서는 **제목·초록·키워드가 직접 일치하는 2022–2026 주요 archival 연구를 포괄하고, NSMT 설계에 직접 영향을 주는 인접 연구를 추가한 실무적 exhaustive review**다. 낮은 신뢰도의 일반 웹 요약은 근거로 쓰지 않았다.

### 1.2 출판 신뢰도 평가 원칙

출판사의 명성은 논문 결과의 진실성을 보증하지 않는다. 다음 두 점수를 분리한다.

**Venue 신뢰도**

- `A`: 오래된 학회/출판사의 엄격한 archival journal 또는 대표 학회. 예: MIT Press `Neural Computation`, IEEE TNNLS/IJCNN, Elsevier `Neural Networks`·`Neurocomputing`, Nature Machine Intelligence.
- `B`: 정상적인 동료평가와 색인이 확인되지만 비교적 젊거나 전문 범위가 좁은 journal. 예: Wiley `Advanced Intelligent Systems`, `Scientific Reports`, `Frontiers in Computational Neuroscience`, MDPI `Entropy`.
- `C`: 짧거나 전문 범위가 좁은 동료평가 conference/book chapter/selected paper. 아이디어 탐색에는 유용하지만 증거량이 작다.
- `D`: arXiv/preprint 또는 출판 절차가 불명확한 매체. 가설 생성에만 사용한다.

**개별 결과의 증거 강도**

- `높음`: 문제와 비교군이 직접적이고, 충분한 데이터·seed·ablation·코드·독립 검증이 있다.
- `중간`: 정상 동료평가이나 데이터/benchmark/재현 범위가 제한적이거나 매우 신작이다.
- `낮음`: 짧은 논문, 단일 synthetic task, 불완전한 코드, preprint, 독립 재현 부재 중 여러 조건에 해당한다.

예를 들어 Nature 계열 venue라도 reservoir가 아닌 BPTT 모델이면 “BPTT-free Neo”의 직접 근거가 아니다. 반대로 전문 venue의 제한된 논문도 구체적인 delay feature 설계에는 유용할 수 있다.

### 1.3 저장소 전체 Python 감사

저장소는 사용자 변경과 미추적 실험 snapshot이 많은 dirty worktree다. 기존 파일은 수정하지 않았다.

- 파일시스템의 `.py`: **728개**
- Git 추적 `.py`: **472개**
- SHA-256 내용 해시 기준 고유 구현: **222개**
- exact duplicate: **506개**
- 모든 728개: `ast.parse` 성공
- `AGENTS.md`: 없음

모든 파일을 “728개를 각각 다른 구현처럼” 세지 않고 다음 방식으로 감사했다.

1. 전체 경로를 수집하고 디렉터리별 수를 대조했다.
2. 내용 해시로 exact clone을 묶었다.
3. 각 task의 표준 파일 역할을 읽었다.
4. 해시가 다른 고유 구현 222개와 실험 전용 모듈을 중심으로 Neo, state, reset, replay, gate, loss, gradient 흐름을 추적했다.
5. 최초 구현 커밋과 현행 root를 비교했다.
6. 과거 `docs/01`–`docs/46`의 실험 결론을 코드와 대조했다.

따라서 아래 저장소 인벤토리는 728개를 누락 없이 디렉터리·동일 역할·고유 실험군으로 포괄한다.

---

## 2. Reservoir computing을 정확히 이해하기

### 2.1 정의

Reservoir computing(RC)은 recurrent dynamical system을 비선형 temporal feature map으로 사용하고, 보통 recurrent core는 고정한 채 readout만 학습하는 패러다임이다. Echo State Network(ESN)와 Liquid State Machine(LSM)이 대표적인 두 계열이다. 이 정의와 고전적 학습 분리는 [Lukoševičius & Jaeger, 2009](https://doi.org/10.1016/j.cosrev.2009.03.005)에 정리되어 있다.

일반적인 discrete-time 표현은 다음과 같다.

\[
r_t = F(W_{\mathrm{in}}u_t + W_{\mathrm{res}}r_{t-1} + b),
\qquad
\hat y_t = W_{\mathrm{out}}\phi(r_t,u_t).
\]

- \(u_t\): 입력
- \(r_t\): reservoir state
- \(W_{\mathrm{in}}, W_{\mathrm{res}}\): 보통 random sparse로 생성하고 고정
- \(\phi\): bias, input, membrane, filtered spike, delay tap 등을 합친 readout feature
- \(W_{\mathrm{out}}\): ridge regression, RLS/FORCE 등으로 학습

핵심은 “모든 reservoir가 무작위여야 한다”가 아니라 **동역학과 readout 학습을 분리한다**는 것이다. topology/scale/tau를 validation으로 고르거나 local plasticity로 precondition할 수 있지만, recurrent core를 task loss로 BPTT 학습하면 strict reservoir computing에서 멀어진다.

### 2.2 ESN과 LSM의 차이

| 항목 | ESN | LSM / spiking reservoir |
|---|---|---|
| state unit | tanh/leaky rate unit 등 연속값 | LIF/ALIF/Izhikevich 등 spike neuron |
| 내부 상태 | activation | membrane, synaptic current, refractory state, spike trace |
| readout 입력 | state 자체 | spike count/rate, filtered spike, membrane, delayed state |
| 시간 표현 | 입력 sample마다 한 update가 일반적 | biological simulation step과 데이터 sample/patch time을 구분해야 함 |
| 장점 | 구현·분석·ridge가 간단 | event-driven hardware 가능성, 풍부한 temporal dynamics |
| 위험 | spectral radius만 맹신 | silent/saturated liquid, spike noise, E/I 불안정, 축 혼동 |

LSM 원전은 [Maass, Natschläger & Markram, 2002](https://direct.mit.edu/neco/article/14/11/2531/6659/Real-Time-Computing-Without-Stable-States-A-New)이다. 고정 random synapse를 쓰는 spiking reservoir의 설계 지표와 변형은 [Soures & Kudithipudi, 2019, IEEE Signal Processing Magazine](https://doi.org/10.1109/MSP.2019.2931479)에 정리되어 있다.

### 2.3 LSM이 계산하는 것

좋은 liquid에는 상충하는 두 성질이 필요하다.

1. **Separation / kernel quality**: 서로 다른 입력 history가 구분되는 reservoir state를 만든다.
2. **Generalization / consistency**: 비슷한 입력과 작은 noise가 지나치게 다른 state로 폭발하지 않는다.

입력 history를 고차원 state로 펼친 뒤 쉬운 readout이 필요한 정보를 선형 분리한다. recurrent feedback, neuron nonlinearity, heterogeneous time constant, E/I interaction이 history-dependent basis를 만든다.

spiking liquid의 한 예는 다음과 같다.

\[
\begin{aligned}
i_n &= \alpha_i i_{n-1} + W_{\mathrm{in}}e_n + W_{\mathrm{rec}}s_{n-1},\\
u_n &= \alpha_u u_{n-1} + i_n - v_{\mathrm{th}}s_{n-1},\\
s_n &= H(u_n-v_{\mathrm{th}}),\\
q_n &= \alpha_q q_{n-1} + (1-\alpha_q)s_n .
\end{aligned}
\]

여기서 `q`는 causal filtered firing rate다. binary spike만 readout에 쓰면 sparse한 정보가 손실될 수 있으므로 membrane `u`, synaptic trace `i`, 여러 decay의 `q`, delayed state를 함께 후보로 둔다.

### 2.4 Echo state property와 fading memory

RC가 같은 입력에 대해 초기 조건과 무관하게 일관된 출력을 내려면 초기 상태의 영향이 사라져야 한다. 이것이 echo state/consistency와 연결된다. [Carroll, 2022](https://doi.org/10.1063/5.0078151)는 계산 일관성을 위해 memory가 fading해야 하며, 너무 길거나 너무 짧은 memory 모두 성능을 떨어뜨릴 수 있음을 설명한다.

[Grigoryeva & Ortega, 2018](https://doi.org/10.1016/j.neunet.2018.08.025)의 universality 결과도 임의의 영구 메모리가 아니라 **fading-memory filter**에 대한 보편 근사다.

따라서 다음 문장은 중요하다.

> 안정적인 pure reservoir의 강점은 “최근 과거를 유용하게 감쇠·변환하는 것”이지, 임의의 과거 사건을 무기한 무손실 저장하는 것이 아니다.

### 2.5 세 종류의 “장기기억”을 구분해야 한다

| 이름 | 정의 | pure LSM으로 가능한가 |
|---|---|---|
| within-window context | 한 look-back 안에서 앞 patch가 뒤 patch state에 영향 | 가능 |
| cross-window fading state | 연속 window 사이에 state를 넘기고 영향이 점차 감쇠 | 가능, state lifecycle 필요 |
| durable episodic/parametric memory | 긴 gap/reset 뒤에도 선택 사건 또는 규칙을 회상 | pure LSM만으로는 불충분 |

사용자가 원하는 “진짜 장기 메모리”가 세 번째까지 포함한다면 explicit slow/external state가 필요하다. [Reservoir Memory Machines, IEEE TNNLS 2022](https://doi.org/10.1109/TNNLS.2021.3094139)는 contractive ESN에 interference-free explicit memory를 붙이면 순수 ESN이 할 수 없는 regular-language 문제까지 다룰 수 있음을 보였다. spiking 전용 논문은 아니지만 구조적 결론은 직접적이다.

### 2.6 Memory capacity

Jaeger의 linear memory capacity는 delay \(k\) 전 입력을 linear readout으로 얼마나 복원하는지 측정한다.

\[
\mathrm{MC}_k =
\frac{\operatorname{cov}(u_{t-k},\hat u_{t-k})^2}
{\operatorname{var}(u_t)\operatorname{var}(\hat u_{t-k})},
\qquad
\mathrm{MC}=\sum_k \mathrm{MC}_k.
\]

고전적 조건에서 linear reservoir의 total MC는 reservoir dimension \(N\)으로 제한된다. 원 보고서는 [Jaeger, “Short Term Memory in Echo State Networks”](https://publica.fraunhofer.de/entities/publication/9dfaead1-4dc0-4e3c-b89b-596f50f671c1)다.

NSMT에서는 total scalar 하나만 보지 말고 다음을 모두 기록해야 한다.

- `MC_k` 대 delay 곡선
- memory half-life
- nonlinear capacity/NARMA
- input separation과 generalization rank
- firing rate, silent neuron 비율, saturation
- state perturbation 뒤 수렴
- stateful와 reset/shuffle 성능 차이

### 2.7 “RC는 BPTT가 필요 없다”의 정확한 의미

맞는 부분:

- `W_in`, `W_res`, neuron dynamics를 고정하면 recurrent temporal graph를 역전파할 필요가 없다.
- state를 한 번 causal forward로 수집하고 ridge로 readout을 닫힌 형태로 구할 수 있다.
- online RLS/FORCE는 현재 feature와 error로 readout을 갱신한다.

주의할 부분:

- reservoir 뒤에 trainable temporal Transformer/Hippo를 두고 end-to-end 학습하면 **Neo reservoir만 BPTT-free**이고 전체 NSMT는 BPTT-free가 아니다.
- gate/readout을 gradient로 학습하는 것은 가능하지만 reservoir state를 `detach`해야 시간축 gradient가 liquid로 들어가지 않는다.
- recurrent weights나 tau를 surrogate gradient로 task 학습하면 그 부분은 더 이상 strict BPTT-free reservoir가 아니다.
- FORCE 문헌 중 recurrent feedback까지 적응시키는 변형도 있으므로 “FORCE=항상 readout-only”라고 일반화하면 안 된다. 본 설계는 readout-only RLS를 기본으로 한다.

---

## 3. 최근 spiking reservoir·time-series 연구 지도

### 3.1 직접 관련 논문 요약표

| 연도 | 연구 | task/핵심 | NSMT에 주는 것 | Venue | 결과 증거 |
|---|---|---|---|---|---|
| 2022 | [Dey et al., Efficient TSC using Spiking Reservoir](https://doi.org/10.1109/IJCNN55064.2022.9892728) | temporal Gaussian/Poisson encoding, LSM, TSC | encoding 대조와 분류 baseline | IEEE IJCNN, A- | 중간 |
| 2022 | [Gaurav et al., Spiking RC on Loihi](https://doi.org/10.1109/SEC54971.2022.00081) | LMU/LDN 기반 ECG5000, Loihi | hardware와 state reset 정책 | IEEE/ACM SEC, C+ | 중간 |
| 2023 | [George et al., Online forecasting](https://doi.org/10.1016/j.neucom.2022.10.067) | temporal encoding + feedback LSM + FORCE/RLS, 9 series | online readout와 concept drift | Neurocomputing/Elsevier, A- | 중간 |
| 2023 | [Gaurav et al., SLRC/LSNN](https://doi.org/10.3389/fncom.2023.1148284) | Legendre delay state, 5 univariate TSC | compact slow state, inhibition/reset | Frontiers FCN, B | 중간 |
| 2024 | [Peng et al., RC based on spiking neural P systems](https://doi.org/10.1016/j.neunet.2023.10.041) | 17 TSC datasets, NSNP membrane computing | 넓은 TSC 결과, 단 LIF/LSM과 다름 | Neural Networks/Elsevier, A | 중간 |
| 2024 | [Kostyukov & Rostov, Spiking Reservoir NN for TSC](https://doi.org/10.1007/978-3-031-52470-7_25) | spike encoding + supervised feature layer | 보조 비교 | Springer CCIS chapter, C | 낮음 |
| 2025 | [Freddi et al., Mean-field criticality](https://doi.org/10.1038/s41598-025-18004-y) | LIF reservoir critical point 근사와 공개 코드 | 초기 weight/criticality 설정 | Scientific Reports, B+ | 중간 |
| 2026 | [Oh et al., Optimizing reservoir connectivity](https://doi.org/10.1016/j.neucom.2025.132037) | spike multiplication factor와 controlled chaos | spectral radius 대신 spike propagation 진단 | Neurocomputing, A- | 중간 |
| 2026 | [Jin et al., Time-Delayed SRC](https://doi.org/10.1002/aisy.70421) | lagged firing-rate stack + ridge, chaotic forecasting | 가장 직접적인 Neo MVP | Wiley AIS, B+ | 중간-낮음 |
| 2026 | [Rosati et al., Entropy/Inhibition/Memory](https://doi.org/10.3390/e28070784) | balanced E/I LIF phase diagram, MC와 separation | sparse input, E/I/asynchronous regime | Entropy/MDPI, B | 중간-낮음 |
| 2026 | [Goto et al., E/I layer spiking reservoir](https://doi.org/10.2299/jsp.30.97) | Lorenz, FORCE/RLS, autonomous stability | closed-loop E/I 안정성 | Journal of Signal Processing short paper, C | 낮음 |
| 2026 | [Freddi et al., Robust Spiking Reservoirs](https://arxiv.org/abs/2604.06395) | robustness interval | hyperparameter plateau 진단 | arXiv/preprint, D | 낮음 |

`A-`, `B+`는 같은 등급 안에서의 상대적 판단일 뿐 정량 점수가 아니다.

### 3.2 기반·인접 연구

| 연구 | 직접성 | 채택할 교훈 |
|---|---|---|
| [Maass et al., 2002 LSM](https://doi.org/10.1162/089976602760407955) | foundational | separation + fading response + readout이라는 원형 |
| [Lukoševičius & Jaeger, 2009 RC review](https://doi.org/10.1016/j.cosrev.2009.03.005) | foundational | 고정 reservoir와 readout 학습의 분리 |
| [Soures & Kudithipudi, 2019](https://doi.org/10.1109/MSP.2019.2931479) | spiking reservoir review | topology, plasticity, hierarchy, neuromorphic 관점 |
| [Carroll, 2022](https://doi.org/10.1063/5.0078151) | memory theory | memory는 길수록 좋은 것이 아니라 task timescale에 맞아야 함 |
| [Paassen et al., 2022 RMM](https://doi.org/10.1109/TNNLS.2021.3094139) | non-spiking external memory | durable memory는 reservoir 밖 명시적 store가 더 정직함 |
| [Sun et al., 2026 Dual Memory Pathways](https://doi.org/10.1038/s42256-026-01255-3) | spiking, reservoir 아님 | fast spikes + compact slow state라는 구조 |

### 3.3 가장 직접적인 연구 1: Time-Delayed Spiking Reservoir Computing

[Jin et al., 2026](https://doi.org/10.1002/aisy.70421)은 causal smoothing한 firing-rate state를 여러 lag에서 꺼내 이어 붙인다.

\[
r_{\mathrm{td}}[t] =
[r[t],r[t-\tau],r[t-2\tau],\ldots,r[t-(K-1)\tau]].
\]

reservoir를 키우지 않고 readout feature dimension을 키우며 ridge regression을 사용한다.

\[
W_{\mathrm{out}}
=Y R_{\mathrm{td}}^\top
\left(R_{\mathrm{td}}R_{\mathrm{td}}^\top+\lambda I\right)^{-1}.
\]

논문은 pulse function, Lorenz, Mackey–Glass에 적용하고 Lorenz 설정에서 baseline 400 neuron 대비 80 neuron으로도 더 나은 closed-loop prediction을 보고한다. 즉 neuron을 80% 줄이고 delay feature로 보상했다.

**NSMT 적용 판단**

- 채택: delayed taps, causal filtered rate, ridge, delay 수와 liquid 크기의 trade-off.
- 추가: membrane과 synaptic trace도 feature에 포함한다.
- 주의: delay stack은 유한 ring buffer이므로 무한 장기기억이 아니다.
- 재현성: [공개 GitHub](https://github.com/Pin-Jin/time-delayed-spiking-reservoir-computing)는 2026-08-11 현재 2 commits, `README.md`, `config.yaml`, `main.py`, `scans/` 중심이고 별도 license 파일이 보이지 않는다. 공개 `main.py`는 Izhikevich reservoir, Gaussian firing-rate smoothing, tapped delay, ridge를 구현하지만 논문의 모든 benchmark pipeline을 완전히 제공하는 형태는 아니다. 따라서 아이디어 재현성은 부분적이다.
- 신뢰도: Wiley의 peer-reviewed journal이고 received/revised/accepted 이력이 있지만 2026년 5월 출판된 매우 신작으로 독립 재현이 없다. **설계 후보로 강하고 성능 보증으로는 중간 이하**다.

### 3.4 가장 직접적인 연구 2: Online forecasting + FORCE/RLS

[George et al., 2023](https://doi.org/10.1016/j.neucom.2022.10.067)은 temporal spike encoder, sparse feedback-enhanced LSM, online FORCE learning을 결합했다. 9개 time series에서 SARIMA, online ARIMA, stacked LSTM과 비교하여 평균 최대 8% 높은 \(R^2\), stacked LSTM보다 약 100배 적은 trainable parameter와 \(10^5\)배 적은 update를 보고했다.

**NSMT 적용 판단**

- 채택: non-stationary stream에는 offline ridge로 초기화한 뒤 readout-only RLS가 적합하다.
- 채택: autonomous/closed-loop forecasting은 teacher-forced 성능과 별도로 검증한다.
- 주의: 이 논문의 “rapidly fading memory 회피”는 feedback과 online adaptation의 조합이지 영구 episodic memory의 증명이 아니다.
- 주의: 비교 baseline과 protocol이 현대 long-horizon Transformer/SNN 기준과 완전히 같지 않다. 주장 숫자를 NSMT 성능 기대치로 옮기지 않는다.
- 신뢰도: `Neurocomputing`의 archival research article로 venue는 강하지만 단일 연구이고 코드/seed/protocol 재현성을 별도 확인해야 하므로 결과 증거는 중간이다.

### 3.5 Legendre slow state: SLRC와 LSNN

[Gaurav, Stewart & Yi, 2023](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1148284/full)은 Legendre Delay Network(LDN)/LMU에서 영감을 받은 두 모델을 제안했다.

- `SLRC`: LDN을 spiking ensembles로 근사하고 readout 연결만 L2 least squares로 학습한다.
- `LSNN`: LDN state를 정적 선형 연산으로 정확히 계산한 뒤 spiking nonlinear decoder를 surrogate gradient로 학습한다.
- ECG5000, Ford-A/B, Wafer, Earthquakes의 univariate binary classification을 사용했다.
- LSNN은 비교 LSM의 2,500–5,000 neuron보다 40배 이상 적은 50–120 spiking neuron을 사용했다고 보고한다.
- 독립 sample 사이에는 reservoir inhibition/reset이 유리하지만, 연속 stream에 reset을 넣으면 해롭다고 논문이 명시한다.

**NSMT 적용 판단**

- 가장 중요한 교훈은 “spikes만으로 모든 slow memory를 만들 필요가 없다”는 점이다. 고정된 저차원 slow state가 더 효율적일 수 있다.
- 초기 MVP는 pure random LSM으로 두되, long-horizon에서 부족하면 fixed LMU/SSM slow path를 병렬로 붙인다.
- LSNN은 hidden connections를 surrogate gradient로 학습하므로 BPTT-free 근거로 인용하면 안 된다.
- Frontiers의 [journal about page](https://www.frontiersin.org/journals/computational-neuroscience/about)는 PubMed/Scopus/SCIE/DOAJ 색인과 single-anonymized peer review를 명시한다. 정상 동료평가 journal이지만 5개 univariate binary task와 일부 off-chip training이라는 제한 때문에 결과 강도는 중간이다.

### 3.6 Criticality, connectivity, E/I balance

#### Mean-field criticality

[Freddi et al., 2025](https://www.nature.com/articles/s41598-025-18004-y)은 일반 topology의 LIF reservoir에 대해 평균 membrane dynamics의 critical point를 근사하고 Python code를 공개했다. 100 neuron 규모에서도 critical behavior를 재현했다고 보고한다.

채택할 것은 “criticality를 초기 탐색점으로 계산한다”는 원칙이다. 최적점을 critical point 하나로 고정하지 않는다.

#### Spike multiplication factor

[Oh et al., 2026](https://doi.org/10.1016/j.neucom.2025.132037)은 spike가 successive propagation에서 얼마나 증식하는지 나타내는 spike multiplication factor \(\lambda\)를 제시하고, controlled chaos와 성능의 관계를 분석했다. spiking reservoir에서는 rate-RNN의 spectral radius 하나만으로 안정성을 판단하기 어렵다는 근거다.

NSMT에서는 다음을 함께 기록한다.

- input spike당 downstream spike 수
- largest Lyapunov proxy 또는 두 초기상태 간 거리
- silent/saturated neuron 비율
- E/I별 firing rate
- perturbation recovery time

#### Balanced E/I and sparse input

[Rosati et al., 2026](https://www.mdpi.com/1099-4300/28/7/784)은 5,000 LIF neuron Brunel E/I reservoir의 phase diagram에서 separation, corrected linear MC, mutual information을 비교했다. inhibition이 커질수록 separation/MC가 개선되고 asynchronous-irregular regime에서 둘이 함께 높았으며, 긴 timescale만 보인 synchronous regime가 반드시 좋은 계산 능력을 뜻하지는 않았다. dense input coupling은 population state를 함께 몰아 memory를 붕괴시키므로 sparse input coupling이 중요하다고 보고한다.

이 논문은 2026년 7월 출판된 특집호 연구이고 downstream time-series 성능을 직접 보이지 않았다. MDPI `Entropy`의 동료평가 논문이지만 매우 신작·무인용에 가까우므로 **초기화·진단 근거**로만 쓴다.

#### Robustness interval

[Freddi et al., 2026 preprint](https://arxiv.org/abs/2604.06395)은 한 점의 최적 criticality 대신 일정 성능 이상을 유지하는 hyperparameter interval을 제안한다. 아직 preprint이므로 최종 근거가 아니라 다음 실험 규칙으로만 채택한다.

> 하나의 seed에서 최고인 weight scale보다 여러 seed·dataset에서 안정적인 plateau를 선택한다.

### 3.7 Dual Memory Pathways: 직접 RC는 아니지만 중요한 설계 근거

[Sun et al., Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-026-01255-3)은 빠른 spike pathway와 layer-shared compact slow state를 병렬로 둔다.

\[
m_k=\bar A m_{k-1}+\bar Bx_k,\qquad
u_k=\beta u_{k-1}+I_k+W_m m_k.
\]

slow state dimension은 hidden width의 작은 비율로 두고, 긴 programmable delay 부담을 줄였다. 논문은 long-sequence benchmark와 post-layout hardware에서 parameter/throughput/energy 이점을 보고한다. Nature Machine Intelligence의 [editorial process](https://www.nature.com/natmachintell/submission-guidelines/editorial-process)는 editor screening과 외부 review 단계를 공개한다.

그러나 이 모델은 SpikingJelly/SLAYER 계열 end-to-end gradient/BPTT를 사용한다. 따라서 다음처럼 사용한다.

- 채택: `fast spikes + compact fixed slow state`의 구조적 아이디어.
- 미채택: 논문의 gradient 결과를 RC의 BPTT-free 증거로 쓰는 것.
- 구현: \(\bar A,\bar B\)를 고정한 LMU/SSM으로 두고 readout만 학습하면 NSMT의 RC 목표와 양립한다.

### 3.8 분류 논문과 인접 연구의 해석 주의

- [Peng et al., 2024](https://pubmed.ncbi.nlm.nih.gov/37918270/)의 “spiking neural P systems”는 membrane-computing rule system이다. conventional LIF/LSM과 같은 물리·수식으로 취급하지 않는다.
- [Kostyukov & Rostov, 2024](https://doi.org/10.1007/978-3-031-52470-7_25)은 7쪽 CCIS chapter로 상세 근거가 작다.
- [Goto et al., 2026](https://www.jstage.jst.go.jp/article/jsp/30/4/30_97/_pdf/-char/en)은 4쪽 selected paper와 Lorenz 중심이다. E/I가 autonomous stability에 중요하다는 보조 근거로만 쓴다.
- neuromorphic energy 효율은 CPU/GPU simulation 속도에서 자동으로 따라오지 않는다. 실제 Loihi/ASIC 측정이나 최소한 AC/MAC/SOP, memory traffic, firing rate가 필요하다.
- 출판 과정이 불명확한 2026 일반 review나 저신뢰 journal은 본 설계의 근거에서 제외했다.

---

## 4. `papers`의 NSMT 논문과 1993 active-memory 논문 분석

### 4.1 NSMT PDF의 출판 상태

`Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer for Time-Series Analysis`는 18쪽, Submission ID 3724의 anonymous manuscript다.

- ACM `acmart` 형식을 사용하지만 conference 이름이 `XX`, DOI가 placeholder다.
- 저자·venue·최종 DOI가 없으므로 **ACM 출판 완료 논문으로 간주하면 안 된다.**
- 이 문서에서는 “NSMT manuscript/draft” 또는 “저장소 논문”이라고 부른다.

형식이 아니라 수식·코드 일치 여부를 근거로 분석한다.

### 4.2 논문의 입력과 두 pathway

입력:

\[
X_{\mathrm{in}}\in\mathbb{R}^{L\times C}.
\]

patching 후 논문은 spiking input을 \(X\in\mathbb{R}^{T\times N\times P}\)로 둔다.

- \(T\): SNN simulation time
- \(N\): patch/token 수
- \(P\): patch 길이

Hippo pathway는 current window의 local patch feature를 만든다.

\[
X^h=\mathrm{SpikingMLP}(X).
\]

Neo pathway는 원래 `T` 축을 평균하고 `N`을 새 spiking time으로 정렬한다. 즉 논문의 핵심 장치가 \(T=N\) alignment다.

\[
X^n = \mathrm{SN}(\mathrm{BN}(\mathrm{Linear}(X))).
\]

여러 spiking MLP layer를 통과한 Neo representation을 DCT 저주파 target으로 학습한다.

\[
\hat X_{\mathrm{low}}
=\mathrm{IDCT}(\mathrm{Mask}_k(\mathrm{DCT}(X_{\mathrm{in}}))),
\]

\[
\mathcal L_{\mathrm{aux}}
=\left\|\mathrm{Decoder}(X_{\mathrm{neo}})-\hat X_{\mathrm{low}}\right\|_2^2.
\]

decoder는 단일 FC이고 inference에서 제거할 수 있다.

### 4.3 논문의 “Memory Replay” 수식

논문은 같은 window의 두 feature를 곱한다.

\[
X^f=X_{\mathrm{neo}}^\top\odot X^h.
\]

Hippo를 query로, \(X^f\)를 key/value로 attention을 만든다.

\[
Q=\mathrm{SN}(\mathrm{BN}(X^hW_Q)),\quad
K=\mathrm{SN}(\mathrm{BN}(X^fW_K)),\quad
V=\mathrm{SN}(\mathrm{BN}(X^fW_V)),
\]

\[
A=(QK^\top)s,\qquad
\mathrm{MemoryReplay}(X^h,X_{\mathrm{neo}})
=\mathrm{Proj}(\mathrm{SN}(A V)).
\]

IAND residual은 다음 형태다.

\[
\mathrm{IAND}(A[t],S[t])=(1-A[t])S[t].
\]

이 연산에는 과거 sample buffer, timestamp, state cache, write, load, retrieval address가 없다. 따라서 정확한 이름은 `WithinWindowNeoFusion` 또는 `ReservoirFusion`이다.

### 4.4 논문의 gradient와 loss

\[
\mathcal L=(1-\alpha)\mathcal L_{\mathrm{task}}
+\alpha\mathcal L_{\mathrm{aux}}.
\]

task gradient는 head → Hippo/Replay → product → Neo로 들어가고 auxiliary gradient는 decoder → Neo로 들어간다. 논문 구조는 surrogate-gradient end-to-end 학습이며 reservoir readout-only 방식이 아니다.

task별 loss:

- forecasting: MAE
- anomaly detection: reconstruction MSE
- UEA classification: label-smoothed cross entropy
- MIT-BIH: CE + class-aware loss

논문의 ablation은 full과 `w/o Neo & Replay`, `w/o low-frequency reconstruction`을 비교하지만 변화가 작고 요소를 완전히 직교 분리하지 않는다. 이 결과만으로 long-term memory를 입증할 수 없다.

논문의 energy 수치는 45 nm AC/MAC 단가를 곱한 theoretical proxy이며 실제 neuromorphic chip 측정이 아니다. 단위와 포함 연산 범위를 재검산하고, `T=N` 제약·depth/hyperparameter sensitivity·실제 chip 미검증이라는 논문 자체의 제한을 새 설계에서 명시해야 한다.

### 4.5 논문 Neo가 장기기억이 아닌 이유

1. current look-back window만 입력한다.
2. state가 다음 window로 전달된다는 수식·알고리즘이 없다.
3. replay buffer가 없다.
4. DCT target 역시 같은 window에서 계산한다.
5. learned weights에 dataset-level slow-trend bias가 남을 수는 있지만 이것은 parametric prior다.
6. delayed cue, reset, state shuffle, cross-window recall 실험이 없다.

가장 정확한 표현:

> 현재 Neo는 **current-window low-frequency reconstruction으로 훈련된 per-window spiking encoder이며, 장기 정보는 state가 아니라 weight에 들어간 parametric/global inductive bias**다.

### 4.6 1993년 short-term active-memory 논문

`A Spiking Network Model of Short-Term Active Memory`는 Zipser et al., *Journal of Neuroscience* 13(8), 1993의 15쪽 논문이다.

핵심:

- fixed recurrent/re-entrant connections와 persistent firing으로 10–20초 active short-term memory를 모델링한다.
- 원 continuous network는 BPTT/optimization으로 weight를 찾은 뒤 고정한다.
- `Info-in`, 명시적 binary `Load-in`, `Memory-out`이 있다.
- spiking conversion은 현대적인 deterministic LIF 하나가 아니라 continuous unit을 probabilistic binary neuron pool로 바꾼 모델이다.
- positive/negative storage unit과 recurrent excitation/inhibition으로 state를 유지한다.
- graded memory는 장시간 뒤 소수 fixed-point attractor로 무너질 수 있고, spike noise가 attractor 사이 jump를 일으킨다.

NSMT에 주는 직접 교훈:

1. write/load gate가 없는 recurrent activity를 “memory storage”라고 부르지 않는다.
2. read gate와 write gate를 분리한다.
3. state를 유지하는 것과 새로운 정보를 덮어쓰는 것을 구분한다.
4. attractor collapse와 noise-induced jump를 반드시 진단한다.
5. 이 논문도 **short-term active memory**이지 무제한 long-term episodic memory가 아니다.

---

## 5. 저장소 전체 Python 코드 감사

### 5.1 디렉터리별 728개 인벤토리

| 디렉터리 | `.py` 수 | 주 역할 |
|---|---:|---|
| `acim_task_general_v1` | 61 | ACIM의 forecast/AD/classification task-general snapshot |
| `ad_recon_gradient_diag_v1` | 2 | AD reconstruction gradient conflict 진단 |
| `anomaly_detection` | 22 | 현행 anomaly detection |
| `archive` | 74 | 폐기·비교·exploratory variant |
| `classification` | 33 | UEA classification 18 + ECG 15 |
| `ecg_valfix_v1` | 15 | ECG validation fix snapshot |
| `forecasting` | 22 | 현행 forecasting |
| `forecasting_hankel_v1` | 22 | causal bi-axial Hankel variant |
| `forecasting_hankel_validation_v2` | 27 | Hankel validation/diagnostic 확장 |
| `initfix_pe_v1` | 24 | initialization fix + TNPE/xvar |
| `initfix_v1` | 22 | name-based local RNG initialization fix |
| `ipgm_task_general_v1` | 62 | IPGM task-general snapshot |
| `neo_grad_rebalance_v1` | 23 | Neo gradient rebalance |
| `neorecall_ad_v1` | 22 | anomaly-detection Neo recall |
| `neorecall_v1` | 23 | forecasting Neo recall + wavelet encoder |
| `rcf_nsmt_v1` | 61 | RCF task-general snapshot |
| `rel_pe_logpe_v1` | 59 | relative/log position encoding |
| root `scripts_dump_best.py` | 1 | best result directory parser |
| `seasonal_neo_v1` | 23 | phase-indexed seasonal Neo |
| `shape_delta_v1` | 22 | patch-delta injection |
| `wavelet2_v1` | 23 | wavelet Neo stream |
| `wavelet3_v1` | 23 | wavelet dual stream |
| `wavelet_ad_v1` | 23 | AD wavelet |
| `wavelet_cls_v1` | 16 | ECG wavelet |
| `wavelet_v1` | 23 | base wavelet variant |
| **합계** | **728** | SHA-256 고유 내용 222개 |

### 5.2 반복되는 표준 파일의 역할

이 매핑은 각 snapshot의 같은 이름 파일을 모두 포괄한다.

| 파일군 | 역할 |
|---|---|
| `config.py` | argparse/config, seed, device, run directory, config save/load |
| `model.py`, `load_model.py` | timm registry, topology 선택, checkpoint load |
| `ours.py` 또는 task `model.py` | NSMT 본체와 Neo/Hippo/fusion/head |
| `layers.py` | spiking linear, attention/MCA/MLP, residual |
| `positional.py` | sinusoidal tAPE |
| `train.py` | optimizer/scheduler/early stop/reset/loss/backward |
| `test.py` | checkpoint 평가, energy/statistics, AD threshold |
| `utils.py` | metric, scheduler, early stopping, plotting/logging |
| `Spikformer.py` | Spikformer baseline |
| `neo_bank.py` | batch-pooled cross-window EMA firing prior |
| `analysis.py` | causal/path/frequency diagnostics |
| `tier1_analysis.py` | simple LIF simulation, half-life/memory capacity |
| `firing_rate.py` | LIF hook와 firing-rate plot |
| `horizon_stats.py` | time-collapse와 horizon별 metric |
| `data_provider/augmentation.py` | jitter, scaling, rotation, permutation, warp, DTW augmentation |
| `data_factory.py` | dataset class와 DataLoader 연결 |
| `data_loader.py` | ETT/custom/M4, PSM/MSL/SMAP/SMD/SWaT, UEA loader |
| `download_data.py` | data archive download |
| `dtw.py` | DTW/shape-DTW |
| `m4.py` | M4 metadata/download |
| `timefeatures.py` | second–month calendar encoding |
| `uea.py` | collate, padding mask, normalization, interpolate/subsample |

Task별 정확한 표준 구성:

- `forecasting/`: model/analysis 14개 + `data_provider` 8개 = 22개.
- `anomaly_detection/`: model/analysis/AD utility 14개 + `data_provider` 8개 = 22개.
- `classification/` root: model 10개 + `data_provider` 8개 = 18개.
- `classification/ECG/`: ECG model/entrypoint/ablation 15개.

### 5.3 고유 실험 모듈과 이미 얻은 교훈

| 실험군 | 구현 아이디어 | 저장소 기록이 주는 판단 |
|---|---|---|
| Hankel | causal block-Hankel two-axis encoding | 별도 validation/readout/spike 진단까지 있음 |
| RCF | multiscale local residual + neocortical context + zero-init additive fusion | 실험상 기각 |
| IPGM | consensus/innovation concat projection 후 Neo suppression | 기각 |
| ACIM | consensus/innovation union을 기존 MCA와 결합 | 기각 |
| Wavelet | à-trous SWT detail을 spike time, approximation을 Neo로 사용 | 여러 stream variant가 있으나 Neo 인과성 확립 못함 |
| Shape Delta | signed/magnitude/shuffle/only patch delta | 기각 |
| Relative PE / LogPE | replay의 relative/log position | 초기화 순서 교란 제거 뒤 headline 철회 |
| Seasonal Neo | phase-indexed global seasonal memory | global prior이지 entity별 episode memory는 아님 |
| Initfix | module name 기반 local RNG | import/order initialization confound 제거 |
| Neo Recall | raw/lowfreq/next target과 mul/none gate | forecasting 개선은 gate가 아니라 self-attention 변경 교란 |
| AD gradient diag | main vs DCT gradient cosine/norm | 두 가설 모두 저장소 commit 기록상 기각 |

`archive/forecasting`에는 align, decomposition, dual-KV, Gaussian DCT taper, HPF, multiscale/coarse Neo, membrane fusion, roll/shift, TN position, teacher–student, twin time-variate, cross-variate, shuffle/no-Neo 등의 과거 모델이 있다. 이 장기간의 실험 기록은 “Neo를 더 복잡하게 만들면 좋아진다”는 prior를 지지하지 않는다.

특히 [docs/46_neorecall_results_and_next.md](/home/yschoi/NSMT/docs/46_neorecall_results_and_next.md)의 causal controls가 중요하다.

- 곱셈 gate는 sparse signal을 죽였다.
- forecasting에서 Neo gate를 꺼도 성능이 같거나 더 좋았다.
- 겉보기 개선은 Memory-Replay를 self-attention으로 바꾼 구조적 교란으로 설명됐다.
- gate magnitude나 learned alpha는 Neo 유용성의 증거가 아니었다.
- 저장소가 보고한 initialization nuisance floor가 약 1.4%이므로 그보다 작은 변화는 강한 인과 주장에 부적합하다.

새 reservoir 실험은 기존 variant 위에 한 번 더 얹지 말고 **최초 모델에서 Neo만 교체하는 orthogonal A/B**로 시작해야 한다.

### 5.4 최초 NSMT 코드의 실제 tensor 흐름

최초 구현 파일은 `79ec94c:forecasting/ours.py`이며 이후 `a3d180b`와 실험 시작 직전 `191f366`의 같은 파일도 동일한 Git blob hash다.

```text
x                           [B,L,C]
  └─ detached mean/std RevIN
repeated x                  [T,B,L,C]
channel-independent         [T,B*C,L]
overlap patch               [T,B*C,N,P]
  ├─ Hippo embedding        [T,B*C,N,D]
  └─ transpose T↔N,
     old-T mean             [N,B*C,1,P]
       └─ Neo spiking MLP   [N,B*C,1,D]
                 │
Hippo × aligned Neo → MCA/IAND → head → [T,B,H,C]
                 └─ FC decoder → current-window low-frequency patch
```

중요한 결합:

- stride는 대체로 `P/2`이고 padding 때문에 patch 수가 코드상 `N+1`이 될 수 있다.
- constructor에서 `T=N`을 맞춘다.
- Hippo의 첫 축 `T`는 simulation step이다.
- Neo의 첫 축 `N`은 실제 patch chronology인데 spiking time으로 재사용한다.
- 두 의미가 숫자상 같아 transpose/broadcast가 작동한다.

reservoir 교체에서는 `simulation step S`, `patch chronology N`, `reservoir width R`를 반드시 별도 기호와 차원으로 둔다.

### 5.5 최초 Memory Replay, gradient, head, loss

최초 `Block.forward`는 forecasting에서:

```python
mx = 0.05 * mx + 0.95 * mx.detach()
g = x * mx.transpose(0, 2)
x = x * (1 - MCA(x, g))
x = x * (1 - MLP(x))
```

- Neo task gradient를 5%로 줄인다.
- AD/classification 계열은 주로 50% throttle을 사용했다.
- feature를 더하는 것이 아니라 Hippo spike를 억제한다.
- 한 번 0이 된 sparse path를 뒤 연산에서 복구하기 어렵다.
- MCA는 softmax attention이 아니라 spiked Q/K의 raw scaled dot-product를 다시 spike/projection한다.

forecast head는 각 `T` step의 flattened `N×D`를 horizon으로 투영한다. train target을 `T`번 반복해 모든 step을 감독하고 test에서는 `T` 평균을 사용한다.

최초 forecasting loss:

\[
\rho=\alpha\frac{H}{720},\qquad
\mathcal L=(1-\rho)\mathcal L_{\mathrm{MAE}}+\rho\mathcal L_{\mathrm{recon}}.
\]

같은 `alpha=0.1`이라도 `H=96`이면 약 0.0133, `H=720`이면 0.1이므로 auxiliary 영향이 7.5배 다르다. `distill=(x_data-x_time)^2`는 계산·logging하지만 실제 loss에는 포함되지 않는다.

최초 train script에는 early-stopping best와 별도로 마지막 state를 best 경로에 다시 저장하던 경로도 있었다. 최초 결과를 재현하는 compatibility run과 이 checkpoint bug를 고친 scientific baseline을 구분해야 하며, 새 reservoir 실험에는 수정된 best-reload 정책을 사용한다.

### 5.6 현행 root의 상태

핵심 위치:

- [forecasting/ours.py](/home/yschoi/NSMT/forecasting/ours.py:237): Block/gate/Memory Replay
- [forecasting/ours.py](/home/yschoi/NSMT/forecasting/ours.py:304): `RecurrentNeoLayer`
- [forecasting/ours.py](/home/yschoi/NSMT/forecasting/ours.py:356): zero-init gated recurrent `NeoLayer`
- [forecasting/ours.py](/home/yschoi/NSMT/forecasting/ours.py:520): 전체 model
- [forecasting/train.py](/home/yschoi/NSMT/forecasting/train.py:63): batch reset와 loss
- [forecasting/layers.py](/home/yschoi/NSMT/forecasting/layers.py:590): MCA
- [forecasting/neo_bank.py](/home/yschoi/NSMT/forecasting/neo_bank.py:82): global EMA bank
- [forecasting/config.py](/home/yschoi/NSMT/forecasting/config.py:80): Neo/replay/bank/gradient flags

현행 `RecurrentNeoLayer`:

- forward 시작에서 LIF를 reset한다.
- `W_in x[t] + W_rec s[t-1]`을 Python loop로 계산한다.
- recurrent weight는 trainable이며 window 내부 surrogate-gradient BPTT를 쓴다.
- 이름과 주석에 attractor/long-term이 있어도 runtime state는 window를 넘지 않는다.

현행 `NeoLayer`:

- `tanh(g) W_rec s[t-1]`, `g=0`으로 passive exact-superset에서 시작한다.
- causal-isolation에는 유용한 설계지만 역시 window-only BPTT다.

현행 `NeoMemoryBank`:

- buffer `[T,1,1,D]`에 batch×channel 평균 firing rate의 EMA를 저장한다.
- existing spike의 off 위치를 thresholded global prior로 보충한다.
- stream ID, timestamp, episode, causal query가 없다.
- shuffled batch 순서와 dataset mix에 따라 달라진다.

따라서 이는 sample memory가 아니라 dataset-level parametric/statistical prior와 가깝다.

### 5.7 task별 차이와 reservoir 도입 전 고쳐야 할 평가 위험

#### Forecasting

- batch마다 global `reset_net`.
- MAE + low-pass reconstruction, optional KD/trend loss.
- train/eval loader가 일부 topology flag를 다르게 전달할 위험이 있다.
- archive로 이동한 optional source의 registry가 남아 특정 flag가 실패할 수 있다.

#### Anomaly detection

- Neo task gradient throttle이 보통 50%.
- window reconstruction MSE + low-pass MSE.
- batch마다 reset.
- 현재 일부 loader의 train/validation slice가 겹칠 가능성을 먼저 확인·수정해야 한다.
- test threshold가 train+test energy percentile을 사용하는 경로가 있어 transductive 평가 우려가 있다.
- test anomaly를 durable bank에 쓰면 anomaly가 normal prototype으로 흡수될 수 있다.

#### UEA/ECG classification

- 각 sample은 일반적으로 독립 episode이므로 sample 사이 state carry는 leakage다.
- label-smoothed CE 또는 weighted CE + auxiliary reconstruction.
- batch마다 reset.
- classification eval의 존재하지 않는 `init_testing()` 호출 가능성과 spike encoder `T` mismatch를 확인해야 한다.
- ECG train은 best early-stop checkpoint 대신 마지막 모델 저장 경로를 점검해야 한다.

#### 공통

- `B*C` flatten 뒤에는 series와 channel identity가 사라진다.
- state cache는 flatten 전에 `stream_id`, `channel_id`, absolute timestamp를 가져야 한다.
- `RecurrentSpikLinear.memory` parameter처럼 선언됐지만 forward에서 사용하지 않는 state도 있어 이름만 보고 memory로 분류하면 안 된다.

---

## 6. “진짜 장기 메모리”의 조작적 정의

새 모델의 성공 조건을 구현 전에 고정한다.

### 6.1 용어

- **Persistent fading memory**: 같은 stream의 연속 window에서 state를 넘기지만 과거 영향은 감쇠한다.
- **Finite explicit memory**: delay taps/LDN으로 정해진 horizon을 보존한다.
- **Bounded episodic memory**: 선택된 과거 state를 timestamp와 함께 외부 bank에 저장하고 causal retrieval한다.
- **Synaptic/statistical memory**: ridge/RLS/STDP로 바뀐 weight에 장기 통계가 남는다.
- **Unlimited permanent memory**: 유한 state와 유한 bank로는 주장하지 않는다.

### 6.2 최소 인과 기준

cross-window memory라 부르려면:

1. delay \(k>L\)에서 stateful 모델의 `MC_k`가 reset LSM과 `W_rec=0`보다 유의하게 높아야 한다.
2. 마지막 \(L\) sample은 같고 그보다 오래된 history만 다른 paired stream을 구분해야 한다.
3. state reset 또는 old-history shuffle에서 task 성능이 하락해야 한다.
4. batch size와 downstream shuffle을 바꿔도 같은 timestamp state가 같아야 한다.

episodic long-term memory라 부르려면 추가로:

1. 실제 retrieval slot의 age가 \(>L\)여야 한다.
2. `age>L` slot 제거, value shuffle, timestamp shuffle, gate-off 중 하나에서 이득이 사라져야 한다.
3. future target/test label이 bank에 먼저 들어가지 않아야 한다.
4. checkpoint save/load나 episode reset 뒤에도 명시된 정책대로 보존되어야 한다.

---

## 7. 제안 아키텍처: Stateful Multi-Timescale Spiking Reservoir Neo

### 7.1 전체 구조

```text
raw chronological stream
        │
        ├─ train-only fixed scaler / causal normalizer
        │
        ▼
fixed sparse E/I spiking liquid ── membrane/current/spike/filtered traces
        │
        ├─ tapped-delay ring 또는 fixed LDN/LMU slow state
        │
        ▼
timestamp-indexed fixed feature Φ[t] ───────────┐
        │                                       │
        ├─ ridge/RLS Neo prediction             ├─ optional causal episodic bank
        │                                       │
window endpoint gather                         │
        │                                       │
        ▼                                       ▼
Reservoir token P_r(Φ) ── safe read gate / additive fusion ── Hippo ── task head
```

이 구조에서 liquid는 transient 계산을, delay/LDN은 finite slow memory를, bank는 durable episode를 담당한다.

### 7.2 tensor 계약

기호:

- `B`: window batch
- `L`: look-back
- `C`: variable/channel
- `P`: patch length
- `q`: patch stride
- `N`: patch 수
- `S`: 기존 Hippo spiking simulation steps
- `D`: Hippo embedding
- `R`: reservoir neuron 수
- `K`: filtered trace 수
- `F`: reservoir feature dimension

기존 Hippo:

```text
x_window        [B,L,C]
patches         [B,C,N,P]
H               [S,B*C,N,D]
```

제안 Neo:

```text
raw stream x                    [T_raw,C]
reservoir state per stream:
  syn, membrane, spike          [C,R]
  traces                        [K,C,R]
timestamp features Φ_all        [T_raw,C,F]
window gather                   [B,C,N,F]
reservoir projection            [B,C,N,D]
old-contract adapter            [N,B*C,1,D]
```

`F=R(3+K)`이면 synaptic current, membrane, spike, K traces를 모두 연결한 경우다. 비용이 크면 `spike + membrane + 2 traces`부터 시작한다.

첫 parity 실험은 현재 channel-independent 구조를 유지한다.

- shared reservoir weights
- state key는 `(stream_id, channel_id)`
- channel별 scalar/patch input

joint multivariate reservoir는 cross-variate modeling을 새로 추가하므로 별도 ablation으로 둔다.

### 7.3 reservoir dynamics

입력 시간축을 먼저 고정한다.

- `sample mode`: \(e_t=[z_t,\Delta z_t,\ldots]\)처럼 causal raw sample feature를 매 timestamp 한 번 넣는다.
- `global-patch mode`: 전체 stream에서 causal patch \(z_{t-P+1:t}\)를 endpoint마다 한 번만 fixed projection하고 넣는다.

최초 NSMT와 compute/input 단위를 맞추는 parity 실험은 `global-patch mode`, cross-window memory의 가장 단순한 구현은 `sample mode`다. 둘을 같은 실험 안에서 몰래 바꾸지 않고 input-mode ablation으로 보고한다.

patch endpoint 또는 raw sample time \(t\)에서:

\[
i_t=\alpha_s i_{t-1}+W_{\mathrm{in}}e_t+W_{\mathrm{rec}}s_{t-1},
\]

\[
\tilde u_t=\alpha_m u_{t-1}+(1-\alpha_m)i_t,
\]

\[
s_t=H(\tilde u_t-\vartheta),\qquad
u_t=\tilde u_t-\vartheta s_t,
\]

\[
q_t^{(k)}
=\lambda_k q_{t-1}^{(k)}+(1-\lambda_k)s_t .
\]

\[
\phi_t=[i_t,u_t,s_t,q_t^{(1)},\ldots,q_t^{(K)}].
\]

권장 초기화:

- sparse input connection
- sparse recurrent graph
- Dale sign constraint
- E/I 비율은 80/20을 시작점으로 두되 fixed truth로 간주하지 않음
- `tau_mem`, `tau_syn`, refractory, trace tau를 log-spaced/분포형으로 설정
- slightly subcritical 또는 balanced~mildly over-inhibited에서 시작
- recurrent scale은 spectral radius 하나가 아니라 firing/separation/consistency로 선택
- random seed와 optimizer seed를 분리

adaptation current가 필요하면:

\[
a_t=\rho_a a_{t-1}+s_t,\qquad
\vartheta_t=\vartheta_0+\beta_a a_t.
\]

느린 adaptive threshold는 memory를 늘릴 수 있지만 silent population을 만들 수 있으므로 vanilla LIF 다음 ablation으로 둔다.

#### 7.3.1 연속값 입력을 spike reservoir에 넣는 방식

LSM의 reservoir neuron이 spike를 낸다고 해서 입력도 반드시 Poisson spike여야 하는 것은 아니다.

1. **기본안**: standardized value를 fixed `W_in`을 통해 analog current로 주입. 가장 deterministic하고 ridge baseline에 적합하다.
2. **signed ON/OFF**: 양/음 변화량을 별도 channel의 event로 변환. sparsity와 변화 감지에 유리하다.
3. temporal Gaussian/latency encoding: Dey/George 계열과의 직접 비교용.
4. Poisson rate encoding: stochasticity와 긴 micro-simulation 비용 때문에 첫 선택이 아니다.

데이터 시간 \(t\) 또는 patch time \(n\)과 neuron micro-simulation \(s\)가 둘 다 필요하면 명시적으로 분리한다.

```text
data/patch time:          n = 1..N
micro simulation per n:  s = 1..S_micro
reservoir width:          R
```

첫 MVP는 `S_micro=1` current injection으로 시작한다. 기존처럼 우연히 `T=N`을 맞춰 두 축을 같은 것으로 취급하지 않는다.

### 7.4 readout feature

binary spike만 쓰지 않는다.

\[
\phi^{\mathrm{delay}}_t =
[\phi_t,\phi_{t-d_1},\ldots,\phi_{t-d_J}].
\]

후보:

1. `spike` only
2. smoothed firing rates
3. membrane + rates
4. membrane + rates + tapped delays
5. 같은 유효 차원의 fixed LDN

TDSRC의 이득이 단순히 feature dimension 증가인지 확인하려면 `R×#lags`가 같은 random feed-forward/delay-only baseline과 비교한다.

### 7.5 fixed LDN/LMU slow path

random liquid의 MC가 부족하면 병렬 slow state를 추가한다.

\[
m_t=\bar A m_{t-1}+\bar B e_t.
\]

- \(\bar A,\bar B\)는 정해진 window \(\theta\)와 order \(d\)의 Legendre delay dynamics로 생성하고 고정한다.
- trainable recurrent dynamics로 만들지 않는다.
- liquid feature와 `m_t`를 이어 붙이거나 별도 ridge prediction을 만든다.
- tapped delay와 동일 feature byte/동일 readout dimension으로 비교한다.

fixed LDN은 영구 기억이 아니라 명시된 \(\theta\) 길이의 compressed finite memory다. 장점은 horizon이 해석 가능하다는 점이다.

### 7.6 normalization

현재 window별 RevIN을 그대로 reservoir에 넣으면 동일 timestamp가 어느 window에 속하느냐에 따라 다른 값이 된다. persistent state가 합성되지 않는다.

Reservoir 입력에는 둘 중 하나만 사용한다.

1. **권장 MVP**: train split에서 channel mean/std를 fit하고 val/test에 고정.
2. causal deployment: 과거만 사용하는 running mean/variance.

Hippo는 기존 RevIN을 유지할 수 있다. 두 pathway의 normalization은 의도적으로 분리하고 metadata/checkpoint에 scaler를 저장한다.

### 7.7 state와 feature API

권장 데이터 구조:

```python
@dataclass
class ReservoirState:
    syn: Tensor          # [C,R]
    membrane: Tensor     # [C,R]
    spike: Tensor        # [C,R]
    traces: Tensor       # [K,C,R]
    delay_ring: Tensor   # optional
    last_t: int
    stream_id: Hashable
```

```python
scan_stream(
    x: Tensor,                  # [T,C]
    initial_state: ReservoirState | None,
) -> tuple[Tensor, ReservoirState]:
    # phi [T,C,F], final state
```

```python
gather_windows(
    phi: Tensor,                # [T,C,F]
    absolute_starts: Tensor,    # [B]
    patch_end_offsets: Tensor,  # [N]
) -> Tensor:
    # [B,C,N,F]
```

필수 method:

- `reset_stream(stream_id)`
- `reset_all_runtime_state()`
- `export_runtime_state()`
- `load_runtime_state()`
- `detach_runtime_state()`
- `state_schema_version`

global `functional.reset_net(model)`과 persistent reservoir reset을 분리한다. Hippo의 per-batch spiking state는 reset하되 reservoir state/cache는 명시적 policy에서만 reset한다.

---

## 8. Window 중복, chronological cache, leakage

### 8.1 가장 큰 구현 함정

현재 forecasting window가 stride 1로 겹친다고 가정하면, 각 full window를 stateful reservoir에 다시 넣을 때 같은 raw sample이 최대 `L`번 적분된다. `shuffle=False`만으로 해결되지 않는다.

batch size 64라면 batch slot 0이 start 0 다음 start 64를 받을 수 있다. batch slot을 state identity로 쓰면 start 1–63의 chronology를 건너뛴다.

### 8.2 권장 방식: offline causal state index

1. train-only scaler를 fit한다.
2. entity/stream별 raw sequence를 timestamp 순서로 **한 번만** scan한다.
3. 각 timestamp의 fixed \(\Phi[t]\)를 cache한다.
4. window의 patch endpoint:

\[
e_{b,n}=\mathrm{absolute\_start}_b+nq+P-1
\]

를 계산한다.
5. `Φ[e[b,n]]`를 gather한다.
6. 그 뒤 Hippo/gate 학습 loader는 cached feature를 함께 주는 조건에서 shuffle해도 된다.

장점:

- 중복 적분 없음
- state graph 없음
- batch size/shuffle 불변
- raw chronology가 명확
- Neo feature cache가 Hippo update와 독립

### 8.3 online 방식이 필요한 경우

full window가 아니라 `last_t+1:end_t`의 새 sample만 reservoir에 전달한다.

- state key: `stream_id`, 절대 batch slot 금지
- out-of-order request: cache lookup 또는 명시적 reset/replay
- time gap: gap 길이에 따라 decay를 analytically 적용하거나 reset
- duplicate timestamp: update 거부

### 8.4 prefix-causality와 cache 불변성 테스트

다음 unit/integration test가 모두 통과해야 한다.

- future suffix를 바꿔도 `Φ[:t]`는 동일
- batch size를 바꿔도 `Φ[t]` 동일
- downstream shuffle을 바꿔도 `Φ[t]` 동일
- state update 횟수 = raw sample 수
- 다른 entity/subject 경계에서 reset
- state와 cache에 `grad_fn is None`
- cache key에 dataset fingerprint, split, scaler, reservoir config, seed, state schema 포함

### 8.5 split별 state protocol

#### Forecasting

두 protocol을 따로 보고한다.

1. `isolated`: split 시작에서 reset하고 과거 input으로 washout.
2. `deployment`: train→val→test의 실제 관측 input을 시간순으로 state에 넣되 model/readout parameter update는 train 뒤 동결.

validation 관측값을 test history로 쓰는 것은 배포 protocol에서는 허용 가능하지만 반드시 명시한다. forecast origin \(t\)에서 허용되는 것은 \(x_{\le t}\)뿐이다.

#### Anomaly detection

정상 stream에서 순서:

1. `state_{t-1}`로 \(x_t\) 예측
2. anomaly score 계산
3. \(x_t\)를 transient state에 반영

test anomaly를 long-term bank에 계속 write하면 anomaly가 normal prototype이 된다. 첫 실험은 train 종료 후 durable bank를 freeze하고 transient liquid만 관측 input으로 갱신한다.

현재 train/validation slice overlap과 train+test threshold 경로를 먼저 제거해야 reservoir 효과를 평가할 수 있다.

#### Classification

UEA/heartbeat sample은 독립 episode다. sample마다 reset한다. 실제 subject ID와 연속 timestamp가 있을 때만 같은 subject 내 state carry를 허용하며 split도 subject 단위여야 한다.

---

## 9. 실제 Memory Replay 모듈

### 9.1 이름 정리

- 기존 `Memory Replay` → `ReservoirFusion` 또는 `WithinWindowNeoFusion`
- 새 module → `EpisodicReplayBank`

이름을 먼저 고쳐야 current-window attention과 historical retrieval이 실험 로그에서 섞이지 않는다.

### 9.2 inference historical bank

stream별 bounded bank:

```text
ReplayBank:
  phi       [M,C,F]      # raw frozen reservoir feature
  time      [M]
  valid     [M]
  novelty   [M]
  quality   [M]          # optional
  stream_id
  schema_version
```

raw `phi.detach()`를 저장하고 projected key/value를 저장하지 않는다. trainable projector가 바뀌어도 old slot이 stale representation이 되지 않게 하기 위함이다.

causal retrieval:

\[
\bar H=\operatorname{Mean}_S(H),
\]

\[
Q=W_q\bar H,\quad K=W_k\Phi_{\mathrm{bank}},\quad
V=W_v\Phi_{\mathrm{bank}},
\]

\[
A=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
+M_{\mathrm{causal}}+b_{\mathrm{age}}
\right),
\]

\[
R=W_o A V.
\]

필수:

- `bank_time < query_time`
- query-before-write
- future target/label 저장 금지
- empty bank의 null read
- retrieval age logging
- `age>=L`인 long bank 대조

### 9.3 short ring과 long bank

- `short_ring`: 최근 state, 정확한 finite context
- `long_bank`: periodic/age-stratified/novelty-selected episode

write priority 예:

\[
p_{\mathrm{write}}
=\sigma(
b_w
+a\,\mathrm{novelty}
+b\,\mathrm{surprise}
+c\,\mathrm{utility}
-d\,\mathrm{redundancy}
).
\]

첫 구현에서는 learned write gate보다 deterministic policy가 낫다.

1. 매 `K_write` patch마다 후보
2. nearest-bank cosine distance가 threshold 이상이면 write
3. capacity 초과 시 age×redundancy 기준 eviction

ground-truth prediction error를 surprise로 쓰려면 target이 실제로 도착한 뒤에만 write/update한다.

### 9.4 training replay와 inference replay는 다르다

- **Inference retrieval**: 과거 state를 현재 예측 feature로 읽는다.
- **Training replay**: 과거 `(phi,target)`로 readout/gate를 다시 적응시킨다.

offline ridge가 전체 train state를 한 번에 풀면 training replay는 필요 없다. online RLS에서 concept drift/rare regime retention을 위해서만 replay한다.

sampling:

- rare/high-surprise
- regime/age stratification
- recent와 old의 비율 고정
- source stream 분리

reservoir/STDP version이 바뀌면 cache와 bank를 전부 무효화한다.

---

## 10. Gate 수정

### 10.1 기존 gate를 버려야 하는 이유

기존:

\[
g=H\odot X_{\mathrm{neo}},\qquad
H'=H\odot(1-\mathrm{MCA}(H,g)).
\]

이는 새 정보를 더하지 못하고 Neo spike가 0이면 product와 gradient가 함께 사라진다. 과거 정보가 Hippo에 없을 때 억제 연산만으로 그것을 복원할 수 없다.

### 10.2 1단계: prediction-level mixture

reservoir가 독립적으로 유용한지 먼저 본다.

\[
\hat y
=\hat y_H+g(\hat y_R-\hat y_H),\qquad
g=\sigma(b_g+w^\top c).
\]

- \(\hat y_R\): frozen reservoir ridge prediction
- \(c\): reservoir confidence, novelty, age, Hippo uncertainty
- `g=0` control이 명확

이 단계가 실패하면 internal cross-attention을 추가하지 않는다.

### 10.3 2단계: additive feature fusion

\[
g_n=\sigma\left(
b_g+\mathrm{MLP}[
\mathrm{LN}(\bar H_n),
\mathrm{LN}(R_n),
|\bar H_n-R_n|,
\mathrm{confidence},
\mathrm{age}]
\right),
\]

\[
Z_n=H_n+\gamma g_n\odot W_R R_n.
\]

권장:

- 첫 버전 gate shape `[B,C,N,1]`
- 기존 head parity를 유지할 때 `R:[B*C,N,D]`를 `[1,B*C,N,D]`로 만든 뒤 `S`에만 broadcast한다. reservoir의 `N`을 Hippo simulation `S`로 재해석하지 않는다.
- 더 깨끗한 후속안은 Hippo를 `S` 평균한 `[B,C,N,D]`에서 reservoir와 합치고 single head를 쓰는 것이다. 이는 head 변경이므로 별도 ablation이다.
- \(b_g\approx-2\)에서 \(-4\), 또는 `W_R` output zero-init
- \(\gamma=0\) 또는 작은 값에서 시작
- empty bank이면 정확히 `g=0`
- Hippo query가 gate를 편법으로 조정하면 query input detach 대조
- `gate=none`, `gate=forced_one`, memory shuffle control

analog sigmoid/softmax/normalization은 MAC을 사용한다. spiking AC-only energy라고 주장하지 않는다.

### 10.4 read gate와 write gate 분리

- read gate: 현재 예측에 과거 memory를 쓸지 결정
- write gate: durable bank/slow state에 현재 episode를 쓸지 결정

1993 active-memory 논문의 `Load-in` 교훈을 반영한다. 하나의 alpha가 두 역할을 동시에 하지 않게 한다.

gate가 커졌다는 사실은 유용성 증거가 아니다. 같은 checkpoint에서 gate-off, memory shuffle, old-slot removal로 counterfactual 성능 저하를 보여야 한다.

---

## 11. Gradient 흐름 수정

### 11.1 권장 경계

```text
x ──> Hippo(θ_H) ──> query/fusion/gate/head ──> L_task
 \
  └─ torch.no_grad()
      fixed W_in/W_rec/tau LSM ──> φ.detach()
                                  ├─> ridge/RLS (autograd 밖)
                                  ├─> raw replay bank.detach()
                                  └─> trainable current-window projection/fusion
```

고정:

- `W_in`
- `W_rec`
- neuron threshold/tau
- Dale mask/topology
- persistent state와 delay ring
- fixed LDN \(\bar A,\bar B\)

학습 가능:

- ridge/RLS readout: autograd 밖
- reservoir-to-`D` projection: ridge로 정하거나 current-window gradient 허용
- read/write gate
- retrieval `Wq/Wk/Wv/Wo`
- Hippo와 task head

### 11.2 코드 규칙

```python
with torch.no_grad():
    phi, state = reservoir.scan(x, state)
phi = phi.detach()
assert phi.grad_fn is None
```

- reservoir parameter는 `nn.Parameter` 대신 persistent buffer가 적합하다.
- trainable module registry에서 제외되었는지 optimizer parameter list로 test한다.
- 과거 state를 현재 graph에 연결하지 않는다.
- 현재의 `0.05*mx + 0.95*mx.detach()` hack은 제거한다.
- stop-gradient boundary는 0% 또는 명시적 trainable path로 이산적으로 정의한다.
- bank에는 raw fixed `phi`만 저장한다.

### 11.3 “전체 NSMT가 BPTT-free”라는 표현 금지

Neo가 frozen reservoir여도 Hippo의 MultiStepLIF와 trainable temporal blocks가 surrogate gradient를 쓰면 전체 모델은 여전히 BPTT를 한다.

정확한 표현:

> Neo recurrent dynamics와 persistent memory state는 BPTT 없이 계산·학습되고, 기존 Hippo/fusion은 현재 방식의 gradient 학습을 유지한다.

전체 BPTT-free NSMT를 원하면 Hippo까지 fixed random/local features와 linear readout으로 바꾸는 별도 연구가 필요하다.

### 11.4 local plasticity를 쓸 때

STDP/homeostasis를 추가한다면:

1. train chronology만 사용
2. Dale sign, clipping, firing target 유지
3. unsupervised phase 종료
4. reservoir 완전 freeze
5. 모든 cache, bank, ridge를 새로 생성

online으로 core가 계속 변하면 과거 raw `phi` bank schema가 stale해지고 fixed RC의 재현성도 사라진다.

---

## 12. Loss function 수정

### 12.1 task loss의 계수를 줄이지 않는다

기존 convex mixture:

\[
(1-\alpha)\mathcal L_{\mathrm{task}}
+\alpha\mathcal L_{\mathrm{aux}}
\]

는 auxiliary를 키울수록 task gradient 자체를 줄이고, forecasting에서는 horizon-dependent ratio까지 있어 해석을 흐린다.

권장:

\[
\mathcal L
=\mathcal L_{\mathrm{task}}
+\lambda_{\mathrm{pred}}\mathcal L_{\mathrm{res\_pred}}
+\lambda_g\mathbb E[g]
+\lambda_{\mathrm{tv}}\mathbb E|g_n-g_{n-1}|
+\lambda_w\mathcal L_{\mathrm{write}}.
\]

각 항의 raw value와 weighted value를 모두 log한다.

reservoir prediction을 ridge/RLS로 학습할 때 \(\mathcal L_{\mathrm{res\_pred}}\)는 별도 closed-form/online objective와 diagnostic이지 PyTorch total loss로 reservoir에 역전파하는 항이 아니다. trainable current-window projection을 둘 때만 downstream gradient loss로 더한다.

### 12.2 Reservoir auxiliary target

current patch를 그대로 복원하면 input shortcut이 강하고 장기기억과 무관하다.

우선순위:

1. \(\phi_t\rightarrow y_{t+1:t+H}\): multi-horizon future prediction
2. \(\phi_t\rightarrow\) future one-sided low-frequency trend
3. \(\phi_t\rightarrow u_{t-k}\): delayed-recall probe
4. pre-trained Hippo의 out-of-fold residual

Hippo residual을 target으로 쓰려면 Hippo를 먼저 학습하고 out-of-fold prediction으로 residual을 만든다. 같은 train prediction의 residual은 overfit을 증폭한다.

per-token target을 만들 때 token \(n\) 뒤의 값을 포함한 full-window DCT를 사용하면 token-level leakage가 생긴다. one-sided causal filter 또는 forecast-origin 이후 target을 사용한다.

### 12.3 task별 main loss

- forecasting: MAE 또는 Huber를 주 loss로 유지; MSE도 dataset protocol에 맞춰 별도 보고.
- anomaly detection: normal train data의 one-step/multi-step prediction residual을 우선; reconstruction은 보조.
- classification: CE/label-smoothed CE; independent sample은 마지막/pooled reservoir feature를 사용.

### 12.4 Gate/write regularization

- `L_gate = mean(g)`는 약한 energy/sparsity cost만 준다.
- gate entropy로 0.5 사용을 강제하지 않는다.
- `L_write`는 write-rate budget, slot coverage, redundancy를 조절한다.
- 너무 강한 sparsity로 gate/bank가 영원히 꺼지는 collapse를 감시한다.

### 12.5 frozen reservoir에는 “가짜 loss”를 주지 않는다

firing rate, E/I balance, branching factor, state norm은 frozen core로 gradient가 가지 않는다. 이들은 loss라고 부르지 말고 hyperparameter selection constraint/diagnostic으로 사용한다.

예:

```text
valid candidate if:
  target_rate_low <= mean_rate <= target_rate_high
  silent_fraction <= threshold
  saturation_fraction <= threshold
  perturbation_distance decays after washout
```

criticality를 정확히 1에 강제하지 않고 안정적인 허용 band를 선택한다.

---

## 13. BPTT 없는 효과적인 학습 방법

### 13.1 1순위: offline ridge

washout 뒤 train feature를 모은다.

\[
\Phi=[\phi_1,\ldots,\phi_M]^\top,\qquad
Y=[y_1,\ldots,y_M]^\top.
\]

\[
W_{\mathrm{out}}
=
(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top Y.
\]

실제 구현은 inverse를 만들지 않고 Cholesky/QR/linear solve를 쓴다.

절차:

1. fixed scaler fit
2. raw train stream causal scan
3. washout 제거
4. train feature만 standardize
5. `lambda`를 validation으로 선택
6. multi-output/multi-horizon을 한 번에 solve
7. readout과 scaler를 checkpoint

feature가 큰 경우:

- chunk로 \(\Phi^\top\Phi\), \(\Phi^\top Y\) 누적
- conjugate gradient
- randomized/low-rank projection
- horizon별 독립 ridge

장점은 deterministic하고 recurrent graph를 저장하지 않으며 reservoir의 독립 유용성을 가장 깨끗하게 측정한다는 점이다.

### 13.2 2순위: online RLS

\[
k_t=
\frac{P_{t-1}\phi_t}
{\gamma+\phi_t^\top P_{t-1}\phi_t},
\]

\[
W_t=W_{t-1}
+k_t(y_t-W_{t-1}^\top\phi_t)^\top,
\]

\[
P_t=\gamma^{-1}
\left(
P_{t-1}-k_t\phi_t^\top P_{t-1}
\right).
\]

- \(\gamma=1\): 누적 least squares
- \(\gamma<1\): concept drift를 위해 과거를 의도적으로 잊음
- forecasting target은 horizon 뒤 실제로 도착한 후 update
- val/test update는 별도 `online-adaptation` protocol에서만 허용

full \(P\)는 \(O(F^2)\) memory/compute다. 큰 feature에서는:

- block-diagonal RLS
- diagonal RLS
- low-rank inverse covariance
- per-channel/per-horizon factorization

을 비교한다.

numerical guard:

- `float64` covariance
- periodic symmetrization
- positive diagonal floor
- regularized initialization \(P_0=\lambda^{-1}I\)

### 13.3 3순위: FORCE

[Sussillo & Abbott, 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2756108/)의 FORCE는 RLS 계열로 output feedback 또는 recurrent weight까지 빠르게 안정화할 수 있다.

사용 시점:

- autonomous chaotic generation
- plain ridge/RLS의 closed-loop가 불안정한 경우

첫 MVP에서는 쓰지 않는다. recurrent core를 바꾸는 FORCE는 fixed reservoir라는 깨끗한 가정을 깨며, feedback exposure bias와 안정성 변수를 동시에 추가한다.

### 13.4 4순위: local unsupervised plasticity

선택지:

- STDP + inhibitory/homeostatic regulation
- intrinsic plasticity
- P-CRITICAL/NALSM 계열 local activity control

이는 representation을 task에 맞추는 supervised readout을 대체하지 않는다. random-fixed LSM보다 실제로 좋아지는지 별도 ablation이 필요하다.

### 13.5 recurrent credit가 꼭 필요하면

- e-prop: eligibility trace × top-down learning signal
- reward-modulated STDP/R-STDP

둘 다 BPTT-free 후보지만 exact gradient가 아니며, 최신 연구에서도 task 이득이 제한적인 경우가 있다. strict reservoir MVP가 실패한 뒤 별도 variant로만 사용한다.

### 13.6 권장 hybrid schedule

```text
Phase A  no_grad chronological reservoir scan/cache
Phase B  ridge로 Neo-only readout 학습
Phase C  reservoir/readout freeze, Hippo + safe fusion gradient 학습
Phase D  필요하면 B와 C를 번갈아 재적합
Phase E  deployment variant에서만 RLS online adaptation
```

Neo recurrent graph를 저장하지 않으므로 Hippo 학습 batch가 shuffled여도 cached feature의 인과성은 유지된다.

### 13.7 reservoir hyperparameter 선택

core가 fixed이므로 gradient 대신 validation 기반 search를 쓴다.

- random search 또는 Bayesian optimization
- 작은 budget의 CMA-ES/evolutionary topology search
- seed 하나의 최고점보다 여러 seed의 robust plateau
- firing/stability constraint를 먼저 적용한 뒤 task metric 순위
- nested validation 또는 고정 validation split
- test set으로 tau/density/delay를 고르지 않음

탐색 순서:

1. input scale와 target firing range
2. recurrent scale/E-I/density
3. heterogeneous tau
4. delay/LDN horizon
5. ridge regularization

모든 축을 한꺼번에 탐색하면 상호작용을 해석할 수 없고 예산도 폭발한다.

---

## 14. 구현 파일과 변경 지점

### 14.1 작업 전략

dirty root를 바로 되돌리거나 기존 46개 실험을 덮어쓰지 않는다.

1. 별도 branch/worktree 또는 `reservoir_nsmt_v1/forecasting` snapshot 생성
2. `79ec94c`/`191f366` 최초 forecasting parity 재현
3. Hippo, DCT auxiliary, head, data split을 고정
4. `encoding_neo + TemporalBlock`만 adapter로 교체
5. 단계마다 기존 출력 shape와 metric parity test

### 14.2 권장 공용 package

여러 task snapshot에 코드를 복제하기 전에 공용 구현을 둔다.

```text
common/reservoir/
  __init__.py
  dynamics.py        # fixed E/I LIF/ALIF
  state.py           # ReservoirState, reset/export/load
  features.py        # filtered traces, tapped delays, LDN
  cache.py           # causal scan, fingerprint, endpoint gather
  readout.py         # ridge, RLS
  replay.py          # EpisodicReplayBank
  metrics.py         # MC, firing, separation, stability
```

forecasting first:

- `forecasting/ours.py`: Neo construction과 fusion adapter 교체
- `forecasting/train.py`: global reset 분리, cached feature input, loss 분리
- `forecasting/data_provider/data_loader.py`: stream ID, absolute timestamp/start metadata
- `forecasting/model.py`: train/eval topology flag parity
- `forecasting/config.py`: reservoir config와 cache fingerprint

tests:

```text
tests/reservoir/test_dynamics.py
tests/reservoir/test_prefix_causality.py
tests/reservoir/test_cache_invariance.py
tests/reservoir/test_state_lifecycle.py
tests/reservoir/test_ridge_rls.py
tests/reservoir/test_no_grad_boundary.py
tests/reservoir/test_replay_causality.py
```

### 14.3 초기 config

값은 탐색 시작점이며 정답이 아니다.

```yaml
neo_backend: reservoir
reservoir:
  neuron: lif
  size: [128, 256, 512]
  recurrent_density: [0.02, 0.05, 0.10]
  input_density: [0.05, 0.10, 0.20]
  excitatory_ratio: [0.75, 0.80, 0.85]
  recurrent_scale: validation_sweep
  tau_mem: log_spaced
  tau_syn: log_spaced
  trace_taus: [short, medium, slow]
  feature: membrane_rate
  state_mode: causal_cache
  reset_policy: stream_boundary
memory:
  kind: tapped_delay
  delays: log_or_ami_selected
readout:
  kind: ridge
  lambdas: logspace
fusion:
  kind: prediction_mix
  init: hippo_only
replay:
  enabled: false
```

hard-coded 80/20 E/I나 critical scale 하나에 고정하지 않는다.

### 14.4 checkpoint schema

다음을 분리 저장한다.

```text
model_trainable.pt
reservoir_config.json
reservoir_fixed_weights.pt
scaler.pt
readout.pt
runtime_state.pt          # deployment only
episodic_bank.pt          # enabled only
cache_manifest.json
```

manifest:

- code commit
- data fingerprint/split
- stream IDs
- scaler fit range
- reservoir seed/config hash
- state schema version
- cache timestamps
- write/reset policy

학습 checkpoint와 runtime memory를 분리해야 reproducible evaluation에서 memory carry 여부를 선택할 수 있다.

---

## 15. 검증 실험과 acceptance criteria

### 15.1 구현 무결성

모델 성능 전에 다음이 모두 통과해야 한다.

- prefix causality
- batch-size invariance
- downstream shuffle invariance
- timestamp 중복 update 없음
- stream boundary reset
- split policy test
- bank의 모든 time `< query_time`
- empty-bank null read
- reservoir parameter gradient 없음
- downstream trainable module gradient 존재
- cache fingerprint mismatch 시 load 거부
- state/cache save-load round trip

### 15.2 최소 baseline matrix

| ID | 조건 | 질문 |
|---|---|---|
| A | 최초 NSMT | 원 구조 |
| B | Hippo-only / no-Neo | Neo 자체가 필요한가 |
| C | original stateless Neo | current-window trend encoder |
| D | matched random feed-forward feature | recurrence가 필요한가 |
| E | non-spiking ESN | spike가 필요한가 |
| F | LSM, every-window reset | within-window liquid |
| G | stateful LSM | cross-window state carry |
| H | `W_rec=0` | recurrence의 인과 효과 |
| I | stateful LSM + slow traces | multi-timescale 효과 |
| J | I + tapped delay | explicit finite memory |
| K | I + fixed LDN, matched dimension | structured memory 대 delay |
| L | J/K + long episodic bank | durable retrieval |
| M | L, state shuffled | state 내용의 인과성 |
| N | L, old bank value/time shuffled | historical content의 인과성 |
| O | L, gate forced off | fusion gate의 인과성 |
| P | delay-only, no recurrence | TDSRC 이득의 단순 차원 증가 여부 |

동일 split, parameter/readout dimension, HP 탐색 예산으로 비교한다.

### 15.3 synthetic memory

필수:

- iid delayed recall `MC_k`
- total linear MC
- nonlinear memory capacity
- NARMA10/30
- delayed XOR
- copy/adding task
- distractor 사이 cue recall
- 마지막 `L`은 같고 old history만 다른 paired sequence
- episode reset 뒤 explicit-bank recall

훈련에서 보지 않은 \(k>L\) delay를 포함한다.

보고:

- delay별 mean/CI
- reservoir seed 10–20개
- state bytes와 effective readout dimension
- firing/silence/saturation

### 15.4 chaotic forecasting

Lorenz/Mackey–Glass:

- one-step teacher-forced MSE/NRMSE
- closed-loop autonomous error
- valid prediction time을 Lyapunov time 단위로 보고
- perturbation과 noise robustness
- teacher-forced와 free-run을 섞어 보고하지 않음

### 15.5 실제 forecasting

- ETT 계열과 기존 NSMT benchmark
- chronological split
- isolated와 deployment state protocol 분리
- 각 horizon MAE/MSE
- stateful 대 reset paired seed
- history \(>L\) 의존성이 실제 있는 synthetic/controlled dataset도 함께 사용
- irregular sampling, missing value, time gap
- concept drift에서 frozen ridge와 RLS 비교

pure accuracy만으로는 장기기억을 증명할 수 없다. state reset/shuffle 성능 저하가 함께 있어야 한다.

### 15.6 anomaly detection

- train/val overlap 제거
- train-only 또는 validation-calibrated threshold
- anomaly point를 state에 넣기 전 score
- durable bank frozen vs guarded write
- point/event F1, AUROC/AUPRC와 delay
- anomaly contamination ablation

### 15.7 classification

- independent sample reset
- subject-level split이 가능하면 사용
- UCR/UEA에서 univariate와 multivariate를 분리
- accuracy/F1와 neuron/state budget
- sample 사이 carry 조건은 leakage control로만 사용

### 15.8 통계

- reservoir seed와 optimizer seed 분리
- real task 최소 5 paired seeds, 핵심 memory test 10–20 seeds
- mean, standard deviation, bootstrap 또는 paired CI
- hyperparameter와 epoch는 validation으로만 선택
- test 최대값 선택 금지
- 기존 저장소 nuisance floor 약 1.4%를 넘는지 별도 표시

### 15.9 energy/efficiency

최소 보고:

- reservoir neuron `R`
- `R×#lags` effective feature dimension
- trainable parameter
- state/cache/bank bytes
- spike count와 firing rate
- synaptic operations/SOP
- analog MAC와 spike AC 분리
- latency/throughput

GPU wall-clock을 neuromorphic energy 효율의 증거로 쓰지 않는다. 실제 hardware가 없으면 theoretical proxy임을 명시한다.

### 15.10 사전 고정한 성공·중단 기준

**P1 stateful LSM 성공**

- \(k>L\)에서 reset/`W_rec=0`보다 MC가 paired CI 기준 높음
- 최소 3개 dataset/설정에서 stateful이 reset보다 일관된 task 이득
- 이득이 nuisance floor와 사전 정한 practical threshold를 넘음
- cache correctness test 전부 통과

**Episodic bank 성공**

- 실제 retrieval median/분포에 `age>L`가 존재
- old slot 제거 또는 shuffle에서 이득 소멸
- gate-off에서 이득 소멸
- leakage 없이 checkpoint/reset 뒤 recall

**중단**

- stateful 대 reset의 \(k>L\) memory 차이가 없으면 gate/replay 확장 중단
- 최근 `<L` slot에서만 이득이면 long-term 주장을 철회
- test-time label update가 있어야 이득이면 `online adaptation`으로 재분류
- batch/order에 따라 같은 timestamp feature가 바뀌면 실험 무효
- non-spiking ESN이 같은 예산에서 우세하면 spiking 채택 이유를 hardware 지표로 별도 입증

---

## 16. 단계별 구현 계획

### Phase 0 — 최초 모델 동결·재현

- `79ec94c`의 forecasting 구조 재현
- 현행 root 수정 금지
- original/no-Neo metric parity
- seed와 initialization fingerprint 고정

산출물: 최초 baseline checkpoint와 tensor-shape test.

### Phase 1 — BPTT-free reservoir 단독 검증

- fixed scaler
- vanilla E/I LIF scanner
- causal timestamp cache
- membrane/rate feature
- ridge readout
- reset/stateful/`W_rec=0`/ESN 비교

gate와 episodic bank는 사용하지 않는다.

Go/no-go: \(k>L\) memory와 Neo-only direct task 성능.

### Phase 2 — finite multi-timescale memory

- heterogeneous tau
- tapped delays
- fixed LDN matched-dimension 대조
- criticality/E-I/firing diagnostics

Go/no-go: vanilla stateful 대비 memory-performance/storage Pareto 개선.

### Phase 3 — Hippo와 안전한 결합

- prediction-level mixture부터 시작
- additive zero-init feature fusion
- gate forced-off/one, state shuffle
- task coefficient 1 유지

Go/no-go: Hippo-only와 reservoir-only를 넘어서는 인과적 이득.

### Phase 4 — 실제 episodic memory

- causal short ring + age-stratified long bank
- deterministic novelty write
- query-before-write
- old-slot ablation
- checkpoint persistence

Go/no-go: `age>L` retrieval이 task를 실제 개선.

### Phase 5 — online adaptation

- ridge initialization
- readout-only RLS
- block/low-rank covariance
- concept-drift protocol
- optional training replay

test-time adaptation 결과는 frozen evaluation과 별도 표로 보고한다.

### Phase 6 — 선택적 local plasticity·hardware

- STDP/homeostasis 또는 e-prop variant
- cache regeneration
- Loihi/ASIC feasibility와 실제 operation accounting

앞 단계가 성공하기 전에는 수행하지 않는다.

---

## 17. 구현자가 바로 사용할 의사코드

### 17.1 causal precompute

```python
scaler.fit(train_raw)
reservoir.freeze()

for stream_id, raw in chronological_streams:
    state = reservoir.initial_state(channels=raw.shape[-1])
    raw_z = scaler.transform(raw)
    with torch.no_grad():
        phi, final_state = reservoir.scan_stream(raw_z, state)
    cache.save(
        stream_id=stream_id,
        phi=phi.detach().cpu(),
        final_state=final_state.detach().cpu(),
        fingerprint=fingerprint,
    )
```

### 17.2 window dataset

```python
def __getitem__(self, idx):
    x, y, stream_id, start = self.base[idx]
    ends = start + self.patch_end_offsets
    phi = self.cache.gather(stream_id, ends)  # [C,N,F]
    return x, y, phi, stream_id, start
```

### 17.3 hybrid forward

```python
hippo = self.hippo(x)                       # [S,B*C,N,D]
phi = phi.detach()                          # [B,C,N,F]
neo_pred = self.reservoir_readout(phi)      # ridge/frozen

if self.fusion_stage == "prediction":
    y_h = self.hippo_head(hippo)
    gate = self.prediction_gate(y_h, neo_pred)
    y = y_h + gate * (neo_pred - y_h)
else:
    r = self.reservoir_proj(phi)             # [B,C,N,D]
    z = self.safe_additive_fusion(hippo, r)
    y = self.head(z)
```

### 17.4 causal episodic query

```python
memory = bank.query(
    stream_id=stream_id,
    query_phi=phi_now,
    before_time=query_time,
    min_age=0,
)
prediction = model(hippo, phi_now, memory)

# query first, write second
bank.maybe_write(
    stream_id=stream_id,
    time=query_time,
    raw_phi=phi_now.detach(),
    novelty=novelty,
)
```

---

## 18. 최종 권고

1. Neo를 LSM으로 교체하는 방향은 타당하다. 현재 Neo보다 reservoir라는 용어와 계산 원리에 더 부합한다.
2. 다만 pure LSM은 fading memory다. “진짜 장기 메모리”는 state lifecycle, finite slow memory, episodic store 중 어느 층을 뜻하는지 명시해야 한다.
3. 가장 강한 첫 설계는 `fixed E/I LIF + membrane/rate + tapped delays + ridge`다.
4. 가장 안전한 data path는 raw chronology one-pass cache다. shuffled/overlapping full-window state carry는 금지한다.
5. 기존 `Memory Replay`는 이름을 바꾸고, 실제 replay는 timestamped causal bank로 새로 만든다.
6. 기존 suppressive multiplication/IAND gate는 제거하고 prediction-level 또는 additive zero-init gate를 쓴다.
7. reservoir core에는 gradient를 주지 않고 5% throttle도 없앤다.
8. loss는 task coefficient를 1로 유지하며, current-window DCT 복원만으로 memory를 주장하지 않는다.
9. 학습은 ridge → RLS → 필요 시 local plasticity/FORCE 순으로 진행한다.
10. state reset/shuffle/old-slot ablation 없이 long-term memory라는 표현을 사용하지 않는다.

---

## 부록 A. 검색된 연구의 연도별 상세 카탈로그

이 부록은 본문에서 깊게 다루지 않은 문헌까지 포함한다. 수치는 저자 보고값이며 독립 재현값이 아니다. `직접`은 NSMT time-series spiking reservoir에 바로 적용 가능, `인접`은 dynamics/학습/hardware 교훈, `대조`는 reservoir가 아니거나 방법론 차이가 큰 연구를 뜻한다.

### A.1 고전·기반 연구

| 연도 | 연구 | 직접성 | 핵심과 한계 | 신뢰 판단 |
|---|---|---|---|---|
| 2002 | [Maass et al., Real-Time Computing Without Stable States](https://doi.org/10.1162/089976602760407955) | 직접/원전 | LSM과 universal analog fading memory. 영구 episodic memory 논문이 아님 | MIT Press Neural Computation, A |
| 2002 | [Maass & Markram, Synapses as Dynamic Memory Buffers](https://pubmed.ncbi.nlm.nih.gov/12022505/) | 인접 | synaptic dynamics가 최근 spike history를 transient buffer로 보존 | Neural Networks, A |
| 2007 | [Legenstein & Maass, Edge of Chaos](https://doi.org/10.1016/j.neunet.2007.04.017) | 인접 | edge-of-chaos와 계산 성능 관계. 모든 task의 고정 최적점은 아님 | Neural Networks, A |
| 2009 | [Lukoševičius & Jaeger, RC Review](https://doi.org/10.1016/j.cosrev.2009.03.005) | 직접/기반 | fixed recurrent expansion과 readout 학습 체계화 | Computer Science Review, A |
| 2009 | [Sussillo & Abbott, FORCE](https://pmc.ncbi.nlm.nih.gov/articles/PMC2756108/) | 직접/학습 | RLS로 chaotic network output/feedback 적응. full covariance와 폐루프 주의 | Neuron, A |
| 2018 | [Bellec et al., LSNN](https://proceedings.neurips.cc/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html) | 대조/인접 | adaptive threshold slow state. surrogate BPTT이므로 BPTT-free RC 근거 아님 | NeurIPS, A |
| 2018 | [Grigoryeva & Ortega, ESN Universality](https://pubmed.ncbi.nlm.nih.gov/30317134/) | 기반 | fading-memory filter에 대한 universality | Neural Networks, A |
| 2019 | [Voelker et al., Legendre Memory Unit](https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html) | 인접 | finite window를 Legendre basis에 압축. fixed LDN만 떼어 쓰면 RC와 양립 | NeurIPS, A |
| 2019 | [Soures & Kudithipudi, Spiking Reservoir Networks](https://doi.org/10.1109/MSP.2019.2931479) | 직접/review | LSM 설계·local plasticity·hierarchy·hardware review | IEEE SPM, A |

### A.2 2020–2023

| 연도 | 연구 | 직접성 | 핵심과 보고 결과/한계 | 신뢰 판단 |
|---|---|---|---|---|
| 2020 | [Cramer et al., Criticality and Computation](https://www.nature.com/articles/s41467-020-16548-3) | 인접 | BrainScaleS-2 32 LIF, plasticity로 criticality 제어; 복잡 task memory는 개선됐지만 simple task는 악화. “criticality는 항상 최적”이 아님 | Nature Communications, A |
| 2020 | [Bellec et al., e-prop](https://www.nature.com/articles/s41467-020-17236-y) | 인접/학습 | eligibility trace×learning signal의 forward 학습. BPTT-free recurrent credit 후보지만 strict fixed RC는 아님 | Nature Communications, A |
| 2020 | [Moriya et al., Modular Spiking Networks](https://www.jstage.jst.go.jp/article/nolta/11/4/11_590/_article) | 인접 | modular IF reservoir와 synchronous burst; spoken-digit 성능이 random reservoir와 비슷 | IEICE NOLTA, B |
| 2020/22 | [Paassen et al., Reservoir Memory Machines](https://pubmed.ncbi.nlm.nih.gov/34255637/) | 인접/핵심 | ESN+explicit memory가 contractive ESN의 계산 한계를 넘음. spiking/time-series 직접 실험은 아님 | IEEE TNNLS, A |
| 2021 | [Ivanov & Michmizos, NALSM](https://proceedings.neurips.cc/paper_files/paper/2021/hash/d79c8788088c2193f0244d8f1f36d2db-Abstract.html) | 인접 | astrocyte-modulated STDP로 branching factor 근처 자가조직화; image/event classification 중심 | NeurIPS, A |
| 2021 | [Garg et al., Regulated Reservoir for sEMG](https://doi.org/10.1145/3477145.3477267) | 인접 | regulated/CRITICAL reservoir, EMG gesture classification. long-horizon forecast 아님 | ACM ICONS, C+ |
| 2022 | [Balafrej et al., P-CRITICAL](https://doi.org/10.1088/2634-4386/ac6533) | 인접 | local unsupervised criticality regulation과 Loihi; N-MNIST/N-TIDIGITS 중심 | IOP NCE, B+ |
| 2022 | [Dey et al., Efficient TSC](https://doi.org/10.1109/IJCNN55064.2022.9892728) | 직접 | E/I LIF, Gaussian temporal encoding, decaying trace; Ford-A/B, Wafer, Earthquakes. 2,500–5,000 neurons와 작은 Earthquakes train set 주의 | IJCNN/IEEE-INNS venue A-; 결과 중간 |
| 2022 | [Banerjee et al., Multivariate Spike Encoding](https://doi.org/10.1145/3517343.3517349) | 직접/encoding | mutual-information 기반 multivariate encoding이 sensor classification reservoir를 개선했다고 보고 | ACM NICE, C |
| 2022 | [Patiño-Saucedo et al., LSM on SpiNNaker](https://pmc.ncbi.nlm.nih.gov/articles/PMC8964061/) | 인접 | fixed reservoir지만 readout을 SNN-adapted BPTT로 학습. “reservoir 논문=무조건 BPTT-free” 반례 | Frontiers in Neuroscience, B |
| 2022 | [Gaurav et al., Spiking RC on Loihi](https://doi.org/10.1109/SEC54971.2022.00081) | 직접/hardware | LMU 기반 ECG5000와 Loihi deployment | IEEE/ACM SEC, C+ |
| 2023 | [George et al., Online Forecasting](https://www.sciencedirect.com/science/article/pii/S0925231222013479) | 직접 | temporal encoding+feedback LSM+FORCE, 9 series. 저자 기준 평균 최대 8% \(R^2\) 개선; 현대 baseline/protocol 대조 필요 | Neurocomputing, A- |
| 2023 | [Gaurav et al., SLRC/LSNN](https://doi.org/10.3389/fncom.2023.1148284) | 직접 | LDN 기반 5 univariate TSC. SLRC는 least squares, 최고 LSNN은 surrogate BPTT. 50–120 neurons 보고 | Frontiers FCN, B |
| 2023 | [Kholkin et al., LSM Fault Detection Comparison](https://doi.org/10.3390/bdcc7020110) | 직접/응용 | 3 fault datasets에서 LSM accuracy 우세 보고, conventional hardware train/test는 더 느림. 단일 setup | MDPI BDCC, B- |

### A.3 2024

| 연구 | 직접성 | 핵심과 한계 | 신뢰 판단 |
|---|---|---|---|
| [Gast et al., Neural Heterogeneity](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801870/) | 인접 | threshold heterogeneity가 동기화를 억제하고 function-generation capacity를 바꾸지만 homogeneous가 유리한 encoding도 있음 | PNAS, A |
| [Woo et al., LSM Dynamics](https://doi.org/10.1016/j.physa.2023.129334) | 인접 | 400E/100I, avalanche/dynamic range/성능 상관; MNIST라 memory 인과성은 제한 | Physica A, B+ |
| [Peng et al., Spiking Neural P-System RC](https://pubmed.ncbi.nlm.nih.gov/37918270/) | 대조/인접 | 17 TSC datasets, 16 baselines. conventional LIF가 아니라 membrane-computing rules | Neural Networks, A |
| [Biswas et al., Temporal/Spatial Reservoir Ensembles](https://arxiv.org/abs/2411.11414) | 인접 | multi-timescale ensemble, N-MNIST/SHD; 4-page conference와 계산량 증가 | IEEE/ACM ICONS, C |
| [Shi et al., Ghost Reservoir](https://doi.org/10.1109/TCSII.2024.3395415) | hardware | on-chip learning과 weight storage 절감; 5-page hardware brief | IEEE TCAS-II venue A-; 결과 범위 낮음 |
| [Lv et al., Efficient Time-Series Forecasting with SNNs](https://proceedings.mlr.press/v235/lv24d.html) | 대조 | time-series SNN framework의 중요한 comparator지만 reservoir/readout-only 모델은 아님 | ICML, A |
| [Kostyukov & Rostov, Spiking Reservoir NN for TSC](https://doi.org/10.1007/978-3-031-52470-7_25) | 직접 | 7-page CCIS chapter; 세부 evidence가 작음 | Springer CCIS, C |

### A.4 2025

| 연구 | 직접성 | 핵심과 보고 결과/한계 | 신뢰 판단 |
|---|---|---|---|
| [Srinivasan et al., Adaptive E/I Balance](https://www.nature.com/articles/s41467-025-64978-8) | 인접/강함 | local inhibitory adaptation, MC와 nonlinear time-series prediction을 globally tuned RC 대비 최대 130% 개선. rate/sigmoid reservoir라 LIF 전이는 검증 필요 | Nature Communications, A |
| [Cazalets & Dambre, Hebbian Reservoir Reshaping](https://www.nature.com/articles/s41467-025-67137-1) | 인접 | gradient-free Hebbian graph growth; Mackey–Glass/Lorenz/Sunspot 개선은 제한적이고 일부에서 GRU/LSTM 우세 | Nature Communications, A |
| [Ma et al., Stochastic Diffusive Memristor SRC](https://doi.org/10.1002/aelm.202400469) | physical RC | waveform classification와 Mackey–Glass device proof; software LSM 일반화 주의 | Wiley AEM, B+ |
| [Cheong et al., E/I Neuransistor](https://pubmed.ncbi.nlm.nih.gov/40195785/) | physical RC | device E/I dynamics, Hénon/Lorenz; hardware-specific | Advanced Materials, A |
| [Lin et al., Resistive Zero-Shot LSM](https://www.nature.com/articles/s43588-024-00751-z) | hardware | 40 nm in-memory macro, event classification; 저자 기준 큰 training/energy 절감, forecasting/LTM 아님 | Nature Computational Science, A |
| [Wan et al., STDP-LSM Fault Diagnosis](https://doi.org/10.1016/j.eswa.2025.126736) | 직접/응용 | hand-engineered features+STDP LSM+SVM; 높은 fault accuracy이나 raw long sequence가 아님 | ESWA, A- |
| [Liu et al., LSM Gaussian Process](https://pubmed.ncbi.nlm.nih.gov/40784191/) | 직접/readout | Bayesian GP readout으로 uncertainty/robustness; exact GP scaling 주의 | Neural Networks, A |
| [Krenzer & Bogdan, Reinforced LSM](https://doi.org/10.3389/fncom.2025.1569374) | 인접/학습 | WTA+eligibility/R-STDP, SHD 한 dataset에서 작은 이득. BPTT-free local learning 후보 | Frontiers FCN, B |
| [Freddi et al., Mean-Field Criticality](https://www.nature.com/articles/s41598-025-18004-y) | 인접 | LIF critical weight 근사와 code. synthetic complexity 중심 | Scientific Reports, B |
| [Karki et al., Neuromorphic On-Chip RC](https://arxiv.org/abs/2407.20547) | 인접 | Loihi/Lava, Hénon/Mackey–Glass, topology meta-learning | arXiv, D |

### A.5 2026

| 연구 | 직접성 | 핵심과 보고 결과/한계 | 신뢰 판단 |
|---|---|---|---|
| [Jin et al., Time-Delayed SRC](https://doi.org/10.1002/aisy.70421) | 직접/최우선 | fixed spiking reservoir+lag stack+ridge. 한 설정에서 total MC 0.22→2.87; finite buffer이며 매우 신작 | Wiley AIS, B+ / 결과 중간-낮음 |
| [Gaurav et al., Deep Legendre-SNN Benchmark](https://doi.org/10.1109/TETCI.2025.3605627) | 인접/대조 | 102 UCR/UEA datasets; fixed LDN이나 spiking hidden은 BPTT. test 최대 선택 편향을 보수적으로 해석 | IEEE TETCI, A- |
| [Oh et al., Reservoir Connectivity](https://doi.org/10.1016/j.neucom.2025.132037) | 직접/dynamics | spike multiplication factor, controlled chaos; spectral radius 단독 진단의 한계 | Neurocomputing, A- |
| [Rosati et al., Entropy/Inhibition/Memory](https://doi.org/10.3390/e28070784) | 직접/dynamics | balanced E/I LIF phase diagram, sparse input와 asynchronous regime. downstream task 없음·매우 신작 | Entropy, B |
| [Goto et al., E/I Spiking Reservoir](https://www.jstage.jst.go.jp/article/jsp/30/4/30_97/_pdf/-char/en) | 직접 | Lorenz+FORCE, autonomous stability의 E/I 민감도; 4 pages | J-STAGE selected paper, C |
| [Freddi et al., Robust Spiking Reservoirs](https://arxiv.org/abs/2604.06395) | 인접 | critical point 대신 robustness interval | arXiv, D |
| [Sun et al., Dual Memory Pathways](https://www.nature.com/articles/s42256-026-01255-3) | 대조/핵심 인접 | fast spike+compact slow state와 hardware co-design. reservoir가 아니며 BPTT 사용 | Nature Machine Intelligence, A |
| [Fernández-Khatiboun et al., SPIRE](https://doi.org/10.1109/TBCAS.2026.3668521) | hardware | 28 nm multi-reservoir accelerator와 online adaptation; 장기 의존 해결 증거는 별도 | IEEE TBCAS, A- |
| [Graded-like Photonic Spiking RC](https://doi.org/10.1021/acsphotonics.5c02170) | physical RC | laser reservoir의 MNIST/bifurcation/Mackey–Glass; digital LSM과 직접 같지 않음 | ACS Photonics, A- |
| [All-Optical Spiking Microring RC](https://doi.org/10.1364/PRJ.558405) | physical RC | 광학 spiking reservoir 구현; NSMT software 설계에는 간접 | Photonics Research, A- |

### A.6 의도적으로 낮은 우선순위를 준 자료

- Procedia의 rainfall “LSM” 연구는 BPTT/MSE를 사용하고 canonical fixed-reservoir 정의와 어긋나며 행사별 review 편차가 커 핵심 근거에서 제외한다.
- 2026년 저신뢰 일반 review와 출판/색인 절차가 불명확한 journal은 인용하지 않는다.
- patent는 기술 소유권 자료이지 성능 재현 근거가 아니다.
- ResearchGate는 원고 접근 경로일 뿐 신뢰 판정은 DOI와 공식 publisher page로 교차한다.
- 검색 시점에 위 핵심 논문의 철회 또는 명백한 predatory venue 정황은 확인하지 못했다. 이는 향후 무결성을 보증하지 않는다.

---

## 부록 B. 후속 세션용 결정 기록

### 확정

- 최초 Neo는 reservoir도 cross-window memory도 아니다.
- 기존 `Memory Replay`는 replay가 아니다.
- pure LSM은 fading memory다.
- raw chronology one-pass cache가 기본 state implementation이다.
- Neo core는 fixed/no-grad, readout은 ridge가 첫 baseline이다.
- suppressive gate는 사용하지 않는다.
- task loss coefficient는 1로 유지한다.
- long-term 주장은 reset/shuffle/old-slot causal controls 뒤에만 한다.

### 아직 실험으로 결정할 것

- vanilla LIF 대 ALIF
- tapped delay 대 fixed LDN
- per-channel 대 joint multivariate liquid
- reservoir size/density/E-I/tau
- stateful feature가 Hippo에 실제 추가 정보를 주는지
- external bank가 필요한지
- RLS forgetting factor와 covariance approximation

### 다음 세션의 첫 작업

1. 새 branch/worktree와 최초 forecasting baseline 위치를 정한다.
2. `stream_id`, absolute timestamp, patch endpoint metadata contract를 test로 고정한다.
3. fixed scaler와 no-grad causal LIF scanner를 구현한다.
4. cache invariance test를 통과시킨다.
5. Hippo와 결합하기 전에 reservoir-only ridge와 MC curve를 측정한다.
