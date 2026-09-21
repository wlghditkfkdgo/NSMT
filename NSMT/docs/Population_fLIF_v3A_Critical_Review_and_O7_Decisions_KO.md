# Population f-LIF v3-A 비판적 검토 및 O7 판정 기준 권고서

## 코딩·연구 에이전트 전달용: 모델 정의, 이론적 타당성, 선행연구, 수치 진단, 수정 우선순위

**작성일:** 2026-09-21  
**검토 대상:** `IDEA LOG — Population f-LIF with Shared History Selection`, v3-A, 2026-09-21 갱신본  
**첨부 원본 파일:** `붙여넣은 마크다운(1)(1).md`  
**문서 성격:** 직전 비판적 검토를 상세화한 검토·의사결정 문서. 실행 완료 보고서나 수정된 최종 모델의 승인서가 아니다.  
**권고 상태:** 연구 방향은 유지. 수식·용어·검증 계약의 충돌을 해소한 뒤 구현 계약을 확정한다.

> **핵심 결론**
>
> 리셋 없는 fractional 가지와 하나의 spiking soma를 결합하는 모델 계열은 연구할 근거가 충분하다. 그러나 현재 v3-A 문서에서는 **정확한 sparsity, 커널 질량 보존, integer-order 환원, 상태 안정성**이 동시에 성립하는 것처럼 설명된 부분을 수정해야 한다. 또한 O7은 **정답 위치를 가리키는 것**, **실제 적분에서 그 위치에 비중을 주는 것**, **그 내용을 사용해서 예측을 개선하는 것**을 분리해 평가해야 한다.
>
> 가장 중요한 질문은 “선택 분포가 희소한가?”가 아니라 **“선택된 과거 dynamics가 현재 상태를 안정적이고 유용하게 바꾸는가?”**이다.

---

## 읽는 순서

| 필요한 정보 | 해당 절 |
|---|---|
| 전체 판단과 승인 범위 | 0–1절 |
| 현재 v3-A의 정확한 모델 정의 | 2절 |
| 가까운 선행연구와 신규성의 위치 | 3절 |
| 반드시 수정할 수식·이론 문제 | 4–10절 |
| 회상 과제, oracle, 평가량의 재정의 | 11–14절 |
| O7 네 항목에 대한 구체적인 답변 | 15절 |
| 비교군·forecasting·통계 | 16–17절 |
| 수치 게이트와 진행 순서 | 18–19절 |
| 코딩 에이전트가 확인받아야 할 결정 | 20절 |
| 결과 해석과 최종 전달 요약 | 21절 |
| 출처와 독립 수치 계산 조건 | 참고문헌·부록 |

---

## 0. 이 문서가 승인하는 것과 승인하지 않는 것

### 0.1. 유지할 연구 방향

현재 v3-A의 중심은 다음과 같다.

1. 하나의 논리 뉴런에는 같은 전류를 받는 여러 fractional 가지가 있다.
2. 가지들은 서로 다른 시간척도로 상태를 형성하지만 직접 발화하거나 reset되지 않는다.
3. population 전체 문맥으로 과거 사건의 dynamics를 공동 평가한다.
4. 선택 또는 재분배된 과거 dynamics가 가지의 fractional integration 내부에 들어간다.
5. 하나의 소마가 가지 출력을 합쳐 발화하고, reset도 소마에서만 수행한다.

이 구조를 이전 v2의 `ordinary LIF + additive retrieved membrane`으로 되돌리지 않는다. 동시에, 현재 v3-A를 **각 구성원이 독립적으로 발화하는 원문 f-LIF의 복제 population**이라고 설명하지도 않는다. 현재 원본 자체가 공유 소마 구조로 모델을 변경했음을 명시하고 있다. [S1, §3.1–3.6]

### 0.2. 그대로 승인할 수 없는 주장

| 원본에서 읽힐 수 있는 주장 | 검토 판단 |
|---|---|
| sparsemax에서 0이면 해당 과거가 최종 적분에서 완전히 제거된다 | 일반적으로 틀림. dense residual `(1−η)`가 남는다 |
| 최종 가지별 커널 질량은 항상 보존된다 | `π_k`를 곱하면 일반적으로 틀림 |
| `α=1, η=0`이면 선언한 Euler leaky integrator로 환원된다 | `g=0` 또는 `π≡1` 조건도 필요 |
| 커널 질량을 보존하므로 상태 증폭 문제는 해결된다 | 성립하지 않음. signed dynamics의 재분배는 큰 증폭을 만들 수 있다 |
| `η>0`이면 모든 상태에서 구성원 간 Jacobian이 비영이다 | 성립하지 않음. 합법적인 퇴화·포화·단일 support 구간이 있다 |
| `η=0`이면 원래 fractional 계수 행렬을 입력에 한 번 곱하면 된다 | 누설 feedback을 반영한 유효 연산자여야 한다 |
| 3-key 회상 과제는 명시적인 과거 bank 조회 없이 해결할 수 없다 | 성립하지 않음. 압축된 recurrent 상태로 해결 가능한 구성적 대안이 있다 |
| oracle은 해당 구조에서 얻을 수 있는 최선의 오차다 | 성립하지 않음. 특정한 특권적 정책이지 최적해가 아니다 |
| O7의 네 숫자만 결정하면 나머지는 바로 실행 가능하다 | 수식·시간 정렬·안정성·평가량 정의를 먼저 확정해야 한다 |

### 0.3. 권고는 자동 모델 변경이 아니다

이 문서는 `g=0`을 첫 기전 검증에서 고정하는 방안과 O7 수정안을 **권고**한다. 사용자 또는 연구 책임자가 확정한 것처럼 원본을 조용히 덮어쓰면 안 된다. 승인된 수정은 별도 revision과 결정 기록에 남기고, 원본 v3-A와 수정안의 차이를 보존한다.

“문서를 정리해 전달한다”는 동의와 “모든 신규 수식·평가 규칙을 최종 승인했다”는 동의를 구분한다. 특히 소마 출력 시점, reset 역전파 규약, `η`의 안전 범위는 이 검토만으로 자동 확정되지 않는다.

---

## 1. 근거와 검토 범위

### 1.1. 근거 종류

| 표기 | 의미 | 사용 방법 |
|---|---|---|
| **[S1]** | 사용자가 제공한 최종 아이디어 문서 | 현재 모델 정의·설정·원본의 보고 내용 |
| **[S2]** | 사용자가 제공한 f-SNN 논문 PDF | 원문 f-LIF와의 관계 및 원문 이론의 적용 범위 |
| **[R번호]** | 공개된 논문·공식 출판처·저자 원문 | 선행연구가 실제로 제시한 모델과 결과 |
| **[D]** | S1의 수식에서 직접 도출한 대수적 결론 | 질량 보존, 환원, Jacobian, fast-path 분석 |
| **[N]** | 이 검토를 정리하면서 독립적으로 다시 계산한 작은 수치 진단 | 계산 조건과 범위를 명시하며 학습 실험과 구분 |
| **[P]** | 검토자가 제시하는 수정·실험·판정 권고 | 확정된 원본 사실이나 검증된 최적 설정으로 제시하지 않음 |

문서의 대수적 도출과 수치 예시는 선행연구의 결론으로 인용하지 않는다. 반대로 선행연구가 유사한 재료를 사용했다는 이유만으로 현재 모델의 안정성이나 forecasting 개선이 입증됐다고 하지 않는다.

### 1.2. 독립 검증하지 않은 사항

S1에 기록된 기존 v2 성능, 27개 reset 조합 진단, 서버 실행 상태, 이전 CPU/GPU gradcheck, 초기 발화율, golden 자료는 **S1이 보고하는 결과**다. 이번 작업에서 그 저장소·checkpoint·학습을 실행해 재현한 것이 아니다.

`docs/PROJECT_LOG.md`와 `Population_fLIF_v3_prereg_KO.md`는 S1이 가리키는 관련 문서다. 그 전체 내용이 이번 검토에서 별도로 제공·열람된 것은 아니므로, S1에 없는 결정이 그 문서에 존재한다고 추정하지 않는다.

### 1.3. 문헌 검색의 범위

fractional neuron, dendrite–soma 모델, 다중시간척도 SNN, fractional/spiking SSM, variable-order FDE, temporal interaction, attention 해석, associative recall을 조사했다. 공개 원문에 접근하여 확인한 근거를 아래 참고문헌에 연결했다.

**전수 검색이나 신규성의 법적·절대적 보증은 아니다.** “검토한 문헌에서 동일한 형태를 확인하지 못했다”와 “세계 최초임이 입증되었다”는 다르다. 2026년 preprint는 정식 출판 논문과 구분한다.

---

## 2. 검토 기준인 v3-A 모델을 먼저 정확히 복원한다

이 절은 원본을 재정의하는 것이 아니라 **S1의 계산을 명확한 기호로 다시 적는 부분**이다. 뒤에서 제시하는 수정 권고와 혼동하지 않는다.

### 2.1. 논리 뉴런과 시간 인덱스

하나의 논리 뉴런에는 `K=4`개의 가지와 소마 하나가 있다. 가지별 시간상수는 `τ=[2,4,8,16]`이다. 같은 입력 전류 `I_n`을 받으며, 각 가지의 상태는 `u_{n,k}`다.

여기서 `n`은 실제 시계열 patch의 순서다. 별도의 반복 simulation 축을 만들지 않는다. 초기 설정은 입력 길이 336, patch 크기 8, 총 42개의 모델 시간 step이다. 서로 다른 입력 sample, 원래 변수, 논리 뉴런의 상태 bank는 분리한다. [S1, §3.1–3.4]

### 2.2. 가지의 dynamics와 fractional 적분

모델 시간격자를 `h=1`로 고정하고 초기 가지 상태를 0으로 두면,

$$
f_{n,k}=\frac{I_n-u_{n,k}}{\tau_k},
$$

$$
b_d^{(\alpha)}=\frac{(d+1)^\alpha-d^\alpha}{\Gamma(\alpha+1)},\qquad d\ge0,
$$

$$
u_{n+1,k}=b_0f_{n,k}+\sum_{j=0}^{n-1}b_{n-j}\rho_{n,j,k}f_{j,k}.
$$

현재 항의 배율은 `ρ_{n,n,k}=1`이다. 과거 선택은 `j<n`만 대상으로 한다. `n=0`에는 과거 후보가 없으므로 확률 정규화나 과거 질량 계산을 호출할 필요가 없다. [S1, §3.2(a)]

중요한 구분은 다음과 같다.

- `u_j`: 그 시점까지의 정보가 반영된 가지 상태.
- `f_j`: 그 시점의 입력과 누설을 포함한 dynamics 평가값.
- `b_d`: dynamics 이력의 fractional 적분 계수.
- `ρ`: 해당 이력항에 대한 내용 의존적 배율.

`f_j`는 원시 입력값도 아니고 항상 양수인 증거량도 아니다. 상태보다 입력이 작으면 음수가 된다.

### 2.3. Population-wide selector

$$
\xi_n=[\mathbf u_n;I_n]\in\mathbb R^{K+1},
$$

$$
e_{n,j}=-\frac{\|W_Q\xi_n-W_K\xi_j\|_2^2}{d_q\vartheta},
$$

$$
p_{n,:}=\operatorname{softmax}(e_{n,:})
\quad\text{또는}\quad
\operatorname{sparsemax}(e_{n,:}).
$$

과거에 대한 기본 커널 질량을

$$
B_n=\sum_{j<n}b_{n-j}
$$

로 정의하고,

$$
\widetilde\rho_{n,j}
=\frac{B_np_{n,j}}{\sum_{\ell<n}b_{n-\ell}p_{n,\ell}}
$$

로 질량을 재분배한다. 원본의 **최종** 배율은

$$
\rho_{n,j,k}
=\left[(1-\eta)+\eta\widetilde\rho_{n,j}\right]\pi_k(n-j),
$$

$$
\pi_k(d)=\exp(-dg/\tau_k),\qquad
\eta=\sigma(\widehat\eta),\quad g=\operatorname{softplus}(\widehat g).
$$

원본 초기값은 `η̂=−4`, `ĝ=−5`이며,

$$
\eta_0\approx0.01798621,\qquad g_0\approx0.00671535.
$$

이다. `η`와 `g`는 층당 학습 가능한 스칼라로 기술되어 있다. [S1, §3.2(b), §3.4]

`p`와 `\widetildeρ`는 population 구성원 전체가 공유한다. 그러나 `π_k`까지 곱한 최종 `ρ_{n,j,k}`는 가지별로 다르다. 이 구분은 tensor 계약과 진단 이름에도 반영되어야 한다.

### 2.4. 소마

원본의 소마는

$$
a_n=\sum_kw_ku_{n,k},
$$

$$
v_n^{\mathrm{charge}}
=v_{n-1}+\frac{a_n-v_{n-1}}{\tau_s},
$$

$$
s_n=H(v_n^{\mathrm{charge}}-\theta),\qquad
v_n=v_n^{\mathrm{charge}}-\theta s_n
$$

로 정의된다. 출력은 가지별 K개 spike가 아니라 논리 뉴런당 spike 하나다. `τ_s=2`, `θ=1`, `w`는 `[D,K]`의 학습 가중치이며 초기값은 1이다. [S1, §3.2(c)–(d), §3.4]

`w`는 정규화된 확률이나 양의 가중치라고 명시되어 있지 않다. 따라서 자동으로 평균 pooling, softmax 혼합, 비음수 혼합으로 변경하면 안 된다. 각 대안은 별도 설계다.

surrogate의 backward 식은 spike 기호와 혼동하지 않도록 scale을 `c_s`로 쓰면,

$$
g_{c_s}(x)=\frac{c_s/2}{1+\left(\frac{\pi}{2}c_sx\right)^2},\qquad c_s=5.
$$

이는 원본이 선택한 backward 규약이다. forward spike는 여전히 hard threshold다. 소마 reset 경로에서 spike를 detach하는지 여부는 surrogate 식만으로 결정되지 않으므로 실행 계약에서 별도로 고정해야 한다. [S1, §3.4; P]

### 2.5. 원문 f-LIF 및 이전 prototype과의 관계

v3-A의 가지는 발화하지 않으므로 그 자체를 완전한 f-LIF spiking neuron이라고 부르면 부정확하다. 더 정확한 기술은 다음이다.

> **Population 문맥으로 dynamics 이력을 조절하는 fractional dendritic/synaptic branches와 ordinary LIF soma의 결합.**

`K=1`의 환원 역시 “원본 scalar f-LIF와 완전히 동일”이 아니라, **scalar fractional branch + soma**다. 원문 및 공개 코드와의 비교는 원본 재현을 위한 별도 reference이며 v3-A의 전체 출력과 무조건 같은 spike train을 요구할 수 없다. [S1, §3.5–3.6; S2]

---

## 3. 선행연구가 뒷받침하는 것과 뒷받침하지 않는 것

### 3.1. 관련 연구 지도

| 연구 | 확인한 관련 내용 | 이 모델에 주는 의미와 한계 |
|---|---|---|
| **f-SNN / Ge et al.** [R1, S2] | Fractional neuronal dynamics와 학습 가능한 SNN framework | 출발점이지만 선택된 커널·별도 소마의 이론을 보장하지 않는다 |
| **Teka et al., 2014** [R2] | Fractional LIF, voltage history, spike adaptation | 기억과 reset의 결합이 중요한 모델링 선택임을 보여준다 |
| **DH-SNN, Nature Communications 2024** [R3] | Dendritic branch의 다양한 timing factor와 soma를 갖는 multi-compartment SNN | “기억 가지 + soma”의 구조적 선례. 사건별 fractional 재분배의 효용은 별도 |
| **TC-LIF, AAAI 2024** [R4] | 두 compartment를 이용한 장기 시퀀스 처리 | 직접적인 compartment dynamics와 비교할 근거 |
| **TS-LIF, ICLR 2025** [R5] | Temporal segment·dual-compartment 뉴런을 시계열 예측에 적용 | Forecasting에 직접 관련된 중요한 비교 연구 |
| **LongSpike, 2026 preprint** [R6] | Fractional SSM, sum-of-exponentials 근사, spiking 출력의 결합 | Fractional 처리 뒤 LIF를 두는 구조 자체는 새롭지 않다 |
| **NvoFDE, AAAI 2025** [R7] | Hidden feature에 의존하는 variable-order fractional dynamics | 내용 의존 fractional memory라는 넓은 주장의 선례 |
| **PSN, NeurIPS 2023** [R8] | 학습된 temporal weighting과 masked/sliding spiking neuron | 시간 mask 또는 temporal weight만으로 신규성을 주장할 수 없다 |
| **TIM, 2024 공개 논문** [R9] | Convolution 기반 temporal interaction 모듈 | 과거 처리 개선이 단순한 temporal feature extraction 때문인지 분리해야 한다 |
| **Mamba** [R10] | 입력 의존 state-space selection | 압축된 상태의 선택적 갱신과 과거 bank 재조회는 다른 비용·표현 선택이다 |
| **FLAMES, 2025 preprint** [R11] | Inter-spike interval에 따른 memory retention을 갖는 spiking–SSM 결합 | 적응적 기억 보존의 관련 선례. 현재 모델의 성능 보증은 아니다 |
| **Zoology, ICLR 2024** [R12] | Multi-query associative recall을 이용한 효율적 sequence model 비교 | 쉬운 recall 성공만으로 명시적 history access 필요성을 증명할 수 없다 |
| **Sparsemax, ICML 2016** [R13] | 확률 simplex 사영과 정확한 0, support 기반 Jacobian | p의 sparsity 근거일 뿐 residual을 포함한 전체 연산의 sparsity 근거는 아니다 |
| **Tempered fractional calculus** [R14] | Power-law kernel에 exponential tempering을 결합하는 수학적 틀 | `π_k`가 남으면 pure power-law와 구분해야 한다 |
| **Attention 해석 논쟁** [R15–R16] | Weight만으로 faithfulness를 판단하는 데 한계가 있으며 추가 검증이 중요 | Weight 분석과 예측 개입을 함께 평가하되 인과적 과장을 피한다 |

### 3.2. LongSpike를 비교할 때의 정확한 범위

LongSpike 공개 원문의 §4.2는 f-SSM 출력을 LIF에 주입하는 전류로 연결한다. 또한 별도의 SDN 기반 spike 계산 경로를 논의하므로, 그 전체 모델이 현재 v3-A의 순차 soma와 동일하다고 해서는 안 된다. §5.3에서 SOE 항 수 `M=2`를 사용한다고 보고하며, Table 1의 LRA Text는 SpikingSSM 80.41%, LongSpike 88.19%다. 이 수치는 원문 PDF의 표와 해당 설명을 다시 확인했다. [R6]

이 결과가 보여주는 것은 **그 논문의 조건에서 fractional/spiking SSM이 경쟁력이 있었다**는 사실이다. 우리의 K=4 고정 누설 bank가 그 SOE 근사나 전체 모델을 재현한다는 뜻은 아니다.

### 3.3. 신규성의 가장 방어 가능한 위치

현재 검토 범위에서 주장할 후보는 재료의 목록이 아니라 다음 질문이다.

> **같은 population의 현재 문맥으로 과거 dynamics의 사건별 기여를 재분배하면, 고정 fractional dynamics 및 유한 지수 필터/SSM보다 관련 정보를 더 효과적으로 활용하는가?**

이를 뒷받침하려면 fractional order, 명시적인 history 접근, population 다양성, 선택 정책, soma readout의 효과가 구분되어야 한다.

아래 주장은 아직 성립하지 않는다.

- “이런 재료를 결합한 최초 모델이다.”
- “Fractional과 소마를 함께 쓰므로 장기 기억이 자동으로 더 좋다.”
- “원문 f-LIF의 robustness 정리가 그대로 적용된다.”
- “Sparsemax를 썼으므로 계산량·에너지가 감소한다.”
- “선택 weight가 높으므로 실제 정답 원인을 찾았다.”

---

## 4. 문제 A — sparse 분포와 최종 적분의 sparsity가 다르다

### 4.1. 직접 계산

어떤 과거 j에서 `p_{n,j}=0`이라고 하자. 그러면

$$
\widetilde\rho_{n,j}=0
$$

이지만,

$$
\boxed{\rho_{n,j,k}=(1-\eta)\pi_k(n-j).}
$$

유한한 sigmoid 파라미터에서 `0<η<1`이고 유한한 lag에서 `π_k>0`이므로, 정확한 실수 연산에서는 최종 계수가 양수다. [D: S1 §3.2(b)]

따라서 **p의 support에서 제외된 과거도 full-history residual 경로로 계속 영향을 준다.** 부동소수점 underflow 때문에 0이 나오는 경우는 의도된 내용 기반 선택과 별개다.

원본 예시의 `p=[0,0.4995,0.5005,0]`에서 `η=0.5`를 적용하면 대략 `ρ=[0.5,1.527,1.529,0.5]`가 되는 것 자체가 이를 보여준다. 그 예시의 첫째·넷째 slot은 선택 분포에서는 제외되지만 최종 적분에서 완전히 제외되는 것은 아니다. [S1, §4]

### 4.2. 모델을 폐기할 문제인가?

아니다. Full-history를 유지하면서 일부 내용을 재분배하는 residual 설계는 일관되게 정의할 수 있다. 다만 정확한 이름이 필요하다.

> **주 모델의 권장 설명:** dense fractional history 위에 sparse content redistribution을 결합한다.

> **별도 hard-mask 조건:** `η=1`을 명시적으로 고정했을 때 p의 0이 최종 계수의 0으로 연결된다.

`sigmoid(η̂)`에서 η̂를 크게 만드는 것과 정확히 `η=1`인 비교 조건은 다르다. hard-mask 조건은 경계값을 어떻게 정의했는지 기록해야 한다. [P]

### 4.3. 코딩 에이전트의 검증 책임

다음 진단을 각각 별도로 남긴다.

| 진단 | 의미 |
|---|---|
| `support(p)` | 선택기 확률 분포에서 살아 있는 과거 |
| `support(ρ̃)` | 재분배 경로의 support |
| `support(ρ)` | residual과 tempering을 포함한 배율의 support |
| `support(bρ)` | 실제 과거 적분 계수의 support |

주 모델에서 앞의 두 support는 sparse하고 뒤의 두 support는 dense일 수 있다. 그것은 정의에 따른 정상 동작이다. G8이 이 네 수준을 혼동해서는 안 된다.

---

## 5. 문제 B — π를 곱한 뒤에는 최종 질량 보존이 깨진다

### 5.1. 보존되는 부분

$$
\sum_{j<n}b_{n-j}\widetilde\rho_{n,j}=B_n.
$$

따라서 `r_{n,j}=(1−η)+ηρ̃_{n,j}`라고 하면,

$$
\sum_{j<n}b_{n-j}r_{n,j}
=(1-\eta)B_n+\eta B_n=B_n.
$$

즉 **π를 곱하기 전까지는** 과거 커널 계수 질량이 보존된다. 현재 항 `b_0f_n`은 이 계산에서 제외되며 원래대로 유지된다. [D]

### 5.2. 최종 질량비

원본의 최종 배율을 사용하면,

$$
\kappa_{n,k}
=\frac{\sum_{j<n}b_{n-j}r_{n,j}\exp[-(n-j)g/\tau_k]}{B_n}.
$$

`g>0`, `n>0`, `τ_k>0`이면 모든 과거 lag가 양수이므로,

$$
\boxed{0<\kappa_{n,k}<1.}
$$

`η=1`로 sparse support만 남아도 질량이 존재하는 모든 과거 lag에서 지수항이 1보다 작으므로 결론은 같다. [D]

따라서 최종 모델의 G8에 **무조건 κ=1**을 요구하면 문서의 수식과 충돌한다.

### 5.3. 원본 초기값의 수치 예시

`n=41`, `α=0.7`, `h=1`, 균일 p, `g=softplus(−5)`에서 계산했다. 균일 p이면 `ρ̃=1`이므로 결과는 η와 무관하다. [N]

| 가지 τ | 최종 과거 커널 질량비 κ |
|---:|---:|
| 2 | 0.9417210670 |
| 4 | 0.9702241956 |
| 8 | 0.9849489556 |
| 16 | 0.9924331728 |

이는 학습되지 않은 계수 계산이다. 초기값부터 이미 κ=1과 다르며, g가 학습되면 달라질 수 있다.

### 5.4. 장기 꼬리의 의미

$$
b_d\pi_k(d)\sim\frac{1}{\Gamma(\alpha)}d^{\alpha-1}\exp(-dg/\tau_k).
$$

따라서 `g>0`이면 장시간에 pure power-law tail을 유지하는 것이 아니라 exponential tempering이 붙는다. Tempered fractional calculus라는 관련 수학적 모델 계열은 존재한다. 다만 위의 이산 계수 곱이 특정 표준 tempered Caputo 수치해법과 정확히 동일하다고 추가로 주장하려면 별도 유도가 필요하다. [D; R14]

### 5.5. 필요한 결정

| 선택 | 결과 | 이 문서의 권고 |
|---|---|---|
| **A. 첫 기전 검증에서 g=0, π≡1 고정** | 질량 보존·중립 극한·공유 배율이 명확해짐 | **권고** |
| B. π를 유지 | 가지별 시간 감쇠가 추가된 tempered 변형 | 별도 비교 모델로 명시 |
| C. π 이후 다시 정규화 | 기준 커널·보존할 질량·공유성의 의미가 다시 바뀜 | 이 문서에서 자동 확정하지 않음 |

A를 채택해도 근거 없이 최적이라고 표현하지 않는다. 목적은 먼저 내용 기반 재분배 효과를 시간 감쇠와 분리하는 것이다. B를 유지한다면 최종 κ를 가지별로 기록하고 “질량 보존 모델”이라는 무조건적 설명을 제거한다.

---

## 6. 문제 C — integer-order 환원에는 g=0이 추가로 필요하다

### 6.1. g가 남은 경우의 정확한 recurrence

`α=1`, `η=0`에서 `r_k=exp(−g/τ_k)`라고 두면,

$$
u_{n+1,k}=\sum_{j=0}^{n}r_k^{n-j}f_{j,k}.
$$

따라서

$$
u_{n+1,k}=r_ku_{n,k}+f_{n,k}
=\left(r_k-\frac1{\tau_k}\right)u_{n,k}+\frac{I_n}{\tau_k}.
$$

이는 일반적인 Euler leaky integrator

$$
u_{n+1,k}=\left(1-\frac1{\tau_k}\right)u_{n,k}+\frac{I_n}{\tau_k}
$$

와 다르다. 두 식이 같으려면 `g=0`이 필요하다. [D]

### 6.2. 수정할 환원 표

| 조건 | 올바른 결과 |
|---|---|
| `η=0, g=0` | 선택 없는 full-history fractional 가지 |
| `α=1, η=0, g=0` | 같은 `/τ` 규약의 Euler leaky-integrator 가지 |
| `α=1, η=0, g>0` | 추가 exponential factor가 들어간 수정 leaky recurrence |
| `K=1` | Scalar fractional branch와 별도 spiking soma |
| `α=1`, 내용 의존 selection 유지 | Integer-order coefficient를 쓰되 명시적 history modulation이 남은 모델 |

S1의 “측정 오차 0.0”은 그 테스트가 어떤 g와 어떤 실행 분기를 사용했는지 확인해야 한다. **보고된 수치만으로 현재 전체 식의 환원이 증명되었다고 하지 않는다.** [S1, §3.5]

### 6.3. α가 바꾸는 것은 꼬리 모양만이 아니다

`/τ` 규약을 고정해도 α는 `Γ(α+1)` 및 유한 구간 총질량을 바꾼다.

$$
\sum_{d=0}^{T-1}b_d=\frac{T^\alpha}{\Gamma(\alpha+1)}.
$$

`T=42, α=0.7`에서는 약 15.0620이고 `α=1`에서는 42다. 따라서 α 실험은 커널 모양·누적 크기·feedback 응답이 함께 바뀌는 실험이다. 이를 단일 “기억 길이” 축으로만 해석하지 않는다. [D, N]

---

## 7. 문제 D — 질량 보존은 상태 안정성 보장이 아니다

### 7.1. 왜 signed dynamics가 중요한가

$$
f_{j,k}=\frac{I_j-u_{j,k}}{\tau_k}
$$

에는 입력과 누설의 음의 feedback이 동시에 있다. 전체 계수 질량을 고정해도 서로 상쇄하던 항의 상대 비중을 바꾸면 상태 크기와 진동이 크게 달라질 수 있다.

$$
\text{계수 총질량 보존}
\not\Rightarrow
\text{합산값 크기 보존}
\not\Rightarrow
\text{시간에 대한 안정성}.
$$

리셋 항을 memory에서 제거한 것은 **직접적인 reset-history 재가중**을 제거한다. 그러나 “순수한 양의 증거만 선택하므로 안정적”이라는 결론은 나오지 않는다.

권장 문구는 다음과 같다.

> 선택자는 직접적인 reset 항이 없는 subthreshold dynamics를 조절한다. 입력과 누설 feedback을 함께 조절하므로 상태 안정성과 정보 해석은 별도로 검사한다.

### 7.2. 독립적인 작은 수치 진단

다음 조건의 가지 적분만 계산했다. 학습, Q/K 최적화, 소마 또는 전체 SNN은 실행하지 않았다. [N]

| 항목 | 값 |
|---|---|
| 초기 상태 | u₀=0 |
| 길이 | 42개의 입력 평가, u₁부터 u₄₂까지 계산 |
| α, τ, h | 0.7, 2, 1 |
| 입력 | I₀=2, 이후 Iₙ=0 |
| 선택 정책 | 가장 최근의 과거 한 항에 p=1, 다른 과거 p=0 |
| 현재 항 | b₀fₙ 그대로 유지 |
| 정규화 | 원본의 Bp/Σbp 및 residual 혼합 |
| tempering | 우선 g=0 |

| η | max₀≤n≤42 \|uₙ\| |
|---:|---:|
| 0 | 1.1005474055 |
| sigmoid(−4) ≈ 0.01798621 | 1.1005474055 |
| 0.3 | 26.5003285353 |
| 0.5 | 397,853.4408079585 |

직전 검토의 반올림 수치를 같은 조건으로 다시 확인했다. 추가로 원본의 초기 `g=softplus(−5)`를 유지해 계산해도 η=0.3에서 약 24.3286, η=0.5에서 약 375,478.1895의 최대 절댓값이 나왔다. 이 추가 계산도 학습 결과는 아니다.

### 7.3. 이 진단이 증명하는 범위

**확인한 것:** 현재 coefficient 규칙이 허용하는 한 정책과 finite horizon에서, 작은 유계 입력에 대해 매우 큰 상태 증폭이 일어날 수 있다. 초기 η가 작다는 사실이나 κ=1이라는 사실만으로 안전하다고 볼 수 없다.

**확인하지 않은 것:** 모든 학습 경로가 불안정하다는 주장, 모든 sparsemax selector가 같은 상태에 도달한다는 주장, 무한 시간에서의 발산 정리, v3-A의 실제 task 성능 실패.

“최근 정책”은 원본이 제시한 비교군이므로 실제 비교 과정에서도 관련 있는 진단이다. 이 비교군이 불안정하면 단순히 그 점수를 제외하지 말고 구조적 원인을 기록해야 한다.

### 7.4. Phase C의 추가 필수 진단

다음은 새 권고이며 아직 실행된 검증이 아니다. [P]

- η를 초기값뿐 아니라 더 큰 값과 hard-mask 경계까지 변화시킨다.
- Full, recent, 먼 단일 slot, random support, oracle-policy, learned-policy를 분리한다.
- 길이 42를 기준으로 더 긴 bank에서도 상태·gradient의 크기와 진동을 검사한다.
- Pulse–silence, constant input, 부호가 바뀌는 입력, 잡음, 반복 사건을 사용한다.
- 가지 상태, f의 양·음 기여, 실제 bρ 계수, max coefficient, 소마 drive와 발화율을 따로 본다.
- 동일 입력에서 g=0과 g>0의 차이를 기록하되 g로 불안정성을 감춘 뒤 선택의 이득이라고 해석하지 않는다.

η의 상한, coefficient cap, support 제약을 새로 넣는 경우 모델 변경으로 기록한다. 단순 clipping은 질량 보존과 neutral limit을 바꿀 수 있으므로 후속 수식 검사가 필요하다. 이 문서는 아직 **검증된 보편적 안전 상한**을 제시하지 않는다.

### 7.5. 선택 집중과 이력 길이의 관계

한 과거 j*에 모든 p가 집중되면,

$$
\widetilde\rho_{n,j^*}=\frac{B_n}{b_{n-j^*}},\qquad
b_{n-j^*}\widetilde\rho_{n,j^*}=B_n.
$$

즉 그 사건 하나가 전체 과거 커널 질량을 넘겨받는다. Bₙ은 이력 길이와 함께 증가하므로, 고정된 짧은 길이에서의 성공을 더 긴 bank의 안정성으로 확대하지 않는다. 이 점은 소수 slot을 읽는 설계의 이점과 안전성 사이의 실제 trade-off다. [D]

---

## 8. 문제 E — fast path는 raw fractional kernel이 아니라 유효 선형 연산자여야 한다

### 8.1. 누설 feedback을 반영해야 한다

선택을 끄고 g와 τ가 고정되어 있으면 가지는 선형 시간불변계다. 그러나

$$
u_{n+1}=\frac1\tau\sum_{j\le n}b_{n-j}(I_j-u_j)
$$

는 단순히 `u=(B/τ)I`가 아니다.

길이 T의 벡터를 `y=[u₁,…,u_T]ᵀ`, 입력을 `I=[I₀,…,I_{T−1}]ᵀ`라 하고, B를 `B_{rc}=b_{r-c}`인 하삼각 Toeplitz 행렬, J를 한 칸 지연시키는 행렬이라고 하자. 초기 상태가 0이면,

$$
y=\frac1\tau B(I-Jy),
$$

따라서

$$
\boxed{
\left(\mathsf I+\frac1\tau BJ\right)y=\frac1\tau BI.
}
$$

유효 연산자는

$$
H_{\mathrm{eff}}=
\left(\mathsf I+\frac1\tau BJ\right)^{-1}\frac1\tau B
$$

다. [D]

이는 역행렬을 특정 코드 방식으로 계산하라는 지시가 아니다. **빠른 경로가 재현해야 할 연산의 의미**를 정의한 것이다. recurrence, 유효 impulse response, 동등한 triangular solve 등은 그 의미와 수치적으로 일치해야 한다.

g가 고정된 tempered full 모델이라면 B의 lag 계수를 `b_d exp(−dg/τ)`로 바꾸어 같은 원리로 분석할 수 있다. 내용에 따라 ρ가 바뀌는 일반 selected 모델에는 하나의 고정 H_eff를 적용할 수 없다.

### 8.2. 정확성과 복잡도는 별개다

유효 연산자를 T×T 행렬로 만들어 한 번 곱하면 GPU에서 효율적일 수 있지만 계산 차수는 여전히 O(T²)이다. H_eff를 구성하는 비용도 별도로 존재한다.

고정 파라미터의 유효 convolution kernel을 이용한 FFT 경로를 정의하면 convolution 적용 비용은 줄일 수 있다. 그러나 **소마의 spike/reset recurrence, kernel 구성 비용, 배치 크기와 메모리 사용**까지 자동으로 없어지는 것은 아니다.

따라서 권장 표현은 다음이다.

> Full-history 선형 가지는 유효 선형계 표현을 이용해 별도로 가속할 수 있다. Content-dependent selection은 그 고정 연산자 표현을 일반적으로 유지하지 않으므로 추가 비용을 유발한다. 실제 차수와 실행 시간은 각 경로를 정의하고 측정한 뒤 보고한다.

### 8.3. 이 문서의 작은 대조 계산

`T=42, α=0.7, τ=4, g=0`, 고정된 임의 입력에서 순차 가지 갱신과 위 유효 선형 연산자를 float64로 비교했으며 최대 절대 오차는 약 `1.11×10⁻¹⁶`이었다. 이는 **분석 식을 확인한 작은 독립 계산**이지, 사용자 구현의 fast path 검증 결과가 아니다. [N]

---

## 9. 문제 F — α=1 필터뱅크와 fitted SOE를 동일시하지 않는다

`α=1, η=0, g=0`이면 가지는

$$
u_{n+1,k}=\left(1-\frac1{\tau_k}\right)u_{n,k}+\frac{I_n}{\tau_k}
$$

이므로, 입력에서 가지까지의 impulse response는 인덱스 지연을 제외하면

$$
h_k[d]=\frac1{\tau_k}\left(1-\frac1{\tau_k}\right)^d.
$$

소마에 들어가는 선형 혼합은

$$
h_{\mathrm{mix}}[d]=\sum_kw_kh_k[d]
$$

로 유한 지수함수 합 형태다. 이 대수적 연결은 타당하다. [D]

그러나 아래는 다르다.

| 대상 | 차이 |
|---|---|
| 고정 τ=[2,4,8,16]의 4개 가지 | 임의로 선택한 네 decay scale |
| 학습 가능한 다중시간척도 bank | 과제에 맞춰 scale·혼합을 조절 |
| Fitted SOE | 정해진 kernel과 시간 범위·오차를 근사하도록 계수 설계 |
| LongSpike 전체 모델 | Fractional SSM 및 별도의 spiking 계산·학습 구조 포함 |

소마가 threshold/reset을 포함하면 전체 시스템은 선형 SOE가 아니다. SOE 비교는 우선 **소마 이전의 선형부**에 대한 관계다. 학습된 w에 비음수 제약이 없으면 “양의 지수 혼합”이라는 더 강한 표현도 자동으로 성립하지 않는다.

**필수 대조:** 같은 soma·입출력·학습 예산의 α=1 조건을 유지한다. 이후 fractional 고유 이점을 강하게 주장하려면 학습 가능 scale 또는 fitted SOE 등 더 적절한 다중시간척도 대조도 검토한다. [P]

α=1을 평균 MSE에서 못 이겼다고 fractional 이론 자체가 무효인 것은 아니다. 반대로 어떠한 사전 선언된 기준에서도 이점이 확인되지 않으면 fractional dynamics가 핵심 이득이라는 주장을 유지할 근거도 없다. 지연 일반화, 잡음 조건, 효율 등은 사전에 정한 경우에만 추가 근거로 사용한다.

---

## 10. 문제 G — 상호작용·활동·gradient 용어를 공유 소마 구조에 맞춘다

### 10.1. Selector-mediated interaction은 일반적으로 가능하지만 항상 비영은 아니다

현재 bank를 고정하고 현재 상태를 국소적으로 교란하는 관점에서, 다른 가지 ℓ의 현재 상태가 가지 k의 다음 상태에 미치는 영향은 선택자의 의존성을 통해 생길 수 있다.

$$
\frac{\partial u_{n+1,k}}{\partial u_{n,\ell}}
=\sum_{j<n}b_{n-j}f_{j,k}
\frac{\partial\rho_{n,j,k}}{\partial u_{n,\ell}},\qquad k\ne\ell.
$$

이는 항상 비영이라는 식이 아니다. 예를 들어 과거 후보 하나, fixed single-support sparsemax 구간, score에 무관한 상태 방향, 상쇄되는 f의 조합에서는 0일 수 있다. Sparsemax는 support에 따라 Jacobian이 달라지는 구간별 연산이다. [D; R13]

따라서 검증은 **비퇴화 입력에서 예상한 경로가 실제로 존재하는가**, **η=0에서 선택을 통한 경로가 사라지는가**를 본다. 모든 입력에서 모든 cross-partial이 비영이어야 한다고 요구하지 않는다.

### 10.2. 소마 혼합은 가지로 돌아가는 feedback이 아니다

`a_n=Σw_ku_{n,k}`는 정보를 하나의 발화 결정으로 합치는 readout이다. 현재 모델에서는 소마 상태나 소마 spike가 가지 dynamics에 직접 되먹임되지 않는다.

정확한 명칭은 다음이다.

> **공유 선택을 통한 내용 의존적 상태 결합 + 소마에서의 공동 readout.**

확산 coupling을 제거한 현재 λ=0 설정을 유지한다면, 직접 compartment-to-compartment recurrent coupling을 가진 모델처럼 설명하지 않는다. 이는 이전 아이디어를 부정하는 것이 아니라 v3-A에서 상호작용의 정의가 달라졌다는 사실을 보존하는 것이다. [S1, §3.8]

### 10.3. 발화율과 가지 활동을 분리한다

가지는 발화하지 않으므로 “구성원별 발화율” 대신 다음을 본다.

| 대상 | 진단 |
|---|---|
| 소마 | 발화율, 침묵·연속 발화 비율, pre/post-reset 상태 |
| 가지 | 상태 분산, f 크기, 유효 rank, 가지 간 상관, 시간 응답 |
| 가지→소마 | w의 크기·부호, 각 가지의 a_n 기여, 하나의 가지 지배 여부 |
| 선택자 | η, g, score scale, p support, 실효 커널 배분 |

초기 w가 모두 1이면 soma drive는 가지들의 합이며 K가 바뀌면 입력 규모도 달라질 수 있다. K 비교에서는 단순 출력 gain 변화와 표현 다양성을 구분하고 calibration 규칙을 맞춰야 한다. 이를 임의로 평균으로 바꾸지는 않는다. [D, P]

### 10.4. Gradient 게이트의 수정

G10은 “모든 parameter의 gradient가 매번 유한·비영”이 아니라 다음이어야 한다.

- 활성 parameter의 gradient가 유한하다.
- 비퇴화 시험에서 의도한 loss 경로가 존재한다.
- Full 모드의 미사용 Q/K, g=0 고정안의 비활성 g, 단일 support 구간 등 정상적인 0-gradient를 허용한다.
- Surrogate backward는 hard-threshold forward의 finite difference와 같을 필요가 없다.
- Reset detachment 정책과 history BPTT 범위는 명시적으로 검사한다.

---

## 11. 합성 회상 과제: 좋은 기능 검사지만, explicit retrieval의 필요조건은 아니다

### 11.1. 유지할 원본 설계

원본은 42개 patch 사건, 3개 key A/B/C, 길이 2–5의 구간, 재등장 전에 1–3개의 중간 구간을 둔다. 최초 구간에는 value가 있고 재등장 구간에서는 입력 value가 0이다. 현재 key에 해당하는 이전 값을 출력하는 과제이며 cue와 value를 같은 원래 채널 안에 둔다. [S1, §6.2]

이 구조는 patch 하나와 사건 slot을 대응시키므로 query 위치·정답 구간·distractor를 분석하기에 좋다. 회상 기능의 초기 진단으로 유지할 가치가 있다.

### 11.2. 반드시 고칠 주장

“선택 없이 기억만 쓰면 신호가 섞여서 풀 수 없다”는 명제는 성립하지 않는다.

예를 들어 recurrent state가 `(v_A,v_B,v_C)`를 저장하고 현재 key에 맞춰 한 값을 출력하면 된다. 명시적으로 과거 모든 time slot을 다시 조회하지 않아도 가능한 구성이다. 이것은 모델의 실제 학습 성공 보장이 아니라 **명시적 bank 검색이 논리적으로 필수는 아니라는 반례**다. [D]

Zoology는 쉬운 synthetic associative recall을 gated convolution도 해결할 수 있음을 논의하고, 더 다양한 질의 위치·거리·key 수를 가진 MQAR로 차이를 분석한다. 그 결과를 현재 소규모 과제의 모델 순위로 그대로 옮길 수는 없지만, 과제 판별력을 점검해야 한다는 근거가 된다. [R12]

따라서 과제의 목적은 다음으로 수정한다.

> **학습된 모델이 현재 문맥에 필요한 값을 회상하며, 그 과정에서 명시적 dynamics 재분배가 matched recurrent/고정기억 대조보다 도움이 되는가?**

### 11.3. 데이터 생성 계약에서 확정해야 하는 사항

| 항목 | 필요한 정의 |
|---|---|
| Value 생성 | 시퀀스마다 key별 값을 새로 생성한다. 전역 A→상수 매핑이 없어야 한다 |
| Value 분포 | 범위·분포·상관을 실행 전 기록한다. 이 문서는 임의의 최종 분포를 추가 승인하지 않는다 |
| 최초 관측과 재등장 | value가 있는 구간과 없는 구간을 명확히 표시한다 |
| 정답 사건 집합 | Raw event-level provenance를 저장하되 유일한 내부 정보 저장 위치라고 간주하지 않는다 |
| 주 오차 집계 | 단순 입력 복사가 가능한 최초 구간이 아니라 recall query를 중심으로 정의한다 |
| Cue 누락·잡음 | 난이도별 생성 규칙을 고정하고 impossible query를 별도 처리한다 |
| Dataset 분리 | 생성 seed를 분리하고 동일 시퀀스가 train/validation/test에 겹치지 않도록 한다 |
| 키 수 확장 | 3-key는 첫 기능 검사. 이후 key 수·delay·distractor를 늘리는 경우 별도 난이도 축으로 기록한다 |

### 11.4. Synthetic readout은 인과적이어야 한다

ETT에서 forecast origin 이전의 모든 input patch를 flatten하는 것은 그 origin 기준으로 가능하다. 그러나 synthetic의 각 시점 target에 전체 시퀀스 flatten을 연결하면 뒤쪽 사건을 미리 볼 수 있다.

Synthetic은 **각 평가 시점까지 도달한 spike/state만 사용하는 readout**으로 정의해야 한다. 연속 membrane readout은 해석용 diagnostic으로 별도 명명하고 주 spike readout을 조용히 대체하지 않는다. [P]

### 11.5. 한 칸 지연과 마지막 입력

원본 식에서는 I_n이 u_{n+1}에 반영되고 s_n은 u_n을 사용한다. 따라서 선택 ρ_n의 효과는 다음 spike에서 나타나는 것이 기본 시간 의미다. [S1, §3.2, §6.2]

다음 중 어느 계약을 쓰는지 명시해야 한다.

- 원본의 한 칸 지연을 유지하고, query q에 대한 response를 q+1에서 평가한다.
- 소마를 새 가지 상태 u_{n+1}로 구동하도록 정렬을 변경한다. 이는 별도의 timing 변경으로 기록한다.

이 검토는 후자를 자동 채택하지 않는다. 원본대로 유지하면 T개 input 뒤 terminal u_T를 이용하는 soma response가 필요한지 명확히 한다. s₀,…,s_{T−1}만 반환하면 마지막 input의 새 정보가 출력에 반영되지 않을 수 있다.

첫 재등장 칸과 두 번째 이후 칸의 결과를 분리하되, 첫 칸을 제외한 성능만으로 전환 즉시 회상했다고 주장하지 않는다. 반복 구간의 두 번째 이후에는 이전 query 처리 결과가 이미 상태에 남을 수 있다.

---

## 12. 정답 사건과 실제 정보 저장 위치는 다르다

원래 값이 들어간 사건 구간을 `\mathcal R_q`라고 하자. 이는 데이터 생성기 수준의 정답 provenance다. 그러나 f_j와 u_j는 이전 사건들의 영향을 담으므로, `j∉\mathcal R_q`인 later state에도 해당 값 정보가 남을 수 있다.

따라서 다음 두 방향 모두 주의한다.

- 정답 구간 weight가 낮음 → 즉시 “기억하지 못했다”로 결론 내리지 않는다.
- 정답 구간 weight가 높음 → 즉시 “그 내용이 실제 예측에 중요했다”로 결론 내리지 않는다.

### 12.1. 서로 다른 개입을 분리한다

| 개입 | 바꾸는 것 | 평가 질문 |
|---|---|---|
| 고정 bank의 value 교체 | 특정 f_j만 교체하고 key/weight를 보존 | 읽은 내용의 직접적인 역할 |
| 고정 bank의 key 교체 | 문맥·점수만 바꾸고 value는 보존 | 검색 결정의 민감도 |
| 과거 event 입력 교체 후 재전개 | 원시 입력을 바꾸고 이후 모든 상태를 다시 계산 | 모델 전체가 그 사건에 의존하는가 |
| Weight/policy 교체 | 같은 checkpoint에 uniform/recent/oracle 적용 | 해당 학습 모델의 read 정책 의존성 |
| 다른 policy로 별도 학습 | 전체 훈련 과정을 다시 수행 | 해당 policy를 포함한 모델의 성능 |

고정 bank의 국소 개입은 자연 궤적 밖의 조합을 만들 수 있다는 한계가 있다. 재전개 개입은 여러 경로를 동시에 바꾸므로 직접 read 효과만 분리하지 못한다. 두 검사를 함께 사용하되 같은 종류의 증거로 합산하지 않는다. [P; R15–R16]

### 12.2. 우연 적중률은 support 크기와 lag를 맞춘다

정답 집합이 r개, 전체 후보가 n개인 균일 무작위 가중치의 기대 비중은 r/n이다. 하지만 실제 기본 fractional kernel은 최근 사건에 더 많은 coefficient를 배정하고, query마다 n과 r이 달라진다.

Support-hit도 support가 클수록 자연히 높아진다. 따라서 “약 0.15가 우연 수준”을 모든 query에 적용하지 말고, **그 query의 후보 수·정답 길이·baseline coefficient allocation**과 비교한다. [D]

---

## 13. Oracle은 최적해가 아니라 정의된 특권적 정책이다

### 13.1. 원본 oracle의 의미

원본 oracle은 정답 집합에 균일한 p를 주고 나머지 `ρ̃→ρ` 경로를 그대로 사용한다. 위치를 안다는 점에서 privileged하지만, 선택된 f_j가 원시 value가 아니고 균일한 가중치도 최적임이 보장되지 않는다. [S1, §7.1]

따라서 `E_oracle`을 최저 가능한 오차, ceiling, 이 구조의 최선으로 정의하지 않는다. Learned 모델이 이 oracle-policy 모델보다 좋을 수 있다.

### 13.2. 반드시 분리할 oracle 세 종류

| 이름 | 정의 | 목적 |
|---|---|---|
| **Oracle-trained** | 처음부터 정답 위치 정책으로 학습하고 validation으로 checkpoint 선택 | 해당 정책과 dynamics를 모델이 활용할 수 있는가 |
| **Test-time oracle intervention** | Learned checkpoint의 p만 정답 정책으로 바꿈 | 이미 학습된 모델에 그 정책을 강제했을 때의 변화 |
| **Generator lookup sanity** | 생성기의 저장값으로 정답을 직접 산출 | 데이터·target 정의의 오류 확인 |

Generator lookup은 비교 모델의 성능이 아니다. Test-time oracle은 학습된 분포와 다른 정책을 강제하므로, 악화됐다고 바로 “정답 memory가 쓸모없다”고 결론 내릴 수 없다.

### 13.3. 작은 η는 완벽한 p의 효과도 약하게 만든다

g=0에서 기본 fractional coefficient가 정답 구간에 두는 비중을 m₀라 하자. p가 정답 집합에 전부 집중하면 최종 effective mass는

$$
\boxed{M_{\mathrm{oracle}}^{\mathrm{eff}}=(1-\eta)m_0+\eta.}
$$

m₀=0.15, η=σ(−4)에서

$$
M_{\mathrm{oracle}}^{\mathrm{eff}}\approx0.16528828.
$$

즉 p는 정답에 100% 집중하지만 실제 적분 계수는 정답 구간에 약 16.5%만 배정할 수 있다. Oracle≈full이 이런 상황에서 발생하면, 검색 위치보다 **그 정책이 dynamics를 얼마나 바꿨는지**를 먼저 확인해야 한다. [D, N]

### 13.4. O7 문턱과 안정성을 함께 확인한다

위 식에서 `M_eff≥m*`를 얻으려면, m₀<m*인 경우 이상적인 p에서도

$$
\eta\ge\frac{m^*-m_0}{1-m_0}
$$

가 필요하다. 예를 들어 m₀=0.15, m*=0.5이면 약 0.4118이다. 이는 이 문서를 정리하면서 명시한 **추가 대수적 점검**이다. [D]

이 값까지 η를 올리라는 권고가 아니다. 오히려 **O7의 coefficient-mass 기준이 어떤 η 범위를 요구하는지, 그 범위가 안전한지를 먼저 확인하라는 의미**다. 7절의 recent 정책 진단은 같은 η에서 항상 oracle이 불안정하다는 증거는 아니지만, 문턱을 맞추려고 무조건 선택 강도를 키워서는 안 된다는 경고다.

---

## 14. 진단량을 다섯 층으로 분리한다

### 14.1. 같은 ‘attention weight’라고 부르지 않는다

| 기호 | 역할 | 정규화 여부 |
|---|---|---|
| e | 학습된 관련성 점수 | 확률 아님 |
| p | softmax/sparsemax 출력 | 과거 후보에 대한 합 1 |
| ρ̃ | 질량 재분배 배율 | 합 1 아님 |
| ρ | residual·tempering을 포함한 배율 | 합 1 아님 |
| bρ | 실제 dynamics 이력의 계수 | 확률 아님 |

p가 정답을 찾고 있는지와 bρ가 실제로 정답 구간을 얼마나 사용하는지는 다른 측정이다. f의 부호와 크기까지 들어간 예측 기여는 또 다른 질문이다.

### 14.2. Effective allocation

n>0에서,

$$
A_{n,j,k}
=\frac{b_{n-j}\rho_{n,j,k}}
{\sum_{\ell<n}b_{n-\ell}\rho_{n,\ell,k}}.
$$

정답 사건 집합 `\mathcal R_n`에 대해,

$$
M_{n,k}^{\mathrm{eff}}
=\sum_{j\in\mathcal R_n}A_{n,j,k}.
$$

이 값은 **최종 과거 적분 계수의 배분 비중**이다. 입력 중요도·예측 기여·원인 확률이라는 뜻은 아니다.

p 자체의 정답 mass도

$$
M_n^p=\sum_{j\in\mathcal R_n}p_{n,j}
$$

로 별도 보고한다. 어느 쪽이 높아지고 낮아지는지 비교하면 검색기와 read-strength의 문제를 구분할 수 있다.

### 14.3. 평균 단위를 미리 정한다

O7을 실행할 때 query response 시점, layer, population, branch의 평균 규칙을 고정해야 한다. 하나의 **권고 집계안**은 다음과 같다. [P]

- M1의 해당 population layer 전체를 사용하고, 결과가 좋은 neuron만 사후 선택하지 않는다.
- Eligible recall query에 대해 가지와 논리 뉴런을 동일 가중으로 평균한다.
- 시퀀스 내 query 평균을 구한 뒤 시퀀스들을 동일 가중으로 평균한다.
- Seed 결과는 별도로 보존하고 사전 정한 방식으로 요약한다.

Soma에 거의 기여하지 않는 population이 이 평균을 왜곡하는지 보기 위한 active-only 또는 w-weighted 분석은 별도 보조 진단으로 둔다. O7을 통과하도록 사후 집계 규칙을 바꾸지 않는다.

---

## 15. O7 승인 권고: 네 숫자와 판정 규칙의 수정안

### 15.1. 권고 요약

| 항목 | 원본 제안 | 최종 권고 |
|---|---|---|
| ① 정답 위치 비중 | ≥0.5 | **0.5 조건부 유지**. Effective allocation 기준과 matched baseline 비교를 명시 |
| ② Oracle 대비 MSE | ≤1.5배 | **Oracle gap 50% 이상 회수로 교체**. Oracle headroom이 유효할 때만 사용 |
| ③ Full 대비 개선 | ≥20% | **예비 진행 문턱으로 유지**. Paired 불확실성과 단순 정책 대비를 함께 본다 |
| ④ Oracle≈full | 5% 이내이면 구조 실패 | **5%는 진단 범위로 유지**. 자동 실패 판정을 삭제 |

이 수치는 문헌의 보편적인 성공 정리가 아니다. 이 연구의 사전 운영 기준으로 채택할 권고다. 수학적 타당성·안정성 검사를 대신하지 않는다.

### 15.2. O7-①: Effective 정답 mass ≥0.5

주 지표는 14절의 `M_eff`로 정의한다. `M^p`는 별도의 검색기 진단이다.

**권고 해석:** 실제 과거 적분 계수 중 적어도 절반이 생성기가 지정한 정답 사건 구간으로 향한다.

**추가 조건:** 같은 query 집합에서 Full 또는 정의된 단순 정책의 mass보다 증가하는지 확인한다. Base mass가 원래 0.5를 넘는 쉬운 lag 배치라면, 0.5 통과만으로 내용 선택의 증거가 되지 않는다.

낮은 M_eff는 **사건별 선택 주장에 대한 증거 부족**이지 회상 기능 전체의 부재가 아니다. 이후 상태로 정보가 전파될 수 있기 때문이다. 높은 M_eff도 예측 개입과 함께 해석해야 한다.

### 15.3. O7-②: Oracle-gap 회수율 ≥0.5

조건별 mean error를 E_full, E_learned, E_oracle-trained라고 하자.

$$
\boxed{
G=\frac{E_{\mathrm{full}}-E_{\mathrm{learned}}}
{E_{\mathrm{full}}-E_{\mathrm{oracle\text{-}trained}}}.
}
$$

**권고 문턱:** G≥0.5.

**전제:** 분모가 양수이며 수치 바닥이나 추정 불확실성과 구분되는 충분한 headroom이어야 한다. 분모의 유효성 기준은 데이터 value scale 및 paired 불확실성을 바탕으로 사전에 정한다. 이 문서는 임의의 절대 오차 epsilon을 보편 기준으로 승인하지 않는다.

| 상황 | 처리 |
|---|---|
| Oracle-trained가 Full보다 분명히 좋음 | G를 계산하고 불확실성을 함께 보고 |
| Oracle-trained≈Full | Headroom 부족. O7-②는 판정 불가 |
| Oracle-trained가 Full보다 나쁨 | G를 성공 지표로 사용하지 않고 oracle 정책·훈련·동역학을 진단 |
| Learned가 Oracle-trained보다 좋음 | G>1 가능. Oracle이 수학적 optimum이 아니므로 허용 |
| 둘 다 오차 바닥에 가까움 | 과제 판별력과 readout 제한을 재평가 |

기존의 E_learned≤1.5E_oracle는 oracle 오차가 0에 가까울 때 불안정하고, 개선 여지와 직접 연결되지 않으므로 주 판정에서 제외한다.

### 15.4. O7-③: Full 대비 평균 개선 ≥20%

Paired seed r의 오류를 사용하여,

$$
R_r=1-\frac{E_{\mathrm{learned},r}}{E_{\mathrm{full},r}}.
$$

한 가지 **명시적인 권고 집계 방식**은 mean(R_r)≥0.2를 쓰고, 동일 seed의 절대 오류 차이

$$
\Delta_r=E_{\mathrm{learned},r}-E_{\mathrm{full},r}
$$

에 대한 95% 신뢰구간이 0보다 낮은지 함께 보는 것이다. Ratio of means를 사용하는 대안도 가능하지만 실행 전에 한 가지를 고정하고 혼용하지 않는다.

이는 “참 개선량이 최소 20%라고 확증”한 것과 다르다. 후자를 말하려면 상대 개선의 신뢰구간 하한이 20%를 넘어야 한다.

또 Full만 이기는 것은 충분한 attribution이 아니다. Recent, 간단한 recurrent 모델, fixed-kernel control이 비슷한 개선을 얻는지 함께 확인한다. 이 문턱은 기전 입증 완료 도장이 아니라 **후속 ETT 평가로 진행할 예비 기준**이다.

### 15.5. O7-④: Oracle≈Full의 5%는 진단 범위

예를 들어

$$
\delta_O=\frac{E_{\mathrm{oracle\text{-}trained}}-E_{\mathrm{full}}}{E_{\mathrm{full}}}
$$

로 상대 차이를 정의하고, ±5%를 사전 진단 범위로 둘 수 있다. 분모가 거의 0인 경우에는 사용하지 않는다.

점추정이 범위 안에 있는 것과 통계적 동등성은 다르다. 동등성을 주장하려면 해당 차이의 불확실성이 사전 선언한 동등성 범위 안에 들어오는지를 별도로 평가해야 한다.

**자동으로 “구조 실패”로 가지 않는다.** 다음 원인을 순서 있게 구분한다.

1. Full이 이미 과제를 거의 완벽하게 푼다.
2. η가 작아 정답 p가 실제 적분에 거의 영향을 주지 않는다.
3. Oracle이 지정한 사건 구간과 유용한 dynamics 저장 위치가 다르다.
4. Oracle-trained가 충분히 학습되지 않았거나 test-time policy 교체만 수행했다.
5. Soma·readout 또는 시간 정렬이 회상 정보를 사용할 수 없게 만든다.
6. 과제가 명시적 event-level 선택을 판별하지 못한다.

최종 판정은 **“oracle과 과제의 진단력이 부족하다”**, **“read-strength 문제가 있다”**, **“현재 구조가 활용하지 못한다”** 등을 증거에 맞춰 구분한다.

### 15.6. 수정된 O7 진행표

| 결과 조합 | 권장 판정 |
|---|---|
| 수학·안정성 게이트 통과 + ①·②·③ 지지 + 내용 개입도 일관됨 | ETT로 진행할 기전 근거 확보. 보편적 성능 우월성은 아직 아님 |
| p는 정답에 집중하지만 M_eff가 낮음 | 선택 강도·residual·tempering의 영향을 먼저 확인 |
| M_eff가 높지만 예측 개선이 없음 | Dynamics value, soma/readout, 과제 적합성 문제를 조사 |
| M_eff는 낮지만 회상 성능이 좋음 | 명시적 정답 사건 선택은 미확인. 압축 기억·간접 경로를 조사 |
| Oracle headroom 없음 | O7-② 판정 불가. 자동 구조 실패 금지 |
| 20% 평균 개선은 있으나 CI가 넓음 | 탐색적 개선. 반복 수와 변동을 더 확인 |
| 선택 정책에서 큰 상태 증폭 발생 | O7 성공 여부와 무관하게 수치·기전 안정성 진단으로 복귀 |
| Recent/간단한 recurrent가 같은 성능 | 제안한 content redistribution의 고유 효과 미입증 |

---

## 16. 비교군과 ablation을 다시 정리한다

### 16.1. 가장 중요한 2×2 비교

| | Full history | Content redistribution |
|---|---|---|
| α=1 | Ordinary multi-timescale branches + soma | Integer-order coefficients + explicit selected history |
| α<1 | Fractional branches + soma | 제안한 fractional selected-history 모델 |

주장하려는 효과를 분리하려면 neuron/soma, data, readout, calibration 규칙과 학습 예산을 동일하게 유지한다. α=1에 selection을 켠 모델은 과거 bank를 사용하는 non-Markovian 모델일 수 있으며, 단순 ordinary LIF라고 부르지 않는다.

### 16.2. 다양성과 선택의 상호작용

현재 λ=0 설정에 맞춰 주 interaction contrast를 다음처럼 정의할 수 있다.

$$
\Delta_{\mathrm{het}\times\mathrm{sel}}
=(E_{\mathrm{het,sparse}}-E_{\mathrm{het,full}})
-(E_{\mathrm{hom,sparse}}-E_{\mathrm{hom,full}}).
$$

음수이면 해당 오류 척도에서 heterogeneity가 있을 때 선택의 추가 이득이 더 크다는 뜻이다. 이것은 통계적 상호작용이며 **직접 recurrent coupling의 증명은 아니다.**

Homogeneous τ의 정확한 값은 원본 실행 계약에서 확인·고정한다. 이전 문서의 harmonic-mean 규약 등을 현재 승인 없이 자동 이식하지 않는다.

### 16.3. 중복 대조군을 제거한다

g=0에서 p가 균일하면 ρ̃=1이며 최종 ρ=1이다. 따라서 Uniform과 Full의 기능이 같다. 또한 정확한 질량 보존안에서는 κ=1이므로 mass-matched uniform도 Full과 같아진다. [D]

같은 dynamics인 조건을 다른 모델 이름으로 반복 학습하여 서로 독립적인 근거로 세지 않는다. 다만 실행 경로가 정말 같은 결과를 내는지 operator test로 확인할 수 있다.

g>0을 유지하면 uniform/full이 어떤 π를 적용하는지 다시 선언한다. `ρ=κ_n`로 치환할 때 π를 이후 또 곱하면 double tempering이 생길 수 있다. 최종 계수를 대체하는 개입인지 내부 확률을 대체하는 개입인지 위치를 고정한다.

### 16.4. 비교군의 역할

| 비교군 | 검증할 질문 |
|---|---|
| Scalar fractional branch + soma | Population 상태가 필요한가 |
| Homogeneous population | 단순 복제/공통 timescale보다 다양한 response가 필요한가 |
| α=1 heterogeneous population | Fractional order 자체가 기여하는가 |
| Dense vs sparse p | Support와 배율 변화가 결합된 효과는 무엇인가 |
| η=1 hard-mask | 완전 배제가 residual 모델보다 필요한가; 안정성은 어떤가 |
| Recent/fixed-lag/random policy | Learned content relation이 단순 정책보다 나은가 |
| Oracle-trained | 지정된 사건 read 정책을 활용할 수 있는가 |
| `[u;I]` vs `[u;I;Δu]` | 현재 값뿐 아니라 변화 문맥이 key에 도움이 되는가 |
| Learned timescale/fitted SOE | 단순 고정 α=1 bank보다 강한 다중시간척도 설명을 배제할 수 있는가 |
| DH-SNN형 branch–soma, TS-LIF 등 | 유사 구조·시계열 특화 neuron과의 실용 관계 |
| Ridge/선형/ANN MLP/window mean | Task 전체의 sanity와 실용 성능 수준 |
| Simple recurrent/gated-convolution recall model | explicit bank 재조회가 아니라 압축 기억으로 해결되는지 |

모든 후보를 한 번에 곱한 대규모 matrix를 만들지 않는다. 먼저 핵심 대비의 정의와 반복 가능성을 확보하고, 주장과 직접 관련된 후보를 추가한다.

### 16.5. Capacity matching은 단일 숫자로 끝나지 않는다

K개의 가지를 가진 D개의 soma와 DK개의 독립 spiking neuron은 출력 spike 수, soma 상태 수, readout 정보량이 다르다. “총 구성원 수만 같음”을 완전한 capacity match라고 하지 않는다.

논리 뉴런 수, branch state 수, soma 수, active parameter 수, output width, recurrent/history 메모리, 계산량을 별도로 기록한다. 서로 다른 fairness 관점의 비교가 필요하면 따로 제시한다. 가중치 수가 같아도 history 접근 방식은 같지 않다.

---

## 17. Forecasting·통계 계획

### 17.1. 유지할 ETT 계약

아래는 S1이 제시한 설정이다. 변경할 때는 별도 결정 기록을 남긴다.

| 항목 | 설정 |
|---|---|
| 개발 데이터 | ETTh1, ETTh2 |
| 구간 | Train [0,8640), validation target [8640,11520), test target [11520,14400) |
| 전처리 | Train 구간만으로 표준화 |
| 입력·patch | L=336, P=8, T=42 |
| Horizon | H96 주판정, H720 보조·스트레스 평가 |
| 창 수 | H96: 8209/2785/2785; H720: 7585/2161/2161 |
| 학습 | AdamW lr 10⁻³, wd 10⁻², batch 128, clip 1 |
| 종료 | 최대 50 epoch, early stop 10, plateau schedule 0.5/5 |
| Checkpoint | 최소 validation MSE |
| Seed | 7, 13, 21, 42, 123, 256, 512, 1024 |
| 구조 확장 | M1 확인 후 M3, TCN, attention |

Validation/test input은 해당 forecast origin 이전의 관측 context를 사용할 수 있지만 target은 지정된 구간 안에 있어야 한다. H96과 H720에서 input history는 42 patch로 같으므로, H720을 “720-step 과거 기억 검사”로 해석하지 않는다.

### 17.2. Flatten과 bottleneck은 서로 다른 연구 질문

ETT의 전체 spike sequence flatten head는 모든 관측 시간 표현을 직접 볼 수 있다. 따라서 내부 retrieval이 없어도 readout이 과거 정보를 활용할 수 있다.

이 실험은 **기존 forecasting 시스템에 추가 이득이 있는가**를 본다. Last/short-terminal readout은 **현재 상태로 정보가 전달되어야 하는가**를 더 직접적으로 본다. 그러나 출력 spike 하나만으로 긴 horizon을 예측하게 하는 매우 강한 bottleneck도 별도 제약이므로, 실패를 곧바로 기억 실패로 단정하지 않는다.

두 head 사이에는 parameter와 정보량 차이가 있으므로, neuron 비교는 우선 같은 head 안에서 수행한다.

### 17.3. MDE 수치의 정확한 의미

S1의 paired SD 0.0043을 가정하고 n=8, 양측 유의수준 0.05, 목표 power 0.8인 paired t-test를 계산하면 최소검출차이는 약 0.00497095다. [N]

이는 다음 조건부 진술이다.

> 새 모델의 paired 차이 분포가 그 가정과 유사하다면, 약 0.005 차이에 대해 80% 검정력을 목표로 한다.

다음 진술은 아니다.

> 0.005보다 작은 차이는 절대로 검출할 수 없다.

또 SD 0.0043은 이전 모델에서 측정했다고 원본이 보고하는 값이지, 현재 v3-A의 확정된 분산이 아니다. 실제 paired variability를 별도로 보고한다.

### 17.4. 판정 단위와 불확실성

- Seed별 paired 차이를 보존한다.
- Dataset별로 먼저 보고하고 H96과 H720을 합쳐 반대 방향 효과를 숨기지 않는다.
- 수천 개의 overlap window를 서로 독립적인 실험 반복으로 세지 않는다.
- Synthetic 데이터의 불확실성은 전체 시퀀스 단위와 훈련 seed 변동을 구분한다.
- 2개 데이터 × 8개 seed를 하나의 균질한 16개 독립 sample처럼 취급하지 않는다.
- α grid, 여러 비교군, architecture 선택에 따른 탐색·다중비교를 기록한다.
- O7이나 test 결과를 본 뒤 threshold·집계법·평가 subset을 바꾸면 새 exploratory revision이다.

### 17.5. 기존 v2 수치는 참고 정보다

S1의 ridge/window-mean 수치는 새 모델과 동일 프로토콜인지 확인할 출발점이다. 이 문서에서 그 원시 결과를 재검증하지 않았으므로 새 성능표에 복사하여 비교 완료라고 하지 않는다.

단순 기준선보다 나쁜 모델에서도 같은 조건 사이의 기전 차이는 분석할 수 있다. 그러나 그 차이를 유용한 장기 forecasting 모델의 입증으로 확대하지 않는다. H720이 약하다는 이유로 사후 삭제하지 않고 한계를 공개한다.

---

## 18. 수정된 수치 게이트

다음은 원본 G1–G10을 보완한 **권고 검증 계약**이다. 모든 검사가 이미 통과했다는 뜻은 아니다.

| 게이트 | 검사와 조건 | 주의사항 |
|---|---|---|
| G1 | 상수 forcing fractional 적분기의 해석해와 비교 | f-LIF 누설·soma reset 전체 검증과 구분 |
| G2 | α=1, η=0, g=0에서 Euler 가지 recurrence와 일치 | g>0 모델에는 수정 recurrence 기준 적용 |
| G3 | Full 가지 loop와 유효 선형 연산자/fast path 일치 | Raw b와 입력만의 convolution으로 대체하지 않음 |
| G4 | Pinned source golden과 명시한 참조 경로 비교 | 전체 v3-A spike가 원문과 동일해야 한다는 뜻 아님 |
| G5 | K=1, homogeneous branches, separate scalar branch 대응 | Scalar branch+soma와 원문 f-LIF를 구분 |
| G6 | 미래 patch 교란 시 과거 state/score/spike 불변 | Synthetic head와 terminal timing도 포함 |
| G7 | η=0의 neutral mode와 선언한 Full 정의 일치 | g 적용 여부를 동일하게 맞춤 |
| G8a | π=1에서 coefficient mass 보존 | n=0은 과거 mass 비율 정의 대상에서 제외 |
| G8b | π>0 변형의 가지별 κ를 이론식과 대조 | 무조건 κ=1을 요구하지 않음 |
| G8c | p support와 최종 bρ support를 따로 검사 | η<1 residual 모델의 최종 dense support는 정상 |
| G9 | Sample/window/channel 간 state isolation | 공유 parameter와 공유 state를 혼동하지 않음 |
| G10 | 유한 gradient 및 비퇴화 조건의 예상 경로 확인 | 모든 parameter가 항상 비영일 필요 없음 |
| G11 | η·support 집중·T 변화의 branch 상태 안정성 | 초기값만 검사하고 완료하지 않음 |
| G12 | 소마 발화율과 branch representation 활동 진단 | 가지 발화율이라는 잘못된 지표 제거 |
| G13 | Query→branch update→soma response 시간 정렬 | 마지막 입력의 response 처리 명시 |
| G14 | Oracle, intervention, retrained policy의 분리 | Privileged information이 learned 경로로 새지 않음 |
| G15 | Synthetic value 독립 생성 및 causal readout | 전역 key→value 학습 또는 미래 누설 차단 |

Bitwise equality는 동일 연산 순서·동일 경로의 neutral switch 등 명확한 경우에만 요구한다. 수학적으로 같은 연산도 loop와 행렬곱은 부동소수점 합산 순서 때문에 미세하게 다를 수 있다. dtype·device·atol·rtol을 선언하고, 기존 보고된 오차 0.0을 모든 경로의 보편적 기준으로 삼지 않는다.

---

## 19. 수정 우선순위와 진행 순서

### Phase A — 모델과 참조 경로

먼저 `/τ` 스케일, zero initial state, branch/soma 상태, reset/surrogate 경로, 현재와 과거 항, 소마 출력 정렬을 고정한다. 원문 reference의 실행 여부에 따라 source parity, source-derived, mathematical validation을 구분한다.

공식 코드 실행과 새 모델 수학 검증은 서로 다른 목적이다. 원문 코드가 특정 결과를 냈다는 사실만으로 v3-A의 모든 식을 승인하지 않는다.

### Phase B — Population 및 calibration

Train 구간의 입력으로 **소마 발화율**과 branch 상태 분포를 측정한다. 원본의 0.1–0.3 발화율 목표는 소마에 적용하는 운영 범위이지 모든 가지를 같은 활동으로 맞추는 목표가 아니다.

하나의 input scale을 선택해 matched 조건에 공유하고, 값·입력 subset·초기화·발화/reset 규약을 기록한다. Per-branch 임의 보정은 새로운 모델 변경이다. 초기 calibration만으로 학습 후 상태를 보장하지 않으므로 훈련 중 진단도 남긴다.

### Phase C — 계수·공유 선택·안정성

π 처리 방안, sparse residual 명명, 최종 coefficient 진단과 안전성 검사를 통과한다. 특히 수치 불안정성을 부정확한 cutoff·state clipping으로 숨긴 채 그대로 성능 실험을 진행하지 않는다. 안정화 수정이 필요하면 새 모델 revision으로 기록한다.

### Phase D — 회상 검증

먼저 generator lookup sanity를 확인하고 causal timing을 맞춘다. Full, α=1, 단순 memory policy, oracle-trained, learned selection을 정의한 조건에서 비교한다. O7은 이 단계의 타당성이 확보된 후에만 적용한다.

### Phase E — ETT M1

H96의 핵심 paired 대비와 H720의 보조 평가를 수행한다. 좋은 seed만 골라 보고하지 않고, 실패와 미완료를 구분한다. 원래 최소 validation checkpoint 규칙을 유지한다.

### 후속 확장

M3·TCN·attention은 mechanism 결과와 분리된 architecture generalization 실험이다. 더 복잡한 backbone에서의 개선이 shallow neuron 자체의 회상 능력을 소급하여 증명하지 않는다.

---

## 20. 에이전트가 연구 책임자에게 확인해야 하는 결정표

### 20.1. 원본의 원칙으로 유지할 사항

| 항목 | 상태 |
|---|---|
| 리셋 없는 fractional branches와 공유 soma | v3-A의 현재 모델 정체성으로 유지 |
| Population 전체 문맥으로 과거를 평가 | 유지 |
| 선택은 dynamics 이력의 적분 계수에 작용 | 유지 |
| 소마 한 곳의 spike 출력 | 유지 |
| M1 우선, H96·H720 보고 | 유지 |
| Train-only scaling, validation checkpoint, 독립 window 상태 | 유지 |
| v2/원문 reference와 새 결과를 구분 | 유지 |

### 20.2. 구현 계약 전에 승인·기록할 사항

| ID | 결정할 사항 | 이 검토의 권고 |
|---|---|---|
| D-A | 첫 주 모델에서 π를 유지할지 | **g=0, π=1을 첫 기전 검증에 사용**, tempered 변형은 별도 |
| D-B | 주 모델이 완전한 sparse history인가 | 아니며 **sparse redistribution + dense residual**로 명명 |
| D-C | η의 안전 범위 및 concentration 처리 | Phase C 결과로 사전 범위를 정의. 이 문서는 보편 상한 미확정 |
| D-D | Query와 soma 출력의 한 칸 지연 | 원본 유지/정렬 변경 중 선택하여 terminal response까지 명시 |
| D-E | Soma reset의 backward detachment | 명시적으로 고정. Surrogate 식과 별개 결정 |
| D-F | Fast path의 의미 | Feedback을 포함한 유효 선형 연산자에 일치 |
| D-G | Interaction의 주장 | Selector-mediated state dependence와 heterogeneity×selection contrast로 명확화 |
| D-H | Oracle 기준 | Oracle-trained, test-time oracle, generator lookup을 분리 |
| D-I | Synthetic value 분포·집계·readout | 독립 생성·causal response·eligible query 규칙을 고정 |
| D-J | Homogeneous τ와 capacity match | Exact value와 soma/branch/output budget 기준을 실행 전에 명시 |
| D-K | O7 숫자와 의미 | **0.5 / oracle-gap 0.5 / 20% / 진단 폭 5%** 권고 |
| D-L | 판정 불가 처리 | Oracle headroom 부족·불확실성·수치 실패를 전체 아이디어 실패와 구분 |

따라서 “O7 네 숫자만 확정하면 구현 착수”는 다음처럼 바꾸는 것이 적절하다.

> **모델 계수와 시간 정렬, 수치 게이트, oracle·metric의 의미를 확정한 뒤 O7을 적용한다. 수학적으로 모순된 테스트를 통과하도록 코드를 맞추지 않는다.**

---

## 21. 결과 해석 및 최종 전달 요약

### 21.1. 결과별 허용되는 해석

| 관측 결과 | 지지되는 해석 | 금지할 단정 |
|---|---|---|
| p에 0이 생김 | 선택기 분포가 sparse함 | 전체 fractional history가 sparse함 |
| p 정답 mass 높음, M_eff 낮음 | 검색기는 집중하지만 residual/gate가 실제 배분을 제한 | 모델이 정답 memory를 충분히 사용함 |
| M_eff 높음, 성능 변화 없음 | 해당 coefficient allocation만으로 실용 효용 미확인 | 정답을 읽으므로 기전 검증 완료 |
| Learned가 Full 개선, α=1도 같은 개선 | Explicit history redistribution의 가능성 | Fractional kernel이 필수 |
| Heterogeneous 개선, selected는 불필요 | Multi-timescale 상태의 가능성 | Content selector의 기여 |
| Oracle≈Full, η가 매우 작음 | Oracle 효과가 충분히 주입되지 않을 가능성 | 구조 자체가 기억을 사용할 수 없음 |
| Simple recurrent도 같은 recall 성능 | 과제가 압축 기억으로 해결될 가능성 | Explicit bank가 필수라는 증명 |
| 수치 게이트 통과, 큰 η에서 증폭 | 현재 안전 범위의 제약 발견 | 계수 질량 보존이 안정성 증명 |
| ETT MSE 개선, 비용 크게 증가 | 정확도–비용 trade-off | Sparsemax를 썼으므로 에너지 효율적 |
| Recall 성공, ETT 개선 작음 | 과제 범위별 메커니즘 효용 차이 | 전체 아이디어 자동 실패 |

### 21.2. 연구 진행 판단

**실현 가능성:** 리셋 없는 여러 기억 가지, 소마 발화, fractional 적분, 내용 기반 선택은 각각 설명 가능한 계산 요소다. 근접한 연구도 존재하므로 모델 계열 자체를 배제할 이유는 없다.

**이론적 타당성:** 현재 식으로 무엇이 성립하고 무엇이 성립하지 않는지를 수정해야 한다. Pure fractional memory의 이론을 content-dependent model에 그대로 이전하거나, mass conservation을 안정성으로 오해해서는 안 된다.

**신규성:** Fractional과 soma의 결합 자체가 아니라, **현재 population 문맥에 따른 과거 dynamics의 사건별 재분배가 실제로 필요한가**에 초점을 맞춘다.

**실험 판단:** Operator correctness → 안정성·시간 정렬 → controlled recall → practical forecasting을 분리한다. 모든 문제를 ETT MSE 하나로 판정하지 않는다.

### 21.3. 코딩 에이전트용 한 문단 요약

> v3-A는 리셋 없는 fractional branch population과 ordinary LIF soma를 결합하고, population 문맥으로 과거 signed dynamics의 적분 계수를 재분배하는 모델이다. 공유 소마 구조는 유지하되 sparsemax p와 최종 history support를 구분해야 한다. 현재 π 항은 kernel mass conservation과 α=1 Euler 환원 조건을 바꾸므로, 첫 기전 검증에서는 g=0을 권고하고 유지하는 변형은 별도로 명명한다. Mass-preserving redistribution에도 recent-slot 정책에서 큰 branch state 증폭이 가능한 수치 예가 있으므로 η·support·history length의 안전성 검사를 필수로 둔다. Full fast path는 raw b를 입력에 곱하는 것이 아니라 누설 feedback을 포함한 유효 연산자여야 한다. Recall task는 causal response, sequence별 독립 value, 명확한 query alignment를 사용하며 oracle은 최적해가 아니라 별도의 privileged policy로 취급한다. O7은 effective 정답 mass 0.5, 유효한 oracle headroom의 50% 회수, 평균 Full 대비 20% 개선과 paired 불확실성, oracle≈Full 5%의 진단 폭으로 수정하는 것을 권고한다. 이 권고를 새 계약으로 승인받기 전에는 원본 확정사항처럼 덮어쓰지 않는다.

---

## 참고문헌 및 출처 기록

### [S1] 사용자 제공 v3-A IDEA LOG

- 제목: *IDEA LOG — Population f-LIF with Shared History Selection*.
- 원본 표기: 2026-09-21, v3-A, 설계 확정·O7 승인 대기.
- 실제 첨부 파일: `붙여넣은 마크다운(1)(1).md`.
- 이 검토에 사용한 파일 크기: 30,321 bytes.
- SHA-256: `e05d22f7fbc359e591636e4ccf8d1d9dd4184d8c3681e54224372f0f14277640`.
- 이 문서의 S1 절 번호는 원본 제목에 따른다. 코드·표·연산 정의와 기존 보고값의 주요 근거다.

### [S2] 사용자 제공 f-SNN 원문

- 파일: `ICLR-2026-fractional-order-spiking-neural-network-Paper-Conference.pdf`.
- 주요 위치: §3의 fractional dynamics와 discretization, Appendix B/C/E.
- 원문 이론과 공개 코드의 reset 규약, v3-A의 별도 soma 규약은 동일하지 않다.

### [R1] Fractional-order Spiking Neural Network

Ge et al. *Fractional-order Spiking Neural Network*. ICLR 2026 표기가 있는 사용자 제공 원문; 공개 arXiv record 2507.16937.

- [공개 원문 기록](https://arxiv.org/abs/2507.16937)
- [공식 spikeDE 저장소](https://github.com/PhysAGI/spikeDE)
- [기존 계획이 고정한 source revision](https://github.com/PhysAGI/spikeDE/tree/fcd743befe504b1a471fa81887e6af7d6789da2e)

### [R2] Fractional LIF와 voltage-history adaptation

Teka, W., Marinov, T. M., and Santamaria, F. (2014). *Neuronal Spike Timing Adaptation Described with a Fractional Leaky Integrate-and-Fire Model*. PLOS Computational Biology 10(3), e1003526.

- [출판사 원문](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003526)

### [R3] DH-SNN / DH-LIF

Zheng, H., et al. (2024). *Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics*. Nature Communications 15, 277. DOI: 10.1038/s41467-023-44614-z.

- [공개 full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- [출판사 DOI](https://doi.org/10.1038/s41467-023-44614-z)

### [R4] TC-LIF

*TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling*. Proceedings of AAAI, 2024.

- [공식 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29625)

### [R5] TS-LIF

*TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting*. ICLR 2025.

- [논문](https://arxiv.org/abs/2503.05108)
- [공식 구현](https://github.com/kkking-kk/TS-LIF)

### [R6] LongSpike

He, X., Kang, Q., Li, X., and Zha, Z.-J. (2026). *LongSpike: Fractional Order Spiking State Space Models for Efficient Long Sequence Learning*. arXiv:2606.12895v1, 2026-06-11 공개 preprint.

- [공개 기록](https://arxiv.org/abs/2606.12895v1)
- [원문 PDF](https://arxiv.org/pdf/2606.12895)
- 관련 위치: §4.2의 f-SSM→LIF 연결, §5.2 Table 1, §5.3의 SOE M=2 설명.
- 논문 표의 수치를 확인했지만 해당 모델을 설치·학습 재현하지는 않았다.

### [R7] NvoFDE

Cui, W., et al. (2025). *Neural Variable-Order Fractional Differential Equation Networks*. Proceedings of AAAI. DOI: 10.1609/aaai.v39i15.33769.

- [공식 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33769)

### [R8] PSN

Fang, W., et al. (2023). *Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies*. NeurIPS 2023.

- [공식 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html)

### [R9] TIM

Shen, S., Zhao, D., Shen, G., and Zeng, Y. (2024). *TIM: An Efficient Temporal Interaction Module for Spiking Transformer*. arXiv:2401.11687.

- [공개 논문](https://arxiv.org/abs/2401.11687)

### [R10] Mamba

Gu, A., and Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.

- [공개 논문](https://arxiv.org/abs/2312.00752)

### [R11] FLAMES

Chakraborty, B., and Mukhopadhyay, S. (2025). *FLAMES: A Hybrid Spiking-State Space Model for Adaptive Memory Retention in Event-Based Learning*. arXiv:2504.01257.

- [공개 preprint](https://arxiv.org/abs/2504.01257)

### [R12] Zoology / MQAR

*Zoology: Measuring and Improving Recall in Efficient Language Models*. ICLR 2024; arXiv:2312.04927.

- [공개 기록](https://arxiv.org/abs/2312.04927)
- [검토한 full text](https://arxiv.org/html/2312.04927v1)
- 관련 위치: §3.2의 쉬운 associative recall과 MQAR 구분, §4의 효율적 모델 비교.

### [R13] Sparsemax

Martins, A. F. T., and Astudillo, R. F. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*. ICML, PMLR 48:1614–1623.

- [공식 논문](https://proceedings.mlr.press/v48/martins16.html)

### [R14] Tempered fractional calculus

Sabzikar, F., Meerschaert, M. M., and Chen, J. (2015). *Tempered fractional calculus*. Journal of Computational Physics 293:14–28.

- [출판사 페이지](https://www.sciencedirect.com/science/article/pii/S0021999114002873)
- 보충 수학 원문: Fernandez, A., and Ustaoğlu, C. *On some analytic properties of tempered fractional calculus*.
- [보충 원문 공개 기록](https://arxiv.org/abs/1912.05482)

### [R15] Attention is not Explanation

Jain, S., and Wallace, B. C. (2019). *Attention is not Explanation*. NAACL-HLT.

- [ACL Anthology](https://aclanthology.org/N19-1357/)

### [R16] Attention is not not Explanation

Wiegreffe, S., and Pinter, Y. (2019). *Attention is not not Explanation*. EMNLP-IJCNLP.

- [ACL Anthology](https://aclanthology.org/D19-1002/)

두 attention 논문은 동일한 결론을 주장하는 것으로 합치지 않았다. 전자는 raw attention의 설명성 한계를 강조하고, 후자는 더 명시적인 평가 조건과 검증을 통해 논의를 구체화한다. 현재 문서가 채택하는 결론은 **가중치만으로 faithfulness를 자동 인정하거나 자동 부정하지 말고, 목적에 맞는 대조와 개입을 수행한다**는 것이다.

---

## 부록 A. 이번 문서에서 다시 계산한 수치와 재현 조건

이 부록은 사용자 저장소 학습 결과가 아니다. 순수한 스칼라·행렬 계산으로 수식 해석을 확인한 기록이다. 새로운 모델 source code를 지시하거나 제공하는 것이 아니라, 동일한 의미의 독립 계산이 가능하도록 조건을 적는다.

### A.1. Fractional coefficient

`α=0.7`, `h=1`, `d=0,…,41`:

| 양 | 값 |
|---|---:|
| b₀ | 1.1005474055236655 |
| b₁ | 0.6872971293568044 |
| b₄₁ | 0.25193961608462245 |
| Σ(d=0…41)b_d | 15.062021305609646 |
| B₄₁=Σ(d=1…41)b_d | 13.96147390008598 |
| sigmoid(−4) | 0.01798620996209156 |
| softplus(−5) | 0.006715348489118068 |

### A.2. 최근 과거 한 항 선택 진단

각 n>0에서 `p_{n,n−1}=1`이고 다른 과거는 0이다. 그러면 `ρ̃_{n,n−1}=B_n/b₁`, 나머지 ρ̃는 0이다. 이 값을 원본 residual 식에 넣고, `f_n=(I_n−u_n)/2`를 매 step 실제 상태로 재평가한다.

`I₀=2`, `I₁… I₄₁=0`, 초기 u₀=0, 42회 갱신, float64 계산을 사용했다. 과거 f_j는 당시 계산된 값을 보관하고 현재 상태로 다시 계산하지 않았다. g=0 조건의 mass 오차는 대략 10⁻¹⁶ 수준이었다.

이 설정은 7절의 표를 재현한다. 소마, 학습된 Q/K, 학습 loss가 없으므로 결과를 trained network의 실패율이나 최종 forecasting 성능으로 쓰면 안 된다.

### A.3. π에 따른 κ

`n=41`, 과거 lag=41,…,1, 균일 p, `g=softplus(−5)`에서

$$
\kappa_{41,k}=\frac{\sum_{d=1}^{41}b_d e^{-dg/\tau_k}}{\sum_{d=1}^{41}b_d}
$$

를 계산했다. 결과는 5.3절 표와 같다. g가 실제로 0이 아닌 초기값이므로, 이를 “수치 오차 때문에 1에서 약간 벗어났다”고 해석하지 않는다.

### A.4. 유효 선형 연산자

T=42, α=0.7, τ=4, g=0, u₀=0에서 B와 J를 8절처럼 구성했다. 고정 seed 7의 표준정규 입력에 대해 순차 recurrence와 유효 선형 연산자의 결과를 비교했으며 최대 절대 오차가 약 1.11×10⁻¹⁶이었다.

이 계산이 확인한 것은 선형 대수식의 대응이다. 실제 구현의 batch, autograd, GPU, memory usage 검증은 별도다.

### A.5. MDE

Paired difference가 정규분포 가정을 만족하고 표준편차가 0.0043이라고 두었다. n=8이면 자유도는 7이고 양측 임계값은 t_{0.975,7}다. 차이 δ의 noncentrality는

$$
\nu=\frac{\sqrt8\,\delta}{0.0043}.
$$

Noncentral t 분포의 양측 rejection probability가 0.8이 되는 δ를 계산하여 약 0.0049709491을 얻었다. 이 값은 **가정한 test의 power 계산**이며 새 모델의 관측 효과나 경험적으로 확인된 분산이 아니다.

### A.6. Oracle effective mass

g=0, p가 정답 사건에만 위치, baseline coefficient mass m₀=0.15인 조건에서

$$
M_{\mathrm{oracle}}^{\mathrm{eff}}
=(1-\sigma(-4))\times0.15+\sigma(-4)
\approx0.1652882785.
$$

따라서 p-perfect와 effective-allocation-perfect를 동일하게 취급하지 않는다.

---

## 부록 B. 원본 문구를 수정할 때의 대응표

| 원본 표현·위치 | 권장 수정 의미 |
|---|---|
| §2 “상태를 물려주지 않는다” | 직접적인 1-step Euler 식으로만 계산하지 않는다는 뜻으로 한정. fₙ에는 현재 uₙ이 들어가므로 상태 의존성은 남음 |
| §3.5 “α=1, η=0 → ordinary integrator” | π=1/g=0 조건을 추가 |
| §3.5 “K=1 → scalar f-LIF+soma” | Scalar fractional branch + soma로 표현 |
| §3.6 “보존되는 것은 멱함수 기억” | g>0이면 tempered kernel이며 순수 멱함수 꼬리가 아님 |
| §3.7 “ρ는 순수 증거만 조절” | Reset 항 없는 입력–누설 dynamics를 조절 |
| §3.7 “O(T²)가 정확히 선택에만 귀속” | 실제 full/selected 계산 경로와 soma 비용을 정의·측정한 뒤 주장 |
| §3.8 “η=0에서만 cross-Jacobian=0” | η>0에서 일반적으로 경로가 가능하지만 합법적인 0 구간 존재 |
| §4 “0·3번 완전 배제” | p의 support에서 제외. Residual이 있으면 최종 기여는 유지 |
| §5 “α=1을 못 이기면 fractional은 빠져야 함” | 사전 선언한 기준에서 추가 역할이 지지되는지 판단 |
| §6.2 “내용을 보고 골라야만 풀림” | Content-conditioned recall을 검사하되 explicit bank 필요성은 비교로 검증 |
| §6.3 Uniform/Full/mass-matched | g=0 질량 보존안에서는 중복 가능성을 식으로 확인 |
| G8 “κ=1, sparse exact zero” | π 전후 질량과 p/최종 coefficient support를 분리 |
| G10 “모든 gradient 비영” | 비퇴화 시험에서 활성 경로의 유효 gradient 확인 |
| O7 “oracle=이 구조의 최선” | 정답 위치를 사용하는 특정 privileged policy |
| O7 “oracle≈full이면 구조 실패” | Oracle headroom·η·과제·timing·학습을 먼저 진단 |

**문서 끝.**
