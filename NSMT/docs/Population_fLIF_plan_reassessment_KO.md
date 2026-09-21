재검토 결과, **기존 계획서는 모델 정의를 확정한 구현 사양으로 사용하기 전에 수정해야 해.** 특히 **τ 규약, 발화·reset 순서, 선택 계수의 정규화**는 설명을 보완하는 수준이 아니라 실제로 어떤 뉴런을 만드는지를 바꾸는 문제야.

기존 영문 계획서, 첨부 논문의 해당 수식, 고정 커밋의 `neuron.py`, `solver.py`, `surrogate.py`를 다시 대조했어. **검토문의 주요 문제 제기는 타당하지만, 제안된 수정안을 그대로 채택하면 생기는 문제도 있어.** 대표적으로 `ρ=np`가 fractional kernel의 총질량을 보존하는 것은 아니고, arctangent의 scale도 **함수 기본값과 실제 LIF 호출값을 구분**해야 해.

아래는 항목별 재판단이야. 새로 제시한 **v2 288회 최종 결과, 검정력, window-mean 수치와 서버 상태는 이번 검토문에서 제공한 정보**로 취급했으며, 원시 결과나 서버 실행으로 독립 검증한 것은 아니야.

---

## 2.1. τ 스케일 — 지적이 맞고, 명시적인 규약 선택이 필요하다

논문 p.5의 이산식과 식 (15)는 $1/\tau$ 규약인데, p.6의 $c_m^{(\alpha)}$와 꼬리식에는 $\tau^\alpha$가 들어가 있어. 실제 페이지에서도 확인했어. 반면 고정된 공개 코드는 뉴런의 dynamics에서 `/tau`를 적용해. **기존 계획서가 이 불일치를 밝히지 않은 채 논문 식 (11)–(13)을 한꺼번에 근거로 사용한 것은 부정확했어.** fileciteturn14file0L60-L109 fileciteturn14file0L140-L183 ([raw.githubusercontent.com](https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py))

### 수정 판단

첫 기준 규약은 다음처럼 명시하는 것이 맞아.

\[
\boxed{
D^\alpha U
=
\frac{-U+I+\text{coupling}}{\tau}
}
\]

즉, **코드 및 논문 p.5와 일치하는 $1/\tau$ 규약을 채택**하고, 식 (13)에 인쇄된 $1/\tau^\alpha$와 다르다는 사실을 적어야 해.

그 결과 predictor coefficient는

\[
b_d^{(\alpha)}
=
\frac{h^\alpha}{\Gamma(\alpha+1)}
\left[(d+1)^\alpha-d^\alpha\right]
\]

로 두고, $1/\tau_k$는 dynamics 안에 한 번만 넣는 기존 §6.8의 계산 자체는 유지할 수 있어. **잘못된 부분은 그 계산을 논문의 모든 수식과 일치한다고 읽히게 한 출처 설명**이야. fileciteturn12file0L646-L676

### 검토문에서도 한 문장은 보완해야 해

“$1/\tau$ 규약에서는 $\alpha$가 커널 모양만 바꾼다”는 표현은 엄밀하지 않아.

$\alpha$는

\[
\frac{h^\alpha}{\Gamma(\alpha+1)}
\]

와 유한 구간의 커널 총질량도 바꿔. 예를 들어,

\[
\sum_{d=0}^{N-1}b_d^{(\alpha)}
=
\frac{(Nh)^\alpha}{\Gamma(\alpha+1)}.
\]

따라서 정확한 구분은 다음과 같아.

> **$1/\tau$ 규약은 별도의 τ 의존 gain을 $\alpha$와 분리하지만, $\alpha$가 전체 시간 응답과 누적 크기에 미치는 영향까지 제거하지는 않는다.**

또한 $1/\tau$가 작다는 것은 dynamics의 입력 계수가 작다는 뜻이지, 무발화 정상상태의 입력–출력 gain이 그만큼 작다는 뜻은 아니야. $-U+I$ 전체에 동일한 계수가 곱해지므로 일정 입력의 정상상태는 여전히 $U_\infty=I$야. 이 구분은 2.7의 발화율 해석에도 필요해.

---

## 2.2. 스파이크 판정량과 reset — 가장 먼저 수정해야 하는 문제다

**이 지적에는 동의해.** 기존 계획서는 source parity를 우선하다가, 사용자가 원하는 “fractional하게 계산된 막전위가 발화를 결정하는 뉴런”과 다른 규약을 주 모델로 고정했어.

고정된 공개 코드의 경로는 실제로 다음과 같아.

\[
D_n=\frac{-U_n+I_n}{\tau},
\]

\[
S_n=H(U_n+D_n-\theta),
\]

\[
F_n=D_n-\frac{\theta S_n}{\tau}.
\]

그다음 solver가 과거 $F_j$를 합성해 실제 상태를 계산해. 따라서 **발화 판정에 사용한 $U_n+D_n$와 적분된 다음 상태가 서로 다른 양**이라는 지적은 맞아. ([raw.githubusercontent.com](https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py))

간단한 계산으로도 차이를 볼 수 있어. $U_0=0$, $\tau=2$, $I_0=1.9$, $\theta=1$, $\alpha=0.7$, $h=1$이면,

\[
D_0=0.95
\]

이므로 local trial에서는 발화하지 않아. 하지만 fractional update는

\[
U_1=b_0D_0
\approx1.100547\times0.95
\approx1.04552
\]

가 돼. **실제 적분 상태가 threshold를 넘었지만, 그 상태를 기준으로 발화한 것은 아닌 것**이지. 이는 위 공개 코드 규칙으로 직접 계산한 예시야.

### 수정 판단: source parity와 주 모델 정의를 분리해야 한다

앞으로는 다음 두 목적을 섞지 않는 것이 맞아.

| 목적 | 기준 |
|---|---|
| 공개 코드 재현 | 고정 커밋의 실제 동작을 그대로 재현 |
| 우리가 제안하는 뉴런 | Fractional charge로 얻은 상태에서 발화를 판정하고, reset을 명확히 정의 |

따라서 **source-compatible 경로는 필수 참조 대조군으로 남기되, 주 모델의 발화 규약으로 자동 채택하지 않겠다.** Fractional charge 이후 발화·reset을 적용하는 경로를 Phase A로 올리고, M1의 필수 비교에 포함해야 해.

### 다만 “논문 식 (13)을 그대로 쓰면 해결된다”도 아직 충분하지 않다

여기서 추가로 발견한 중요한 문제가 있어.

식 (13)의 $U_0$를 고정 초기값으로 두고, 합 안의 $U_j$를 **post-reset 상태**로 해석하여 매번 전체 합을 다시 계산하면, **$\alpha=1$에서도 일반적인 reset LIF recurrence가 자동으로 복원되지 않아.**

이를 확인하기 위해 다음 조건을 사용하자.

\[
\alpha=1,\quad h=1,\quad \tau=2,\quad
I_t=2.4,\quad \theta=1,\quad U_0=0.
\]

첫 번째 step은 두 방식 모두

\[
V_1=1.2,\qquad S_1=1,\qquad U_1=0.2
\]

야.

그러나 두 번째 step에서, 식 (13)을 위 방식으로 문자 그대로 다시 합산하면

\[
V_2
=
\frac{2.4-0}{2}
+
\frac{2.4-0.2}{2}
=
2.3.
\]

반면 일반적인 Euler charge–reset LIF는

\[
V_2
=
U_1+\frac{2.4-U_1}{2}
=
1.3
\]

이야. **이전 reset의 상태 감소량 1만큼 차이가 생겨.**

이것은 논문의 모든 해석이 틀렸다는 단정이 아니라, **“post-reset 상태를 저장한다”만으로는 전체 이력 적분에서 reset을 어떻게 누적할지가 완전히 정의되지 않는다는 점**을 보여줘. 논문은 식 (13)과 그 뒤에서 charge–spike–reset 및 integer-order 환원을 함께 설명하므로, 이 경계조건을 반드시 검사해야 해. fileciteturn14file0L95-L124

### 따라서 Phase A의 필수 검증을 더 강화해야 해

**공개 코드 경로**, **식 (13)의 명시적인 해석**, **우리의 자기일관적인 charge–reset 경로**를 구분해야 해. 마지막 경로에는 다음 조건이 필요해.

- 발화는 실제 fractional charge에 의해 결정된다.
- 과거 reset의 직접적인 상태 감소를 선택자가 임의로 지우지 않는다.
- $\alpha=1$, full history에서 선언한 ordinary charge–reset 규약으로 환원된다.

예를 들어, 별도의 **새 이산 hybrid 모델 후보**로 다음처럼 과거 reset 점프를 따로 보존할 수 있어.

\[
V_{n+1}
=
U_0+
\sum_{j=0}^{n}
b_{n-j}\rho_{n,j}D_j
-
\sum_{r=1}^{n}\theta S_r,
\qquad \rho_{n,n}=1,
\]

\[
S_{n+1}=H(V_{n+1}-\theta),
\qquad
U_{n+1}=V_{n+1}-\theta S_{n+1}.
\]

이 후보는 $\alpha=1$, $\rho=1$에서

\[
V_{n+1}=U_n+hD_n
\]

을 복원해. 다만 **이 식을 원문에 있던 정답이나 canonical Caputo reset이라고 부르면 안 돼.** 우리가 명시적으로 정의하고 검증할 별도 reset 모델이야.

즉, 수정의 핵심은 **“논문 규약으로 무조건 교체”가 아니라 “발화·reset의 수학적 일관성을 주 모델 선택의 선행 조건으로 격상”**하는 것이야. Synthetic 성능이 좋아도 이 조건을 통과하지 못하면 주 모델로 선택하지 않는 편이 맞아.

---

## 2.3. 선택자가 감쇠만 가능 — 맞지만, `ρ=np`도 그대로 채택하기에는 부족하다

기존 계획의

\[
\rho_{n,j}=\frac{p_{n,j}}{\max_\ell p_{n,\ell}}
\]

는 실제로

\[
0\le\rho_{n,j}\le1
\]

을 강제해. Dense에서는 지적대로

\[
\rho_{n,j}
=
\exp(e_{n,j}-\max_\ell e_{n,\ell})
\]

가 돼. 따라서 **원래 fractional coefficient보다 특정 과거의 계수를 키울 수는 없는 감쇠 전용 모델**이야. 이 설계 선택을 아이디어의 일반적인 구현처럼 제시한 것은 수정해야 해. fileciteturn12file0L602-L644

다만 두 가지는 구분해야 해.

**첫째**, 먼 과거가 가까운 과거보다 상대적으로 중요해지는 것은 기존 설계에서도 가능해. 가까운 과거를 충분히 억제하면 되니까. 그러므로 이는 곧바로 구현 오류라기보다, **검증하는 가설을 감쇠 기반 선택으로 제한한 설계**야.

**둘째**, 계수 총질량이 줄어드는 것과 실제 막전위 기여가 줄어드는 것도 같지 않아. $D_j$나 $F_j$는 부호가 있으므로, 음의 기여를 제외하면 최종 막전위는 오히려 증가할 수 있어.

### 제시한 대안의 장점과 한계

\[
\rho_{n,j}=np_{n,j}
\]

는 균일한 $p$에서 $\rho=1$을 복원하고, 선택된 과거를 원래보다 증폭할 수 있어. 이 장점은 맞아.

그러나 이것이 보존하는 것은

\[
\sum_j\rho_{n,j}=n
\]

이지,

\[
\sum_jb_{n-j}\rho_{n,j}
=
\sum_jb_{n-j}
\]

가 아니야. Fractional coefficient가 lag마다 다르므로 **여전히 선택 위치에 따라 전체 커널 질량이 변해.**

### 내가 제안하는 수정 방향

감쇠 전용을 삭제하기보다 **감쇠 대조군**으로 유지하고, 증폭을 허용하는 후보를 명시적으로 분리하는 것이 좋아.

특히 “선택 위치의 효과”를 분리하기 위한 진단 후보로는,

\[
B_n=\sum_{j<n}b_{n-j},
\]

\[
\boxed{
\rho_{n,j}
=
\frac{B_n\,p_{n,j}}
{\sum_{\ell<n}b_{n-\ell}p_{n,\ell}}
}
\]

를 고려할 수 있어.

이 후보는

\[
\sum_{j<n}b_{n-j}\rho_{n,j}=B_n
\]

을 만족하고, 균일 점수에서 $\rho=1$, sparsemax의 0도 유지해. 특정 과거의 원래 계수를 증폭하는 것도 가능해.

단, 이 역시 안정성 보장은 아니야. 먼 과거에 확률이 몰리면 $\rho$가 매우 커질 수 있어. 상한 $\rho_{\max}$까지 두려면 선택된 support $\mathcal S_n$가 최소한

\[
\rho_{\max}
\sum_{j\in\mathcal S_n}b_{n-j}
\ge B_n
\]

을 만족해야 해. 그렇지 않으면 **정확한 sparse support, 상한, 총질량 보존을 동시에 만족시킬 수 없어.**

따라서 새 계획에서는 다음을 사전에 선언해야 해.

> **감쇠만 허용하는가, 재분배·증폭을 허용하는가, 어떤 질량을 보존하는가, 상한과 보존 조건이 충돌하면 무엇을 완화하는가?**

이 질문을 해결하지 않은 채 `max`를 `mean`으로만 바꾸면 문제가 다른 형태로 이동할 뿐이야.

---

## 2.4. Diffusive coupling — 단일 후보로 고정한 것은 수정하되, 항상 해롭다고 보지는 않는다

기존의

\[
-\lambda L_{\mathrm{pop}}\mathbf U
\]

가 구성원 차이를 줄이는 방향으로 작용한다는 지적은 맞아. 계획서도 이를 인정하면서 $\lambda=0.1$을 첫값으로 고정했는데, **그 강도가 실제로 의미 있는지 확인하는 선행 절차가 부족했어.** fileciteturn12file0L467-L513

다만

> 차이를 줄인다 → key 품질이 반드시 나빠진다

까지는 아직 결론 내릴 수 없어. 무관한 변동을 줄여 관련 패턴을 더 안정적으로 만들 가능성도 있으므로, **상관 증가와 정보 품질 저하를 동일시하면 안 돼.**

### 수정 판단

제안한

\[
\lambda\in\{0,0.05,0.1,0.3\}
\]

의 소규모 진단을 M1 이전에 수행하는 방향은 타당해. 발화율·상태 상관뿐 아니라 **전체 dynamics에서 coupling 항의 크기**, effective rank, 구성원 교란이 다른 구성원에 미치는 영향도 함께 봐야 해.

또 비확산형 후보를 하나 추가하는 것은 검토할 가치가 있어.

### 반대칭 결합에서 주의할 점

\[
A^\top=-A
\]

이면

\[
\mathbf U^\top A\mathbf U=0
\]

이므로, 선형 ODE에서 순수 반대칭 항은 유클리드 상태 에너지를 직접 증가·감소시키지 않아. 하지만 이를 **fractional integration, heterogeneous $\tau$, spike/reset까지 포함한 모델의 다양성 보존·안정성 보장**으로 확대하면 안 돼.

또 일반적인 반대칭 행렬은

\[
A\mathbf1\ne0
\]

일 수 있어. 이 경우 **동질적인 population도 coupling 때문에 서로 다른 상태로 갈라질 수 있어.** 그러면 기존 계획의 “homogeneous coupling은 항상 0이므로 중복 실험을 생략한다”는 조건이 더 이상 성립하지 않아. fileciteturn12file0L761-L769

동질성 불변성을 유지하려면 $A\mathbf1=0$ 같은 추가 조건을 두거나, 불변성이 깨지는 모델임을 인정하고 비교군을 다시 구성해야 해.

**수정된 해석 기준은 다음이 적절해.**

> Diffusive coupling의 무효과는 상호작용 전체의 반증이 아니다. 반대로 반대칭 coupling의 이득도 단순한 추가 상태 다양성 때문인지 분리해야 한다.

두 형태를 비교할 때는 $\lambda$ 숫자만 맞추지 말고 연결 행렬의 norm이나 실제 coupling 크기도 맞춰야 해.

---

## 2.5. 실행 환경 — CPU reference 경로를 Phase A의 정식 절차로 넣는다

제시한 서버 상태를 전제로 하면 **별도 CPU reference에서 원본 궤적을 확보하는 방안이 타당해.** 기존 환경의 핵심 제약은 `torch.fx` 자체보다 고정 소스에 존재하는 `torch.compile` 호출이야. `torch.compile`은 PyTorch 2.0에서 도입됐고, 해당 저장소는 import·wrapper 경로에서 이를 사용해. ([raw.githubusercontent.com](https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/snn.py))

CUDA 13.x의 일반적인 minimum-driver 기준은 580 이상이므로, 제시된 535 계열 드라이버와 cu130의 조합에 문제가 있다는 설명도 공식 호환성 문서와 부합해. 다만 **`nvidia-smi`의 CUDA 표시값을 모든 minor-version 조합의 절대 상한으로 해석해서는 안 돼.** ([docs.nvidia.com](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html))

### 수정할 reference 등급

| 등급 | 실제로 확인한 것 |
|---|---|
| **Source parity** | 고정 원본을 실행한 궤적과 대조 |
| **Source-derived** | 원본 수식을 옮겼지만 원본 실행 비교는 하지 못함 |
| **Mathematical validation** | 독립적인 해석해·수치해법과 비교 |

CPU golden 자료에는 출력 spike뿐 아니라 **입력, local trial, 적분 상태, dynamics, reset 관련 항**도 함께 남기는 것이 좋아. 출력만 같아서는 내부 불일치를 발견하기 어려워.

또 CPU golden과 일치했다고 **GPU 전체 학습 경로까지 검증된 것은 아니므로**, 본 실험 환경에서는 짧은 궤적의 device/dtype 대조를 추가해야 해.

설치가 불가능한 경우 `source-derived`라고 부르자는 제안에도 동의해. 다만

\[
U(t)=U_0+\frac{ct^\alpha}{\Gamma(\alpha+1)}
\]

하나만 통과했다고 f-LIF 전체를 검증했다고 해서는 안 돼. 이것은 **상수 forcing 적분기 검사**이고, 누설이 있는 relaxation, $\alpha=1$ 환원, 반복 발화·reset은 별도 검사야.

---

## 2.6. 통계 규모 — 실험 종류를 늘리기보다 핵심 대비의 반복 수를 늘려야 한다

기존 계획은 개발 3 seed, 확인 5 seed로 두고 아키텍처당 108회 이상의 matrix를 제시했어. 다수 조건을 돌리지만 **정작 핵심 효과 하나를 판정하는 반복 수는 작은 설계**였어. 이 부분은 수정하는 것이 맞아. fileciteturn13file0L152-L200

### 다만 8–10 seed가 검정력 문제를 해결한다고 말할 수는 없다

제시한 검정력 계산대로 H720에서 1% 효과에 수백 쌍이 필요하다면, 8–10 seed는 개선이지만 그 효과를 검출하기에 충분하다는 뜻은 아니야.

또한

\[
2\ \text{datasets}\times3\ \text{seeds}=6
\]

을 동일한 모집단의 독립적인 6쌍으로 취급해서도 안 돼. Dataset 차이와 초기화 seed 변동은 다른 종류의 변동이야.

### 수정 판단

핵심은 seed 숫자 하나가 아니라 **사전 목표 효과와 판정 단위**를 정하는 것이야.

우선 M1에서 주 대비를

\[
\Delta_{\mathrm{selection}}
=
E_{\mathrm{coupled,sparse}}
-
E_{\mathrm{coupled,full}},
\]

\[
\Delta_{\mathrm{interaction}}
=
(E_{\mathrm{coupled,sparse}}-E_{\mathrm{coupled,full}})
-
(E_{\mathrm{uncoupled,sparse}}-E_{\mathrm{uncoupled,full}})
\]

로 좁히는 것이 좋아. 여러 아키텍처의 전체 조합을 늘리는 대신, 이 대비의 **dataset별 paired 차이와 신뢰구간**에 예산을 먼저 배분하는 방향이야.

주 메커니즘 판정은 **controlled recall**, forecasting의 우선 판정은 **H96**, H720은 **별도로 공개하는 보조·스트레스 평가**로 두는 제안에 동의해. 다만 H720에서 불리한 결과가 나왔다는 이유로 사후 제외해서는 안 돼.

### Window-mean보다 나쁘다는 결과의 의미

제공된 ETTh2 H720 수치가 같은 프로토콜에서 계산된 것이라면, 실용성에 중요한 경고야. 하지만 다음 두 질문은 분리해야 해.

> 선택 기능이 같은 모델의 오차를 줄였는가?

> 그 모델 자체가 단순 예측기보다 유용한가?

단순 기준선보다 나쁜 모델에서도 첫 질문의 답은 분석할 수 있어. 다만 그것을 **유용한 장기 forecasting 모델의 증거로 제시하면 안 돼.**

Window-mean, persistence, 선형 예측기는 먼저 동일 split·scaling·metric으로 고정해야 하고, 8–10 seed는 우선 변동 추정의 출발점이지 “통계적 확증 완료”의 숫자로 쓰지 않는 것이 맞아.

---

## 2.7. 발화율 보정 — 필요하지만, 이질성 자체를 지우는 보정은 피해야 한다

학습 전에 실제 입력 분포에서 구성원별 발화 상태를 확인하자는 지적은 타당해. 기존 계획은 `input_scale=2`를 고정한 뒤 뒤늦게 nontrivial spike activity를 검사하도록 되어 있었어. fileciteturn12file0L815-L819 fileciteturn12file0L952-L956

다만 다음 두 작업은 달라.

| 작업 | 의미 |
|---|---|
| 모든 구성원이 죽거나 포화되지 않도록 입력 범위를 확인 | 유효한 동작 구간 확보 |
| 구성원별 발화율을 강제로 동일하게 맞춤 | Population의 반응 차이까지 제거할 수 있음 |

따라서 **train 구간만 사용하는 사전 calibration**을 넣되, 첫 단계에서는 비교군이 공유하는 input scale을 정하고 기록하는 것이 좋아. 구성원마다 별도 gain이나 threshold를 조정하면, “같은 입력에 대한 서로 다른 temporal response” 외에 다른 설계 요인이 추가돼.

또 $1/\tau$가 작다는 이유만으로 느린 구성원이 죽는다고 단정할 수는 없어. 누적, 누설, reset, 입력 분포가 함께 결정하므로 실제 궤적 검사가 필요해.

참고로 제시한 커널 총질량은 $h=1$, $\alpha=0.7$, 42개 항을 기준으로 계산하면

\[
\frac{42^{0.7}}{\Gamma(1.7)}
\approx15.062
\]

야. 약 15라는 직관은 맞지만, 이것을 일정한 입력 gain처럼 사용하는 것은 누설과 reset을 무시한 근사야.

---

## 2.8. Key 표현력 — ΔU 비교를 추가하는 것이 타당하다

기존 계획의

\[
\xi_n=[\mathbf U_n;I_n]
\]

와 선형 Q/K는 제한적인 현재 문맥이야. 그리고 coupling·$\tau$가 고정된 기존 설계에서는

\[
\mathbf D_n
=
\mathsf T^{-1}
[-\mathbf U_n+\mathbf1I_n-\lambda L\mathbf U_n]
\]

가 $(\mathbf U_n,I_n)$의 선형함수이므로, **이를 단순히 추가한다고 선형 projection의 표현 가능한 정보가 새로 늘지는 않는다**는 지적이 맞아. fileciteturn12file0L549-L598

반면

\[
\Delta\mathbf U_n=\mathbf U_n-\mathbf U_{n-1}
\]

는 현재 상태와 입력만으로 일반적으로 결정되지 않는 이전 상태 정보를 포함해.

따라서 아래 비교를 추가하겠다.

\[
\xi_n^{\mathrm{base}}=[\mathbf U_n;I_n],
\]

\[
\boxed{
\xi_n^{\Delta}=
[\mathbf U_n;I_n;\Delta\mathbf U_n].
}
\]

첫 시점의 차분값, pre/post-reset 중 어떤 상태의 차분인지, 모델 시간격자에 따른 스케일은 고정해야 해. Query와 historical key도 동일한 시점 의미를 사용해야 하고.

다만 **v2에서 해당 key가 도움이 적었다는 것이 새로운 fractional state에서도 실패한다는 증거는 아니야.** 상태가 만들어지는 dynamics 자체가 바뀌었으므로, 이 부분은 “실패가 확정된 설계”가 아니라 **우선순위 높은 표현력 비교**로 올리는 것이 정확해.

---

## 2.9. Arctangent surrogate — 실제 식은 정정하되, scale 5가 무조건 틀린 것은 아니다

여기는 검토문의 사실 정정을 그대로 수용하면 오히려 새로운 오류가 생겨.

### 확인된 사실

`surrogate.py`의 함수는

```text
arctan_surrogate(input, scale=2.0)
```

를 기본값으로 가지며, 실제 backward는

\[
\boxed{
g_s(x)
=
\frac{s/2}
{1+\left(\frac{\pi}{2}sx\right)^2}
}
\]

야. 논문 B.2.2의

\[
\frac{\kappa}{1+(\kappa x)^2}
\]

와 다른 식이라는 지적은 맞아. ([raw.githubusercontent.com](https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/surrogate.py)) fileciteturn14file1L227-L235

**하지만 `LIFNeuron`이 상속하는 `BaseNeuron`의 기본값은 `surrogate_grad_scale=5.0`이고, LIF는 이 값을 surrogate 함수에 명시적으로 전달해.** 따라서 기본 `LIFNeuron` 호출 경로에서는 arctangent 함수의 기본값 2.0이 적용되지 않고, 전달된 5.0이 사용돼. ([raw.githubusercontent.com](https://raw.githubusercontent.com/PhysAGI/spikeDE/fcd743befe504b1a471fa81887e6af7d6789da2e/spikeDE/neuron.py))

### 수정 판단

기존 문구는 다음처럼 바꾸는 것이 정확해.

> **Pinned implementation의 arctangent backward 식을 사용한다. Scalar source reference에서는 실제 호출값을 기록하며, 기본 `LIFNeuron` 경로의 값은 5.0이다. 함수 단독 호출의 기본값 2.0 및 논문 식 (30)과는 구분한다.**

첫 reset 비교에서는 surrogate까지 동시에 바꾸지 말고, **정확한 코드 식과 동일한 scale을 두 경로에 공통 적용**하는 것이 좋아. Scale 2를 비교하려면 별도의 명시적인 민감도 분석으로 두면 돼.

이 항목의 문제는 “5라는 값 자체가 반드시 틀렸다”가 아니라, **함수 기본값·호출값·논문 수식을 구분하지 않은 기술**이었어.

---

## 3. 논문 쪽 사실에 대한 판단

제시한 세 가지 해석도 대체로 동의해.

**Fractional이 항상 우월하다는 전제는 두면 안 돼.** 논문 내부에서도 데이터에 따라 $\alpha=1$이 선택되고, fractional order는 실험적으로 조정돼. 따라서 $\alpha=1$ 대조는 주 결과에서 유지해야 해. 다만 선택 기능을 켠 $\alpha=1$ 모델은 여전히 내용 의존적 이력 접근을 가지므로, 이것을 일반 Markovian LIF와 동일시하면 안 돼. fileciteturn0file0 fileciteturn12file0L737-L759

**원 논문에는 ETT forecasting 검증이 없다.** 따라서 그 논문의 분류·그래프 결과는 우리 forecasting 가설의 근거가 아니라 출발점이야. 다만 “이 논문에서 검증하지 않았다”와 “관련 분야 전체에 선례가 없다”는 다른 주장이라 후자까지 확대하지 않는 것이 맞아. fileciteturn0file0

**내용 의존적 계수에는 원문의 고정 convolution 가속 주장을 그대로 적용할 수 없다.** 다만 계수에 특별한 구조를 부여한 새로운 가속법의 가능성까지 부정하는 것은 아니고, 현재 설계에 그런 근거가 없다는 의미로 쓰면 돼. fileciteturn13file0L330-L346

---

## 4. 재검토 후의 수정된 진행 순서

기존의 **“source-compatible 모델을 먼저 고정하고 크게 확장”**하는 순서를 다음처럼 바꾸는 것이 맞아.

| 단계 | 먼저 결정·검증할 것 | 다음 단계로 넘어가는 기준 |
|---|---|---|
| **A. Scalar 모델 정의** | τ 규약, 실제 charge와 spike의 관계, reset 이력, surrogate, CPU reference | 출처별 차이가 명시되고 integer-order·reset 검사를 통과 |
| **B. Population 동작 진단** | 발화율 calibration, heterogeneity, coupling 형태·강도 | 죽음·포화·동질화와 실제 정보 교환을 구분 |
| **C. 선택 기능 진단** | 감쇠와 증폭, kernel-mass 변화, key와 ΔU, reset 보호 | 중립 극한·인과성·공동 선택·정확한 0·상태 크기 검사를 통과 |
| **D. Controlled recall + M1** | 알려진 관련 이력의 선택, H96 핵심 대비, H720 보조 평가 | 선택·결합·fractional order의 역할을 분리한 결과 확보 |
| **E. 반복·확장** | 목표 효과에 맞춘 seed와 데이터 확인, 이후 M3·TCN·attention | 불확실성과 반복 가능성을 확인한 뒤 확장 |

**기존 108회/아키텍처 matrix는 위의 모델 정의가 끝나기 전에는 실행 기본안에서 내려야 해.** 새로 추가되는 reset·gain·coupling 후보를 모두 곱해서 더 큰 matrix로 만드는 것도 피해야 해. 앞 단계의 수치·기전 진단으로 후보를 명시적으로 좁힌 뒤, 고정된 핵심 비교에 반복 예산을 쓰는 것이 맞아.

## 최종 판단

이번 검토에서 가장 중요한 교정은 **“공식 코드와 같으면 우리가 원하는 f-LIF의 확장이다”라는 기준을 버리는 것**이야.

앞으로는

\[
\boxed{\text{공식 코드와의 일치}}
\]

를 **출처 재현 문제**로,

\[
\boxed{\text{fractional charge·발화·reset의 자기일관성}}
\]

을 **모델 정의 문제**로,

\[
\boxed{\text{population 공동 선택과 상호작용의 유효성}}
\]

을 **연구 검증 문제**로 분리해야 해.

**아이디어 자체를 철회할 이유는 없지만, 기존 계획의 주 모델 계약은 다시 확정해야 해.** 특히 reset 규약을 먼저 해결하지 않으면, 이후의 성능 변화가 관련 기억을 선택한 효과인지, 과거 reset을 재가중한 효과인지조차 구분하기 어려워져. 이번 수정의 최우선은 아키텍처나 실험 횟수가 아니라 **그 구분이 가능한 뉴런 정의를 확보하는 것**이야.
