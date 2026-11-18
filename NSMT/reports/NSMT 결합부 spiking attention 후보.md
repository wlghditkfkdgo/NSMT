# Attention 종류보다 score 식을 먼저 고쳐라

2026-11-17 · 질문: "Hippocampus와 Neocortex 경로를 잇는 지금의 Spikformer식 softmax 없는 spiking cross-attention이 최선인가, 아니면 다른 spiking attention이 나은가?" · 근거: research note 4편(`research_notes/NSMT 결합부 spiking attention 후보/`), 이전 조사의 note 5편(`research_notes/NSMT query 구조 개선 제안/`), 코드(`nsmt_flif_v1/classification/layers.py`의 `SSA_rel_scl`, `energy.py`), 프로젝트 기록([MODI_PROJECT_LOG](../docs/MODI_PROJECT_LOG.md))

**지금의 결합 attention이 최선이라는 증거도, 다른 spiking attention이 ETT에서 더 낫다는 증거도 없다.** spiking cross-attention을 forecasting에서 시험한 논문은 찾지 못했다. ETT 계열에서 spiking attention을 다른 mixer와 바꿔 본 peer-reviewed 결과는 SpikF(ICML 2025) 하나뿐이고, 그 차이도 MSE 0.3–2.7 %였다. 대신 문헌은 우리 score 식의 구체적인 약점을 짚는다. NeurIPS 2025 forecasting 실험에서 "0/1 dot product + 더하는 상대 위치 bias" 조합은 무너지거나(ALiBi를 더한 Spikformer의 평균 R²가 0.733에서 0.413으로) 수렴하지 못했다. 우리 코드는 head당 channel이 8개(ETTh2) 또는 4개(ETTm2)라 query–key 한 쌍의 내용 점수가 최대 1.0 또는 0.5다. 반면 학습되는 bias에는 상한이 없다. 11-08 측정은 그 증상을 그대로 보여 준다. ETTm2에서는 attention 출력의 크기를 bias 항이 지배했고(Neocortex가 바꾸는 항의 비중 9–16 %), ETT 위치의 70–83 %에서는 query가 통째로 0이었다. 그러므로 attention 계열을 바꾸기 전에 score 식부터 고쳐야 한다. 가장 믿을 만한 교체는 같은 NeurIPS 2025 논문의 XNOR 유사도 + 정수 log 거리 bias(Log-PE)이고, 그다음이 발화율로 정하는 배율·정규화다. 기대 크기는 작게 잡아야 한다. 같은 논문에서 원래 Spikformer 대비 이득은 평균 R² +0.017(RSE 약 −6 %)이었지만, ETT에서는 순서가 준 몫 자체가 작았다(SpikF 실험에서 MAE 0.8 %, 우리 순서 대조는 "차이 없음"). 따라서 학습 없는 진단을 먼저 하고, 교체는 Spikformer식 비교 model에도 똑같이 넣어 "Neocortex가 만드는 차이"가 커지는지로 판정해야 한다. 또 우리 energy 계산은 attention 곱과 bias 덧셈을 전혀 세지 않으므로, 이 부분을 따로 세기 전에는 교체끼리 energy를 비교할 수 없다.

## 읽는 법과 용어

문장마다 근거의 종류를 나눴다. **논문 측정**은 논문이 자기 실험에서 잰 숫자다. **우리 기록**은 우리 log와 코드에서 확인한 사실이다. **추론**은 이 둘을 바탕으로 우리 model에 대해 내린 예상이며, 아직 시험되지 않았다. 논문 숫자 대부분은 영상 분류(CIFAR, ImageNet)에서 나왔다. seed를 여러 번 돌린 것은 Lv 연구진의 시계열 논문들(seed 3–5개) 정도이고, 영상 쪽 attention ablation은 대부분 한 번 돌린 숫자다. 우리 판정 규칙은 문헌의 어떤 ablation보다 엄격하다. 4칸 × seed 3개 = 12쌍에서 test MSE의 평균 상대 변화가 −1 % 이하이고 8쌍 이상에서 낮아야 "좋음"이다. note에 출처 URL 없이 적힌 수치는 쓰지 않았다.

| 용어 | 쉬운 뜻 |
|---|---|
| softmax | 점수를 지수 함수로 바꾼 뒤, query 하나의 점수 합이 1이 되도록 나누는 정규화. 지금 우리 module에는 없다 |
| Spikformer식 attention (SSA) | Q, K, V를 모두 0/1 spike로 만들고, softmax 없이 Q·Kᵀ·V에 고정 배율 s를 곱해 spiking neuron에 넣는 attention(ICLR 2023) |
| cross-attention | query와 key·value가 서로 다른 경로에서 오는 attention. 우리는 query = Neocortex, key·value = Hippocampus다 |
| base | 지금 기준 model. Neocortex 층 사이에 membrane shortcut을 둔 query population model(pop LIF, pop f-LIF) |
| Spikformer식 비교 model | 같은 network에서 Neocortex를 빼고 q, k, v를 모두 Hippocampus에서 만든 model |
| dot product | 0/1 벡터 두 개에서 "둘 다 1"인 channel 수 |
| XNOR 유사도 | 값이 같은 channel 수. "둘 다 1"과 "둘 다 0"을 모두 센다 |
| Hamming 유사도 (부호형) | 0/1을 −1/+1로 바꿔 곱해 더한 값. 같은 channel은 +1, 다른 channel은 −1이 된다 |
| head 차원 d | head 하나가 쓰는 channel 수. 우리는 64/8 = 8(ETTh2), 64/16 = 4(ETTm2)다 |
| 상대 위치 bias b_h(j − i) | head마다, token 사이 거리마다 하나씩 학습되는 실수. query와 무관하다 |
| ALiBi | 거리에 비례해 점수를 깎는 고정 bias |
| Log-PE | 거리의 log 값을 올림한 고정 정수 bias. 가까울수록 크다 |
| Gray-PE | 위치를 Gray code(이웃 위치끼리 한 bit만 다른 이진 부호)로 바꿔 q와 k 뒤에 붙이는 방법 |
| 발화율 | 0/1 출력 가운데 1의 비율 |
| 수렴하지 못함 | 학습 loss가 내려가지 않아 쓸 만한 model이 나오지 않음 |
| surrogate gradient | spike(계단 함수)의 미분 대신 쓰는 부드러운 대용 미분. 입력이 문턱에서 멀면 0에 가까워져 학습 신호가 사라진다 |
| IAND gate | x ← x·(1 − y). y가 1인 곳의 Hippocampus spike를 지운다 |
| membrane shortcut (MS) | 층 사이 residual을 spike 대신 막전위 쪽 실수 흐름으로 넘기는 방식 |
| AC / MAC | 덧셈만 하는 연산(0.9 pJ로 계산) / 곱셈과 덧셈을 함께 하는 연산(4.6 pJ로 계산) |
| syops | 우리 energy 계산 도구. Linear·BN module에 hook을 걸어 연산 수를 센다 |
| popcount | 1의 개수 세기. 0/1 dot product는 AND 뒤 popcount다 |
| quantized ANN | 활성값을 몇 bit 정수로 줄인 일반 신경망. T step SNN은 보통 ⌈log₂(T+1)⌉ bit짜리와 비교한다 |
| R² / RSE | forecasting 지표. R²는 1에 가까울수록, RSE는 낮을수록 좋다 |
| 다른 분야 지표 | perplexity·BPC(언어 model, 낮을수록 좋음), mIoU(segmentation, 높을수록 좋음), pose 오차(낮을수록 좋음), AUC(tracking 성공률 곡선 아래 넓이, 높을수록 좋음) |
| RoPE | query·key 벡터를 위치에 따라 회전시켜 위치를 넣는 방법. 일반 ANN Transformer에서 흔하다 |
| FLOPs / SOP | 실수 연산 수 / spike가 일으키는 synapse 연산 수 |
| transfer 설정 | 다른 데이터로 미리 학습한 가중치에서 시작하는 학습 |
| batch 불변성 | 같은 입력을 batch 1로 평가하든 batch 64로 평가하든 출력이 같아야 한다는 조건 |
| fail-closed | 값이 빠지거나 NaN이면 통과로 보지 않고 실패로 처리하는 판정 방식 |

## 나쁜 소식부터: 어떤 교체도 ETT에서 이긴다는 근거가 없다

**논문 측정 — 두 경로를 잇는 spiking cross-attention.** 이런 사례는 대부분 audio-visual 분류와 tracking에 있다. 그리고 거의 모두 지금 우리와 같은 Spikformer 형태를 쓴다. 한 경로에서 0/1 Q를, 다른 경로에서 0/1 K·V를 만들고, Q·Kᵀ·V에 배율을 곱하며, softmax는 없다. S-CMRL은 audio를 query, video를 key·value로 썼다. 직접 합치기와 비교해 CREMA-D 정확도가 **69.62 %에서 72.72 %**로 올랐다고 보고했다. 그러나 이 논문은 arXiv에만 있다. 또 다른 논문이 다시 돌렸을 때는 보고값 73.25 % 대신 **71.59 %**가 나왔다([S-CMRL](https://arxiv.org/abs/2502.12488); [재실행](https://arxiv.org/abs/2602.00701)). SpikeFET(저자 표기 NeurIPS 2025)에서는 token을 이어 붙여 함께 attention하던 것을 단순 더하기로 바꾸었다. 그 결과 FE108 AUC가 68.5에서 68.0으로 바뀌었다([SpikeFET](https://arxiv.org/abs/2505.20834)). 모두 seed 한 번의 숫자다. **regression이나 forecasting에서 spiking cross-attention을 다른 결합 방식과 비교한 논문은 찾지 못했다.**

**논문 측정 — attention 형태는 작은 손잡이다.** STEP(NeurIPS 2025)은 여러 spiking Transformer를 하나의 학습 recipe로 다시 돌렸다. Q와 K를 무작위 projection으로 고정해도 CIFAR-10 정확도는 **0.35점 미만** 떨어졌다(Spikformer 95.12 → 94.96 %). 이미지를 열 단위로 차례로 넣는 sCIFAR에서 Spikformer는 84.26 %였다. spike를 쓰지 않는 같은 구조는 85.62 %로 더 높았다. 저자들은 spike 기반 attention이 "spatial rather than temporal modeling"에 더 맞는다고 결론지었다([STEP](https://arxiv.org/html/2505.11151v2)). 논문 안에서 attention만 바꾼 효과도 작다. Spike-driven Transformer에서 SSA를 SDSA로 바꾸면 CIFAR-10/100이 **−0.03 / −0.34점**이었다. 같은 표에서 residual을 membrane shortcut으로 바꾼 효과(+0.81 / +1.46)가 더 컸다([SDT, NeurIPS 2023](https://arxiv.org/abs/2307.01694)).

**논문 측정 — 시계열.** 시계열 spiking 연구의 기준인 Lv 외(ICML 2024)의 iSpikformer는 SSA를 변수(channel) token 사이에 쓴다. ETT 결과는 없다([Lv 외](https://arxiv.org/abs/2402.01533)). 표준 ETT 숫자가 있는 peer-reviewed spiking forecaster는 SpikF(ICML 2025) 하나다. SpikF는 attention을 쓰지 않는다. 그 이유로 spiking self-attention이 순서에 둔감하다는 점을 든다. token을 섞으면 출력도 같은 방식으로 섞일 뿐이라는 것이다. SpikF는 자기 mixer를 변수 token SSA(iSSA)로 바꿔 보았다. MSE는 ETTh1 **0.440 → 0.452**, ETTm1 0.388 → 0.389, Weather 0.245 → 0.247이었다. 입력 순서를 고르게 섞는 대조에서 SpikF가 순서로 얻은 몫은 **MAE 0.8 %**였다. iTransformer는 0이었다([SpikF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)).

**우리 기록.** base와 Spikformer식 비교 model의 차이는 입력 96에서 pop LIF −0.12 %, pop f-LIF −0.21 %로 "차이 없음"이었다. 입력 336에서 pop LIF는 −0.71 %(12쌍 중 10쌍 낮음)로 역시 차이 없음이었다. pop f-LIF는 −1.57 %(8/12)로 "좋음"이었지만, 이 결과는 ETTh2 H96 한 칸(−4.48 %)에 기댄다. 그 칸은 Spikformer식의 seed 편차(표준편차/평균)가 0.43 %에서 4.10 %로 커진 칸이다. 그 칸을 빼면 −0.60 %(5/9)다([MODI 11-17 10:40](../docs/MODI_PROJECT_LOG.md)). Neocortex의 memory를 끈 대조와 token 순서를 섞은 대조도 base와 차이가 없었다. Neocortex 층 사이 membrane shortcut은 ETTm2에서 Neocortex 발화를 약 2배로 늘렸지만 정확도는 그대로였다([MODI 11-09 01:37](../docs/MODI_PROJECT_LOG.md)). Neocortex 입력을 주파수 대역으로 나눈 DCT 시험과 STFT 시험에서도 ETT는 "차이 없음"이었다([MODI 11-13 21:55, 11-14 01:26](../docs/MODI_PROJECT_LOG.md)).

**추론.** 우리 기록으로는 두 해석이 다 가능하다. 하나는 ETT가 Neocortex의 memory와 순서를 거의 보상하지 않는다는 해석이다. SpikF의 0.8 %가 이쪽과 맞는다. 다른 하나는 결합부가 Neocortex 신호를 출력까지 나르지 못한다는 해석이다. Neocortex 발화를 늘려도(MS) 결과가 바뀌지 않았다는 사실은 두 해석 모두와 맞는다. attention 교체가 의미 있는 경우는 둘째 해석이 맞을 때뿐이다. 둘을 가르는 일은 교체 전에, 마지막 절의 학습 없는 진단으로 할 수 있다.

## 지금 score 식은 문헌이 실패 조건으로 꼽은 조합이다

### Neocortex가 출력에 닿는 통로는 하나뿐이다

**우리 기록 (코드).** query 형태(`q_from_kv`)에서 `SSA_rel_scl`이 하는 계산은 다음과 같다([layers.py](../nsmt_flif_v1/classification/layers.py)).

```
q = LIF(BN(Linear(Neocortex spike)))      # Neocortex 출력 [N, B', 1, 64]를 4 step에 복사한 입력
k = LIF(BN(Linear(Hippocampus spike)))    # [4, B', N, 64]
v = LIF(BN(Linear(Hippocampus spike)))
score_ij = (q_i · k_j) · 64^-0.5 + b_h(j − i)     # head 하나당 channel 8개(ETTh2) 또는 4개(ETTm2)
z_i      = Σ_j score_ij · v_j
         = 0.125 · q_i (Σ_j k_jᵀ v_j)  +  Σ_j b_h(j − i) · v_j
y_i  = LIF(BN(Linear(LIF(z_i))))           # 모든 LIF: spikingjelly, τ 2, 문턱 1, 발화 뒤 0으로 reset
x_i ← x_i · (1 − y_i)                       # y가 1인 곳의 Hippocampus spike를 지움
```

배율은 `dim ** -0.5`이고 dim은 전체 embedding 크기 64이므로 0.125다. 그런데 이 값이 곱해지는 대상은 head 하나의 dot product(channel 8개 또는 4개)다. 그래서 query–key 한 쌍의 내용 점수는 0, 0.125, …, 최대 **1.0(ETTh2) 또는 0.5(ETTm2)**다. bias 표는 0에서 시작해 상한 없이 학습된다(실행 인자 `ssa_pe 1`, `attn_bn False`). attention 출력 neuron은 spikingjelly 기본 문턱 1을 쓴다. Spikformer 공식 code는 같은 배율 0.125를 쓰지만 이 neuron의 문턱은 0.5로 둔다([Spikformer code](https://raw.githubusercontent.com/ZK-Zhou/spikformer/main/imagenet/model.py)).

**추론.** 둘째 줄의 분해가 핵심이다. 첫째 항에서 query i는 Hippocampus 창 전체를 위치 없이 요약한 Σ_j k_jᵀv_j 가운데 어떤 channel을 꺼낼지 고른다. Neocortex가 출력에 닿는 통로는 이것 하나다. 둘째 항은 query와 무관하다. Hippocampus value에 학습된 token convolution을 건 것과 같고, Spikformer식 비교 model에도 같은 모양으로 들어 있다. 그러므로 두 model의 차이는 첫째 항이 둘째 항보다 얼마나 큰지에 달려 있다. LIF 식(τ 2, 문턱 1, 같은 입력 4 step)을 풀면, attention 출력 neuron이 4 step 안에 한 번이라도 발화하려면 입력이 16/15 ≈ 1.07 이상이어야 한다. query와 key가 channel 하나만 겹치는 흔한 경우 점수는 0.125다. 이때 내용 항만으로 이 문턱을 넘으려면 같은 channel에서 value spike를 낸 token이 9개 이상 있어야 한다.

### dot product에 위치 bias를 더하는 조합은 forecasting에서 무너졌다

**논문 측정.** Lv 외(NeurIPS 2025)는 Spikformer 계열에 상대 위치 정보를 넣는 방법들을 비교했다. 실험은 forecasting 데이터 4개(Metr-la, Pems-bay, Solar, Electricity) × horizon 4개 × seed 3개였다([Lv 외 2025](https://arxiv.org/abs/2501.16745); [DOI](https://doi.org/10.52202/085713-2601)). 평균 R² / RSE는 원래 Spikformer(convolution 위치 encoding)가 0.733 / 0.541이었다. **dot product에 ALiBi를 더하면 0.413 / 0.909로 무너졌고**, Solar에서는 모든 horizon에서 R²가 0.080이었다. Log-PE를 dot product Spikformer에 더하면 Electricity(prediction 24)에서 수렴하지 못했다(R² 0.710*, RSE 1.03*, *는 수렴 실패). QKFormer + Log-PE도 같았다. 반면 XNOR 유사도에 같은 Log-PE를 더하면 Electricity R² 0.974, RSE 0.300이었다. XNOR를 쓰더라도 정수가 아닌 선형 거리 bias(C-RPE)를 더하면 Metr-la R² 0.158*로 실패했다. 저자들은 이 bias가 spiking attention 활성을 "catastrophically overshadow"한다고 적었다. 결론 문장은 "the dot product, which does not use Hamming distance to measure relative distance, is not suitable for RPE methods"다. 성공한 조합 XNOR + Log-PE는 평균 **0.750 / 0.509**로, ANN Transformer + RoPE(0.756 / 0.507)에 가까웠다.

**추론.** 우리 score는 "0/1 dot product + 더하는 상대 위치 bias"를 그대로 따른다. 다른 점은 bias가 고정값이 아니라 0에서부터 학습된다는 것이다. 그래서 무너지지는 않는다. 대신 학습이 bias를 쓸모 있는 만큼 키우고, bias가 내용 항을 덮은 상태로 자리 잡을 수 있다. 우리 model은 학습이 잘 되고 MSE도 Spikformer식과 같다. 그러니 이 경우 증상은 "실패"가 아니라 "Neocortex가 있으나 없으나 같다"로 나타날 것이다.

### 배율과 "둘 다 0" 문제는 영상에서 여러 번 확인됐다

**논문 측정 — 배율.** Spikformer는 Q·Kᵀ·V 값이 너무 커서 출력 neuron의 surrogate gradient가 사라지는 문제를 막으려고 배율 s를 넣었다. ImageNet에서 배율 없는 변형(I, ReLU, LeakyReLU)은 수렴하지 못했다([Spikformer](https://arxiv.org/abs/2209.15425)). QKFormer의 재실험에서는 배율을 뺀 SSA가 CIFAR-100에서 76.95 %로, 배율을 둔 78.21 %보다 낮았다([QKFormer](https://arxiv.org/abs/2403.16552)). SpikingResformer(CVPR 2024)는 spike 입력의 평균이 0이고 분산이 1이라고 가정할 수 없으므로 1/√d가 맞지 않는다고 보았다. 그래서 발화율로 배율을 유도했다. 배율을 빼거나 1/√d로 바꾸면 **수렴하지 못했다.** 유도한 배율을 Spike-driven Transformer에 넣자 ImageNet-100 정확도가 83.06 %에서 83.96 %로 올랐다([SpikingResformer](https://arxiv.org/abs/2403.14302)). SpikeVideoFormer(ICML 2025)에서는 문턱 배율을 1/(2D)에서 예전 관행인 1/8로 바꾸자 pose 오차가 39.8에서 46.7로, segmentation mIoU가 73.1에서 68.5로 나빠졌다([SpikeVideoFormer](https://arxiv.org/abs/2505.10352)).

**논문 측정 — "둘 다 0".** 0/1 dot product는 "둘 다 1"만 센다. 발화가 드물면 대부분을 차지하는 "둘 다 0" 일치는 버려진다. α-XNOR(CVPR 2025)는 둘 다 0인 channel에 가중치 α를 준다. Spikformer CIFAR-100에서 기준은 78.21 %였다. 발화율로 정한 affine 변환(k·score + b)만 넣고 dot product는 그대로 두면 79.04 %였다. α = 1(XNOR)이면 79.33 %, α = 0.3이면 **80.22 %**였다([α-XNOR](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html)). SpikeVideoFormer에서는 Hamming 유사도를 dot product로 되돌리면 pose 오차가 39.8에서 45.7로, mIoU가 73.1에서 65.9로 나빠졌다. SNN-ViT(ICLR 2025)의 표에서 dot product를 유지한 변형은 기준과 거의 같았다(76.95 → 77.12 %)([SNN-ViT](https://arxiv.org/abs/2502.12677)). BSA(NeurIPS 2025)는 0/1 곱의 기댓값이 d·p²(p는 발화 확률)라는 것을 보였다. 즉 기댓값이 0이 아니고 한쪽으로 치우친다([BSA](https://proceedings.neurips.cc/paper_files/paper/2025/file/9316cfcb0a81e53a1f35b4353f115571-Paper-Conference.pdf)).

**추론.** 우리 head 차원 4–8은 이 논문들의 설정보다 훨씬 작다. 0/1 dot product는 channel 4개에서 0부터 4까지 다섯 값만 가질 수 있다. 발화율이 몇 % 수준이면 대부분의 쌍은 0이고, 정보는 대부분 "둘 다 0" 쪽에 있다. 이런 조건을 시험한 논문은 없다.

### 11-08 측정은 예상한 증상을 보여 준다

**우리 기록.** MS를 넣기 전 query population LIF model의 checkpoint 15개를 학습 없이 잰 결과다([MODI 11-08 00:17](../docs/MODI_PROJECT_LOG.md); [neo-path-probe](../nsmt_flif_v1/classification/results/flif-v1-neo-path-probe.md)). ETT에서 Neocortex 마지막 층의 발화율은 2–4 %였다. 첫 token의 발화율은 ETT run 12개 모두에서 정확히 0이었다. 첫 token에서 발화하려면 입력 전류가 단위 중앙값 기준 8.6–10.0이어야 한다. 그런데 BN을 거친 전류는 평균이 약 0, 표준편차가 약 1이다. 앞쪽 5–7 token은 거의 발화하지 않았다. 그 뒤에서도 (행, token) 위치의 **70–83 %**에서 Neocortex 출력 전체가 0이었다(ETTm2 H96의 뒤쪽 절반만 약 40 %). 이런 위치의 query는 고정 벡터 q0가 되는데, ETT에서는 q0도 0이었다. attention 출력 neuron에 들어가기 직전 값을 원소별 절댓값 평균으로 나누면 아래와 같다.

| 칸 | Neocortex가 바꾸는 항 | bias 항 | 바꾸는 항의 크기 비중 |
|---|---|---|---|
| ETTh2 H96 (head 8개) | 0.0267 | 0.0168 | 0.612 |
| ETTh2 H720 (head 8개) | 0.0275 | 0.0224 | 0.550 |
| ETTm2 H96 (head 16개) | 0.0293 | **0.2993** | 0.089 |
| ETTm2 H720 (head 16개) | 0.0161 | **0.0972** | 0.159 |

이 측정에는 두 가지 빈칸이 있다. Neocortex를 지웠을 때 MSE가 얼마나 바뀌는지는 재지 않았다. 지금 base(Neocortex MS)에서 다시 재지도 않았다.

**추론.** 내용 점수 상한이 0.5인 ETTm2에서는 bias가 Neocortex 항의 약 6–10배로 자랐다. 상한이 1.0인 ETTh2에서는 두 항이 비슷했다. head 차원이 작을수록 bias가 더 지배하는 방향이다. 이것은 NeurIPS 2025 논문이 본 "bias가 내용을 덮는" 현상이, bias를 학습하는 경우에는 무너지지 않은 채 나타난 모습으로 읽힌다. 다만 칸이 넷뿐이고 데이터도 다르다. 그러므로 head 차원이 원인이라고 단정할 수는 없다. 또 하나, query가 0인 70–83 % 위치에서는 어떤 유사도를 써도 Neocortex가 바꿀 수 있는 것이 없다. 이 문제는 attention이 아니라 Neocortex 앞부분, 즉 첫 token이 발화할 수 없는 구조에 있다.

### 복사된 4 step과 gate 때문에 쓸 수 있는 후보가 줄어든다

**우리 기록.** Neocortex 출력 [N, B', 1, 64]는 4 step에 그대로 복사된다. 11-17 점검에서 6/6 PASS로 확인했다([MODI 11-17 22:10](../docs/MODI_PROJECT_LOG.md)). Hippocampus도 같은 token embedding을 4번 본다.

**추론.** q의 Linear와 BN에는 4 step 모두 같은 입력이 들어간다. 그래서 q LIF(τ 2, 문턱 1, 발화 뒤 0으로 reset)가 4 step에 걸쳐 적는 것은 시간이 아니라 전류의 크기다. 일정한 전류 I를 넣고 LIF 식을 풀면 다음과 같다.

| 전류 I | spike 수 | 발화 step |
|---|---|---|
| I < 16/15 | 0개 | 없음 |
| 16/15 ≤ I < 8/7 | 1개 | 넷째 |
| 8/7 ≤ I < 4/3 | 1개 | 셋째 |
| 4/3 ≤ I < 2 | 2개 | 둘째·넷째 |
| I ≥ 2 | 4개 | 모든 step |

k와 v도 같은 입력을 반복해서 받으므로, step 사이의 차이는 역시 크기를 나타내는 부호다. 그래서 spiking step 축을 섞는 것이 핵심인 방법(STAtten, TIM, SSSA의 step 혼합)은 섞을 새 정보가 없다.

**논문 측정 — gate.** 출판된 사례에서 두 spike 경로를 합치는 방식은 두 가지다. 하나는 cross-attention 출력을 강도 α로 더하는 것이다. S-CMRL에서는 α = 1.5가 가장 좋았고, 더 키우면 나빠졌다([S-CMRL](https://arxiv.org/abs/2502.12488)). 다른 하나는 상대 경로의 발화율로 만든 배율을 곱하는 것이다(CMA, Neural Networks 2024)([CMA](https://arxiv.org/abs/2410.15689)). x·(1 − y)처럼 억제하는 gate를 쓴 사례는 찾지 못했다. **우리 기록.** Hippocampus의 IAND 경로를 membrane shortcut으로 바꾼 시험에서 ETT는 차이가 없었다([MODI 11-09](../docs/MODI_PROJECT_LOG.md)).

**추론.** gate의 효과는 y가 발화하는 만큼만 생긴다. y가 거의 0이면 gate는 아무것도 지우지 않으므로 두 model이 같아진다. y가 너무 자주 발화하면 Hippocampus spike가 지워져 정보가 사라진다. 지금 base에서 y의 발화율과 gate가 지우는 Hippocampus spike 비율은 잰 적이 없다. 어떤 교체든 attention 출력의 크기 분포를 바꾸므로, 가장 먼저 바뀌는 것은 이 두 값이다. 교체 효과를 해석하려면 이 두 값을 함께 봐야 한다.

## Energy 숫자에서는 attention 교체가 공짜로 보인다

**우리 기록.** 우리 energy 계산에서 syops는 nn.Linear와 BatchNorm module에 hook을 걸어 연산을 센다. 입력이 0/1이면 AC 0.9 pJ, 아니면 MAC 4.6 pJ로 계산한다. neuron 상태 갱신 비용은 따로 더한다([energy.py](../nsmt_flif_v1/classification/energy.py)). 그런데 q·kᵀ, score × v, bias 덧셈은 `SSA_rel_scl.forward` 안의 tensor 연산이다. module이 아니라서 hook에 잡히지 않는다. **즉 지금 계산에서 attention 핵심 연산의 energy는 0이다.** projection 수가 같다면 어떤 attention으로 바꿔도 energy 숫자는 거의 그대로 나온다. 특히 bias 덧셈은 spike와 상관없이 매 step, 매 head, 모든 (i, j) 쌍에서 일어나는 실수 덧셈이다. 내 계산으로는 N = 24, head 8개, 4 step일 때 창 하나·channel 하나당 18,432번이고, N = 180, head 16개일 때는 2,073,600번이다.

**논문 측정 — 계산 관행이 서로 다르다.** Spikformer는 attention 비용을 발화율 × FLOPs × 0.9 pJ로 셌다. 배율, BN, AND 연산, memory 비용은 넣지 않았다([Spikformer](https://arxiv.org/abs/2209.15425)). Spike-driven Transformer V1은 spike끼리의 원소별 곱을 mask로 보고 energy를 0으로 셌다([SDT](https://arxiv.org/abs/2307.01694)). NeuroBench는 0/1이 아닌 입력과의 연산을 MAC로 센다([NeuroBench](https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html)). Spike-driven Transformer 유지 관리자는 GitHub issue에서, 정수 spike를 만드는 SEW residual model들이 이 점을 보통 무시한다고 썼다. 그 이유로 든 말이 "so that the energy values don't look too high"다([SDT #23](https://github.com/BICLab/Spike-Driven-Transformer/issues/23)). QKFormer는 그림에서 attention 입력을 spike라고 했지만, 공개 code의 입력은 spike가 아니다([QKFormer #5](https://github.com/zhouchenlin2096/QKFormer/issues/5)).

**논문 측정 — 상수와 memory.** 45 nm에서 0.9 pJ는 FP32 덧셈의 비용이다. 4.6 pJ는 FP32 곱셈 3.7 pJ에 덧셈 0.9 pJ를 더한 값이다. 같은 표에서 8-bit 정수 덧셈은 0.03 pJ, 32-bit 정수 덧셈은 0.1 pJ다. 8 KB cache에서 64-bit를 한 번 읽는 데는 10 pJ가 든다([Horowitz, ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323)). memory까지 센 두 연구는 spiking attention에 불리한 결과를 냈다. AICAS 2024에서는 T = 10일 때 attention block 하나의 energy가 INT8 ANN **97.73 µJ**, Spikformer **109.05 µJ**였다. 차이는 memory 접근(89.96 대 102.85 µJ)에서 나왔다([Song 외](https://arxiv.org/abs/2402.09109)). STEP에서 memory까지 센 Transformer-8-512는 quantized 16.51 mJ, spiking 17.16 mJ였다([STEP](https://arxiv.org/html/2505.11151v2)). IEEE TCAD 2026의 분석은 T = 5인 SNN이 같은 정보량의 quantized network를 이기려면 평균 발화율이 **5.7 % 미만**이어야 한다고 계산했다([Yan 외](https://doi.org/10.1109/TCAD.2026.3718799)). 하드웨어 논문들도 spike × spike attention을 공짜 mask로 보지 않는다. Bishop(ISCA 2025)은 attention 점수에 6–10 bit가 필요하다고 보고했다([Bishop](https://arxiv.org/html/2505.12281v1)).

**추론.** syops 관행은 popcount(정수 덧셈)를 FP32 덧셈 값으로 과대 계산하고, memory 비용은 0으로 과소 계산한다. memory를 센 두 연구(AICAS 2024, STEP)에서 memory 쪽이 지배했으므로, 둘을 합친 오차는 SNN에 유리한 쪽으로 날 가능성이 높다. 우리 model에서 잰 것은 아니다. 교체 시험에서는 attention 핵심 연산을 두 arm 모두에서 명시적으로 세야 한다. 후보의 energy 순위는 계산 관행에 따라 바뀐다. XNOR는 0인 channel도 점수에 넣으므로, "발화하지 않으면 공짜"라는 spike 계산의 장점을 잃는다. 반대로 Log-PE는 정수 bias라서 지금의 실수 bias보다 정확하게 셀 수 있다. QKTA와 SDSA-2는 projection 하나를 없애므로 지금 계산에서도 energy가 줄어든 것으로 나온다.

## 후보별 분석: 1순위는 XNOR + Log-PE, 나머지는 문제를 바꾸거나 근거가 약하다

아래 표는 후보 전체를 한눈에 보도록 정리한 것이다. "쓸 수 있나"는 q를 Neocortex에서, k·v를 Hippocampus에서 가져오는 지금 배치를 유지할 수 있는지를 뜻한다. 기대효과와 우려는 모두 추론이다.

| 순위 | 후보 (venue, 확인 수준) | 무엇이 바뀌나 | 쓸 수 있나 | 기대효과 | 가장 큰 우려 |
|---|---|---|---|---|---|
| 1 | XNOR + Log-PE (NeurIPS 2025, DOI·code) | 유사도와 위치 항 | 예 | bias가 내용을 덮는 현상이 줄어듦 | d = 4이면 Log-PE 범위(0–8)가 XNOR 범위(0–4)보다 넓다 |
| 2 | 발화율 기반 배율·정규화 (α-XNOR CVPR 2025; SpikingResformer CVPR 2024; 우리 `--attn_bn`) | score의 크기 | 예 | 내용 항의 크기 회복, 학습 안정 | CinC에서 attention BN은 차이 없음. 0인 query는 못 고친다 |
| 3 | Hamming 유사도 SDHA (ICML 2025, code) | 부호 있는 유사도 | 예 | 음수 점수로 억제 가능 | d = 4이면 점수가 다섯 값뿐 |
| 4 | key 쪽 거리 감쇠·window (RetNet arXiv; SpikeGPT TMLR 2024) | 점수에 거리 감쇠를 곱함 | 예 (band 코드 있음) | 내용 항이 위치를 갖게 됨 | ETT는 순서를 거의 보상하지 않음 |
| 5 | DSSA (CVPR 2024, code) | 0/1 attention map과 발화율 배율 | 예 (입력 둘이 달라도 정의됨) | 수렴 근거가 가장 강함 | 재현 격차, 1-D 판 미검증, 위치 항 없음 |
| 6 | A²OS²A (CVPR 2025, code 없음) | 실수 K, −1/0/+1 V | 예 | Hippocampus 쪽 정보 손실 감소 | spike만 쓰는 구조가 아니고, 두 model에 똑같이 듣는다 |
| 대조용 | QKTA·QKCA (NeurIPS 2024), SDSA-1·2 (NeurIPS 2023, ICLR 2024) | token·channel gate, routing 없음 | N이 같으면 되지만 의미가 바뀜 | energy 감소, "routing이 필요한가"를 확인 | 영상에서 −0.8 ~ −3.4점, QKFormer code 결함 |
| 낮음 | STAtten (CVPR 2025), TIM (IJCAI 2024), SSSA (ICLR 2025) | spiking step 축 섞기 | 예 | 거의 없음 | 4 step이 복사본이다 |
| 낮음 | softmax 유지 (BSA NeurIPS 2025, SpikedAttention NeurIPS 2024, SpikeZIP-TF ICML 2024) | 행 정규화 | 예 | 0/1 q·k에서 이득 없음 | 변환 방식은 T 40–128 필요 |
| 범위 밖 | SpikF 주파수 선택기 (ICML 2025), SSM·LMU, STMixer (NeurIPS 2024) | attention이 아님 | 아니오 | ETT 근거는 SpikF뿐 | 결합 자체를 바꾼다. 우리 STFT 시험은 음성 |

### 지금 형태(SSA)는 가장 많이 재현된 기본값이다

**논문 측정.** Spikformer SSA는 ICLR 2023 공식 사이트에서 확인되고 code도 있다([ICLR](https://iclr.cc/virtual/2023/poster/12139)). CIFAR-100에서 softmax attention과 거의 같은 정확도(77.86 대 77.92 %)를 냈다. 이때 attention 한 번의 energy 추정은 1.170 µJ로, softmax 쪽 30 µJ보다 훨씬 낮았다([Spikformer](https://arxiv.org/abs/2209.15425)). Spike-driven Transformer V2는 네 가지 연산을 비교한 뒤 SSA와 같은 SDSA-3를 기본값으로 골랐다. 정확도가 가장 높았기 때문이다(ImageNet, T = 1: SDSA-1 −0.8점, SDSA-2 −1.2점)([SDT-V2](https://arxiv.org/abs/2404.03663)). 재현은 완전하지 않다. STEP에서 다시 돌린 Spikformer는 CIFAR-100 77.37 %로 논문값 78.21 %보다 낮았다([STEP](https://arxiv.org/html/2505.11151v2)). 공개 설정으로 CIFAR-10 91 %만 나왔다는 issue도 답이 없는 채 닫혔다([Spikformer #33](https://github.com/ZK-Zhou/spikformer/issues/33)). 언어 과제에서는 Spikingformer가 softmax를 다시 넣어야 수렴했다고 밝혔다([Spikingformer](https://arxiv.org/abs/2304.11954)).

**추론.** 계열 자체가 틀린 선택은 아니다. 두 경로를 잇는 spiking 연구들도 이 형태를 쓴다. 문제는 우리 쪽 설정 세 가지다. 서로 다른 경로를 잇는다는 점, 학습되는 위치 bias를 더한다는 점, 그리고 배율 0.125를 아주 작은 head 차원(4–8)에 쓴다는 점이다. 아래 후보들은 이 세 가지를 얼마나 직접 고치는지를 기준으로 순위를 매겼다.

### 1순위: XNOR 유사도 + Log-PE

**논문 측정.** 이 조합은 위 NeurIPS 2025 논문이 dot product + bias의 실패를 보고 내놓은 처방이다. 점수는 Σ_D ¬(Q ⊕ K), 즉 값이 같은 channel 수다. Log-PE는 R_ij = ⌈log₂((L − 1)/(|i − j| + 1))⌉를 이 점수에 더한다. 그래서 정수는 정수로 남는다. Spikformer의 평균 R² / RSE는 원래 형태(0.733 / 0.541)에서 XNOR + Log-PE(0.750 / 0.509)로 좋아졌다. QKFormer는 XNOR + Log-PE로 0.695에서 0.746으로, SDT-V1은 Log-PE로 0.721에서 0.734로 좋아졌다. Gray-PE(0.748 / 0.513)는 위치 길이가 100 이상이면 약해진다고 저자들이 밝혔다. code는 microsoft/SeqSNN에 있다([Lv 외 2025](https://arxiv.org/abs/2501.16745); [DOI](https://doi.org/10.52202/085713-2601)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | score_ij = q_i·k_j + (1 − q_i)·(1 − k_j) + R_\|i−j\|. head별 값은 0–d이고, R은 고정 정수다. 학습 bias 표를 없앤다. 배율은 attention 출력 neuron의 문턱으로 접는다 |
| 신뢰도 | NeurIPS 2025, DOI와 code 확인. forecasting 4 데이터 × 4 horizon × seed 3개. ETT, cross-attention, head 차원 4–8은 시험하지 않았다 |
| cross-attention으로 쓸 수 있나 | 예. XNOR와 Log-PE는 q, k가 어디서 왔는지와 무관하게 정의된다. Log-PE는 위치 i, j만 쓰고, Neocortex token i와 Hippocampus token j는 같은 번호 체계를 쓴다. Gray-PE는 q와 k에 같은 부호를 붙이는 대칭 설계라, cross에서는 양쪽 경로 모두에 붙이도록 고쳐야 한다 |
| 4 step·bias·gate와의 관계 (추론) | Log-PE는 지금 bias처럼 모든 step에서 같다. query가 0인 위치(70–83 %)의 점수는 d − \|k_j\|, 즉 key에만 달린 값이 된다. 따라서 그 위치에서 Neocortex의 영향은 늘지 않는다. q가 channel 몇 개에서만 발화하는 위치에서는 "둘 다 0" 일치가 점수에 들어가 내용 항이 커진다. 거의 모든 쌍의 점수가 0이 아니게 되므로 attention 출력 neuron의 입력이 커진다. 그 결과 y 발화율과 gate가 지우는 비율이 크게 바뀔 수 있어, 정규화를 함께 넣어야 한다 |
| 기대효과 (추론) | ETTm2처럼 bias가 내용을 덮던 칸(bias 0.30 대 Neocortex 항 0.03)에서 q가 출력에 닿는 몫이 커질 수 있다. MSE 이득의 크기는 불확실하다. 논문에서는 원래 Spikformer 대비 RSE가 약 6 % 줄었지만(R² +0.017), ETT에서는 순서가 준 몫 자체가 작았다(SpikF MAE 0.8 %, 우리 순서 대조 "차이 없음"). 우리 문턱 −1 %를 넘을지는 열려 있다 |
| 우려되는 점 | 정확도: ETTm2(d = 4)에서 XNOR 점수 범위는 0–4인데, Log-PE 범위는 N = 24에서 0–5, N = 84에서 0–7, N = 180에서 0–8이다(내 계산). 위치 항이 다시 내용 항을 덮을 수 있고, 이는 논문이 C-RPE의 실패 원인으로 꼽은 바로 그 현상이다. 또 Spikformer식 비교 model도 학습 bias를 잃으므로 두 model이 함께 바뀐다. energy 계산: 0인 channel도 점수에 들어가므로 spike 기반으로 세면 dot product보다 비싸 보인다. 정수 Log-PE는 실수 bias보다 정확하게 셀 수 있다. 구현·계산: matmul 두 번과 고정 buffer 하나면 된다. N ≤ 180, D = 64라 계산 부담은 무시할 만하다 |
| 우선순위 | 1 |

### 2순위: 발화율로 정하는 배율·정규화

**논문 측정.** α-XNOR(CVPR 2025)는 Attn′ = (k·Score + b)V로 쓴다. k와 b는 Q, K, V의 발화율로부터 E[Attn′] = 0, Var[Attn′] = 1이 되도록 정하고, 곱셈은 neuron 문턱에 합친다. dot product를 그대로 두고 이 변환만 넣어도 CIFAR-100이 78.21 %에서 79.04 %로 올랐다. code는 찾지 못했다([α-XNOR](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html)). SpikingResformer는 c₁ = 1/√(f_X·d)를 쓴다. f는 학습 중 지수 이동 평균(momentum 0.999)으로 추적한 발화율이다. 이 배율이 없으면 수렴하지 못했다([SpikingResformer](https://arxiv.org/abs/2403.14302)). **우리 기록.** attention 출력 neuron 앞에 BN을 두는 `--attn_bn` 경로가 이미 코드에 있다. 10-21에 사전 고정한 CinC 시험에서 Spikformer식에 넣었을 때는 "차이 없음"이었다. query model과 ETT에서는 시험하지 않았다([MODI 10-23 21:30](../docs/MODI_PROJECT_LOG.md)). 같은 묶음에서 Spikformer식의 상대 위치 bias를 빼면 CinC가 −1.11 %p로 "나쁨"이었다. 즉 CinC에서는 bias가 쓸모 있는 정보를 날랐다.

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | 고정 배율 0.125 대신 attention 출력 neuron의 입력을 평균 0·분산 1 근처로 맞춘다. 방법은 셋이다. (가) α-SSA식 k·S + b(발화율로 계산), (나) SpikingResformer식 1/√(f_q·d), (다) 학습되는 BN(`--attn_bn`, inference 때 문턱에 접힘) |
| 신뢰도 | α-XNOR: CVPR 2025(CVF), code 없음. SpikingResformer: CVPR 2024(DOI), code 있음. 다만 STEP의 end-to-end 재현에서 CIFAR-100이 논문값 85.98 %가 아니라 79.45 %였다(원 논문은 transfer 설정) ([STEP](https://arxiv.org/html/2505.11151v2)). 둘 다 영상 실험만 있다 |
| cross-attention으로 쓸 수 있나 | 예. 발화율을 q(Neocortex)와 k(Hippocampus)에서 따로 잰다 |
| 4 step·bias·gate와의 관계 (추론) | Neocortex처럼 발화율이 낮은(2–4 %) 쪽은 1/√(f_q·d)가 크게 나와, 드문 Neocortex spike의 점수를 키운다. Neocortex 신호를 키우는 자동 장치로는 이것이 유일하다. bias를 그대로 두면 배율이 커지는 만큼 bias의 상대 크기가 줄어든다. query가 0인 위치는 여전히 0이다. y 발화율이 바뀌므로 gate가 지우는 비율도 함께 바뀐다 |
| 기대효과 (추론) | 내용 항의 크기가 회복되고 학습이 안정된다. 가장 싸게 시험할 수 있다. 1순위의 구성 요소이기도 해서, 1순위 효과가 어디서 오는지 나누는 대조로도 필요하다 |
| 우려되는 점 | 정확도: CinC에서 attention BN이 차이 없었으므로 ETT에서도 그럴 수 있다. energy 계산: 배율과 offset을 문턱에 접으면 비용이 0이지만, 접지 않으면 MAC다. 구현·계산: 낮다. (다)는 스크립트에 연결만 하면 되고, (가)·(나)는 발화율을 추적하는 buffer가 필요하다 |
| 우선순위 | 2 |

### 3순위: 부호 있는 Hamming 유사도 (SDHA)

**논문 측정.** SpikeVideoFormer(ICML 2025)의 SDHA는 SN((2Q − 1)(2K − 1)ᵀV / (2D))다. 0/1을 −1/+1로 바꾸는 일은 bit shift로 한다. KᵀV를 먼저 계산하면 비용이 token 수에 비례한다. Hamming을 dot product로 바꾼 ablation은 앞에서 본 대로 크게 나빴다(pose 39.8 → 45.7, mIoU 73.1 → 65.9). code가 있다([SpikeVideoFormer](https://arxiv.org/abs/2505.10352); [PMLR](https://proceedings.mlr.press/v267/zou25b.html)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | 같은 channel은 +1, 다른 channel은 −1로 더한다. 점수 범위는 −d부터 +d이고 부호가 있다. 배율 1/(2d)는 문턱에 접는다 |
| 신뢰도 | ICML 2025(PMLR), code 있음. 영상(Kinetics, pose, segmentation) 실험만 있다 |
| cross-attention으로 쓸 수 있나 | 예 |
| 4 step·bias·gate와의 관계 (추론) | 담는 정보는 XNOR(α = 1)와 같다(2·XNOR − d). 다만 기준이 0이라 음수 점수가 생긴다. 그래서 bias가 양수여도 내용 항이 attention 출력 neuron을 억제할 수 있다. query가 0이면 점수가 d − 2\|k_j\|가 되어 key에만 달린 값이다. q와 k가 둘 다 드물게 발화하면 대부분의 쌍이 +d 근처에 몰린다. 학습 bias를 남겨 두면 NeurIPS 2025가 경고한 "더하는 bias" 문제도 그대로 남는다 |
| 기대효과 (추론) | XNOR와 비슷하다. 부호 덕분에 y가 더 선택적으로 발화할 수 있다 |
| 우려되는 점 | 정확도: d = 4이면 점수가 −4, −2, 0, 2, 4의 다섯 값뿐이다. 영상 밖의 근거가 없다. energy 계산: 0인 channel이 모두 −1로 계산에 들어간다. 구현·계산: 낮다 |
| 우선순위 | 3. 1순위가 신호를 보일 때 두 번째 형태로 시험한다 |

### 4순위: key 쪽에 거리 감쇠나 window를 곱한다

**논문 측정 (주로 spike가 아닌 model).** RetNet은 attention 점수에 γ^(i−j) 형태의 거리 감쇠를 곱한다. ablation에서 in-domain perplexity가 전체 model 26.05였고, 감쇠를 빼면 27.86, head마다 다른 γ를 빼면 27.02였다([RetNet](https://arxiv.org/abs/2307.08621)). SWAX에서는 attention window를 짧게 두자 함께 쓴 recurrent memory가 더 잘 쓰였다. window 128이 window 2048보다 긴 맥락 검색에서 16점 높았다([SWAX](https://arxiv.org/abs/2509.24552)). spiking 쪽 선례로 SpikeGPT(TMLR 2024)가 있다. token 하나를 spike step 하나로 돌리면서 감쇠가 있는 RWKV mixer를 썼다. Enwik8 BPC는 1.283으로 Transformer 1.137보다 나빴다. 내부 상태에 exp와 나눗셈이 있어 spike만으로 계산하지는 않는다([SpikeGPT](https://arxiv.org/abs/2302.13939)). **우리 기록.** 곱하는 방식의 band mask(`attn_window`)가 PPG 시험 때 코드에 들어갔다. 대조 run이 없어 판정은 하지 못했다([MODI 10-28](../docs/MODI_PROJECT_LOG.md)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | score_ij에 γ_h^\|i−j\| 또는 1[\|i−j\| < W]를 곱한다. query i가 i 근처의 Hippocampus 요약을 읽게 된다 |
| 신뢰도 | 감쇠의 근거는 spike가 아닌 언어 model(RetNet은 arXiv)에서 나왔다. spiking 선례는 SpikeGPT(TMLR 2024, code)다. 0/1 q, k에서 시험한 논문은 없다 |
| cross-attention으로 쓸 수 있나 | 예 |
| 4 step·bias·gate와의 관계 (추론) | 위치 정보를 더하지 않고 곱한다. 그래서 내용 항 자체에 위치가 생긴다. 학습 bias 없이 쓰면 query가 0인 token의 출력은 0이 되어, bias처럼 query를 덮는 일이 없다. bias를 남기면 지금 코드에서는 band가 bias를 더한 뒤에 곱해지므로 bias도 같은 범위로 잘린다 |
| 기대효과 (추론) | Neocortex가 token i 근처의 내용을 고르게 된다. 그러면 memory·순서 대조가 반응할 가능성이 생긴다. ETT에서 정확도 이득은 작을 것이다(SpikF의 순서 몫 0.8 %, 우리 순서 대조 "차이 없음") |
| 우려되는 점 | 정확도: 전역 요약이 줄어 MSE가 나빠질 수 있다. energy 계산: γ를 곱하는 것은 MAC다. γ = 1 − 2⁻ⁿ으로 두면 shift와 뺄셈으로 할 수 있다. 구현·계산: band 코드가 있어 낮다 |
| 우선순위 | 4. "순서를 쓰게 만들 수 있는가"를 물을 때 쓴다 |

### 5순위: DSSA (0/1 attention map)

**논문 측정.** SpikingResformer(CVPR 2024)의 DSSA는 attention map을 SN(X·Wᵀ·Yᵀ·c₁)로 만든다. 두 spike가 함께 있을 때만 가중치를 더하는 방식이고, map 자체가 0/1 spike다. 출력은 SN(map·f(Y)·c₂)다. 논문 정의(식 7–8)에 따르면 X와 Y는 서로 다른 spike 입력이어도 된다. 이 model 안에서 DSSA를 SSA나 SDSA로 바꾸면 수렴하지 못했다. 저자들은 그 이유를 다중 해상도 입력과 배율에서 찾았다([SpikingResformer](https://arxiv.org/abs/2403.14302)). SOP를 어떻게 세었는지 묻는 후속 질문에는 답이 없다([issue #7](https://github.com/xyshi2000/SpikingResformer/issues/7)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | map_ij = SN(c₁·x^Neo_i·W·x^Hip_jᵀ)는 0 또는 1이다. 출력은 out_i = SN(c₂·Σ_j map_ij·(W₂x^Hip_j))다. q·k의 정수 점수 대신 0/1 map이 생기고, 실수 가중치 덧셈은 AC로 셀 수 있다 |
| 신뢰도 | CVPR 2024(DOI), code 있음. 재현 격차(CIFAR-100 85.98 → 79.45 %)가 있고 영상 실험만 있다 |
| cross-attention으로 쓸 수 있나 | 예, 논문 정의 그대로 |
| 4 step·bias·gate와의 관계 (추론) | query가 0인 행은 map 전체가 0이라 출력도 0이다. DSSA에는 위치 항이 없으므로 위치는 따로 넣어야 한다. c₁ = 1/√(f_Neo·d)가 드문 Neocortex spike를 증폭한다. map이 0/1이라 출력 크기는 c₂로 고정되고, 그만큼 y 발화를 예측하기 쉽다 |
| 기대효과 (추론) | 학습 안정. spike만 쓰는 계산에 가장 가깝다 |
| 우려되는 점 | 정확도: 논문의 p×p downsampling은 2-D용이다. 1-D stride 판은 시험된 적이 없다. 위치 항도 없다. energy 계산: map은 셀 수 있지만 원 논문의 SOP 규칙이 불분명하다. 구현·계산: 중간이다. 발화율 추적과 SN 두 단계가 필요하다. 연산은 N²에 비례하지만 N ≤ 180이라 문제없다 |
| 우선순위 | 5. 1·2순위에서 학습이 불안정할 때 꺼낸다 |

### 6순위: A²OS²A (실수 K, −1/0/+1 V)

**논문 측정.** A²OS²A(CVPR 2025)는 Q는 0/1, K는 ReLU(BN(XW_K))인 실수, V는 −1/0/+1 spike로 두고 SN(Q·Kᵀ·V)를 계산한다. K에 상한이 없으므로 배율을 쓰지 않는다. 비교 대상은 Spikformer가 아니라 SDT다. CIFAR-100에서 2-256은 76.00 → 76.96 %, 2-512는 78.83 → 79.90 %로 올랐다. 2-512의 기준값은 본문에 78.43 %로 적혀 있어 표와 다르다. energy 숫자와 code는 없다([A²OS²A](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.html)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | k를 실수(음수 아님)로, v를 −1/0/+1로 바꾼다. q는 Neocortex의 0/1 그대로다 |
| 신뢰도 | CVPR 2025(CVF, DOI). code가 없고 표와 본문이 맞지 않는다 |
| cross-attention으로 쓸 수 있나 | 예 |
| 4 step·bias·gate와의 관계 (추론) | query가 0이면 점수도 0이라 bias만 남는다. 실수 k는 Hippocampus 쪽 정보 손실을 줄이는데, Spikformer식 비교 model에도 똑같이 듣는다. 그래서 Neocortex의 몫은 그대로일 가능성이 높다. −1/0/+1 v는 음수 출력을 만들어 y를 억제할 수 있다 |
| 기대효과 (추론) | 두 model의 정확도가 함께 조금 오를 수 있다. 그러나 질문("Neocortex가 쓰이는가")에는 답하지 못한다 |
| 우려되는 점 | 정확도: 근거가 SDT 기준 1점 안팎이고 code가 없다. energy 계산: k가 실수라 "spike만 쓴다"는 주장이 약해지고, NeuroBench 규칙에서는 MAC 논쟁이 생긴다. 구현·계산: −1/0/+1 neuron을 새로 만들어야 한다 |
| 우선순위 | 6 (낮음) |

### 대조로만 쓸 것: gate형 (QKTA·QKCA, SDSA-1·2, SSSA)

**논문 측정.** QKFormer(NeurIPS 2024)의 QKTA는 A_t = SN(Σ_channel Q)로 token별 0/1 mask를 만든 뒤 K에 곱한다. V도, 배율도, token 사이 routing도 없다. QKTA만 쓰면 SSA만 쓸 때보다 CIFAR-100 79.09 대 81.23 %, CIFAR10-DVS 80.70 대 84.10 %로 낮았다([QKFormer](https://arxiv.org/abs/2403.16552)). 공개 code에서는 attention neuron이 batch 안의 표본 사이에 막전위를 섞는다는 결함 보고가 두 번, 서로 따로 올라왔다. 이를 고치면 공개 checkpoint 정확도가 78.732 %에서 74.196 %로, 큰 model은 84.248 %에서 78.698 %로 떨어졌다. 저자는 답하지 않았다([#14](https://github.com/zhouchenlin2096/QKFormer/issues/14); [#17](https://github.com/zhouchenlin2096/QKFormer/issues/17)). SDSA-1은 Q ⊗ SN(Σ_token K⊗V)이고 SDSA-2는 SN(Σ_token Q) ⊗ V다. ImageNet에서 각각 −0.8점(7.2 mJ), −1.2점(6.3 mJ)이었다. 기준은 75.4 %, 7.8 mJ다([SDT-V2](https://arxiv.org/abs/2404.03663)). SSSA(ICLR 2025)는 token별 spike 수와 step을 섞는 행렬로 점수를 만든다. CIFAR-100 79.60 %로 SSA 76.95 %보다 높았다. code는 찾지 못했다([SNN-ViT](https://arxiv.org/abs/2502.12677)). **우리 기록.** 10-17 분석에서 원래 NSMT의 AND mask(Neocortex spike를 Hippocampus key·value에 곱하는 방식)는 "창마다 거의 같은, 위치별 희소 gate"로 쓰였다. 다른 창의 Neocortex 출력으로 바꿔도 ETT MSE 변화는 ±0.3 % 안이었다([MODI 10-17 21:16](../docs/MODI_PROJECT_LOG.md)).

| 항목 | 내용 |
|---|---|
| 계산에서 바뀌는 것 | query–key 쌍의 점수가 없어진다. QKTA는 token i의 Neocortex spike 수로 0/1 token mask를 만들어 Hippocampus k_i에 곱한다. SDSA-2는 Neocortex 전체 token의 합으로 channel mask를 만들어 Hippocampus v에 곱한다. SDSA-1은 Hippocampus의 K⊗V 합으로 channel mask를 만들어 Neocortex q에 곱한다 |
| 신뢰도 | QKFormer: NeurIPS 2024(DOI), code 있음, 결함 보고는 미확인. SDT: NeurIPS 2023과 ICLR 2024(자기 표기), code 있음. SSSA: ICLR 2025, code 없음 |
| cross-attention으로 쓸 수 있나 | QKTA는 두 경로의 N이 같으면 되고, 우리는 이 조건을 만족한다. 그러나 token i의 출력이 token i만 보게 되어 의미가 바뀐다. SDSA-2는 token 축으로 합하므로 Neocortex의 순서 정보가 정의상 사라진다 |
| 4 step·bias·gate와의 관계 (추론) | QKTA는 원래 NSMT의 AND mask가 실제로 하던 일과 사실상 같다. 이미 본 구조로 돌아가는 셈이다. query가 0인 70–83 % 위치는 mask도 0이다. bias를 넣을 자리가 없다 |
| 기대효과 (추론) | energy가 줄어든다(V 또는 K projection과 score 곱이 없어짐). 무엇보다 "token 사이 routing이 필요한가"를 확인하는 대조로 가치가 있다. QKTA gate가 지금 attention과 같은 MSE를 내면, Σ_j k_jᵀv_j를 거치는 routing은 아무것도 더하지 않는다는 뜻이다 |
| 우려되는 점 | 정확도: 영상에서 −0.8 ~ −3.4점이었다. energy 계산: QKFormer의 attention 입력은 code에서 spike가 아니므로, 논문 숫자를 그대로 가져올 수 없다. 구현·계산: 낮다. 다만 QKFormer식으로 [T·B]를 합쳐 neuron에 넣으면 같은 결함이 생긴다 |
| 우선순위 | 대조로만 |

### 낮음: spiking step 축을 섞는 방법 (STAtten, TIM)

**논문 측정.** STAtten(CVPR 2025)은 spiking step을 B개씩 묶어, 묶음 안에서 attention을 계산한다. 이미지를 열 단위로 넣는 sCIFAR에서 Spikformer 기준 s-CIFAR10이 79.29 → 82.45 %, s-CIFAR100이 57.17 → 58.75 %로 올랐다. energy는 SDT 12.42 mJ, STAtten을 붙인 SDT 12.36 mJ였다([STAtten](https://arxiv.org/abs/2409.19764)). TIM(IJCAI 2024)은 query에 이전 step query의 1-D convolution을 섞는다. SHD는 85.1 → 86.3 %, CIFAR10-DVS는 78.5 → 81.6 %였다([TIM](https://arxiv.org/abs/2401.11687)).

**추론.** 둘 다 spiking step 축을 쓴다. 그런데 우리 4 step은 같은 입력의 복사본이고, step 사이 차이는 크기 부호뿐이다(앞 절의 표). STAtten을 쓸모 있게 하려면 묶음을 step이 아니라 token으로 바꿔야 한다. 그러면 4순위의 window attention과 같아진다. TIM이 query에 주는 "이전 기록"은 우리 Neocortex가 이미 token 축을 따라 주고 있다. TIM이 SHD에서 얻은 +1.2점은, query에만 memory를 줄 때의 이득이 작다는 우리 결과와 맞는 방향이다. 우선순위는 낮다.

### 낮음: softmax를 되살리는 방법

**논문 측정.** BSA(NeurIPS 2025)의 CIFAR-100 ablation에서 0/1 Q·Kᵀ에 softmax나 Shiftmax(2의 거듭제곱으로 근사한 softmax)를 붙여도 이득이 없었다(79.13, 79.07, 79.14 %). q, k를 −1/0/+1로 바꾼 뒤에야 79.27 % → 80.76 %(softmax), 80.48 %(Shiftmax)로 올랐다([BSA](https://proceedings.neurips.cc/paper_files/paper/2025/file/9316cfcb0a81e53a1f35b4353f115571-Paper-Conference.pdf)). ANN을 변환해 softmax를 정확히 흉내 내는 방법은 step이 많이 든다. SpikedAttention은 T = 40에서 ImageNet 80.0 %를 냈다. 이를 위해 전용 neuron 회로가 면적 9.88 %, 전력 12.35 %를 더 쓴다([SpikedAttention](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c9341ad0263428b5057d92f4d88dfa0-Paper-Conference.pdf)). SpikeZIP-TF는 T = 64–128을 썼다([SpikeZIP-TF](https://arxiv.org/abs/2406.03470)).

**추론.** 우리 q, k는 0/1이므로 BSA 결과대로라면 softmax만 붙여서는 얻을 것이 없다. −1/0/+1 neuron을 두 경로에 모두 넣는 것은 attention 교체보다 큰 변경이다. 변환 방식은 T_h = 4 예산을 크게 넘는다. 우선순위는 낮다.

### 범위 밖: 결합을 attention이 아닌 것으로 바꾸기

**논문 측정.** SpikF의 주파수 선택기는 ETT에서 근거가 있는 유일한 spiking mixer다. ETTh2 평균 MSE / MAE가 0.372 / 0.394, ETTm2가 0.281 / 0.320이었다(입력 96, horizon 4개 평균)([SpikF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)). LMUFormer는 같은 network에서 LMU를 SSA로 바꾸자 Speech Commands 35-class가 96.12 %에서 93.38 %로 떨어졌다([LMUFormer](https://arxiv.org/abs/2402.04882)). STMixer(NeurIPS 2024)는 query 없이 학습된 token 혼합 행렬을 쓰므로 cross-attention이 될 수 없다([STMixer](https://doi.org/10.52202/079017-4092)). **우리 기록.** Neocortex 입력을 STFT 대역으로 바꾼 11-14 시험은 ETT에서 "차이 없음"이었다. **추론.** 이 방법들은 결합부를 바꾸는 것이 아니라 model 구조를 바꾼다. 이번 질문의 범위 밖이다.

## 추천 순서와 공정한 교체 시험

### 순위

| 순위 | 할 일 | 학습 | run 수 | 이유 |
|---|---|---|---|---|
| 0 | 학습 없는 진단. 지금 base(MS)와 Spikformer식 checkpoint(입력 96, 336)에서 y 발화율, gate가 지우는 비율, attention 분해, 시험 때만 q → q0와 bias → 0으로 바꾼 효과, batch 불변성, attention 핵심 연산의 energy를 잰다 | 없음 | 0 | 교체가 의미 있는 조건인지 먼저 확인한다 |
| 1 | S-X: XNOR + Log-PE + 발화율 배율 | 있음 | 36 | 우리 실패 조건을 문헌이 직접 처방한 방법이다 |
| 2 | S-N: 발화율 배율만. S-B: 학습 bias만 제거 | 있음 | 36 + 36 | S-X의 효과가 어디서 오는지 나눈다 |
| 3 | S-X가 Neocortex 판정을 통과하면 memory 끔·순서 섞기 대조를 돌리고, 입력 336으로 넓힌다 | 있음 | 48 + 입력 336 몫 | Neocortex 몫을 확인한다 |
| 선택 | Hamming SDHA, 거리 감쇠·window, QKTA gate 대조 | 있음 | 각 36 | 형태 확인, routing이 필요한지 확인 |
| 미룸 | STAtten·TIM·SSSA, softmax 복원·변환, DSSA, A²OS²A | — | — | 근거나 적합성이 부족하다 |

arm 하나는 base pop LIF, base pop f-LIF, Spikformer식 비교 model 3개 × ETT 4칸 × seed 3개 = 36 run이다. ETT run 시간은 11-09 pool 기준 163–1,179초였다([MODI 11-09](../docs/MODI_PROJECT_LOG.md)). 입력 720 pool이 도는 동안에는 코드를 scratchpad 사본에서 만들고, pool이 끝난 뒤 합친다. 합친 뒤에는 기존 경로가 비트 단위로 같은지 확인한다.

### 무엇을 고정하고 무엇을 바꾸나

**고정할 것.** base 구조는 그대로 둔다. Neocortex MS query population(pop LIF와 pop f-LIF), Hippocampus IAND gate, T_h = 4, D = 64, head 8 / 16개, 층 수, patch 8 stride 4, embedding, head Linear를 유지한다. 학습 recipe도 그대로다. L1 손실, 50 epoch, lr 5e-4, batch 64, weight decay 0.06, reduce scheduler, patience 10, best-validation 가중치를 쓴다. 입력은 96부터 시작한다. 데이터는 ETTh2·ETTm2 × H96·H720, seed는 42 / 7 / 2026이다. 모든 arm은 같은 code sha에서 돌린다. 교체는 **base와 Spikformer식 비교 model 양쪽에 똑같이** 넣는다. 한쪽에만 넣으면 attention 개선과 Neocortex 효과를 구분할 수 없다. 계산 관행(syops)과 판정 규칙도 바꾸지 않는다.

**판정.** R은 12쌍 test MSE 상대 변화의 평균이다. R ≤ −1 %이고 8/12쌍 이상에서 낮으면 "좋음", 부호를 뒤집은 대칭 조건이면 "나쁨", 나머지는 "차이 없음"이다. 11-17에 정한 읽는 법도 그대로 따른다. 한 neuron에서만 좋음이거나, 한 칸을 뺐을 때 결과가 바뀌면 "약함"으로 읽는다. 비교는 넷이다. **V1** base_new − base_now(base의 attention 개선). **V2** Spikformer식_new − Spikformer식_now(비교 model의 attention 개선). **V3** base_new − Spikformer식_new(새 attention 아래 Neocortex의 몫). **V4** V3와 지금 격차(base_now − Spikformer식_now)의 차이를 쌍마다 계산한 것(격차 변화).

**사전 예상.** V1–V4는 모두 "차이 없음"일 가능성이 가장 높다. 무언가 움직인다면 bias가 지배하던 ETTm2 칸에서 먼저 움직일 것이다. y 발화율과 gate가 지우는 비율은 S-X에서 크게 바뀔 것이다.

### 기록할 진단

모든 run에서 test split과 best-validation 가중치로 아래 값을 기록한다. model 선택에는 쓰지 않는다. tensor 이름은 명시적으로 붙인다. Spike-driven Transformer에서는 사용자가 논문의 발화율을 재현하지 못했는데, 나중에 유지 관리자가 code의 Q가 논문의 V 역할을 한다고 설명한 사례가 있다([SDT #24](https://github.com/BICLab/Spike-Driven-Transformer/issues/24); [#25](https://github.com/BICLab/Spike-Driven-Transformer/issues/25)).

| 진단 | 내용 | 왜 |
|---|---|---|
| 발화율 | q, k, v, attention 출력 neuron, y(attention gate와 MLP gate)의 발화율. query가 0인 (행, token) 비율과 q0의 발화율 | 교체가 바꾸는 첫 값들. energy의 입력이기도 하다 |
| gate가 지우는 비율 | 블록마다 Σ x·y / Σ x (attention gate와 MLP gate 각각) | 0에 가까우면 gate가 아무 일도 하지 않는다. 너무 크면 정보를 지운다 |
| attention 분해 | 11-08과 같은 방식으로 고정 q 항, Neocortex가 바꾸는 항, 위치 항의 크기 비중을 잰다 | bias 지배가 풀렸는지 직접 본다 |
| 점수 분포 | head별 점수 값의 종류 수와 histogram, 학습 bias의 head별 크기 | d = 4에서 점수가 얼마나 거친지 본다 |
| 시험 때만 하는 ablation | q → q0(Neocortex 출력 전체 0), 위치 항 → 0, 내용 항 → 0으로 바꿨을 때의 ΔMSE | Neocortex가 출력에 닿는지 본다. 시험 때만 바꾸면 처음 보는 입력이라 하락이 과장되므로, 절대값보다 arm 사이 차이로 읽는다 |
| batch 불변성 | 같은 test 입력을 batch 1, 7, 64, 순서를 섞은 batch로 평가해 출력 최대 차이가 1e-6 이하인지 본다. 넘으면 그 run의 결과를 쓰지 않는다 | QKFormer처럼 [T·B]를 합쳐 neuron에 넣는 오류를 잡는다. 우리 Neocortex는 token 축으로 돌기 때문에 특히 필요하다 |
| energy | 기존 syops 숫자(연속성 유지용)와, attention 핵심 연산을 명시적으로 센 숫자를 함께 적는다. 핵심 연산은 q·kᵀ = popcount, score × v, 위치 항, 문턱에 접은 배율이다. 0/1이 아닌 피연산자는 NeuroBench 규칙에 따라 MAC로 센다. 정수 덧셈 0.03·0.1 pJ와 memory 읽기 10 pJ/64-bit로 바꿔 순위가 유지되는지도 본다 | 교체가 공짜로 보이는 것을 막는다 |
| seed 편차 | 칸별 표준편차/평균 | 입력 336의 ETTh2 H96처럼 한 칸이 결과를 끌고 가는지 본다 |

### 무엇을 양성·음성으로 볼 것인가

**Neocortex에 대한 양성.** 다음이 모두 성립해야 한다. 첫째, V3가 "좋음"이고 V4도 "좋음"(격차가 커짐)이다. 두 neuron 모두이거나, 한 neuron이라도 한 칸을 빼도 R ≤ −1 %가 유지되어야 한다. 둘째, 진단에서 Neocortex가 바꾸는 항의 비중이 지금보다 크고, 시험 때 q → q0로 바꾼 ΔMSE도 지금 base보다 크다. 셋째, y가 퇴화하지 않았다. 즉 거의 항상 0이거나 Hippocampus spike를 대부분 지우는 상태가 아니다. 그다음 단계 3에서 memory 끔·순서 섞기 대조가 base_new보다 "나쁨"이어야 "새 score에서 Neocortex의 memory와 순서가 쓰인다"고 말할 수 있다.

**attention만의 개선.** V1과 V2가 함께 "좋음"이고 V3·V4가 "차이 없음"이면, 더 나은 attention을 찾은 것이다. 두 model에 공통으로 채택할 수 있다. Neocortex 주장의 근거로는 쓰지 않는다.

**음성.** V3·V4가 "차이 없음"이고 진단 비중도 그대로이면, ETT에서 Neocortex를 가린 것은 score 식이 아니었다는 뜻이다. 그러면 attention 교체는 멈춘다. 다음 손잡이는 Neocortex 앞부분(첫 token이 발화할 수 없는 구조)이나 과제 선택이다. 어떤 run이든 NaN, 결과 누락, 학습 발산이 있으면 실패로 센다. 그 run을 빼고 판정하지 않는다(fail-closed). batch 불변성을 통과하지 못한 arm의 결과는 쓰지 않는다.

## 결론

"어떤 spiking attention이 최선인가"는 우리 model에서는 단위가 틀린 질문에 가깝다. 출력에 영향을 주는 것은 attention의 이름이 아니라 세 숫자다. 내용 점수의 크기와 위치 항의 크기 비율, 살아 있는 query의 비율, 그리고 y가 Hippocampus spike를 실제로 지우는 비율이다. 문헌에서 우리 설정에 가장 강하게 적용되는 신호는 뜻밖에도 부정적인 결과다. 0/1 dot product에 위치 bias를 더하면 forecasting에서 무너진다는 것이다. 11-08 측정은 우리 model이 무너지지 않은 대신 바로 그 증상, 즉 bias 지배와 침묵하는 query를 보인다는 것을 보여 준다. 그래서 1순위 교체의 가치는 MSE 이득보다 해석의 정리에 있다. 위치 항이 내용을 덮지 않는 score에서 나온 "차이 없음"이라야 비로소 Neocortex에 대한 결론이 된다. 지금의 "차이 없음"은 score의 산수에 대한 결론일 수도 있다.

두 번째 함의는 순서다. 학습 없는 진단만으로 교체가 쓸모 있는 조건인지 알 수 있다. 시험 때 q를 q0로 바꿔도 MSE가 그대로라면, 결합부는 지금 Neocortex를 쓰지 않는 것이다. 그 원인은 두 갈래로 가를 수 있다. query 대부분이 0이라면 문제는 attention보다 앞, 즉 첫 token부터 침묵하는 Neocortex 출력에 있다. 이 경우 어떤 유사도로 바꿔도 결과는 같다. 반대로 query는 살아 있는데 위치 항이 지배하거나 y가 거의 발화하지 않는다면, 그때가 score 교체가 쓸모 있는 경우다. energy 쪽도 같은 순서를 요구한다. attention 핵심 연산을 세기 전까지는 어떤 교체도 "energy 중립"으로 보인다. 따라서 진단과 energy 재계산을 먼저 하는 것은 교체 시험의 준비 단계가 아니라, 교체 시험을 해석하기 위한 전제다.
