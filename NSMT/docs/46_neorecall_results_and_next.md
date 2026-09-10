> 폴더 정리 후 현재 실행·출력 경로는 [project_layout.md](project_layout.md)를 참고한다. 아래 실험 기록의 과거 경로·건수는 당시 기준이다.

# 46. Neo-as-recall — 현재 결과 종합과 다음 실험 방향

- **최종 갱신**: 2026-08-06
- **브랜치**: `exp/init-order-fix-v1` · fingerprint `e0c6fcc8…` 불변
- **상태**: forecasting **176 runs 완료**, AD **15/24** (SKIP 6 = MSL 전체, 원인 규명됨)
- **선행 문서**: [docs/45](45_neorecall_handoff.md) (모델 상세·환경·실행 방법)

> 이 문서는 **결과 해석과 다음 실험 설계**에 집중한다. 모델 구조·환경 구축·실행 명령은
> [docs/45](45_neorecall_handoff.md)를 참조.

---

## 0. 한 문단 요약

**forecasting에서는 게이트 통제가 핵심 주장을 기각했다**(게이트를 꺼도 같거나 더 좋음).
**AD에서는 개선이 나온다**(SMD +2.26pp 3/3, PSM +0.49pp 2/2) — task에 따라 부호가 다르다는
제안자의 직관이 방향으로는 맞을 수 있다. **그러나 AD의 두 통제(`recall_rec`, `recall_none`)가
모두 seed 2개에 부호가 갈려 판정 불가**다. **다음 실험의 최우선은 이 두 통제의 seed 보강**이며,
그것 없이는 "+2.26pp가 Neo 반전 덕분"이라고 말할 수 없다.

---

## 1. forecasting 최종 결과 (176 runs, 3 seed)

### 1.1 데이터셋별

| 데이터셋 | Δ vs baseline | 우세 |
|---|---:|---:|
| **ETTh1** | **−1.99%** | **21/24** |
| ETTh2 | −0.66% | 16/24 |
| ETTm2 | −0.22% | 20/24 |
| **ETTm1** | **+1.23%** | **5/23** |
| **전체** | **−0.43%** | (n=95) |

**전체 평균이 nuisance(1.4%) 미만**이다. ETTh1에서만 견고하고 ETTm1에서는 오히려 해롭다.

### 1.2 ★ 게이트 격리 통제 — 결론을 뒤집은 실험

`recall_none` = Neo가 복원 학습은 하되 **Hippo로 되읽히지 않음**.

| | ETTh2 p96 | ETTh1 p720 |
|---|---:|---:|
| **게이트 없음 (통제)** | **−2.11%** (3/3) | **−2.90%** (3/3) |
| recall_raw | −1.89% (3/3) | −2.82% (3/3) |
| recall_low | −2.21% (3/3) | −1.79% (3/3) |

**게이트를 꺼도 같거나 더 좋다** → 개선의 원천은 Neo 게이트가 아니라 **Hippo의 융합 방식을
Memory-Replay에서 self-attention으로 바꾼 것**이다.

### 1.3 곱셈 게이트 반증
`recall_mul` (ETTh2 p96): **+5.35%, 0/3**. Block의 모든 연산이 곱셈이라
(`x = x*(1−mca); x = x*(1−mlp)`) 한 번 0이 되면 복구되지 않는다. 측정된 Neo 발화율 **0.13%**
(Hippo 12.83%).

### 1.4 α 해석 — 정정
α가 0.982 → 0.967로 감소한 것을 "모델이 Neo를 쓴다"고 읽었으나 **틀렸다**. ETTm 계열이 α를 가장
많이 낮추는데(0.90~0.94, Neo 비중 6~10%) **성능은 가장 나쁘다**(+1.23%). **α 변화는 유용성의
증거가 아니다.**

---

## 2. AD 반전 결과 (15/24)

### 2.1 SMD (F1 point-adjusted)

| 조건 | F1 | vs baseline | n | α |
|---|---:|---:|---:|---:|
| baseline | 0.844176 | — | 3 | — |
| **recall** (Neo=다음패치) | **0.866806** | **+2.26pp** (3/3) | 3 | 0.9891 |
| recall_rec (통제: Neo=복원) | 0.859252 | +0.79pp (2/2) | **2** | 0.9894 |
| recall_none (통제: 게이트 없음) | 0.867491 | +1.61pp (2/2) | **2** | 0.9820 |

seed별 `recall` vs baseline: **+0.27 / +2.61 / +3.91 pp** (3/3 개선)

### 2.2 PSM

| 조건 | F1 | vs baseline | n |
|---|---:|---:|---:|
| baseline | 0.954983 | — | 3 |
| **recall** | **0.959897** | **+0.49pp** (2/2) | 2 |

### 2.3 ★ 두 통제 — 모두 판정 불가

| 질문 | 비교 | 결과 | seed별 (7/13/21) | 판정 |
|---|---|---:|---|---|
| **반전이 필요한가?** | `recall` vs `recall_rec` | **+1.32pp** | −0.78 / **+2.09** / **+2.65** | ⚠️ **2/3** |
| **게이트가 필요한가?** | `recall` vs `recall_none` | −0.17pp | **−1.75** / +1.41 / *(중단)* | ❌ 1/2 |

> **중단 직전 `SMD recall_rec seed21 = 0.842350`이 추가**되면서 반전 통제가 3 seed로 채워졌고,
> **평균 +1.32pp, 2/3 우세로 반전 쪽으로 기울었다.** 다만 사전 기준(**3/3 + 1.0pp**)에는 미달이고
> seed 7이 여전히 반대다. 게이트 통제는 seed 21이 중단으로 SKIP돼 여전히 n=2다.
>
> **→ 8.7에서 해소**: 이 서버에서 `SMD recall_none seed21 = 0.855047`을 실행해 **n=3을 채웠다**.
> 결과는 **+0.35pp, 2/3**으로 게이트 통제도 사전 기준 미달이 확정됐다.

**두 통제 모두 seed 2개이고 부호가 갈린다.** 특히 `recall_none`(0.867491)이 `recall`(0.866806)과
사실상 동일해, **forecasting에서 결론을 뒤집었던 패턴이 AD에서도 재현될 조짐**이다.

> **현재로선 "+2.26pp가 Neo 반전 덕분인지, forecasting과 마찬가지로 self-attention 전환 덕분인지
> 구분할 수 없다."** 이것이 다음 실험이 반드시 답해야 할 질문이다.

### 2.4 α — task에 따라 방향이 다르다

| task | α 변화 (초기 0.9820) | Neo 비중 |
|---|---|---:|
| forecasting | **감소** (→0.90~0.98) | 2~10% |
| **AD** | **증가** (→0.9891) | 1.80% → **1.09%** |

AD에서는 모델이 Neo 비중을 **줄인다**. §1.4에서 α가 유용성의 증거가 아님을 확인했으므로 과대
해석은 금물이나, **부호가 task별로 반대라는 사실 자체는 기록해둘 가치가 있다.**

### 2.5 MSL 실패 (SKIP 6건) — 원인 규명됨

**데이터·설정 문제가 아니라 CUDA OOM**이다:
```
neuron_kernel.py:806 backward -> RuntimeError: CUDA out of memory.
Tried to allocate 2.18 GiB (GPU 0; 47.41 GiB total; 43.32 GiB already allocated)
```
MSL은 **55채널**(SMD 38, PSM/SMAP 25)이라 `[T, B·C, N, D]` 텐서가 가장 크고, 당시 같은 GPU에
다른 실험이 올라가 있었다. 데이터(`MSL_train.npy` shape `(58317, 55)`)와 `c_out=55` 설정은 정상.

**대처**: MSL은 **전용 GPU에 단독 배치**하거나 `-bs 128 → 64`로 낮춘다. batch를 바꾸면 다른
데이터셋과 레시피가 달라지므로, **단독 배치를 우선**한다.

---

## 3. 종합 해석

### 3.1 확립된 것

| # | 사실 | 근거 |
|---|---|---|
| 1 | **곱셈 게이트는 신호를 죽인다** | `recall_mul` +5.35% (0/3), Neo 발화율 0.13% vs Hippo 12.83% |
| 2 | **forecasting에서 Neo 게이트는 무효** | 게이트 통제가 같거나 더 좋음 (−2.11%/−2.90%) |
| 3 | **살아남은 처방은 Memory-Replay → self-attention** | params 0 추가, −2.1~−2.9%, 전 seed 일관 |
| 4 | **AD에서는 개선이 나온다** | SMD +2.26pp (3/3), PSM +0.49pp (2/2) |
| 5 | **α 변화 ≠ 유용성** | ETTm이 α 최저인데 성능 최악 |

### 3.2 미결

| # | 질문 | 왜 미결인가 |
|---|---|---|
| A | AD 개선이 **Neo 반전** 때문인가? | `recall_rec` 통제가 n=2, 부호 혼재 |
| B | AD 개선이 **게이트** 때문인가? | `recall_none` 통제가 n=2, 부호 혼재 |
| C | self-attention 처방이 일반적인가? | 2설정만 검증 |
| D | 에너지 이득이 있는가? | 미측정 |

**A·B가 풀리지 않으면 AD 결과는 보고할 수 없다.** forecasting에서 정확히 같은 구조의 통제가
핵심 주장을 기각한 전례가 있기 때문이다.

---

## 4. 다음 실험 — 우선순위와 구체적 설계

### ★ ①  AD 통제 seed 보강 (최우선, 12 runs)

**목적**: §3.2의 A·B를 해결한다. 이것 없이는 AD 결과 해석 불가.

```bash
Q=neorecall_ad_v1/anomaly_detection/scripts/queues/queue_ad_ctrl.txt; mkdir -p "$(dirname "$Q")"; : > "$Q"
# SMD 통제 2종을 3 seed 로 채우고, PSM 에도 같은 통제를 건다
for S in 7 13 21; do
  for C in recall_rec recall_none; do
    echo "bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh SMD $C $S" >> $Q
    echo "bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh PSM $C $S" >> $Q
  done
done
nohup setsid bash scripts/gpu_pool2.sh $Q 1,2,3,5 > neorecall_ad_v1/anomaly_detection/log/adctrl.log 2>&1 &
```
이미 있는 `SMD recall_rec {7,13}`, `SMD recall_none {7,13}`은 중복 실행돼도 무해하다
(분석이 `(dataset, cond, seed)` 키로 멱등).

**판정 기준 (미리 고정)**
- `recall` > `recall_rec` 가 **3/3 + 1.0pp 초과** → **반전이 필요하다**는 주장 성립
- `recall` > `recall_none` 가 **3/3 + 1.0pp 초과** → **게이트가 기여한다**
- 둘 중 하나라도 실패 → **AD 개선도 self-attention 전환 효과**로 귀결 (forecasting과 동일 결말)

### ② MSL 재실행 (6 runs, 전용 GPU)

```bash
Q=neorecall_ad_v1/anomaly_detection/scripts/queues/queue_msl.txt; mkdir -p "$(dirname "$Q")"; : > "$Q"
for S in 7 13 21; do for C in baseline recall; do echo "bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh MSL $C $S" >> $Q; done; done
nohup setsid bash scripts/gpu_pool2.sh $Q 5 > neorecall_ad_v1/anomaly_detection/log/msl.log 2>&1 &   # GPU 1개 단독 (OOM 회피)
```
MSL이 채워져야 AD 주장이 3개 데이터셋(SMD/PSM/MSL)에 근거하게 된다. SMAP까지 넣으면 4개.

### ③ self-attention 처방 확장 (실제로 살아남은 발견, 96 runs)

`recall_none`이 곧 "Memory-Replay → self-attention" 조건이다. **2설정에서만 −2.1~−2.9%가
확인됐으므로 논문에 쓰려면 전 그리드가 필요**하다.

```bash
for S in 7 13 21; do for DS in ETTh1 ETTh2 ETTm1 ETTm2; do for PL in 96 192 336 720; do
  for C in baseline recall_none; do echo "bash neorecall_v1/forecasting/scripts/run_recall.sh $DS $PL $C $S"; done
done; done; done > neorecall_v1/forecasting/scripts/queues/queue_selfattn.txt
```
`baseline`은 이미 176 runs 안에 있으므로 실제로는 `recall_none` 48 runs만 새로 필요하다.

**이 실험이 중요한 이유**: 이 저장소는 Neo를 살리려는 시도가 15종+ 실패했다. `recall_none`은
**Neo를 쓰지 않는 방향**의 처방이고, 그것이 유일하게 일관된 이득을 낸다. 확장에서 유지되면
**논문의 실질적 기여**가 될 수 있다(단 CLS 이중경로 프레이밍과 충돌 — §5 참조).

### ④ 에너지 측정 (추가 구현 불필요)

`forecasting/test.py:303`이 `print_per_layer_stat=True`로 **레이어별 AC/MAC 분해를 이미 출력**한다
(`syops`). `utils.py:265`의 `get_energy_consumption(O_ac, O_mac, E_ac=0.9, E_mac=4.6)`로 μJ 산출.

**측정 대상**: `baseline` vs `recall_none` vs `recall`. self-attention 전환이 에너지에 미치는
영향을 확인한다. 특히 **비-spiking head의 MAC 비중**을 함께 보고하면 [docs/44](44_session_wavelet_dual_recall.md)에서
제기한 "무엇이 실제로 에너지를 쓰는가" 질문에 답할 수 있다.

### ⑤ (선택) forecasting의 ETTm1 붕괴 규명

ETTm1만 +1.23%(5/23)로 일관되게 나쁘다. ETTm은 15분 샘플링이라 96 step이 24시간뿐이다.
Neo의 T=N LIF가 이 짧은 실시간 범위에서 다르게 거동할 가능성이 있다. **가설 검증용**이지
성능 개선 목적은 아니다.

---

## 4.5 ★ 실험 설정 명세 — 조건 → 플래그 → 가중치

> 이 절은 러너 스크립트를 열지 않고도 실험을 정확히 재현·변형할 수 있도록 **모든 값을 명시**한다.

### 4.5.1 조건 → CLI 플래그 매핑

**forecasting** (`neorecall_v1/forecasting/scripts/run_recall.sh`, 워크스페이스 `neorecall_v1/forecasting/`)

| 조건 | 추가 플래그 | 의미 |
|---|---|---|
| `baseline` | *(없음)* | 원본 Memory-Replay 융합 |
| `recall_raw` | `--neo_recall --neo_recall_tgt raw` | 제안 설계, Neo가 **원 패치** 복원 |
| `recall_low` | `--neo_recall --neo_recall_tgt lowfreq` | Neo가 **DCT 저주파** 복원 |
| `recall_mul` | `--neo_recall --neo_recall_gate mul` | 곱셈 게이트 (반증용, **쓰지 말 것**) |
| `recall_none` | `--neo_recall --neo_recall_gate none` | **게이트 격리 통제** |

**AD** (`neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh`, 워크스페이스 `neorecall_ad_v1/anomaly_detection/`)

| 조건 | 추가 플래그 | 의미 |
|---|---|---|
| `baseline` | *(없음)* | — |
| `recall` | `--neo_recall` | Neo = **다음-패치 예측** (반전) |
| `recall_rec` | `--neo_recall --no_neo_recall_next` | **반전 통제**: Neo를 복원으로 되돌림 |
| `recall_none` | `--neo_recall --neo_recall_gate none` | **게이트 통제** |

`--neo_recall`을 켜면 **Hippo의 `data_block`이 `neo_fuse='xattn'`(self-attention)으로 전환**된다.
따라서 `recall_none`은 "Neo는 복원 학습만 + Hippo는 self-attention" 조건이며, 이것이 §1.2에서
결론을 뒤집은 통제다.

### 4.5.2 ★ 손실 가중치 — horizon에 따라 7.5배 변한다 (반드시 인지할 것)

`forecasting/train.py:276`, `anomaly_detection/train.py:136`:
```python
ratio = args.alpha * (args.pred_len / 720)          # forecasting/train.py:276
loss  = ce * (1. - ratio) + rec * ratio             # ce=Hippo, rec=Neo
```

**`--alpha`는 상수 가중치가 아니라 `pred_len`에 비례해 스케일된다.** `--alpha 0.5` 기준 실제 값:

| pred_len | ratio | Hippo 비중 | **Neo 비중** |
|---:|---:|---:|---:|
| 96 | 0.0667 | 0.9333 | **0.0667** |
| 192 | 0.1333 | 0.8667 | 0.1333 |
| 336 | 0.2333 | 0.7667 | 0.2333 |
| 720 | 0.5000 | 0.5000 | **0.5000** |

**Neo의 복원 손실 가중치가 p96과 p720 사이에서 7.5배 차이 난다.** 이는 §1의 horizon별 결과를
해석할 때 **교란 변수**다 — 예컨대 ETTh1이 p720에서 가장 좋았던 것(−2.82%)이 Neo 가중치가
가장 큰 설정이라는 사실과 분리되지 않았다. **이 문서 이전의 어떤 실험도 이 가중치를 통제하지
않았다.**

> **후속 실험 제안(신규)**: `--alpha`를 조정해 **모든 horizon에서 ratio를 고정**한 뒤 재측정한다.
> 예: p96에서 ratio 0.5를 만들려면 `--alpha 3.75`(= 0.5 × 720/96). 이것이 §4-⑤(ETTm1 붕괴)의
> 대안 가설이기도 하다.

AD는 `pred_len=0`이므로 위 스케일링이 적용되지 않고 **`--alpha`가 그대로 ratio**가 된다
(SMD 0.1397 / MSL·PSM·SMAP 0.1).

### 4.5.3 α (`recall_alpha`) — 게이트 가중치 (손실 가중치와 다른 것)

```python
self.recall_alpha = nn.Parameter(torch.tensor(4.0))    # sigmoid(4.0) = 0.98201
x_hippo = sigmoid(recall_alpha) * h + (1 - sigmoid(recall_alpha)) * sg(n)
```
- **학습 가능**, 초기 Neo 비중 = **1.799%**
- 추가 파라미터 **1개** (forecasting 187,265 / AD 95,937)
- **§1.4 경고**: α가 움직여도 유용성의 증거가 아니다

> **혼동 주의**: `--alpha`(CLI, 손실 가중치)와 `recall_alpha`(모델 파라미터, 게이트 가중치)는
> **완전히 다른 것**이다. 이름이 겹치므로 코드 수정 시 특히 주의할 것.

### 4.5.4 하이퍼파라미터 전문

**forecasting** (전 조건 동일, 배포 스크립트와 일치)
```
--gating attn --no-bias --scheduler reduce -s --test --warm_up_epoch 0 --init_order_fix
-bs 64 -emb 64 -nh 8 --max_ratio 2 --patch_size 8 --seq_len 96 --patience 3 --c_in 7
--alpha 0.5 --keep_ratio 0.25 --mlp_ratios 1 -lr 0.001 --time_layers 2 -e 50
--freq h   (ETTh*)  |  --freq t  (ETTm*)
```
`--keep_ratio 0.25` = DCT 저주파 타깃의 보존 비율(`recall_low`에서만 의미 있음).
`--time_layers 2` = Neo `TemporalBlock`의 층 수. `--layers`(=`depths`)는 기본 1.

**AD** (전 조건 공통)
```
--gating attn --no-bias --scheduler reduce -s --test --warm_up_epoch 0
-e 100 -bs 128 -emb 32 -nh 16 --mlp_ratios 4 --keep_ratio 0
--patch_size 8 --seq_len 100 --pred_len 0 --features M --time_layers 2
```
데이터셋별:

| DS | `--c_out` | `--anomaly_ratio` | `-lr` | `--alpha` |
|---|---:|---:|---:|---:|
| SMD | 38 | 0.5 | 0.000869989 | 0.1397 |
| MSL | 55 | 1.0 | 0.0005 | 0.1 |
| PSM | 25 | 1.0 | 0.0005 | 0.1 |
| SMAP | 25 | 1.0 | 0.0005 | 0.1 |

> **AD는 `--init_order_fix`가 없다.** AD 워크스페이스에는 그 기구가 이식되지 않았으므로,
> 모듈을 추가하는 변경을 할 때는 **전용 RNG generator**를 써야 한다([docs/45](45_neorecall_handoff.md) §7-6).
> 현 `neo_recall`은 `nn.Parameter` 1개만 추가하고 buffer를 만들지 않으므로 이 문제가 없다.

### 4.5.5 seed와 판정
- seed: **7, 13, 21** (3 seed). 초기 2설정 탐색에는 42, 2026을 더해 5 seed를 썼다
- 판정 임계: **nuisance 1.4%** ([docs/41](41_init_order_reproducibility.md)) — 이보다 작은 효과는 신뢰하지 않는다
- forecasting 지표: `mse_overall` (로그 마지막 줄)
- AD 지표: **point-adjusted F1** — 로그의 `adj : acc=.. pre=.. rec=.. f1=..` 줄

> **이 서버에서 추가된 오버라이드**(모델 코드 무변경, 러너 환경변수):
> `PY` · `EPOCHS` · `PATIENCE` · `WARMUP` · `NEO_TAU` · `NEO_FULL_GRAD` · `TAG` · `RES` · `LOG`.
> `TAG`는 `cond` 필드와 로그 디렉터리에 접미사를 붙여 스윕 결과가 섞이지 않게 한다 (8.10 참조).

---

## 5. 전략적 판단 — 이 방향이 논문에 갖는 의미

**살아남은 것은 "Neo를 쓰는 방법"이 아니라 "Hippo의 융합을 바꾸는 방법"이다.**

이는 이 저장소의 축적된 증거와 일관된다:
- Neo 살리기 레버 **15종+ 실패**
- Neo 통째 제거해도 **+0.34%**
- 융합 형태 **6종 전부 inert** (docs/17)
- Memory Replay가 신호의 **절반 소실** (docs/43 §D)
- Neo 발화율 **0.13%** (Hippo의 1/100)

**그러나 이 처방은 Neocortex를 제거하는 방향을 가리키며, 논문의 CLS 이중 경로 프레이밍
(`paper/MASTER.md`)과 정면 충돌한다.** 두 선택지가 있다:

1. **정직한 재프레이밍**: "CLS 영감 이중 경로를 검증한 결과 Neocortex 경로는 인과적으로 무효였고,
   대칭적 self-attention이 더 낫다" — 부정적이지만 증거가 압도적이고 방법론적 가치가 있다
2. **AD로 활로 찾기**: AD에서 반전 설계가 통제를 통과하면 "task에 따라 두 경로의 역할이 바뀐다"는
   주장이 가능하다 — **①의 결과에 전적으로 달려 있다**

**따라서 ①이 단순한 통제 실험이 아니라 방향 결정의 분기점이다.**

---

## 6. 실행 순서 제안

```
① AD 통제 보강 (12 runs, ~4h)  ─┐
② MSL 재실행   ( 6 runs, ~3h)  ─┼─ 병렬 가능 (GPU 4개 분배)
③ self-attn 확장(48 runs, ~3h) ─┘
              ↓
        ① 결과로 방향 결정
              ↓
   통과 → AD 확장(SMAP 추가) + 논문 재프레이밍 2안
   실패 → ③ 확장 결과로 논문 재프레이밍 1안 + ④ 에너지
```

**GPU 배분 예시** (제한 4개):
- GPU 5 단독 → MSL (OOM 회피)
- GPU 1,2 → AD 통제
- GPU 3 → self-attention 확장

---

## 7. 재현에 필요한 정보

모델 구조·환경 구축·플래그·함정 목록은 **[docs/45](45_neorecall_handoff.md)**에 있다. 요약만:

| 항목 | 값 |
|---|---|
| forecasting 워크스페이스 | `neorecall_v1/forecasting/` |
| AD 워크스페이스 | `neorecall_ad_v1/anomaly_detection/` |
| 러너 | `neorecall_v1/forecasting/scripts/run_recall.sh`, `neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh` |
| 결과 | `neorecall_v1/forecasting/results/raw.txt` (176), `neorecall_ad_v1/anomaly_detection/results/raw.txt` (15) |
| 추가 파라미터 | **α 1개** (forecasting 187,265 / AD 95,937) |
| 판정 임계 | **nuisance 1.4%** ([docs/41](41_init_order_reproducibility.md)) |

**반드시 지킬 것**: `--init_order_fix` 유지, 절제 실험 전 `functional.reset_net()`,
smoke test는 **실제 1-epoch 학습**으로, 큐 파일 외부 재작성 금지.

---

## 8. 이 서버(A6000 ×4) 적용 시 차이 — 2026-08-06 추가

> §1~§7은 원 서버 기준으로 작성됐다. 이 절은 **현재 서버에서 §4의 명령을 그대로 붙여넣으면
> 안 되는 이유**만 적는다. 원 서버로 돌아가면 이 절은 무시하면 된다.

### 8.1 GPU 번호 — **GPU 5는 없다**

이 서버는 **RTX A6000 ×4, id 0·1·2·3**이다. §4·§6의 `1,2,3,5` / `5`는 존재하지 않는 장치라
워커가 즉시 실패한다. 아래 8.4의 배분을 쓸 것.

`nvidia-smi`로 확인: 4장 모두 48GB, 현재 전부 유휴.

### 8.2 결과 파일 이관 완료 — 검증 결과 (2026-08-06)

`neorecall_v1/forecasting/results/raw.txt`(**176 RESULT, SKIP 0**)와 `neorecall_ad_v1/anomaly_detection/results/raw.txt`(**17 RESULT,
SKIP 6**)가 원 서버에서 도착했다.

**진위 확인**: forecasting 파일에서 §1.1을 seed-paired로 재계산한 결과가 **완전히 일치**한다 —
ETTh1 −1.99%(21/24), ETTh2 −0.66%(16/24), ETTm1 +1.23%(5/23), ETTm2 −0.22%(20/24),
전체 −0.43%(n=95). 진본이다.

**⚠️ AD 파일은 이 문서 §2보다 2 runs 최신이다.** `SMD recall_rec seed21`과 `PSM recall seed21`이
문서 작성 후 도착해, **§2.1·§2.2·§2.3의 수치와 판정이 바뀐다**. 아래 8.7 참조.

### 8.2.1 실제 잔여 작업 (재산정)

| 실험 | 문서 §4 | **실제 필요** | 근거 |
|---|---:|---:|---|
| ① A·B (AD 통제) | 12 | **0** | 두 통제 모두 이미 실패 판정 (8.7) |
| ③ self-attn `recall_none` | 48 | **42** | baseline 16설정×3seed 전부 있음; `recall_none`은 ETTh1_p720·ETTh2_p96 2설정만 있음 |
| ④ 에너지용 baseline 체크포인트 | — | (48) | 이관된 것은 `raw.txt`뿐 — **체크포인트는 없다** (8.9) |
| ② MSL | 6 | 6 | 조건부 가치 (8.9) |
| ① PSM 통제 (선택) | — | 6 | 전무. run당 131분으로 최고가 |
| ① `SMD recall_none` s21 | — | 1 | 판정 불변, 보고 표 완결성용 |
| **합계** | 66 | **42 필수** | 나머지는 전부 조건부·선택 |

**①의 임계 경로는 12 runs가 아니라 0 runs다.** 판정은 이미 나 있다.

### 8.3 MSL OOM — 이 서버에서는 재현되지 않을 수 있다

§2.5의 OOM은 "같은 GPU에 다른 실험이 올라가 있었다"가 직접 원인이다(43.32 GiB 선점).
이 서버는 4장 모두 비어 있으므로 **전용 GPU 배정만으로 충분할 가능성이 높다**.
`-bs 64`로 낮추는 것은 레시피를 바꾸므로 **OOM이 실제로 재발한 뒤에만** 고려한다.

### 8.4 GPU 배분 (수정판) 과 소요 추정

이관된 `time_s`에서 뽑은 실측 중앙값:

| | 중앙값 | 비고 |
|---|---:|---|
| fc ETTh1 / ETTh2 | 179s / 128s | 빠름 |
| fc ETTm1 / ETTm2 | **963s** / 550s | ETTm이 5~7배 비싸다 |
| AD SMD | 1937s (32분) | |
| AD PSM | **7842s (131분)** | max 15184s — **PSM 통제 6 runs는 ~13 GPU-h** |

```
GPU 0 단독 → ② MSL                     (6 runs)   ← 타 실험과 공유 금지
GPU 1      → ① SMD recall_none s21 (1) → 이후 ③ 합류
GPU 2, 3   → ③ self-attn recall_none   (42 runs, ~5.8 GPU-h)
```
필수분(①B + ② + ③)은 **약 9 GPU-h → 4장으로 2.5~3시간**. PSM 통제까지 넣으면 +13 GPU-h.
③은 ETTm1·ETTm2를 먼저 넣어야 꼬리가 짧아진다(긴 작업 우선 배치).

①·②가 끝나는 대로 GPU를 ③에 합류시킬 때는 §45 §1.4 경고대로 **큐 파일을 다시 쓰지 말고**,
남은 작업을 담은 **별도 큐 파일로 새 워커**를 띄울 것.

### 8.5 환경 — `snn_jelly`가 아니라 `snn_recall`

이 서버의 `snn_jelly`는 torch가 1.12.0 → 2.11.0+cu130으로 덮여 있어 **CUDA를 못 잡는다**
(드라이버 535.113.01 = CUDA 12.2). `snn_jelly` 클론에 torch 1.12.0+cu113을 되돌린
**`snn_recall`**이 이 서버의 정답이며, 두 러너의 기본 `PY`가 이미 그것을 가리킨다.

러너에 이 서버에서 추가된 변경:
- 절대경로 하드코딩 제거 → 스크립트 위치 기준(`$R`). **`scripts/SETUP_ON_TARGET.sh`를 다시 실행하지 말 것**
- `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 자동 export (PIL의 `GLIBCXX_3.4.29` 문제)
- `PY` / `EPOCHS` / `RES` / `LOG` 환경변수 오버라이드 지원
- `scripts/gpu_pool2.sh`는 번들에 없어 §45 §1.4 명세대로 재작성됨

### 8.6 검증 완료 상태 (실제 1-epoch 학습, §45 §7.2 준수)

| exp | cond | params | α | 지표 | 1 ep |
|---|---|---:|---:|---:|---:|
| fc | `baseline` | 187264 | na | mse 0.286693 | 31s |
| fc | `recall_raw` | 187265 | 0.9807 | mse 0.282468 | 33s |
| fc | `recall_none` | 187265 | **0.9820** | mse 0.283017 | 35s |
| AD | `recall` | 95937 | 0.9826 | f1(adj) 0.839730 | 122s |

params가 §7의 187,264 / 187,265 / 95,937과 정확히 일치한다.
`recall_none`의 α가 초기값 `sigmoid(4.0)=0.9820`에서 **한 치도 움직이지 않은 것**은 게이트가
실제로 끊겨 α에 그래디언트가 흐르지 않는다는 뜻으로, §1.2·§2.3의 통제가 의도대로 구현돼
있음을 확인해 준다.

**주의**: 위 값은 1 epoch짜리 배관 검증용이며 **결과로 인용해서는 안 된다**.
스크래치로 격리했으므로 이관된 `neorecall_v1/forecasting/results*/raw.txt`에는 섞여 있지 않다.

### 8.7 ★ AD 재계산 — §2.1·§2.2·§2.3 갱신

이관 파일에 `SMD recall_rec seed21`(f1=0.842350)과 `PSM recall seed21`(f1=0.957183)이 들어 있어
문서 §2의 표가 갱신된다.

| 조건 | 문서 §2 | **갱신** |
|---|---|---|
| SMD `recall_rec` | 0.859252 (n=2) | **0.853618 (n=3)** |
| PSM `recall` | 0.959897 (n=2) | **0.958992 (n=3)** |
| SMD `recall_none` | 0.867491 (n=2) | 변동 없음 (n=2) |

**통제 재판정** (seed-paired, §4①의 사전 기준 = 3/3 AND >1.0pp):

| 질문 | 비교 | n | 평균 | seed별 | 판정 |
|---|---|---:|---:|---|---|
| **A. 반전이 필요한가** | `recall` vs `recall_rec` | **3** | +1.32pp | **−0.78** / +2.09 / +2.65 | ❌ **2/3 → 실패** |
| **B. 게이트가 필요한가** | `recall` vs `recall_none` | **3** | **+0.35pp** | **−1.75** / +1.41 / +1.39 | ❌ **2/3 → 실패 (실측 확정)** |
| (참고) SMD 본효과 | `recall` vs `baseline` | 3 | +2.26pp | +0.27 / +2.61 / +3.91 | ✅ 3/3 |
| (참고) PSM 본효과 | `recall` vs `baseline` | 3 | **+0.40pp** | +0.50 / +0.48 / +0.23 | ✅ 3/3 (§2.2의 +0.49pp에서 하향) |

**★ 두 통제 모두 사전 기준(3/3 AND >1.0pp)을 이미 실패했다. 추가 run으로 뒤집을 수 없다.**

- **A**: seed 3개가 다 있고 seed7이 −0.78pp 패배 → **2/3 확정**.
- **B**: seed7이 −1.75pp 패배 → 3/3은 산술적으로 불가능했고, **2026-08-06 seed21 실측(+1.39pp)으로
  2/3 확정**. 평균 +0.35pp로 "평균 >1.0pp" 완화 기준에도 미달한다.

**★ AD의 살아남은 증거 (SMD, n=3 완성)**

| 비교 | 평균 | 우세 | seed별 |
|---|---:|---:|---|
| `recall_none` vs `baseline` (**self-attention 단독**) | **+1.92pp** | **3/3** | +2.02 / +1.20 / +2.52 |
| `recall` vs `baseline` (게이트까지) | +2.26pp | 3/3 | +0.27 / +2.61 / +3.91 |
| 차이 = **게이트의 기여** | +0.35pp | 2/3 | — |

**AD +2.26pp 중 +1.92pp가 self-attention 전환만으로 나온다.** 게이트가 더하는 +0.35pp는
3 seed 중 1개에서 부호가 뒤집히며, forecasting의 게이트 통제(§1.2)와 **정확히 같은 패턴**이다.

§4①이 미리 정해둔 귀결이 그대로 발동한다 — **"둘 중 하나라도 실패 → AD 개선도 self-attention
전환 효과로 귀결"**. 따라서:

> **AD +2.26pp는 Neo 반전이나 게이트의 산물이 아니라, forecasting과 동일하게
> Hippo의 Memory-Replay → self-attention 전환의 산물로 귀결된다.**
> **이 판정에 필요한 추가 실험은 0 runs다.**

이로써 §5의 선택지 2("AD로 활로 찾기")는 **닫혔다**. §6 분기도 "실패" 경로로 확정된다.
`SMD recall_none seed21` 1 run은 판정을 바꾸지 못하며, 보고 표의 n을 3으로 맞추는
**완결성 목적**으로만 가치가 있다.

### 8.8 ⚠️ sha 혼재 — `recall_none` 비교의 출처 문제

이관 파일의 `sha=`는 4종(`890eab1` 149, `fb1c783` 25, `ef8b712` 2, `fded303` 1)이 섞여 있다.
seed-paired 쌍 104개 중 **7개가 커밋 불일치**인데, 그 중 **6개가 하필 `recall_none`**이다:

```
ETTh1_p720 recall_none seed7/13/21 : baseline=890eab1  cond=fb1c783
ETTh2_p96  recall_none seed7/13/21 : baseline=890eab1  cond=fb1c783
```

**즉 §1.2에서 핵심 주장을 뒤집은 −2.11% / −2.90%가 전부 커밋 교차 비교다.** 그리고 문제의 격차
(`recall_none` vs `recall_raw`)는 ETTh2_p96 0.22%, ETTh1_p720 0.08%로 **nuisance 1.4%의 1/6 이하**다.

**다만 과잉 해석은 금물이다.** 러너의 `SHA`는 `git rev-parse --short HEAD` = **감싸는 저장소**의
HEAD인데, 실험 코드가 있는 `NSMT/` 워크스페이스는 **git 추적 대상이 아니다**(`git ls-files NSMT`가
빈 출력). 게다가 위 4개 커밋은 이 저장소에 **존재하지 않는다**(원 remote가 `main`만 보유).
따라서 sha 차이가 모델 코드 차이를 뜻하는지 **여기서는 검증할 수 없다** — 오염의 증거가 아니라
**출처 검증 불가**다.

**실무적 결론 두 가지**
1. **표현을 낮춰야 한다.** 0.08~0.22% 격차로는 "게이트를 끈 쪽이 **더 좋다**"를 지지할 수 없다.
   지지되는 것은 **"게이트 기여가 검출되지 않는다"**이며, §1.2의 결론(개선 원천 = self-attention
   전환)은 이 약한 형태로도 그대로 성립한다.
2. **③을 돌릴 때 선택이 필요하다.** 이 서버의 새 run은 `sha=189bf59`(브랜치 `main`)로 기록되므로,
   이관된 baseline과 비교하면 **또 커밋 교차**가 된다. 서버 내 정합을 원하면 baseline 48 runs를
   같이 재실행해야 하고(실측 6.1 GPU-h), 감수하면 42 runs로 끝난다.
   → **8.8.1에서 이 문제는 실증적으로 해소됐다.**

### 8.8.1 ★ 해소 — 재실행 baseline 이 이관본과 **비트 단위로 일치**

P2(baseline 재실행)가 돌기 시작하자마자 결론이 났다. 이 서버(`sha=189bf59`)의 재실행 결과가
원 서버 값과 **소수점 6자리까지 완전히 동일**하다:

| 설정 | 이관본 | 신규(189bf59) | |
|---|---:|---:|---|
| ETTm1_p720 baseline s7 | 0.458105 (`890eab1`) | 0.458105 | **IDENTICAL** |
| ETTm1_p720 baseline s21 | 0.455896 (`fb1c783`) | 0.455896 | **IDENTICAL** |

**서로 다른 두 원본 커밋(`890eab1`, `fb1c783`)이 제3의 커밋(`189bf59`)에서 똑같이 재현된다.**
따라서 이 세 sha 사이에 **결과를 바꾸는 코드 차이는 없다**. 8.8이 지적한 `recall_none`의
커밋 교차 비교(−2.11% / −2.90%)는 **유효하며, 그 caveat은 철회한다**.

부수적으로 이것은 (a) `snn_recall` 환경이 원 서버 환경을 **정확히 재현**하고,
(b) 이 파이프라인이 seed 고정 하에 **완전 결정적**임을 뜻한다.

**P2의 위상 변화**: P2의 1차 목적(출처 문제 제거)은 달성됐고, MSE 관점에서 P2의 남은 48 runs는
**새 정보를 만들지 않는다**(결정적이므로 이관값과 같을 것이다). 남는 유일한 가치는
**④ 에너지 측정에 필요한 baseline 체크포인트 생산**이다. 6.1 GPU-h를 아끼려면 P2를 줄이고
④에 필요한 대표 설정 몇 개만 남겨도 된다.

---

## 8.9 ★ 중요도 기준 우선순위 (2026-08-06 확정)

**전제**: 8.7로 AD 반전 가설이 판정 종료됐다. 살아남은 주장은 **"Hippo의 Memory-Replay →
self-attention 전환"** 하나뿐이며, forecasting·AD 양쪽에서 동일하게 그 결론이 났다.
따라서 **우선순위는 "이 하나 남은 주장을 얼마나 빨리 반증에 노출시키는가"로 정렬**한다.

| 순위 | 작업 | runs | GPU-h | 왜 이 순위인가 |
|---:|---|---:|---:|---|
| **P0** | ③ **ETTm1·ETTm2** `recall_none` | 24 | 5.0 | **유일한 반증 시험**. 검증된 2설정이 둘 다 ETTh인데, 게이트판(`recall_raw`)은 ETTm1에서 **+1.23%로 해로웠다**. 여기서도 실패하면 처방은 "ETTh 전용"이 되어 기여가 붕괴한다 |
| **P1** | ③ ETTh1·ETTh2 `recall_none` | 18 | 0.8 | 그리드 완성. **가장 싸다** — 42 runs 중 43%인데 비용은 13% |
| **P2** | ③ `baseline` 16설정×3seed 재실행 | 48 | 6.1 | 8.8의 출처 문제를 **완전히 제거**(동일 서버·동일 커밋)하고, 동시에 **④ 에너지에 필요한 baseline 체크포인트**를 만든다. 이관된 것은 `raw.txt`뿐이라 체크포인트가 없다 |
| **P3** | ④ 에너지 (AC/MAC → μJ) | 0 | ~0 | P2가 끝나면 **추가 학습 없이** 측정 가능. `baseline` vs `recall_none` 비교 |
| **P4** | AD `SMD recall_none` seed21 | 1 | 0.5 | **살아남은 주장의 AD측 증거**를 n=3으로 완성. 현재 `recall_none` vs `baseline`은 **+1.61pp (2/2)** — AD판 self-attention 효과다 |
| **P5** | AD `PSM recall_none` ×3 | 3 | 6.5 | 위 효과의 2번째 데이터셋 복제. 가치는 높으나 run당 131분으로 비싸다 |
| **P6** | ② MSL `baseline` + **`recall_none`** | 6 | ~3 | AD 데이터셋 커버리지. **`recall`이 아니라 `recall_none`을 돌릴 것** — `recall`의 가설은 8.7에서 죽었다 |
| **P7** | ① PSM `recall_rec` ×3 | 3 | 6.5 | **이미 기각된 가설(A)의 재검증**. 최저 정보/비용비 |
| **P8** | ⑤ ETTm1 붕괴 진단 | — | — | 가설 검증용. P0 결과가 나온 뒤에야 질문이 구체화된다 |

### 권장 실행 묶음

**P0+P1+P2 = 90 runs / 11.9 GPU-h → 4 GPU로 약 3시간.** 이 한 묶음이
**단일 서버·단일 커밋의 자족적 실험**이 되어 8.8의 caveat 없이 논문에 바로 쓸 수 있다.
§4③의 원래 명령(baseline+recall_none 전 그리드)이 **이 서버에서는 정확히 옳은 선택**이다 —
42 runs만 돌리는 것보다 6.1 GPU-h 더 들 뿐이다.

**긴 작업 우선 배치**: ETTm1(963s) → ETTm2(550s) → ETTh1(179s) → ETTh2(128s) 순으로 큐를 쌓아야
꼬리가 짧아진다. 무작위로 쌓으면 마지막에 ETTm1 한 개가 남아 1시간을 낭비한다.

**P4는 P0~P2와 병렬 가능**하다(AD는 별도 GPU 1장, 32분).

### ★ 8.9.1 실행 결과 (2026-08-06 19:37 완료, 91 runs, SKIP 0)

P0·P1·P2·P4를 전부 완주했다. **③의 전 그리드 판정이 났고, 결론은 부정적이다.**

**recall_none(self-attention 단독) vs baseline — 16설정 × 3seed, 커밋 정합**

| 설정 | Δ% | 우세 | | 설정 | Δ% | 우세 |
|---|---:|---:|---|---|---:|---:|
| ETTh1_p96 | −0.63% | 2/3 | | ETTm1_p96 | −0.79% | 2/3 |
| ETTh1_p192 | −0.43% | 1/3 | | ETTm1_p192 | +1.19% | 1/3 |
| **ETTh1_p336** | **−2.23%** | **3/3** ★ | | **ETTm1_p336** | **+1.52%** | 1/3 ★ |
| **ETTh1_p720** | **−2.90%** | **3/3** ★ | | ETTm1_p720 | +0.97% | 1/3 |
| **ETTh2_p96** | **−2.11%** | **3/3** ★ | | ETTm2_p96 | +0.22% | 2/3 |
| ETTh2_p192 | −1.12% | 2/3 | | ETTm2_p192 | +0.07% | 1/3 |
| ETTh2_p336 | −1.08% | 3/3 | | ETTm2_p336 | −0.03% | 1/3 |
| ETTh2_p720 | +0.54% | 1/3 | | ETTm2_p720 | −0.04% | 2/3 |

(★ = nuisance 1.4% 초과)

| | recall_none | recall_raw | 차이 = **게이트 기여** |
|---|---:|---:|---:|
| ETTh1 | −1.55% | −2.08% | −0.53pp |
| ETTh2 | −0.94% | −0.60% | +0.34pp |
| ETTm1 | +0.72% | +0.79% | +0.06pp |
| ETTm2 | +0.05% | −0.23% | −0.28pp |
| **전체(16설정)** | **−0.43%** | **−0.53%** | **−0.10pp** |

**판정 1 — ③ 실패. self-attention 처방은 일반적이지 않다.**
전 그리드 평균 **−0.43%로 nuisance 1.4%에 한참 못 미친다.** nuisance를 넘는 설정은
**16개 중 3개뿐**(ETTh1_p336·p720, ETTh2_p96)이고, 반대 방향으로 넘는 것도 1개 있다(ETTm1_p336).
기존에 "검증된 2설정"이라 부르던 것이 **바로 그 3개 중 2개**였다 — 즉 전 그리드로 넓히자
**선택 효과였음이 드러났다**.

**어떤 축으로도 설명되지 않는다.** ETTh1은 **긴** horizon에서 강하고(p336·p720), ETTh2는 **짧은**
horizon에서 강하다(p96, 반면 p720은 +0.54%). **horizon 방향이 정반대**여서 "장기에 유리하다"는
해석은 성립하지 않는다.

**판정 2 — 게이트 무효가 전 그리드에서 확정됐다.**
게이트 기여가 전체 **−0.10pp**이고 데이터셋별 부호가 제각각(−0.53 / +0.34 / +0.06 / −0.28pp)이다.
기존에는 2설정만으로 주장하던 §1.2의 결론이 이제 **16설정 × 3seed로 확립**됐다.
AD에서도 동일하다(게이트 기여 +0.35pp, 2/3 — 8.7).

**따라서 §5의 선택지 2는 AD 통제로, 선택지 1의 "대칭적 self-attention이 더 낫다"는 부분은
③으로 각각 닫혔다.** 남는 기여는 성능이 아니라 **인과 규명과 방법론**이다:

1. **Neocortex 경로는 인과적으로 무효** — forecasting 16설정×3seed와 AD 양쪽에서 게이트 기여 ≈ 0
2. **곱셈 게이트는 신호를 죽인다** — +5.35%(0/3), Neo 발화율 0.13% vs Hippo 12.83%
3. **방법론** — 사전 등록 판정 기준, nuisance 임계, 게이트 격리 통제, 결정성 검증(8.8.1)

### 우선순위에서 내려간 것들과 이유

- **AD 통제(①) 전체**: 판정이 끝났다. 추가 run은 결론을 바꾸지 못한다
- **MSL**: §4②는 "AD 주장이 3개 데이터셋에 근거하게 된다"를 노렸으나, 그 주장 자체가 기각됐다.
  이제 MSL은 **null 결과의 데이터셋 수를 늘리는 일**이라 급하지 않다
- **PSM `recall_rec`**: 죽은 가설에 13 GPU-h의 절반을 쓰는 일

---

## 8.10 ④ 에너지 측정 결과 (추가 학습 0)

`--test`가 이미 켜져 있어 90 runs 각각에 `final+result.csv`(AC/MAC/µJ/발화율)와
`model+info+per+layer.txt`(레이어별 분해)가 **이미 생성돼 있었다**. 재실행 불필요.

### 8.10.1 head 는 에너지 병목이 아니다 — 파라미터 비중으로 추정하면 틀린다

```
(head): Linear(1.11 M, 96.526% Params, 11.66 M Ops, 48.594% ACs, 0.0 Ops, 0.000% MACs)
```
`head`는 **파라미터의 96.5%**지만 입력이 스파이크라 **MAC이 0%**이고 전부 AC다.
에너지 기여는 ETTm1_p720 기준 **21.7%**. **params 비중으로 에너지를 추정하면 안 된다.**

MAC(에너지의 52~76%)의 실제 출처: `emb_linear` 36.9% + **`BatchNorm1d` 9개 63.1%**.
단, **BN을 MAC으로 세는 것은 이 분야의 합의된 회계 관행이므로 건드리지 않는다**(사용자 판단).

### 8.10.2 ★ `recall_none`은 에너지를 20% 더 쓰고 성능은 그대로다

두 조건의 BN 개수가 같으므로 **회계 방식과 무관한 결과**다.

| | 평균 (14설정 × 3seed) |
|---|---:|
| 에너지 | **+20.1%** (전 설정 증가, +12.6% ~ +26.5%) |
| 발화율 | +47.9% |
| MSE | −0.13% (nuisance 이하 = 무효) |

**살아남은 처방이라 불리던 `recall_none`은 사실 "더 비싼 동률"이었다.** §8.9.1의 성능 기각과
합쳐, self-attention 전환은 성능·에너지 **양쪽 모두에서 근거를 잃는다.**

---

## 8.11 ★ 구조 유지 성능 개선 시도 (2026-08-08)

**목표**: 구조를 최대한 유지한 채 성능을 올린다. 모델 코드는 **한 줄도 바꾸지 않았고**,
러너에 환경변수 오버라이드만 추가했다.

### 8.11.1 학습 스케줄 — 기각

전 90 runs가 `-e 50`을 허용받고도 **중앙값 8 epoch**(최소 4, 최대 18)에서 조기종료된다.
`EarlyStopping patience=3` vs `ReduceLROnPlateau patience=2`(utils.py:183 하드코딩)로 간격이
1 epoch뿐이라, lr을 낮추자마자 학습이 끊긴다(로그에 `lr=0.001`만 관측).

`--patience 10`으로 늘리자 학습은 **15~32 epoch**으로 늘었으나 성능은:

| 설정 | pat3 | pat10 | Δ |
|---|---:|---:|---:|
| ETTh1_p96 | 0.381151 | 0.377708 | −0.90% (2/3) |
| ETTh2_p96 | 0.290564 | 0.290981 | +0.14% (2/3) |

**평균 −0.38%로 nuisance 이하 → 기각.** 모델은 8 epoch에서 이미 수렴해 있었다.

> **부수 발견**: **`--warm_up_epoch`은 아무 동작도 하지 않는다.** `config.py:70`에서 인자를
> 정의만 하고 어디서도 읽지 않는다. `pat10`과 `pat10wu3`의 MSE가 소수점 6자리까지 동일하다.
> 러너가 넘기던 `--warm_up_epoch 0`은 무의미했다.

### 8.11.2 ★ `neo_tau` 8.0 → 2.0 — 이 세션 최고의 개입 (48 runs, 전 그리드)

`neo_tau`는 **Neo LIF의 막시정수**로 `ours.py:664`에서만 쓰인다. 순전파 동특성만 바꾸며
파라미터·구조 변경이 없다. 기존값 8.0의 근거("probe optimum ~8")는 저장소에 기록이 없다.

| | Δ vs `neo_tau=8` | 우세 |
|---|---:|---:|
| **ETTh1** | **−1.94%** | **11/12** |
| ETTh2 | −0.36% | 8/12 |
| ETTm2 | −0.12% | 8/12 |
| ETTm1 | +0.32% | 4/12 |
| **전체** | **−0.52%** | 31/48 |

| | 변화 |
|---|---:|
| 발화율 | **+43.6%** |
| 에너지 | **+2.3%** |

**ETTh1 −1.94%(11/12)는 nuisance 1.4%를 넘는다.** 그리고 `recall_none`(ETTh1 −1.55%, 에너지
+20%)보다 **성능이 좋고 비용은 1/9다.** 다만 전 그리드 평균 −0.52%는 여전히 nuisance 이하다.

> **판정 범위의 정정**: `neo_tau`는 Neo LIF만 건드리는데 결과가 바뀌었다. 따라서
> **"Neo 경로는 인과적으로 무효"는 경로 B(recall, 이중 stop-grad)에 한정된 판정**이며,
> 경로 A(baseline, Memory-Replay)에는 그대로 일반화되지 않는다.

### 8.11.3 ★ 2×2 요인 설계 — `neo_tau` × 그래디언트 스로틀 (ETTh1, n=12)

`--neo_full_grad`는 `ours.py:269`의 5% 스로틀(`mx = 0.05*mx + 0.95*mx.detach()`)을 제거한다.
**이 저장소에서 한 번도 켜진 적이 없던 플래그다.**

| ETTh1 | `neo_tau=8` | `neo_tau=2` |
|---|---:|---:|
| **grad 5%** (기본) | 0.00% (기준) | **−1.94%** (11/12) |
| **grad 100%** | −0.95% (9/12) | −1.14% (9/12) |

```
주효과 grad  −0.95pp | 주효과 tau  −1.94pp
단순 합 예측 −2.88pp | 실제        −1.14pp
교호작용     +1.75pp   ← 강한 음의 시너지
```

**두 개입이 각각은 도움이 되나 합치면 상쇄된다.** `neo_tau=2` 단독이 최선이고
**`--neo_full_grad`는 켜지 말 것**(tau=2와 병용 시 −0.80pp 손해).

### 8.11.4 왜 그런가 — 융합 구조가 설명한다

기본 융합(`neo_fuse='gate_attn'`, `ours.py:289-291`):
```python
g = x * mx.transpose(0, 2).contiguous()   # Neo 시간축 T -> Hippo 패치축 N (T=N 규약)
x = x * (1. - self.mca(x, g))             # IAND: 억제만 가능
x = x * (1. - self.mlp(x))
```

1. **`g`에 Neo 고유 정보가 없다.** Neo는 Hippo를 마스킹할 뿐이고 내용물은 전부 Hippo에서 온다.
   Neo가 침묵하면(0.13%) `g≈0` → `x*(1−0)=x` → **항등**.
2. **`(1−·)` 형태라 Neo는 억제만 할 수 있고 덧셈 주입 경로가 없다.**
3. **따라서 Neo가 예측 손실을 줄이는 최적해는 "정보 전달"이 아니라 "침묵"이다.**
   → `--neo_full_grad`가 해로운 이유. 5% 스로틀은 그 붕괴를 늦추던 장치였을 가능성이 크다.
4. 실패한 융합 목록(`gate_only`·`gate_mix`·`xattn_hq`·`xattn_nq`·`cmpl`)이 **전부 `x*(1−·)`**다.
   유일한 예외가 `cmpl_add`(`x = x + a`)이나 다른 변수와 얽혀 결론이 나지 않았다.

**해석**: `neo_tau`의 이득은 정보량 증가가 아니라 **억제의 강도·시간 분포 변화**일 가능성이 높다.
발화율 +44%가 정보였다면 학습 압력(grad 100%)을 더했을 때 강화됐어야 하는데 상쇄됐다.

### 8.11.5 다음 후보

1. **`neo_tau` 미세 탐색** (1.0 / 1.5 / 4.0) — 2.0이 최적인지. 코드 변경 0, ETTh1 36 runs ≈ 15분
2. **`--alpha`로 ratio 고정** (§4.5.2) — horizon 교란을 제거한 뒤 위 결과들을 재측정
3. **가법 잔차** (`x = x + mca(x,g)`) — 8.11.4의 가설을 직접 시험. **구조 변경에 해당**

---

## 8.12 §4.5.2 손실 가중치 스케일링 — 코드 검증 및 파급

`train.py:276`에서 **확인됨**:
```python
alpha = args.alpha * (args.pred_len/720)     # 276
train_one_epoch(model, train_loader, optimizer1, alpha, args.device)   # 287
val_one_epoch(model, val_loader, alpha, args.device)                   # 288
```
`--alpha 0.5` 기준 실제 Neo 손실 비중 p96 **0.0667** → p720 **0.5000** (**7.5배**). §4.5.2의
표와 정확히 일치한다.

**이미 관측된 결과에 미치는 영향**

| | 영향 |
|---|---|
| **같은 horizon 내 조건 비교** | **무영향** — 비교 대상이 같은 ratio를 공유한다. §8.9.1·§8.11.2의 설정별 판정은 그대로 유효 |
| **horizon 간 패턴 해석** | **교란됨** — "긴 horizon에서 효과가 크다"가 "Neo 손실 가중치가 크다"와 분리되지 않는다 |

구체적으로 §8.9.1의 `recall_none`(ETTh1 p96 −0.63% → p720 −2.90%)은 horizon 단조 증가라
이 교란과 형태가 일치한다. 반면 ETTh2는 p96 −2.11% → p720 +0.54%로 **정반대**여서 교란만으로는
설명되지 않는다. §8.11.2의 `neo_tau`도 p336이 최대(−2.48%)로 단조가 아니다.

**→ 교란은 실재하나 관측 패턴을 전부 설명하지는 못한다.** §4.5.2의 제안대로 ratio를 고정한
재측정이 필요하며, 이는 8.11.5의 후보 2번이다.

---

## 8.13 조건 B 개입 5연속 실패 — 종합 (2026-08-09 완료)

`recall_raw` (`stop` + `pool`) 를 기준으로, 모든 개입을 시드 페어링 48런 전 그리드에서 측정한 결과.
`Δ` 는 MSE 변화율이므로 **양수가 나쁨**.

| 개입 | 무엇을 바꿨나 | 전체 Δ | 개선 | 판정 |
|---|---|---:|---:|---|
| `recall_none` | Neo→Hippo 게이트 제거 | ~0 | — | 무효 (게이트 기여 −0.10pp) |
| `neo_tau` 2 vs 8 | Neo LIF 시상수 | −0.00% | 24/48 | 무효 |
| `neo_recall_tokens=full` | Neo 토큰 축 24배 확대 | +0.00% | 24/48 | 무효 |
| `neo_recall_grad=to_neo` | 예측 손실이 Neo 에 도달 | +0.31% | 20/48 | 약하게 해로움 |
| `neo_recall_grad=full` | 양방향 detach 제거 | +0.78% | 17/48 | 해로움 |

**`to_neo` 가 가른 것.** `full` 의 손해 +0.78% 는 두 성분의 합이다.
- Neo 에 예측 압력을 준 것 자체: 약 **+0.31%** (`to_neo` 가 격리 측정)
- 복원 손실이 Hippo 로 역류한 오염: 나머지 약 **+0.47%**

오염 가설의 직접 증거는 horizon 의존성이다. `full` 은 `ratio` 가 큰 곳에 피해가 집중되어
ETTm1_p720 에서 +4.69% 인데, `to_neo` 는 p96 +0.82% / p720 +0.68% 로 **평평하다**.
Hippo 를 복원 손실에서 차단하니 `ratio` 연동 피해가 정확히 사라졌다 (§8.12 의 `ratio` 스케일링과 일관).

**학습된 α 의 방향.** `stop` 평균 0.9551 (최소 0.8899) → `to_neo` 평균 0.9429 (최소 0.8329).
예측 압력을 받은 Neo 는 자기 비중을 **키웠고** (1−α: 4.5% → 5.7%) 그 결과가 성능 악화다.
조건 A 의 2×2 (§8.11.3) 에서 곱셈 게이트가 Neo 를 침묵으로 수렴시킨 것과 부호는 반대지만
결론은 같다 — **예측 손실에 노출된 Neo 는 유용한 방향으로 학습되지 않는다.**
(단 §1.4 대로 α 의 움직임 자체는 유용성의 증거가 아니다.)

**결론:** 통로를 넓혀도 무효, 학습 압력을 줘도 해로움. 절연을 유지한 원설계가 최선이다.
남은 가설은 개입의 *경로* 가 아니라 Neo 의 *과제* 다 → §8.14.

## 8.14 ★ auxiliary loss 재설계 (2026-08-09 착수)

### 8.14.1 전제 정정 — "저주파 복원" 은 조건 B 의 현재 설정이 아니다

`ours.py` 의 저주파(DCT lowpass) 분기는 조건 B 에서 **막혀 있다**:

```python
if (self.train_mode == 'training') and (self.keep_ratio > 0) and (not self.neo_aux_next) \
        and not (self.neo_recall and self.neo_recall_tgt == 'raw'):    # 조건 B 에서 False
```

`recall_raw` = `--neo_recall --neo_recall_tgt raw` 이므로 `org_x` 는 **정규화된 원 패치**로 남는다.
즉 "원 시계열 복원" 은 이미 현재 기준선이다. 저주파 버전(`recall_low`) 은 이미 측정되어 있다:

```
recall_low vs recall_raw   n=47   전체 +0.18%   개선 20/47
```

저주파 타깃이 근소하게 더 나쁘다. **타깃을 raw 로 바꾸는 것은 새 실험이 아니다.**

### 8.14.2 "Hippo 표현 복원" 은 조건 B 에서 항등함수로 퇴화한다

`distill` 은 이미 매 스텝 계산되고 있고 (`train.py:88`) 역전파만 되지 않는다. 전개하면:

- Neo 입력 `_nin = _h.transpose(0,2).mean(2,keepdim=True)`
- `x_data` = 게이트 **이후** x_hippo 를 같은 방식으로 pooling = `α·_nin + (1−α)·x_neo`
- `x_time` = `x_neo`
- ⇒ `distill = α²·‖_nin − x_neo‖²`

**Neo 의 입력과 Neo 의 출력의 거리다.** 조건 B 에서는 Neo 의 입력이 곧 Hippo 표현이므로
"Hippo 표현 복원" 이 자동으로 자기복사가 되고, 2층 LIF 가 항등에 수렴하면 0 이 된다.
(덤으로 `x_data` 가 detach 되어 있지 않아 Hippo 로도 흘러 `grad=full` 실패를 재연한다.)
→ **문자 그대로는 실행할 가치가 없다.**

### 8.14.3 진짜 가설 — 과제가 너무 쉽다

지금 Neo 의 과제는 "패치 n 의 Hippo 부호에서 패치 n 의 raw 8값 복원", 즉 **답이 입력에 들어 있는
오토인코딩**이다. 이것이 조건 B 5연속 무효(§8.13)를 설명하는 선도 가설이다: Neo 는 Hippo 가
이미 가진 것 이상을 배울 이유가 없다. 필요한 것은 타깃 교체가 아니라 **병목**이다.

| | 과제 | 왜 비퇴화인가 | 플래그 |
|---|---|---|---|
| **C** | 패치 n → 패치 n+1 의 raw 값 | 시간 방향 이동, 답이 입력에 없음 | `--neo_aux_next` |
| **A** | Neo 입력 패치의 절반을 0 으로 가리고, **가려진 패치만** 채점 | 답이 입력에서 제거됨 (MAE 식) | `--neo_mask_ratio 0.5` |
| B | 패치 n → 패치 n+1 의 Hippo 표현 | 8.14.2 의 비퇴화 버전 | 보류 (A/C 결과 후 판단) |

C→A 순으로 실행. 각 arm 48런 (4 데이터셋 × 4 horizon × 3 시드), 기준 arm 도 동일 코드로 재실행.

### 8.14.4 코드 정리 — 조건 B 에서 버려지던 Neo 패스 제거

조건 B 에서도 raw 패치 Neo 경로(`encoding_neo` → `time_block`)가 실행된 뒤 recall 블록에서
**덮어써지고 있었다.** LIF 는 forward 마다 리셋되므로 결과가 오염되지는 않았으나, 순수한 낭비였고
Neo BatchNorm 의 running statistics 를 갱신하고 있었다.

정리 내역:
- 조건 B 에서 raw 패치 Neo 패스 스킵. 진단/탭 블록은 `_neo_post()` 로 추출해 두 조건 모두 **최종**
  `x_neo` 를 보도록 배선.
- `--neo_recall_tokens` 제거 (실패한 축, §8.13), 원래의 `.mean(2, keepdim=True)` 풀링으로 복귀.
- `aux_l1` 을 `neo_aux_next` 에서 **분리**. 종전에는 하드와이어라 "다음 패치 타깃" 과 "MSE→MAE" 가
  교락되어 있었다. 이제 `--aux_l1` 은 독립 플래그이고, C 는 MSE 를 유지한 순수 타깃 교체다.
- 죽은 파라미터 **4,864개** 삭제: `Block.time_recon` (4,224 — 전 조건에서 미사용), 조건 B 의
  `encoding_neo` (640). 파라미터 187,265 → **182,401**.

**회귀 검증 (ETTh1_p96, 3 시드):**

| | seed 7 | seed 13 | seed 21 | 평균 Δ |
|---|---:|---:|---:|---:|
| 기준 `recall_raw_t8` | 0.371521 | 0.386043 | 0.370930 | — |
| 버려지던 패스 제거 후 | 0.371586 | 0.386008 | 0.370801 | **−0.009%** |
| 죽은 파라미터 삭제 후 | 0.371586 | 0.386008 | 0.370801 | **0.000%** (비트 일치) |

버려지던 패스 제거는 **비트 단위로 일치하지 않는다** (BN running stats 경유). 크기는 nuisance
임계(1.4%)의 1/70 이라 기존 결론은 모두 유지되지만, 엄밀성을 위해 **기준 arm(`cl0`) 을 동일
코드로 재실행**하여 C/A 와 짝지운다. 죽은 파라미터 삭제는 `--init_order_fix` 의 이름 시드 초기화
덕분에 비트 단위로 일치했다 — §41 의 "미사용 모듈 삭제가 초기화를 밀지 않는다" 를 재확인.

**§8.10 에너지 수치 무효화.** §8.10.2 의 "`recall_none` 이 에너지를 +20.1% 더 쓴다" 는 버려지던
Neo 패스를 포함해 측정한 값이다. 정리 후 조건 B 는 그만큼 싸졌으므로 **재측정이 필요하다**
(학습 불필요, syops 만 다시 돌리면 된다). 성능 결론에는 영향이 없다.

### 8.14.5 ★ 실행 결과 (2026-08-10 완료, 144 runs, SKIP 0)

기준 `cl0` 는 §8.14.4 의 정리된 코드로 재실행한 `recall_raw` (`stop`+`pool`+raw). 세 arm 모두
동일 코드·동일 시드(7/13/21)·전 그리드(4 데이터셋 × 4 horizon). **양수가 나쁨.**

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| **C `auxnext`** (다음 패치 예측) | +0.46% | +0.01% | +0.08% | +0.01% | **+0.14%** | 20/48 |
| **A `mask50`** (마스킹 복원) | −0.60% | +0.18% | +0.70% | +0.03% | **+0.08%** | 21/48 |

horizon 별:

| | p96 | p192 | p336 | p720 |
|---|---:|---:|---:|---:|
| C `auxnext` | +0.12% | +0.52% | −0.07% | −0.02% |
| A `mask50` | +0.01% | +0.18% | +0.50% | −0.38% |

**둘 다 무효다.** 개선 20/48, 21/48 은 동전 던지기이고 크기는 nuisance 임계(1.4%)의 1/10 이하다.
중간 점검에서 관측된 신호는 모두 부분집합 편향이었다.
- ETTm1_p720 의 C −0.56% (n=3): 전 그리드에서 +0.14% 로 소멸.
- ETTm1 의 A +0.70%: ETTh1 −0.60% 가 상쇄. **데이터셋 방향 역전** (§8.9.1 과 같은 패턴).
- C 의 horizon 단조성(p96 +0.46% → p720 −0.33%, ETTm 부분집합): 전 그리드에서 사라짐.

학습된 α: cl0 0.9508 / auxnext 0.9424 / mask50 0.9546. 세 arm 의 서열은 스윕 내내 안정적이었으나
(C 가 Neo 비중을 키우고 A 가 줄임) **성능은 셋 다 구분되지 않는다.** α 와 유용성이 분리되어 있다는
§1.4 의 경고에 대한 직접 증거다.

### 8.14.6 ★ 가설 기각 — 그리고 구조적 상한

§8.14.3 의 가설 *"Neo 의 과제가 답이 입력에 든 오토인코딩이라 너무 쉽다"* 는 **기각한다.**
과제를 시간 방향으로 밀어도(C) null, 답을 입력에서 제거해도(A) null 이다. 조건 B 의 개입은 이제
**일곱 번 연속 무효**다 (게이트, `neo_tau`, `tokens=full`, `grad=to_neo`, `grad=full`, `auxnext`, `mask50`).

원인은 aux 손실이 아니라 **읽기 경로의 구조적 상한**으로 보인다. 조건 B 에서:

- `_g = x_neo.transpose(0,2)` 는 `[1,BC,N,D]` 이므로 **스파이킹 시간축 T 에 대해 상수**다.
- `self.head` 는 `nn.Linear(embed_dim*num_patches, pred_len, bias=False)`, 비선형이 전혀 없다.
- 테스트 지표는 T 평균이다 (`reduce_T="mean"`, test.py).

따라서 `z_t = α·head(_h_t) + (1−α)·head(_g)` 이고 **두 번째 항은 모든 t 에서 동일하다.**
즉 **Neo 가 예측에 미치는 영향 전부는 시간축에 대해 상수인 덧셈 벡터 하나**다. Neo 는 T 축에서
아무것도 변조할 수 없고 예측을 평행이동만 시킬 수 있다. head 에 `bias=False` 라 별도 편향 항이
없다는 점까지 겹쳐, "모델이 편향이 필요해서 α 를 내렸다" 와 "Neo 내용이 유용해서 내렸다" 가
**원리적으로 구분되지 않는다.**

Neo 가 무엇을 배우든 상수 편향으로만 전달된다면, aux 손실을 어떻게 바꿔도 개선이 안 나오는 것이
당연하다. 다음 후보는 Neo 의 *과제* 가 아니라 *전달 경로* 여야 한다.

---

## 8.15 ★ Neo → 고정 스파이킹 reservoir 교체 (2026-08-11 착수)

`docs/reservoir_summary.md` 의 설계 중 **"Neo network 를 reservoir network 로 교체"** 한 조각만
떼어내 조건 B 에 적용한다. 문서의 나머지 층(cross-window state, chronological cache, ridge/RLS,
episodic bank, gate·loss 재설계)은 **가져오지 않는다.**

### 8.15.1 현재 Neo 의 정체 — 재귀가 아예 없다

`TemporalBlock` = 2 × `NeoLayer` 이고 `neo_recurrent` 기본값이 `False` 이므로 **`W_rec` 이 없다.**
유일한 시간 결합은 LIF 막전위의 누설 적분이고, `W_in`·BN 은 학습되며, 출력은 이진 스파이크만 쓴다.
따라서 교체가 바꾸는 것은 정확히 셋이다.

| | 현재 Neo (`neolayer`) | reservoir |
|---|---|---|
| 재귀 연결 | **없음** | 고정 sparse E/I `W_rec` (Dale, ρ=0.9) |
| 코어 학습 | `W_in`·BN 학습 | **전부 buffer, no_grad** |
| readout | 이진 스파이크만 | spike + 막전위 + filtered trace ×2 |

### 8.15.2 조건 B 가 오히려 적합한 host 다

문서 §14.1 은 최초 모델에서 시작하라고 권고하지만, 최초 모델(조건 A)은 task gradient 가 5%
throttle 로 Neo 에 들어간다 — 문서 §11.2 가 제거하라고 지목한 바로 그 hack 이다. 조건 B 는
`--neo_recall_grad stop` 이라 **Neo 경로의 task gradient 가 0** 이고, aux 손실만 `weak_decoder`
(단일 Linear)를 통해 들어온다. 코어를 얼리면 남는 학습 대상은 `SpikLinearLayer(1024→64)` 투영뿐이라,
**고정 feature 위의 선형 readout** 이라는 RC 구조와 정확히 일치한다 (ridge 의 gradient 버전).

### 8.15.3 구현

`SpikingReservoir` (ours.py). R=256, φ=[s, u, q₁, q₂] → F=1024 → `SpikLinearLayer`(학습) → 스파이크 [N,BC,1,64].
출력을 스파이크로 유지해 α 게이트·`weak_decoder`·AC 에너지 회계가 종전과 동일한 것을 본다.

```
i_t = a_s·i_{t-1} + W_in e_t + W_rec s_{t-1}      # 전부 buffer, no_grad
u_t = a_m·u_{t-1} + (1-a_m)·i_t
s_t = H(u_t - 1);  u_t ← u_t - s_t
q_t^k = λ_k q_{t-1}^k + (1-λ_k) s_t               # K=2
```

- 위상 RNG 는 **로컬 generator**. 전역 스트림에서 뽑으면 이후 구축되는 모든 모듈의 초기화가 밀려
  `--init_order_fix` 계약이 깨진다.
- `res_seed` 기본 −1 = 실행 시드 사용 → 3 시드가 3 개의 서로 다른 위상을 표집한다 (문서 §15.8).
- 학습 파라미터 182,401 → **239,617** (투영층 57,216 증가). `res` 와 `res_norec` 은 파라미터가
  **정확히 같으므로**, 재귀의 인과 대조만 파라미터 매칭이다. reservoir vs `cl0` 는 매칭이 아니다.

### 8.15.4 ⚠️ 발화율 교정 — 기본값으로는 liquid 가 완전히 침묵했다

코드 기본 `res_in_scale=1.0` 에서 **발화율 0.0000, 침묵 뉴런 100%** 였다. 문서 §12.5 가 경고한
실패 모드 그대로이고, 이대로면 `W_rec·s = 0` 이라 `res` 와 `res_norec` 이 동일해져 실험이 무의미해진다.

| `in_scale` | 1 | 3 | 10 | **30** | 100 | 300 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 발화율 | 0.0000 | 0.0027 | 0.0557 | **0.2205** | 0.3838 | 0.4493 | 0.4751 |
| 침묵 | 1.000 | 0.750 | 0.203 | **0.070** | 0.027 | 0.016 | 0.016 |
| 포화 | 0 | 0 | 0 | **0.000** | 0.059 | 0.102 | 0.133 |

`in_scale=30` 을 기본값으로 확정 (발화율 0.22 / 침묵 7% / 포화 0%). 재귀 유무가 실제로 동역학을
바꾸는 것도 확인했다: 발화율 0.2205(full) vs 0.1882(zero), 침묵 7.0% vs 10.5%.
고정 코어라도 **읽는 Hippo 표현이 학습 중 변하면 발화율이 드리프트**하므로 `train.py` 가 매 epoch
`[reservoir] rate/silent/saturated` 를 찍는다. 1 epoch 실학습 후 0.2224 / 0.195 / 0.008 로 대역 유지.

### 8.15.5 회귀 검증

`cl0` (ETTh1_p96, 3 시드) 0.371586 / 0.386008 / 0.370801 — **bit-exact 유지.** 리저버 코드 경로
추가가 기존 초기화를 밀지 않았으므로 §8.14.5 의 `cl0` 48런을 그대로 기준선으로 재사용한다
(재실행 불필요, 96런만 신규).

### 8.15.6 이 실험이 답하는 것과 답하지 않는 것

**답한다:** 고정 랜덤 재귀 liquid 가 학습된 비재귀 spiking MLP 보다 Neo encoder 로 나은가.
그리고 (파라미터 매칭 대조로) 그 차이가 **재귀 연결** 때문인가.

**답하지 않는다:** 장기기억. batch 마다 `functional.reset_net` 이 그대로이므로 state 가 window 를
넘지 않는다. 문서 §2.5 의 **within-window context** 층에만 해당하며, cross-window·episodic 주장은
불가능하다. 스캔 축도 patch 24 스텝뿐이라 MC 곡선류 진단은 의미가 약하다 (`S_micro=1`, 문서 §7.3.1).

**미리 적어두는 대안 설명:** §8.14.6 의 구조적 상한이 그대로 남는다. Neo 가 무엇이 되든 예측에는
**T-상수 덧셈 벡터 하나**로만 전달된다. 무효가 나오면 "reservoir 가 나쁘다" 와 "전달 경로가 막혀
있다" 를 구분할 수 없으므로, 그 경우 다음 후보는 코어가 아니라 읽기 경로여야 한다.

### 8.15.7 ★ 실행 결과 (2026-08-11 완료, 96 runs, SKIP 0)

기준 `cl0` 는 §8.14.5 의 48런을 그대로 재사용 (§8.15.5 에서 bit-exact 확인). 전 그리드
4 데이터셋 × 4 horizon × 3 시드, 시드 페어링. **양수가 나쁨.**

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| `res` vs `cl0` (교체 전체 효과) | +0.55% | −0.09% | +0.38% | +0.08% | **+0.23%** | 20/48 |
| `resnorec` vs `cl0` | +0.25% | +0.21% | +0.48% | −0.03% | **+0.23%** | 21/48 |
| **`res` vs `resnorec`** (재귀만, 파라미터 매칭) | +0.30% | −0.30% | −0.09% | +0.11% | **+0.00%** | 22/48 |

horizon 별:

| | p96 | p192 | p336 | p720 |
|---|---:|---:|---:|---:|
| `res` vs `cl0` | +0.07% | +0.41% | +0.22% | +0.21% |
| `res` vs `resnorec` | −0.23% | −0.14% | +0.53% | −0.14% |

nuisance 임계(1.4%) 를 넘는 셀: `res` vs `cl0` **0/16**, `resnorec` vs `cl0` **0/16**,
`res` vs `resnorec` 1/16 (ETTm1_p336 +1.52%, 다른 15 셀은 전부 임계 미만).

**두 결론이 서로 다른 강도로 성립한다.**

1. **재귀 연결은 정확히 무효다 (강한 결론).** `res` vs `resnorec` 는 파라미터가 **완전히 동일**하고
   위상 시드까지 공유하는 대조인데 전체 **+0.00%, 개선 22/48** 이다. 데이터셋별 부호도 갈린다
   (+0.30 / −0.30 / −0.09 / +0.11). 고정 sparse E/I `W_rec`(ρ=0.9, 뉴런당 25.5 연결)이 발화율을
   0.209 → 0.242 로 실제로 바꾸는데도 성능은 움직이지 않는다.
2. **교체 전체는 무효~미세하게 해롭다 (약한 결론).** `res`/`resnorec` 모두 `cl0` 대비 **+0.23%** 로
   완전히 같다. 파라미터가 57,216 개 많은 비매칭 비교이므로 "리저버가 나쁘다" 로 읽을 수는 없고,
   **"57k 파라미터를 더 쓰고도 이득이 없다"** 가 정확한 진술이다.

두 arm 이 `cl0` 대비 소수점 둘째 자리까지 같은 값(+0.23%)이라는 사실 자체가 정보다. `cl0` 와의
차이 전부가 backend 교체(코어 동결 + 풍부한 readout + 큰 투영층)에서 오고 **재귀는 한 톨도
기여하지 않는다.**

**건전성 (전 96런 최종 epoch):** `res` 발화율 0.184~0.305 (평균 0.242), 침묵 평균 9.5% 최대 24.6%,
포화 최대 4.7%. `resnorec` 0.161~0.260 (평균 0.209), 침묵 평균 13.5%. 전 런에서 대역을 유지했고
두 arm 의 동역학 분리도 유지됐다. **"사실은 같은 모델이었다" 로는 무효를 설명할 수 없다.**

학습된 α: cl0 0.9508 / res 0.9408 / resnorec 0.9423. 리저버가 Neo 지분을 4.9% → 5.9% 로 약간
키우지만 두 리저버 arm 이 서로 같으므로, α 이동은 **재귀가 아니라 backend 통계 변화**에 반응한 것이다.
성능이 무효인 것과 합치면 §8.14.6 의 "α ≠ 유용성" 을 다시 확인한다.

### 8.15.8 해석 — 코어를 바꿔도 안 되는 이유

조건 B 개입은 이제 **아홉 번 연속 무효**다 (게이트, `neo_tau`, `tokens=full`, `grad=to_neo`,
`grad=full`, `auxnext`, `mask50`, `reservoir`, `reservoir+recurrence`).

이번 결과가 앞의 일곱과 다른 점은 **Neo 의 계산 방식을 통째로 갈았는데도** 같은 자리에 왔다는 것이다.
학습된 비재귀 spiking MLP → 고정 랜덤 재귀 liquid + 막전위/trace readout 은 문헌 기준으로 완전히
다른 모델인데 전 그리드 차이가 +0.23% 다. 과제(§8.14), 그래디언트 경계(§8.13), 이제 코어(§8.15)까지
바꿔도 움직이지 않는다.

§8.14.6 에 적어둔 구조적 상한이 이 패턴을 그대로 설명한다. `_g` 는 T 축에 대해 상수이고 head 는
비선형 없는 Linear 이며 지표는 T 평균이므로, **Neo 가 무엇을 계산하든 예측에는 T-상수 덧셈 벡터
하나로만 도달한다.** 리저버가 아무리 풍부한 표현을 만들어도 이 64→pred_len 선형 사상 하나를 통과해야
하고, 그 사상은 이미 Hippo 가 채우고 있다. 코어 교체 실험은 사전에 적어둔 대안 설명("전달 경로가
막혀 있다")을 **기각하지 못하며, 오히려 그쪽을 지지한다.**

다음 후보는 Neo 의 *계산* 이 아니라 *읽기 경로* 여야 한다. 최소 요건은 Neo 기여가 T 축에서 상수가
아니게 만드는 것 — 예: 게이트를 `[T,BC,N,1]` 로 확장하거나, Neo 를 head 이전이 아니라 예측 수준에서
혼합하거나(reservoir_summary §10.2 의 prediction-level mixture), Hippo 를 T 평균한 뒤 결합하는
single-head 재설계(§10.3 의 후속안). 셋 다 head 를 건드리므로 별도 ablation 이 필요하다.

---

## 8.16 ★ stop-gradient 전면 해제 + aux OFF + 진짜 cross-attention (2026-08-12 완료, 144 runs, SKIP 0)

### 8.16.1 구성

| | Block fusion | Neo 읽기 | aux | Hippo↔Neo grad | params |
|---|---|---|---|---|---:|
| `h0` | self-attn | **없음** (`gate=none`) | **OFF** | — | 182,401 |
| `g0` | self-attn | α 게이트 | **OFF** | **전면 개방** | 239,617 |
| `x0` | **cross-attn (q=Hippo, kv=Neo)** | Block 내부 | **OFF** | **전면 개방** | 239,617 |

Neo 코어는 세 arm 모두 **고정 리저버**(R=256, no_grad). `--alpha 0` 으로 `ratio=0` → `loss = ce`.
`--neo_recall_grad full` + `--neo_full_grad` 로 인터페이스 detach 와 5% throttle 을 모두 제거.

**순환 의존 해소:** cross-attention 은 Neo 를 key/value 로 읽으므로 Neo 가 Block 보다 먼저 있어야
하는데, Neo 의 입력은 Block 의 출력이다. forward 안 **2-pass** 로 해소했다 — pass 0 은 패치 임베딩에서
Neo 를 부트스트랩, pass 1 은 pass 0 의 Block 출력에서 Neo 를 재구성. **학습과 추론이 동일한 패스 수**를
돌므로 train/test 프로토콜 불일치가 없다 (에폭 간 캐시 방식은 테스트 샘플에 캐시가 없어 이 불일치를
피할 수 없다). 2 패스 모두 query 는 `x_emb` 로 고정해 **모델 깊이를 baseline 과 같게** 유지했고,
패스 사이에 Block·readout 의 LIF 막전위를 리셋해 "같은 함수의 재적용" 을 보장했다.

**사전 검증:** refine 0→1 출력 최대차 0.168 (2-pass 가 실제로 계산을 바꿈).
task 손실이 도달하는 파라미터 = `reservoir.proj.{linear.weight, bn.weight, bn.bias}`.
**이 세션에서 예측 손실이 readout 을 학습한 최초의 조건이다** (조건 B 에서는 `recall_alpha` 하나뿐이었다).

### 8.16.2 결과 — 처음으로 임계를 넘었고, 방향이 반대다

| 비교 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| `x0` vs `h0` (Neo 가 값을 하는가) | +2.66% | +0.97% | +1.82% | +0.68% | **+1.53%** | 6/48 |
| `g0` vs `h0` | +0.83% | +0.32% | +2.08% | +0.47% | **+0.92%** | 9/48 |
| `x0` vs `g0` (cross-attn 인과, 파라미터 매칭) | +1.82% | +0.65% | −0.25% | +0.21% | **+0.61%** | 17/48 |
| `h0` vs `cl0` (aux OFF 효과) | −0.56% | −0.00% | −0.04% | +0.03% | **−0.14%** | 26/48 |

**용량-반응이 단조롭다.** Neo 의 예측 영향력이 커질수록 예보가 나빠진다:

| Neo 의 영향력 | vs `h0` | 악화 |
|---|---:|---:|
| 없음 (`h0`) | +0.00% | 0/48 |
| α 게이트 (`g0`) | **+0.92%** | 39/48 |
| cross-attention (`x0`) | **+1.53%** | 42/48 |

`x0` 는 nuisance 임계(1.4%)를 넘는다. **이 세션에서 임계를 넘긴 유일한 결과이고, 부호가 해로운 쪽이다.**
horizon 의존도 단조롭다 (`x0`: p96 +1.59% → p720 +2.62%).

### 8.16.3 세 가지 결론

**1. auxiliary loss 는 성능에 기여하지 않는다.** `h0` vs `cl0` = **−0.14%, 개선 26/48**. 복원 손실을
완전히 끄고 Neo 읽기까지 없앤 순수 Hippo 가 현행 조건 B 와 구분되지 않는다 (오히려 근소 우위).
논문의 auxiliary 서사를 지탱하는 **성능 근거가 없다.**

**2. 예측 손실을 Neo readout 에 연결해도 개선되지 않는다 — 해롭다.** §8.15.8 에서 "아직 시험하지 않은
유일한 축" 으로 지목했던 것이 이것이다. 고정 코어 + 과제로 학습되는 readout 이라는 정통 RC 구성을
만들었고, 결과는 `g0` +0.92%(39/48), `x0` +1.53%(42/48) 다. **가설은 기각된다.**

**3. 통로를 넓힐수록 나빠진다.** cross-attention 은 α 게이트보다 **+0.61% 더 나쁘다**. Neo 가 Block 안에서
key/value 로 직접 읽히는, 대역폭이 가장 큰 배선이 가장 해롭다.

### 8.16.4 학습된 α — 모델이 Neo 를 능동적으로 채택하고 나빠졌다

| | 평균 α | 최소 | Neo 최대 지분 |
|---|---:|---:|---:|
| `cl0` (grad 차단) | 0.9508 | 0.8899 | 11.0% |
| **`g0` (grad 개방)** | **0.8607** | **0.4837** | **51.6%** |
| `x0` / `h0` | 0.9820 | 0.9820 | (게이트 미사용, 초기값 유지) |

`g0` 에서 α 는 **예측 손실이 직접 학습한다.** 그런데 예측 손실은 α 를 1 로 밀어 Neo 를 꺼버리는 대신
**0.48 까지 내려 Neo 지분을 51.6% 로 키웠고, 그 결과가 +0.92% 악화다.** 즉 Neo 는 무시당한 것이 아니라
**채택되었고 손해를 냈다.** 이는 단순 무효보다 강한 진술이며, §8.14.6 의 "α ≠ 유용성" 을 가장 극적으로
보여준다 — 이번에는 α 를 움직인 주체가 예측 손실 자신이다.

(`x0` 의 α 가 정확히 sigmoid(4)=0.9820 에 머문 것은 cross-attention 모드가 α 게이트를 우회한다는
배선 확인이기도 하다.)

### 8.16.5 조건 B 계열의 종합

열 번째 개입이고, 처음으로 **명확히 해로운** 결과다. 앞의 아홉은 전부 무효였다.

지금까지 바꿔본 것: 과제(§8.14), 그래디언트 경계(§8.13), 코어(§8.15), 융합 배선과 손실 구성(§8.16).
남은 것은 Neo 를 쓰지 않는 것뿐이다 — 그리고 `h0` 가 바로 그것이며 전 그리드에서 가장 좋다.

---

## 8.17 ★ Neo 는 무엇을 배우는가 — 학습된 `x0` 체크포인트 진단 (2026-08-12)

재학습 없이 `x0` (cross-attn + aux OFF + grad 개방 + 고정 리저버) 체크포인트에서 측정.
`diag_neo.py`. ETTh1_p96 / ETTm1_p96, seed 7, train 적합 · test 평가.

### 8.17.1 인과 대조 — Neo 는 **쓰인다**

| x_neo 치환 | ETTh1_p96 | ETTm1_p96 |
|---|---:|---:|
| 원본 | 0.381835 | 0.321316 |
| **0 으로 대체** | 0.432410 (**+13.25%**) | 0.362730 (**+12.89%**) |
| **배치 셔플** (다른 샘플의 Neo) | 0.417786 (+9.42%) | 0.351724 (+9.46%) |
| 패치축 셔플 | 0.381835 (+0.00%) | 0.321316 (+0.00%) |

Neo 를 지우면 13% 가 무너진다. **모델은 Neo 에 확실히 의존한다.**
배치 셔플이 이득의 (13.25−9.42)/13.25 = **29%** 만 회복하므로, 나머지 71% 는 **해당 샘플의**
Neo 여야 한다 — Neo 는 샘플별 정보를 실제로 나르고 있다.

**패치축 셔플 0.00% 는 발견이 아니라 구조적 항등이다.** 이 어텐션은 softmax 가 없는
`(q@kᵀ)@v` 이고 key 인덱스에 대해 합산하며 key 에 위치 부호화가 없다. 따라서 Neo 의 패치 순서를
섞는 것은 **정확히 항등 연산**이다. 이는 Neo 에 대한 증거가 아니라 **배선에 대한 발견**이다 —
cross-attention 은 Neo 의 시간 순서를 원리적으로 버린다.

### 8.17.2 표현 통계 — 희소하고, 시간 구조가 없고, 백색이다

| | ETTh1_p96 | ETTm1_p96 |
|---|---:|---:|
| 발화율 | **0.0221** | **0.0161** |
| 패치축 autocorrelation | −0.0227 | −0.0165 |
| DCT 저주파 비중 (keep=0.25) | **0.2415** | **0.2401** |
| participation ratio | 257/1536 | 256/1536 |

- liquid 자체는 0.22 로 발화하는데 **과제로 학습된 readout 이 2% 로 압축**했다.
- autocorrelation ≈ 0, DCT 저주파 비중 ≈ 0.24 는 균등 스펙트럼(백색)의 값 0.25 와 사실상 같다.
  **저주파 추세가 전혀 아니다.** aux 손실을 끄면 Neo 는 느린 성분을 부호화하지 않는다 —
  원래 설계가 상정한 "Neocortex = 느린 추세" 는 aux 손실이 강제하던 것이지 자발적 성질이 아니다.

### 8.17.3 선형 probe — 모든 타깃에서 Hippo 보다 못하다

ridge, λ 그리드 최적 (고정 λ 는 열이 2배인 concat 을 넓다는 이유만으로 불리하게 만든다).

| probe | ETTh1_p96 | ETTm1_p96 |
|---|---:|---:|
| H̄ → 미래 y | **+0.2435** | −0.0394 |
| Neo → 미래 y | +0.0503 | −0.0791 |
| **[H̄, Neo] → 미래 y** | **+0.2374** | **−0.0963** |
| Neo → 원 패치 | +0.2362 | +0.6367 |
| H̄ → 원 패치 | **+0.9454** | **+0.9464** |
| Neo → H̄ | +0.1547 | +0.3999 |

(ETTm1 의 음수 R² 는 train 적합 · test 평가에서의 분포 이동 탓이다. 절대값이 아니라 **순서**만 읽는다.)

**세 가지가 확정된다.**
1. **Neo 를 Hippo 에 이어붙여도 예보가 나아지지 않는다** (0.2374 ≤ 0.2435, −0.0963 ≤ −0.0394).
   §8.15.8 에서 제기한 "과제로 학습된 readout 이 Neo 에서 뭔가 뽑아낼 수 있는가" 의 답은 **아니오**다.
2. **Neo 는 자기 본래 과제(윈도우 복원)조차 Hippo 보다 훨씬 못한다** (0.24 vs 0.95, 0.64 vs 0.95).
3. Neo → H̄ 가 0.15~0.40 이므로 Neo 는 Hippo 의 **손실 많은 부분 함수**다. 당연하다 — 입력이 Hippo 다.

### 8.17.4 종합 — "쓰인다" 와 "쓸모 있다" 는 다르다

| | |
|---|---|
| Neo 를 지우면 | **+13% 악화** (강하게 의존) |
| Neo 를 아예 안 쓰고 학습하면 (`h0`) | **−1.53% 개선** (§8.16.2) |

모순이 아니다. **모델이 Neo 에 공적응(co-adapt)한 것이다.** Neo 가 있는 채로 학습하면 기능의 상당
부분이 그 경로를 지나가도록 배선되고, 추론 시 끊으면 무너진다. 그러나 처음부터 없이 학습한 해가
더 낫다. Neo 는 정보원이 아니라 **모델이 통과하도록 강제된 희소 병목**으로 기능한다.

이 세션에서 "겉보기 의존성 ≠ 유용성" 의 **세 번째** 사례다 — 게이트 크기(§45), 학습된 α(§8.14.6),
그리고 이제 ablation 비용. 인과 ablation 조차 단독으로는 유용성의 증거가 아니며, **처음부터 없이
학습한 대조군(`h0`)이 있어야만** 판정할 수 있다.

---

## 8.18 ★★ 설계 의도 대 실제 — 리저버는 옳았고, readout 이 지우고 있었다 (2026-08-12)

### 8.18.1 의도

사용자가 리저버를 고른 이유: 창 간 저장이 아니라, **현재 윈도우 안에서 Neo 가 더 긴 시정수로
주기·추세 같은 평균적 성분을 담고, Hippo 는 그 위의 특이점·고주파를 담는 주파수 역할 분담.**
누설 재귀 동역학이 자연히 긴 시정수를 갖기 때문이다.

### 8.18.2 측정 — 액체는 의도대로다. readout 이 백색으로 만든다

`x0` 학습 체크포인트, 패치축 DCT 저주파 비중(keep=0.25, **백색 = 0.25**)과 1-lag autocorrelation.

| 신호 | ETTh1_p96 | | ETTm1_p96 | |
|---|---:|---:|---:|---:|
| | 저주파 | autocorr | 저주파 | autocorr |
| Neo 입력 (= pooled Hippo) | 0.3830 | 0.1241 | 0.6754 | 0.5354 |
| **리저버 액체 상태 φ** | **0.9837** | **0.9798** | **0.9908** | **0.9819** |
| Neo 출력 (readout 후) | 0.2408 | −0.0229 | 0.2397 | −0.0173 |
| Hippo 최종 표현 | 0.3830 | 0.1241 | 0.6754 | 0.5354 |

**리저버 내부는 정확히 의도대로 동작한다.** φ 는 DCT 에너지의 98~99% 가 최저 1/4 대역에 있고
autocorrelation 이 0.98 이다 — 극도로 느리고 매끄러운 추세다. **리저버 선택은 옳았다.**

readout 단계별로 좁히면 원인이 한 층에 특정된다.

| readout 단계 | ETTh1 저주파 | autocorr | ETTm1 저주파 | autocorr |
|---|---:|---:|---:|---:|
| 액체 φ (readout 입력) | 0.9837 | 0.9798 | 0.9908 | 0.9819 |
| + Linear(1024→64) | 0.9894 | 0.9884 | 0.9876 | 0.9784 |
| + BatchNorm | 0.9684 | 0.9620 | 0.9803 | 0.9672 |
| **+ LIF = x_neo** | **0.2408** | **−0.0229** | **0.2397** | **−0.0173** |

Linear 와 BN 은 추세를 온전히 보존한다. **마지막 LIF 한 층이 0.97 을 0.24 로 무너뜨린다.**

**메커니즘.** 이 readout LIF 의 시간축은 **패치축 자체**다. 과제 손실이 수렴시킨 발화율이 2.2% 이고
패치축 길이가 24 이므로, 각 유닛은 윈도우당 **0~1 개의 스파이크**를 낸다. rate code 가 아니라
"임계를 넘었는가" 라는 사실상 이진 코드이고, 매끄러운 파형이 임펄스가 된다. **임펄스의 스펙트럼은
원리적으로 평평하다.** 24 스텝에 0.5 스파이크로는 시간 프로파일을 실을 수 없다.

### 8.18.3 역할이 뒤집혀 있었다

| | 저주파 비중 (ETTm1) | autocorr |
|---|---:|---:|
| Hippo 최종 표현 | **0.6754** | **0.5354** |
| Neo 출력 | 0.2397 | −0.0173 |

의도는 Neo=추세 / Hippo=고주파였는데, 실제로는 **Hippo 가 느린 성분을 들고 있고 Neo 출력이 희소한
이벤트 코드**다. `x0` 가 `h0` 보다 +1.53% 나쁜 것(§8.16.2)과 정합한다 — Hippo 가 이미 가진 추세 위에
정보가 대부분 소실된 이벤트 코드를 섞고 있었다.

이것은 reservoir_summary §2.3·§7.4 가 미리 경고한 실패 모드다: *"binary spike 만 readout 에 쓰면
sparse 한 정보가 손실될 수 있다"*. 리저버 **내부** 특징 φ 에는 막전위와 trace 를 모두 넣었지만
(그래서 φ 가 0.98), **출력에서 다시 이진화**했다 — α 게이트/cross-attention 의 스파이크 계약과
AC 에너지 회계를 유지하려는 선택이었고, 그것이 원인이었다.

### 8.18.4 수정 — `--res_readout analog`

readout 의 LIF 를 제거하고 Linear+BN 만 남긴다 (`AnalogReadout`). LIF 는 파라미터가 없으므로
**두 arm 은 파라미터가 정확히 같다** (239,617). 초기화 시점 검증:

| readout | params | φ 저주파 | x_neo 저주파 | x_neo autocorr | 이진? |
|---|---:|---:|---:|---:|---|
| spike | 239,617 | 0.9756 | 0.2493 | −0.0098 | True |
| **analog** | 239,617 | 0.9751 | **0.9434** | **+0.9233** | False |

**MCA 내부 k/v LIF 는 재이진화하지 않는다.** 그 LIF 의 시간축은 T 인데 kv 는 T 에 대해 상수
(`repeat(T,...)`)이므로, 각 (패치, 차원)이 독립적으로 rate code 가 되어 **패치 프로파일이 살아남는다**:

| readout | x_neo 저주파 | MCA k (T평균) | MCA v (T평균) |
|---|---:|---:|---:|
| spike | 0.2513 | 0.3474 | 0.3441 |
| **analog** | **0.9424** | **0.8629** | **0.8619** |

즉 어텐션이 실제로 읽는 key/value 까지 추세가 도달한다.

**에너지 회계 주의:** Neo 출력이 이진이 아니므로 MCA 의 k/v Linear 는 AC 가 아니라 MAC 이다.
Hippo 경로는 전부 스파이킹 그대로다. reservoir_summary §10.3 이 허용하되 **AC-only 에너지 주장을
하지 말라**고 명시한 경우에 해당하므로, 이 arm 의 에너지는 별도 재측정이 필요하다.

실행: `x0an` 48 런 (analog), 기준은 기존 `x0`(spike, 파라미터 동일)과 `h0`(Neo 없음).

---

## 8.19 ★ Max-Former 고주파 믹서 + 직렬 리저버 (2026-08-12 완료, 144 runs, SKIP 0)

### 8.19.1 원문 확인과 세 가지 정정

**Max-Former** ([arXiv 2505.18608](https://arxiv.org/abs/2505.18608), NeurIPS 2025) 의 실제 구현:

```python
class Max_Mixer(nn.Module):                       # self-attention 을 대체
    self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)   # 파라미터 0개, 크기 보존
class Block_Max(nn.Module):
    def forward(x): return self.mlp(self.mixer(x))
```

1. **토큰(공간)축**을 stride 1 로 훑는 슬라이딩 max 이고 다운샘플이 아니다. 믹서 자체는 파라미터가 없다.
2. **저장소의 `SpikLinearMaxLayer` 는 축이 틀렸다.** `MaxPool1d(2, stride=2)` 를 **특징축 D(64→32)**
   에 적용한다. `HighFreqAmp`(v1)은 이를 3번 쌓아 D 를 64→8 로 줄인다. 시간축과 무관한 채널 병목이라
   주파수 해석이 성립하지 않는다.
3. **논문은 막전위 도메인, 우리는 스파이크 도메인이다.** 원문 `S_MLP` 는 LIF 가 맨 앞이고 잔차가
   덧셈이라 max 가 **연속값**에 걸린다. 우리 블록 입력은 이진 스파이크이고, **이진값의 max 는 OR
   (형태학적 팽창)이라 평활화**된다 — 논문 의도와 반대다. 그래서 `Linear → BN → MaxPool(N축, k=3,
   s=1) → LIF` 순서로 두어 max 가 BN 직후 연속값에 걸리게 했다 (원문 `Max_Embed` 순서와 일치).

**부수 발견: 모델에 패치 순서 정보가 전혀 없다.** `pe=False`, 시간근접 마스크는 layers.py:666 에서
주석 처리, `use_pe`/`use_tn_pos` 모두 off. 어텐션이 패치축에 대해 **완전히 순열 불변**이며, §8.17 의
patch-shuffle 개입이 정확히 0.00% 였던 이유가 이것이다. 슬라이딩 max 와 리저버 스캔은 이 모델에
처음 들어가는 순서 인식 연산이다.

### 8.19.2 결과 — 세 요인이 깨끗하게 분리된다

| 대조 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| `sA` vs `h0` — **T-접기** | +1.05% | +0.88% | +2.03% | +0.93% | **+1.22%** | 8/48 |
| `sB` vs `sA` — **고주파 믹서** | +2.11% | +0.01% | −0.07% | +0.55% | **+0.65%** | 16/48 |
| `sC` vs `sB` — **직렬 리저버** | +1.32% | −0.10% | +0.80% | +0.25% | **+0.57%** | 22/48 |
| `sC` vs `h0` — 누적 | +4.50% | +0.77% | +2.76% | +1.74% | **+2.44%** | 2/48 |

**손해의 최대 성분은 T-접기(+1.22%)다.** 대조군 `sA` 가 없었다면 누적 +2.44% 를 "고주파 믹서나
리저버 탓" 으로 잘못 귀속했을 것이다. 현행 구조는 T 스텝마다 head 를 적용하고 손실이 24 스텝을 모두
감독하는데(앙상블/정규화 효과), head 앞에서 T 를 접으면 그것이 사라진다.

**고주파 믹서는 어텐션보다 나쁘다 (+0.65%, 16/48).** 파라미터가 8,448 개 적다는 이점은 있으나
성능은 하락한다. ETTm1 −0.07% / ETTh2 +0.01% 는 중립이지만 ETTh1 +2.11% 가 전체를 끌어올린다.

**직렬 리저버도 해롭다 (+0.57%, 22/48).**

### 8.19.3 ⚠️ γ 의 크기는 α 와 비교하면 안 된다

학습된 `tanh(γ)`: 평균 −0.0073, 범위 [−0.066, +0.066], **48런 전부 |γ| < 0.07**.
값만 보면 "모델이 무료 덧셈 경로를 거절했다" 로 읽힌다. **그러나 실제 기여 크기를 재면 다르다:**

```
ETTh1_p96 seed7:  tanh(gamma) = +0.02789
                  || tanh(g)*x_neo || / || hbar || = 0.16135
```

**γ 가 0.028 인데 상대 기여는 16% 다.** `hbar` 는 이진 스파이크의 T-평균이라 값이 [0,1] 이고 평균
0.2 수준인 반면, analog readout 의 BN 출력은 대략 단위분산이기 때문이다. 즉 **모델은 리저버를 거절한
것이 아니라 16% 크기로 쓰고 있고, 그 대가가 +0.57% 다.**

게이트 파라미터의 raw 값으로 "얼마나 쓰는가" 를 읽으면 안 된다는 것이 §8.14.6(α), §8.16.4(α),
§8.17(ablation 비용) 에 이은 **네 번째 사례**다. 두 경로의 스케일이 다르면 게이트 값은 비교 가능한
양이 아니며, **‖기여‖/‖기준‖ 을 직접 재야 한다.**

### 8.19.4 종합

세 개입 모두 해롭고, `h0`(원래 어텐션 + Neo 없음 + T-접기 없음)가 여전히 최선이다.
설계 의도의 두 축 — Hippo 를 고주파로 특화, Neo 가 추세를 담당 — 을 아키텍처 수준에서 구현했으나
성능으로 이어지지 않았다.

---

## 8.20 ★★ 두 번째 경로에 여유가 있는가 — 선형 probe 상한 (2026-08-12)

변형 (1)(Neo 가 원 시계열을 자체 임베딩으로 받음)에 48런을 쓰기 전에, 값싼 형태로 먼저 물었다.
학습된 `h0` 체크포인트, 재학습 없음, `probe_headroom.py`. p96, seed 7, 4 데이터셋.

**핵심 논리:** 모든 Neocortex 후보 특징은 현재 윈도우의 함수다. 따라서 **윈도우 자체가 Hippo 표현에
더해주는 것이 없다면, 그 어떤 특징도 선형 readout 에 더 보탤 수 없다.** `[H, R]` 이 그 상한이다.

### 8.20.1 방법론 주의 — train 적합/test 평가는 이 질문에 부적합하다

처음에 train 에 ridge 를 적합하고 test 로 평가했더니 ETTh2/ETTm1/ETTm2 에서 R² 가 0 이하였다.
분포 이동 때문에 **선형 사상이 전이되지 않는** 것이고, 그러면 델타를 해석할 수 없다. 우리가 묻는 것은
"정보가 있는가" 이지 "전이되는가" 가 아니므로, **test 분할 내부 2-fold CV** 로 다시 쟀다.

### 8.20.2 결과 (test 내부 2-fold CV)

| probe | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
|---|---:|---:|---:|---:|
| H → y (Hippo 표현, 1536차원) | **+0.2568** | **+0.0076** | **+0.0563** | **−0.0007** |
| R → y (원 윈도우, 192차원) | +0.2468 | +0.0035 | +0.0565 | +0.0009 |
| **[H, R] → y ★ 상한** | +0.2556 | +0.0075 | +0.0662 | +0.0004 |
| **Δ [H,R] − H** | **−0.0012** | **−0.0000** | **+0.0098** | **+0.0011** |
| G → y (리저버가 본 원 신호) | +0.2187 | −0.0005 | −0.0126 | −0.0005 |
| Δ [H,G] − H | −0.0213 | −0.0073 | −0.0470 | +0.0002 |

**전체 원 윈도우를 Hippo 표현에 이어붙여도 얻는 것이 사실상 없다** (Δ = −0.0012 / −0.0000 /
+0.0098 / +0.0011). ETTm1 만 +0.0098 (기준 0.0563 대비 17% 상대)로 유일하게 여지가 보인다.

**리저버가 본 원 신호(G)는 Hippo 보다 정보가 적다.** 네 데이터셋 모두에서 `G → y` < `H → y` 다.
저역통과가 예보에 필요한 성분을 버린다는 직접 증거다.
(단 `[H,G]` 의 음수는 일부 방법론 산물이다 — λ 를 블록별이 아니라 공유해서 스윕했으므로 1536 열을
덧붙이면 희석 페널티가 붙는다. `R` 은 192 차원이라 이 문제가 작고, 그래서 **`[H,R]` 이 신뢰할 수 있는
수치**다.)

### 8.20.3 부수 발견 — 학습된 표현이 원 윈도우보다 낫지 않다

`R → y` 와 `H → y` 가 사실상 같다: ETTh1 0.2468 vs 0.2568, ETTm1 0.0565 vs 0.0563,
ETTm2 +0.0009 vs −0.0007. **스파이킹 트랜스포머 전체가 만들어낸 1536 차원 표현이, 선형 예보 정보량
기준으로 192 개 원 값과 구별되지 않는다.** 이 실험 계열의 범위를 넘는 관측이지만 기록해 둔다.

### 8.20.4 판단

**(1) 에 48런을 쓰는 것을 권하지 않는다.** 상한이 0 근처이고, 리저버 특징은 특히 Hippo 보다 정보가
적다. 다만 두 가지 한계를 명시한다.

1. **선형 상한이다.** cross-attention 으로 들어가는 Neocortex 는 비선형으로 작용하므로 이것은
   증거이지 증명이 아니다.
2. **학습된 `h0` 를 기준으로 한다.** 공동 학습된 모델의 Hippo 표현은 다를 수 있다.

ETTm1 이 유일하게 여지를 보이므로, 그래도 진행한다면 **ETTm1 만 12런**으로 축소해 먼저 확인하는 것이
비용 대비 합리적이다.

---

## 8.21 ★★ 세션 종합 (2026-08-14)

이 서버에서 실행한 forecasting 런: `results_tune` 1,037 + `neorecall_v1/forecasting/results` 266. 모든 비교는 시드
페어링(7/13/21) 전 그리드(4 데이터셋 × 4 horizon) 48런 단위. nuisance 임계 1.4%.

### 8.21.1 12개 변형, h0 대비 (양수가 나쁨)

`h0` = 원래 어텐션 믹서 + Neo 읽지 않음 + aux 손실 없음.

| 변형 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| 조건 B 기준선 (aux ON, α게이트) | +0.59 | +0.01 | +0.05 | −0.02 | **+0.16** | 22/48 |
| + 마스킹 aux | −0.03 | +0.19 | +0.74 | +0.01 | **+0.23** | 23/48 |
| + next-token aux | +1.05 | +0.03 | +0.13 | −0.02 | **+0.30** | 17/48 |
| + 리저버 코어 | +1.12 | −0.08 | +0.42 | +0.05 | **+0.38** | 15/48 |
| 리저버 = 믹서 (spike, IAND) | +1.31 | +0.43 | +0.22 | +0.11 | **+0.52** | 20/48 |
| aux OFF + grad 개방 + α게이트 | +0.83 | +0.32 | +2.08 | +0.47 | **+0.92** | 9/48 |
| 리저버 = 믹서 (analog, 덧셈) | +3.52 | +1.05 | **−0.40** | **−0.07** | **+1.02** | 21/48 |
| 단일경로 T-접기 | +1.05 | +0.88 | +2.03 | +0.93 | **+1.22** | 8/48 |
| + cross-attn (spike readout) | +2.66 | +0.97 | +1.82 | +0.68 | **+1.53** | 6/48 |
| + cross-attn (analog readout) | +2.11 | +1.77 | +2.44 | +0.84 | **+1.79** | 3/48 |
| 단일경로 + Max 믹서 | +3.18 | +0.88 | +1.95 | +1.49 | **+1.87** | 5/48 |
| 단일경로 + Max 믹서 + 직렬 리저버 | +4.50 | +0.77 | +2.76 | +1.74 | **+2.44** | 2/48 |

**열두 변형 모두 `h0` 보다 나쁘다.** 유일하게 일부라도 개선인 것은 `리저버=믹서(analog)` 의 ETTm 계열
(−0.40 / −0.07) 인데, 같은 arm 의 ETTh1 이 +3.52 라 사전에 정한 성공 기준("네 데이터셋 부호 일관")을
충족하지 못한다. **최선의 구성은 Neocortex 도 auxiliary loss 도 없는 원래 Hippo 단독이다.**

### 8.21.2 무엇이 왜 안 되는지 — 확립된 메커니즘

1. **정보 천장 (§8.14.2, §8.17, §8.20).** Neo 의 입력이 Hippo 출력의 결정적 함수인 한 Neo 는 정보를
   더할 수 없고 계산만 더한다. 선형 probe 로 확인: `[H̄, Neo] → 미래 y` 가 `H̄` 단독을 넘지 못한다.
   그리고 §8.20 에서 **원 윈도우 전체**를 Hippo 에 이어붙여도 R² 증가가 −0.0012~+0.0098 이다 —
   윈도우의 어떤 함수도 선형 readout 에 보탤 것이 없다.
2. **readout 이 추세를 지웠다, 그러나 고쳐도 무관했다 (§8.18).** 리저버 액체는 DCT 저주파 0.98 로
   의도대로 추세를 담고 있었고, readout 의 LIF 가 그것을 0.24(백색) 로 만들고 있었다. analog readout
   으로 고쳐 어텐션이 읽는 k/v 까지 0.86 을 전달했으나 성능은 +1.79% (3/48) 로 더 나빠졌다.
3. **T-접기가 비싸다 (§8.19).** head 앞에서 스파이킹 축을 접는 것만으로 +1.22%. 대조군 `sA` 가
   없었다면 단일경로 실험의 +2.44% 를 믹서나 리저버 탓으로 잘못 귀속했을 것이다.
4. **Hippo 의 표현이 원 윈도우보다 낫지 않다 (§8.20.3).** `R → y` 와 `H̄ → y` 가 사실상 같다
   (ETTm1 0.0565 vs 0.0563). 1536 차원 학습 표현이 192 개 원 값과 선형 예보 정보량에서 구별되지 않는다.

### 8.21.3 반복해서 틀린 네 가지 대리 지표

이 세션에서 **"쓰는 것처럼 보인다" 가 "쓸모 있다" 를 뜻하지 않은 사례가 네 번** 나왔다.

| 대리 지표 | 무엇을 시사하는 듯했나 | 실제 |
|---|---|---|
| 학습된 α 가 내려감 (§8.14.6) | 모델이 Neo 를 채택 | 성능 무변화 |
| 예측 손실이 α 를 0.48 까지 내림 (§8.16.4) | 예측 손실이 Neo 를 원함 | +0.92% 악화 |
| x_neo 제거 시 +13% 악화 (§8.17) | Neo 가 필수적 | 처음부터 없이 학습하면 −1.53% 더 좋음 |
| γ ≈ 0.03 (§8.19.3) | 모델이 리저버를 거절 | 실제 기여 크기는 16% |

**게이트 값·ablation 비용은 단독으로 유용성의 증거가 아니다.** 판정에는 (a) 처음부터 없이 학습한
대조군, (b) `‖기여‖/‖기준‖` 직접 측정이 필요하다.

### 8.21.4 방법론 — 부분집합 신호는 다섯 번 증발했다

중간 점검에서 유망해 보인 신호가 전 그리드에서 사라진 사례: `to_neo` p720, `auxnext` ETTm1_p720,
`mask50` ETTm1, Max 믹서 "중립", `리저버=믹서` −0.34%. **모두 ETTm↔ETTh 방향 역전이 원인이었다.**
중간 수치는 진행 확인용이며 결론으로 인용하면 안 된다.

착수 전 건전성 점검도 두 번 실험을 구했다 — 리저버 입력 스케일 미교정 시 발화율 0.000 / 침묵 100%
(§8.15.4), raw 패치 입력에 대한 재교정 필요(§8.20.1). 교정 없이 돌렸으면 두 스윕 모두 공허했다.

### 8.21.5 코드 상태

`neorecall_v2/forecasting` 는 정리된 스냅샷이다. 이번 세션에서 제거한 죽은 코드/파라미터:
- `Block.time_recon` 4,224 + 조건 B 의 `encoding_neo` 640 (§8.15.5, bit-exact)
- 조건 B 의 버려지던 raw 패치 Neo 패스 (§8.14.4, −0.009%)
- `--no_aux`: 도달 불가한 Neo pathway 8,961 파라미터 (bit-exact 검증, 런당 시간 242s → 122s)

진단 도구: `diag_neo.py` (인과 ablation, 선형 probe, 표현 통계), `probe_headroom.py` (상한 probe).

---

## 8.22 ★★★ P2 — 병목은 Neo 가 아니라 96 스텝 윈도우였다 (2026-08-14)

창 간 상태(reservoir_summary §7~9)에 착수하기 전, 그 전제를 학습 없이 물었다:
**"96 스텝 밖에 예보 정보가 있는가?"** 여러 lookback 길이의 정규화된 원 윈도우에서 ridge 로 미래를
예측하고 test 내부 2-fold CV 로 채점 (§8.20.1 과 같은 방법론).

| lookback | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
|---:|---:|---:|---:|---:|
| **96** (현재 설정) | 0.2470 | 0.0036 | 0.0389 | 0.0001 |
| 192 | 0.2829 | 0.0947 | **0.4094** | 0.0041 |
| 336 | **0.3154** | 0.1830 | **0.4523** | 0.0095 |
| 720 | 0.3048 | **0.2099** | 0.4083 | **0.2854** |

**네 데이터셋 모두에서 96 스텝 밖의 정보가 안쪽보다 압도적으로 많다.** ETTm1 은 R² 가 11 배
(0.0389 → 0.4523), ETTh2 는 58 배 (0.0036 → 0.2099), ETTm2 는 0.0001 → 0.2854 다.

### 8.22.1 이것이 세션 전체를 재해석한다

§8.20 은 "현재 윈도우 안에서 Hippo 에 더 보탤 것이 없다" 를 보였고 그 판정은 옳았다. 그러나
**윈도우 자체가 병목이었다.** 12 개 변형(§8.21.1)은 전부 여유가 거의 없는 96 스텝 안에서 정보를
재배치하고 있었고, 그래서 전부 ±1~2% 안에서 움직였다. 그 사이 336 스텝 원 윈도우에 대한 **선형**
모델이 ETTm1 에서 R² 0.45 를 낸다.

**이는 reservoir_summary §7~9(창 간 상태)의 전제를 직접 검증한다.** §8.20 의 상한은 "현재 윈도우의
함수" 에 대한 것이므로 창 밖 정보를 나르는 상태에는 적용되지 않는다.

### 8.22.2 최적 lookback 은 무한이 아니다

ETTh1(336 에서 정점, 720 에서 하락)과 ETTm1(336 정점) 은 비단조다. ETTh2·ETTm2 는 720 까지 상승한다.
"길수록 좋다" 가 아니라 **데이터셋별 최적 horizon 이 있다** — reservoir_summary §2.4(Carroll: memory 는
길수록 좋은 것이 아니라 task timescale 에 맞아야 한다)와 일치한다.

### 8.22.3 주의

**선형 probe 이지 모델 성능이 아니다.** 긴 lookback 을 트랜스포머가 실제로 활용하기 어렵다는 것이
LTSF-Linear 계열의 핵심 관찰이다 — 많은 트랜스포머는 lookback 을 늘리면 오히려 나빠진다. 따라서
"정보가 있다" 는 "우리 모델이 쓸 수 있다" 를 뜻하지 않는다. 다만 **정보 천장이 데이터가 아니라
윈도우 설정의 산물이었다** 는 것은 확정된다.

## 8.23 리저버 = 믹서 최종 (96 runs, 정리 후 재실행)

정리(`--no_aux`)가 `rs`/`ra` 에서도 **bit-exact** 였다 (+0.000%, 48/48 각각). 최종 수치는 §8.21.1 과 동일:
`rs` +0.52% (20/48), `ra` +1.02% (21/48). `ra` 는 ETTm1 −0.40 / ETTm2 −0.07 로 ETTm 계열에서만
개선이고 ETTh1 +3.52 라 사전 성공 기준(네 데이터셋 부호 일관)을 충족하지 못한다.

---

## 8.24 ★★★ lookback 스윕 — 세션 최초의 성공 (2026-08-14, 48 runs, SKIP 0)

§8.22 의 probe 가 "96 스텝 밖에 정보가 있다" 를 보였다. 이 스윕은 **"우리 아키텍처가 그 정보를 쓸 수
있는가"** 를 답한다.

### 8.24.1 설계 — N 을 고정한 통제

seq_len 을 직접 늘리면 `T = N = num_patches` 규약 때문에 어텐션이 T·N² 으로 폭발한다. 실측:
seq_len=336 은 **bs=8 에서도 43.8GB**, bs=64 는 OOM. 배치를 8 배 줄이면 학습 동역학이 달라져 비교가
무의미해진다.

대신 **패치를 함께 키워 N 을 고정**했다 (PatchTST 방식). 결과적으로 거의 완벽한 통제가 된다:

| seq_len | patch | N=T | params | peak mem |
|---:|---:|---:|---:|---:|
| 96 | 8 | 24 | 173,440 | 3.41 GB |
| 192 | 16 | 24 | 173,952 | 3.42 GB |
| 336 | 28 | 24 | 174,720 | 3.43 GB |
| 720 | 60 | 24 | 176,768 | 3.46 GB |

파라미터 +1.9%, 메모리 동일, N·T 동일. **lookback 과 패치 해상도만 변한다.**
`L96` arm 이 기존 `h0` 를 **bit-exact 로 재현**하여 배선을 검증했다.

### 8.24.2 결과 — 사전 성공 기준을 충족한다

L96 대비 (음수가 개선):

| lookback | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | +1.97 | +0.58 | −3.42 | −2.95 | **−0.95** | 7/12 |
| 336 | +1.49 | −0.24 | **−9.12** | −5.30 | **−3.29** | 8/12 |
| **720** | **−1.21** | **−5.12** | −6.54 | **−6.44** | **−4.83** | **12/12** |

**L=720 은 네 데이터셋 모두에서 개선이고 12/12 런 전부 개선이다.** 사전에 정한 성공 기준
("네 데이터셋 부호 일관 + 전체 평균 음수")을 **처음으로 충족한다.** −4.83% 는 nuisance 임계(1.4%)의
3.5 배이고, 이 세션 12 개 변형이 전부 ±2% 안에 갇혀 있던 것과 자릿수가 다르다.

데이터셋별 최적 L 을 고르면 **−5.47%** (ETTh1 720, ETTh2 720, ETTm1 336, ETTm2 720).

원 MSE (시드 평균):

| | L96 | L192 | L336 | L720 |
|---|---:|---:|---:|---:|
| ETTh1 | 0.3699 | 0.3772 | 0.3754 | **0.3654** |
| ETTh2 | 0.2868 | 0.2884 | 0.2861 | **0.2721** |
| ETTm1 | 0.3125 | 0.3017 | **0.2840** | 0.2920 |
| ETTm2 | 0.1739 | 0.1688 | 0.1647 | **0.1627** |

### 8.24.3 probe 의 사전 예측 대조 — 3/4 적중

§8.22 의 probe 는 데이터셋별 최적 L 을 **성능 측정 전에** 예측했다.

| | probe 예측 | 실제 최적 | |
|---|---|---|---|
| ETTm1 | 336 정점, 720 하락 | **336** (−9.12), 720 −6.54 | ✓ 정확 |
| ETTh2 | 720 까지 상승 | **720** (−5.12) | ✓ |
| ETTm2 | 720 필요 | **720** (−6.44) | ✓ (다만 336 도 −5.30 으로 이미 크다) |
| ETTh1 | 336 정점 | **720** (−1.21), 336 은 +1.49 | ✗ 빗나감 |

선형 probe 가 최적 lookback 을 3/4 맞혔다. ETTh1 은 반대로 갔다 — probe 상 336 이 정점(0.3154)인데
모델은 336 에서 오히려 나빠지고 720 에서만 개선된다. **선형 정보량과 모델의 활용 능력이 일치하지
않는 사례**이며, LTSF-Linear 계열이 지적한 "트랜스포머는 긴 lookback 을 잘 못 쓴다" 의 잔재로 보인다.

### 8.24.4 함의 — Neo 판정의 유효 범위

**이 세션 12 개 Neo 변형의 판정은 전부 seq_len=96 에서 나왔다.** 그리고 §8.20 은 그 조건에서
`[H̄, R] → y` 가 `H̄` 를 넘지 못함을 보였다 — **두 번째 경로가 나를 정보 자체가 없는 조건**이었다.

따라서 "Neo 는 쓸모없다" 는 **"96 스텝 윈도우 안에서는 쓸모없다"** 로만 유효하다. L=720 에서는
선형 정보량이 크게 늘고(§8.22) 실제 성능도 −4.83% 개선되므로, **Neo 를 최적 L 에서 다시 시험해야
한다.**

반론도 명시한다: 긴 lookback 에서는 **패치도 커진다**(8→60). 패치가 커지면 Embedding 이 더 넓은 시간
창을 한 토큰으로 보므로 **Hippo 가 이미 저주파를 더 담게 되어** Neo 의 여지가 오히려 줄 수 있다.
다음 단계에서 (a) L=720 에서 `[H̄, R]` 여유 재측정, (b) L 별 Hippo 표현의 DCT 저주파 비중으로 직접
확인한다.

## 8.25 L=1440 과 L=720 에서의 Neo 재시험

### 8.25.1 lookback 곡선은 720 에서 정점

| lookback | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | +1.97 | +0.58 | −3.42 | −2.95 | −0.95 | 7/12 |
| 336 | +1.49 | −0.24 | **−9.12** | −5.30 | −3.29 | 8/12 |
| **720** | **−1.21** | **−5.12** | −6.54 | −6.44 | **−4.83** | **12/12** |
| 1440 | −0.08 | +0.78 | −4.19 | **−7.25** | −2.69 | 8/12 |

1440 에서 되돌아온다 — ETTh2 가 −5.12% → +0.78% 로 뒤집히고 ETTh1 도 이득이 사라진다.
**L=720 만이 12/12 로 부호 일관성 기준을 충족한다.** 패치가 함께 커지므로(L=1440 이면 patch=120)
**lookback 을 늘리는 이득과 패치 해상도를 잃는 손해가 720 에서 교차**한다. 이 교차점을 옮기려면
`T = N` 결합을 끊어 패치를 키우지 않고 N 을 늘려야 한다 — 별도 후보로 남긴다.

### 8.25.2 L=720 에서의 여유 재측정 — 여전히 닫혀 있다

`h0`@L720 체크포인트에서 §8.20 을 반복 (test 내부 2-fold CV):

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
|---|---:|---:|---:|---:|
| H̄ → y (기준) | +0.3190 | +0.1852 | +0.4126 | +0.2866 |
| **[H̄, R] − H̄ (여유)** | **−0.0016** | **+0.0217** | **+0.0060** | **+0.0142** |
| G → y − H̄ (리저버) | −0.0769 | −0.0725 | −0.1114 | −0.0765 |

L=96 에서의 여유(−0.0012 / −0.0000 / +0.0098 / +0.0011)와 사실상 같다. ETTh2 만 +0.0217 로 조금 늘었다.

**그러나 Hippo 자체가 극적으로 좋아졌다:**

| H̄ → y | L=96 | L=720 |
|---|---:|---:|
| ETTh1 | 0.2568 | **0.3190** |
| ETTh2 | 0.0076 | **0.1852** (24배) |
| ETTm1 | 0.0563 | **0.4126** (7배) |
| ETTm2 | −0.0007 | **0.2866** |

**−4.83% 개선의 실체가 이것이다** — Hippo 가 긴 윈도우의 정보를 실제로 흡수했고, 그 결과 두 번째
경로의 여유는 다시 닫혔다. 리저버 특징은 L=720 에서 Hippo 와의 격차가 **더 벌어졌다**(−0.077~−0.111):
긴 윈도우일수록 저역통과가 버리는 것이 많다.

(앞서 세운 반론 "패치가 커지면 Hippo 가 추세를 흡수해 Neo 여지가 준다" 는 스펙트럼 측정으로
지지되지 않았다 — ETTm1 의 H̄ 저주파 비중이 0.6754(L96) → 0.4244(L720) 로 오히려 낮아졌다. 다만
DCT 축이 패치 24 개로 같아도 한 패치가 8→60 스텝이라 절대 시간 규모가 달라 직접 비교는 조심해야 한다.)

### 8.25.3 L=720 에서 Neo 재시험 — 여전히 무효

가장 유망했던 `p3n`(예측 수준 덧셈 추세, L96 에서 24/48) 을 L=720 에서 12 런:

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | **전체** | 개선 |
|---|---:|---:|---:|---:|---:|---:|
| p3n@L720 vs L720 | +0.36 | −0.41 | +0.63 | −0.28 | **+0.08** | 6/12 |

**6/12, 정확히 동전 던지기.** probe 가 예측한 대로다. ETTh2 가 −0.41% 로 유일한 음수인데, 공교롭게도
probe 에서 유일하게 여유(+0.0217)를 보인 데이터셋이다 — 다만 임계 아래이고 2/3 이다.

**따라서 "Neo 는 96 스텝 윈도우에서만 쓸모없다" 는 가능성은 기각된다.** L=720 에서 Hippo 가 훨씬
많은 정보를 갖게 된 뒤에도 두 번째 경로의 여유는 열리지 않았다.
