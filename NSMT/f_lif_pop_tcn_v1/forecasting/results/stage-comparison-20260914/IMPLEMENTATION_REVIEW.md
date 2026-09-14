# PopulationLIF: 제공된 리뷰와 실제 코드 대조

2026-09-14. 사용자 `NSMT/docs/PopulationLIF_implementation_review.md`와 원 concept 문서의 §7–11, §24.5 및 실제 두 PopulationLIF 구현을 대조했다. 사용자 리뷰/원문은 수정하지 않았다. 이 문서는 학습 실험이 아닌 구현·개념 검토다.

## 판정

사용자 리뷰의 핵심 지적에 동의한다. 현재 구현은 **population-based content-adaptive dense membrane-memory retrieval**이다. 입력공유/이질적 tau/population query/과거 post-reset value/K 공유 temporal weight/가산식 evidence/별도 gate/population spike/forward-local history는 구현되어 있다. 관련 없는 memory slot을 정확히 제외하는 content-dependent read mask는 없다. 1차/2차의 PopulationLIF 클래스는 AST가 동일하므로 두 실험에 모두 해당한다.

원문 §24.5는 soft/hard 선택을 미정으로 두므로 dense pilot 자체를 무조건 잘못된 구현이라고 하지는 않는다. 다만 §10의 read selection과 read strength 구분, 원문의 weighted subset 및 irrelevant-history exclusion이라는 더 강한 주장은 현재 실험으로 확인하지 못했다. 기존 결과를 sparse retrieval의 성공/실패로 설명하면 부정확하다.

## 1. 핵심 누락: 과거 memory slot의 선택·제외

두 task의 `layers.py:64–76`에서 모든 j<t history에 score.softmax를 적용하고 모두 weighted sum에 포함한다. `layers.py:91`의 F.pad는 retrieval이 끝난 뒤 diagnostic attention에 미래 위치의0을 채우는 동작이다. 과거의 무관한 slot을 마스킹하지 않는다. `memory_mode='recent'`는 최근slot만 읽는 고정 평가 개입이며 학습된 내용 기반 mask가 아니다.

CPU counterexample(seed7,T8,BC2,D3)에서 과거168개 weight가 모두 양수였고 최소값은6.967689e-5였다. 현재 temperature.25/정규화 cosine의 유한 score 범위에서는 softmax가 매우 작은 값으로 underflow해서0이 되는 것을 sparse selection으로 기대할 수도 없다.

Soft mask를 추가해도 값이 모두 양수라면 여전히 dense하다. 실제 sparse retrieval을 주장하려면 제외된 slot의 가중치가 정확히0인지 확인해야 한다.

## 2. 설계 제약: cosine relevance가 절대 크기를 버린다

`layers.py:34–35,67–69,87`: bias 없는 선형 Q/K 뒤에 L2 정규화를 적용한다. 양의 상수 c에 대해 정규화 epsilon에 걸리지 않는 비영 벡터라면 normalize(W(cu))=normalize(Wu)이므로 같은 방향의 상태는 크기가 달라도 key가 같다. 이는 Q/K를 학습해도 이 구조 내에서 유지되는 성질이다. Memory value 자체의 크기는 보존되므로 **점수에서 크기를 구분하지 못한다**는 뜻이지 전체 모델이 진폭 정보를 전부 잃는다는 뜻은 아니다.

실제 module의 초기 Q/K로 [1,2,3,4]와 그10배를 비교한 weight는 약[.5,.5]였다. 크기 불변성이 의도라면 유효한 설계 선택이다. 서로 다른 크기의 과거 막전위를 구분해야 하는 task에는 표현 제약이다. H720 악화의 원인이라고 입증한 것은 아니다.

Homogeneous 대조에서는 같은 beta/input/initial state/threshold/shared memory weight 때문에 K개 상태가 동일하게 유지된다. 위 cosine 구조에서는 같은 부호의 비영 memory들은 크기가 달라도 같은 점수를 받는다. 이 대조는 redundant population의 기준으로 유효하지만 같은 유효 표현 차원의 대조는 아니다. Heterogeneous 개선을 선택적 검색의 성공으로 곧바로 해석할 수 없다.

## 3. 설계 제약: memory-use gate는 검색 결과를 보지 않는다

`layers.py:78–79`의 gate는 sigmoid(Linear(u_bar))이며 memory vector/후보 score/선택된 개수 등의 함수가 아니다. 학습으로 기억 기여를 약하게 만들 수는 있다. 다만 동일 u_bar에 대해 memory bank만 바뀌면 gate 출력은 같아서 후보의 질 변화에 직접 대응할 수 없다. 초기 gate=.5이며 검색을 완전히 거부하는 null/empty-support 선택은 없다. Gate와 slot mask는 역할이 다르다.

초기 homogeneous 모델에 [.5,.4,-20.]를 넣으면 마지막 query와 두 과거 key의 cosine은 모두−1인데도 weight=[.5,.5], gate=.5, 각 구성원의 evidence=+.00325379였다. 이는 모두 낮은 score여도 정규화로 총weight1이 되는 구조를 보여준다. 반대 방향이 반드시 task에 무관하다는 주장이나 학습된 checkpoint의 실패율 측정이 아니다.

추가로 두 입력 history에서 현재 charge를 모두[.1,.1,.1,.1]로 맞추고 과거 state 부호를 바꾸면 gate는 둘 다.5이지만 evidence는 ±.00247737이었다. Same-query/different-bank의 gate 한계에 대한 작은 반례다.

## 4. 기존 검사와 실험 해석 범위

`f_lif_pop_v1/forecasting/check_model.py:28–35,39–44`는 미래 memory 접근 금지/미래 patch 교란 불변성, 나머지는 수식·gradient·상태 초기화 등을 검사했다. 관련 없는 과거를 제외하는지 또는 정답 memory를 선택하는지를 검사하지 않았다. 따라서 기존 passed 결과는 sparse/semantic selectivity를 보증하지 않는다.

H720에서 관찰한 검색 추가의 악화(.679887→.711945)는 현재 dense retrieval 변형의 결과다. Masked/sparse 원안이 실패했다는 증거도, mask를 넣으면 개선된다는 증거도 아니다. Fractional prior는 원문에서 선택 사항이므로 누락 버그가 아니다.

## 다음 구현에서 분리할 사항 (제안; not run)

- Dense v1을 그대로 기준으로 보존하고 별도 버전에서 mask를 추가한다. 과거 j<t라는 causal 조건과 무관한 과거를 제외하는 relevance mask를 별개로 둔다.
- 가중치를 정확히0으로 만드는 subset 방식과, 아무 후보도 사용하지 않는 경우 M=0을 정의한다. Top-k만으로는 무관한 후보가 k개 선택되는 문제를 해결하지 못한다.
- Mask 후 선택된 후보 내에서 정규화할지 명시한다. 단순히 기존 softmax에 mask를 곱하면 전체 memory strength도 바뀌므로 selection과 gain 효과가 섞인다. 전부 mask된 경우 NaN 없이 M=0이어야 한다.
- 같은 backbone/seed/뉴런/학습 budget에서 dense와 masked를 비교한다. 그 뒤 norm 정보가 있는 relevance나 retrieval-aware gate를 한 가지씩 분리한다.
- 검증: 제외 slot weight=0, 후보가 없으면 evidence=0, 모든 K가 동일 slot mask 공유, 미래 접근 금지, 제외된 stored value에 대한 **직접 read** 불변성, 알려진 relevant lag와 distractor를 가진 synthetic recall. Stored-state masking은 그 과거 raw input이 이후 recurrent state에 남긴 영향까지 삭제하지 않는다.
- Sparse use와 efficient search는 별개다. 모든 과거와 score를 계산한 뒤 top-k를 골라도 dense search 비용은 남는다.

## 실행·보존

CPU 진단만 수행했다. 환경 `/home/yschoi/.conda/envs/snn_recall/bin/python`, LD_LIBRARY_PATH 동환경/lib, torch1.12.0+cu113, seed7,threads2. 재학습/new checkpoint/GPU evaluation: not run. 수치/입력/재현 recipe/source hashes는 [implementation_review_probes.json](implementation_review_probes.json).

검토 중 H96 24개 완료/H7208개 완료·4개 running·12개 pending 상태를 확인했다. 학습 중 두 `layers.py`/나머지 실행 source/HEAD를 변경하지 않았다. 현 branch `exp/f-lif-pop-tcn-v1`, training snapshot `794a29693a97dc1246c97ccf42d37ff6fd0ac585`. 검토 결과는 canonical PROJECT_LOG에도 append하며 기존 완료 후처리가 이 비교 디렉터리/로그를 보존하도록 한다.
