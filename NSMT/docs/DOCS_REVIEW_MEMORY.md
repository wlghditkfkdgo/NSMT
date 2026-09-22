# 문서별 검토 메모 — 2026-09-21

이 문서는 `docs` 문서들의 내용을 다음 세션에서도 복원하기 위한 지속 기록이다. 원문을 대체하거나 과거 결론을 최신 설계로 소급 변경하지 않는다. 현재 구현 감사는 [ASSESMENT.md](ASSESMENT.md)에 있다.

검토 기준은 `exp/f-lif-pop-v3`, HEAD `df3a3407b9ae653b4b1031320c4e8240b4c943a0`이다. `NSMT/docs`의 하위 archive와 JSON까지 조사했고, `PROJECT_LOG.md` 링크가 가리키는 저장소 루트 문서 및 루트 docs의 환경 목록도 포함했다. Markdown은 본문·수식·결론·개정 이력을 검토했다. 긴 PROJECT_LOG의 반복 실행 표는 전체 행을 파싱하여 수치 범위와 v2 macro를 재계산했고, 이동 manifest는 모든 항목을 구조적으로 검사했다. 과거 수백 회 학습이나 모든 논문을 다시 실행·재현한 것은 아니다.

문서/코드/기존 결과의 최초 SHA256은 [감사 inventory](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/inventory.json)에 보존했다. `NSMT/docs/PROJECT_LOG.md`와 루트 `docs/PROJECT_LOG.md`는 같은 문서다.

## 1. 현재 결정을 읽는 순서

1. [Population_fLIF_v3_prereg_KO.md](Population_fLIF_v3_prereg_KO.md)의 날짜부 추가 결정 §2A·§3A·§9A·§2B를 먼저 읽는다. 이전 본문과 충돌하면 후속 결정이 우선한다.
2. [IDEA_LOG.md](IDEA_LOG.md)는 현재 모델을 이해하기 위한 통합 설명이다. 아래의 잔존 불일치는 감사 A09에 기록했다.
3. [PROJECT_LOG.md](PROJECT_LOG.md)는 실제 수행 여부·실패·정정·commit·artifact를 확인하는 유일한 누적 기록이다.
4. 초기 개념, EN 계획, KO 재검토, 비판적 검토, rev.0 archive는 변경 이유와 폐기된 대안을 설명한다. 초기 값을 실행 설정으로 복사하지 않는다.

현재 검증할 주장은 **“리셋하지 않는 이질적 fractional 가지의 과거 dynamics를 내용에 따라 재배분하고, 갱신된 가지를 공유 LIF 소마가 읽을 때, 선택·fractional 차수·이질성 각각이 회상과 예측에 어떤 기여를 하는가”**이다. 이 세 기여는 별도로 입증해야 한다.

## 2. population_selective_membrane_memory_snn_concept.md

[원문](population_selective_membrane_memory_snn_concept.md)

출발점은 입력 Gaussian population coding이 아니라 **뉴런 내부의 시간 상태를 K개 구성원으로 표현**하는 것이다. 동일 입력을 서로 다른 시간상수로 처리하여 현재·과거 막전위 벡터를 만들고, 현재 문맥으로 과거 내부 상태를 검색한다. 원문이 우선한 Branch B는 직접 막전위 기억을 현재 충전에 더하는 구조다. Fractional prior, soft/hard selection, reset 전후 저장, K개 출력 대 단일 출력은 당시 미결이었다.

핵심 구별은 “어느 과거를 읽는가”와 “검색 결과를 얼마나 쓰는가”, 그리고 원시 사건과 누적된 내부 상태의 차이다. 슬롯을 제외해도 그 사건의 영향이 이후 상태에 남을 수 있다. 모든 과거를 scoring한 뒤 sparse weight를 만드는 것만으로 검색 비용 절감을 주장할 수 없다. 실제 시계열 또는 patch 순서를 시간축으로 삼고 같은 window의 과거만 읽어야 한다.

현재에 남는 요구는 shared input, heterogeneity, content-dependent selection, causal memory, 통제된 recall 검증이다. v3의 `f_j` 재적분과 공유 소마는 이 원문의 모든 미결 선택을 그대로 재현한 것이 아니라 후속 결정으로 바뀐 설계다.

## 3. PopulationLIF_implementation_review.md

[원문](PopulationLIF_implementation_review.md)

v1의 PopulationLIF와 원 개념을 대조한 구현 리뷰다. 동일 입력, 서로 다른 tau, 과거 post-reset 막전위, 현재 query, 공유 시간 가중치, 가산적 memory와 사용 gate는 구현되었다. 반면 softmax는 모든 과거에 양의 가중치를 주므로 **명시적 read mask나 완전 제외는 없다**. 미래 padding mask, 고정 recent 개입, 학습된 의미 선택은 서로 다른 기능이다.

따라서 v1을 content-adaptive dense membrane retrieval로 부르는 것은 타당하지만 선택적 배제까지 검증했다고 부를 수 없다. 원 개념이 soft/hard를 미정으로 남겼으므로 dense pilot 전체를 무조건 오구현으로 취급하지 않는 균형도 필요하다. Fractional prior의 부재는 당시 필수 기능 누락이 아니었다.

현재 감사에 적용할 교훈은 “가중치가 달라진다”, “정확한 0이 있다”, “정답 기억을 골랐다”, “과제에 유용하다”를 각각 확인하는 것이다. v3의 sparsemax도 dense residual 때문에 이 구분이 필요하다.

## 4. PopulationLIF_research_feasibility_assessment.md

[원문](PopulationLIF_research_feasibility_assessment.md)

v2에서 sparse/null 선택과 학습 가능한 scorer가 작동한 것과 연구 가설이 입증된 것을 분리한다. Sparse support와 empty read가 관측되지만, ETT의 정답 기억 위치가 없고 선택기의 변화가 성능 개선으로 이어졌다는 증거도 약하다. Homogeneous 반복 상태는 유효 차원이 작으므로 단순 파라미터 수 일치만으로 이질성의 효과를 고립할 수 없다.

Flatten head는 과거 전체를 직접 읽어 뉴런 기억을 우회할 수 있다. H720은 미래 예측 길이이며 “720 step 과거 기억”을 의미하지 않는다. 같은 checkpoint의 기억 제거로 성능이 나빠지는 결과는 공적응과도 양립하므로 처음부터 기억 없이 학습한 비교가 필요하다.

권고는 대규모 backbone 확장보다 정답 slot이 있는 synthetic recall, oracle, 제한된 causal readout, 학습된 recent/질량 대조, capacity 대조를 먼저 수행하라는 것이다. 구현 가능성은 있지만 이질성×선택의 상승효과와 실용적 유용성은 미입증이라는 판정이다.

## 5. Population_fLIF_Group_Interaction_Implementation_Plan_EN.md

[원문](Population_fLIF_Group_Interaction_Implementation_Plan_EN.md)

fractional dynamics bank를 실제 구현·학습 체계로 옮기려던 상세 초기 계획이다. 고정된 spikeDE 출처, scalar parity, `f_j`의 의미, 인과적 population-shared selection, 내부 상호작용, reset과 memory, logger·결과 보존, 비교 행렬을 구체화한다. 당시에는 성분별 발화, 확산 결합, tau=[2,4,8,16], 감쇠형 `p/max(p)` 등 현재와 다른 후보를 포함했다. 3-seed의 큰 실행 행렬도 계획되었다.

수식·코드·논문이 같다는 가정, reset을 적분 안팎 어디에 넣는가, alpha=1 환원 조건, 선택으로 커널 질량이 변하는 교란이 후속 논쟁의 중심이다. 해당 계획의 명령이나 수치가 최신 실행 계약이라고 해석하면 안 된다.

현재 유지할 부분은 출처 commit 고정, 내부 궤적 검사, 실제 dynamics를 value로 쓰는 것, matched control, window별 상태 격리, 최소 validation checkpoint 복원, 완전한 결과 보존이다. 확산·성분별 spike·옛 tau·옛 matrix는 최신 사전등록으로 대체되었다.

## 6. Population_fLIF_plan_reassessment_KO.md

[원문](Population_fLIF_plan_reassessment_KO.md)

EN 계획을 수학·검증 관점에서 재평가한다. `/tau`와 `/tau**alpha`는 같은 규약이 아니고, alpha는 커널 모양뿐 아니라 유한 구간 총질량도 바꾼다. 인쇄된 식의 post-reset state 재합산은 alpha=1에서 통상 LIF로 환원되지 않는 반례가 있다. 출처 재현과 원하는 모델의 자기일관성을 분리해야 한다.

`n*p`는 확률 배율 합을 맞출 뿐 fractional 질량을 보존하지 않는다. `B*p/sum(b*p)`가 그 문제를 해결하지만, 희소 support·상한·총질량을 동시에 강제할 수 없는 경우가 있다. 확산은 다양성을 줄일 수 있으나 곧바로 성능 악화를 뜻하지 않고, 반대칭 결합도 해당 fractional system의 안정성 정리가 아니다.

추가 교훈: CPU reference도 가능, seed 8개 자체가 power 보장은 아님, train-only 공유 scale로 발화 보정, `f=(I-u)/tau`를 `[u,I]`에 더하는 것은 선형 projection의 정보 증가가 아님, `Δu`는 다른 정보일 수 있음. Surrogate 함수 기본값과 실제 호출값도 구분해야 한다. 단계 A→B→C→D→E가 큰 행렬보다 우선한다.

## 7. Population_fLIF_v3A_Critical_Review_and_O7_Decisions_KO.md

[원문](Population_fLIF_v3A_Critical_Review_and_O7_Decisions_KO.md)

rev.0을 독립적으로 비판한 수학·연구 설계 검토서다. `f_j`에는 부호가 있는 누설이 있으므로 질량 보존이 안정성을 보장하지 않는다. 가지별 lag prior는 중립극한과 질량을 바꾸며, sparsemax의 0도 eta<1의 residual을 거치면 실제 적분에서는 0이 아니다. Full-history 빠른 계산은 단순 `B@I/tau`가 아니라 누설 feedback을 포함한 유효 연산자가 필요하다.

소마가 갱신 전 상태를 읽으면 현재 query와 출력 사이에 지연이 생긴다. Oracle은 최적해가 아니라 특권적 정책이며, oracle-trained·test-time oracle·generator lookup은 질문이 다르다. 3-key recall은 압축 recurrent state로도 풀 수 있으므로 명시적 검색의 필요성을 증명하지 못한다. Shared selection의 교차 미분도 모든 입력에서 nonzero인 것은 아니다.

O7을 raw attention mass·oracle 배수에서 실제 적분 질량·oracle-gap·paired CI로 바꾸자는 핵심이 후속 승인에 반영되었다. 이 문서의 권고 자체와 이후 승인된 R3/F1/0.5 기준은 구분한다. 이 검토서만으로 저장소 학습 실행을 검증한 것은 아니다.

## 8. Population_fLIF_v3_prereg_KO.md

[원문](Population_fLIF_v3_prereg_KO.md)

최신 실행 계약이다. 기존 본문을 남긴 채 뒤에 결정을 추가하므로 앞부분만 읽으면 틀린 모델을 만들 수 있다. 최신 주 모델은 K4 non-reset fractional branches, shared pre-update query `[u_n,I_n]`, `f=(I-u)/tau`, tau=[4,8,16,32], alpha=.7, g=0/pi=1, R3 `c=min(b*rho,b0)`, eta_hat=-4, 갱신 후 `u_(n+1)`을 읽는 단일 soma다. 최신 §9A의 soma 초기 혼합은 1/K다.

회상은 patch8=event, T42, C1, value를 처음 보여준 뒤 cue만 주는 과제다. 최신 D-O 난이도는 3/5/8-key, D-N은 train-only frozen projected-current normalization이다. 기본 scale은 recall8, ETTh1 6. 같은 encoding을 써야 난이도와 encoding이 혼동되지 않는다.

O7은 M_eff≥.5, 유효 oracle headroom이 있을 때 gap 회수≥.5, full 대비 평균 개선≥20% 및 paired CI 상한<0이다. ±5% oracle 차이는 자동 실패가 아니다. 8 training seeds, max50/early10/scheduler5, AdamW .001/.01, batch128, clip1, 최소 val 복원; H96 주/H720 보조다. Gate와 로그의 실제 구현은 미완성이므로 계약만으로 통과를 선언할 수 없다. Cap 뒤 M_eff 공식과 D-O chance 근거의 오류는 감사 A02/A04에서 정정 필요로 표시했다.

## 9. IDEA_LOG.md

[원문](IDEA_LOG.md)

LIF→fractional→population의 직관, 완전한 뉴런 수식, tensor 축, 선택 계산 예, 문헌 관계, 검증 순서와 모듈 계획을 한곳에 모았다. rev.1은 cap R3, tau 이동 F1, pi 제거, 소마 정렬, 유효 연산자, GRU와 mass-matched 대조, oracle 분리, fractional 기여의 제한된 해석을 반영한다.

핵심은 단일 spike 출력 앞에 K개의 연속 기억 가지를 둔 구조이며, 원본 f-SNN과 발화/reset 규약이 같은 재현 모델이라고 부르면 안 된다는 것이다. eta=0은 full fractional branch, alpha=1·eta=0·g=0은 ordinary leaky branches+소마로 환원한다. Sparse 정책에도 dense residual이 남는다.

남은 불일치: 표의 w 초기값1.0 대 최신 §9A의1/K, 후반 eta=.5 설명 대 eta_hat=-4, cap 이전 oracle 질량 공식이다. 이런 부분은 실행 코드가 모두 잘못됐다는 뜻이 아니라 통합 설명의 정정이 필요하다는 뜻이다. 주장은 정확도·기전·효율·생물학적 유사성을 분리해야 한다.

## 10. archive/IDEA_LOG_rev0_20260921.md

[원문](archive/IDEA_LOG_rev0_20260921.md)

개정 이전 445줄의 보존본이다. tau2..16, 가지별 pi, cap 이전 질량보존, 갱신 전 소마 정렬, 옛 oracle 배수 판정 등이 남아 있다. 그 내용은 당시 상태를 재구성하는 데 중요하며 최신 설정으로 사용하면 안 된다.

이 파일의 의미는 “왜 rev.1에서 tau·cap·정렬·O7을 바꿨는가”를 추적하는 데 있다. 역사 문서 자체를 현재 수식으로 고치지 말고, 후속 문서에서 대체 관계를 설명한다.

## 11. IDEA_SUMMARY_FOR_REPORT.md

[원문](IDEA_SUMMARY_FOR_REPORT.md)

보고서용 쉬운 설명이다. 과거 Neocortex/검색 실험의 한계, fractional 기억·이질 가지·선택적 회상의 동기와 최종 아이디어를 서술한다. 수식보다 문제의식과 연구 이야기를 전달하는 용도다.

다만 현재 상태보다 오래된 부분이 있다. “구현 전”, oracle 1.5배 기준, 소수 key 회상이 명시적 선택으로만 가능하다는 설명, 최초성이나 SNN 일반의 한계로 읽힐 수 있는 표현을 그대로 인용하면 부정확하다. v2의 288회는 v3의 성공/실패 결과가 아니다.

보고서 갱신 시 최신 구현 단계와 not-run을 반영하고, novelty는 문헌과 구별되는 좁은 조합·검증에 한정한다. 이 문서는 사전등록을 덮어쓰는 권위 문서가 아니다.

## 12. 46_neorecall_results_and_next.md

[원문](46_neorecall_results_and_next.md)

초기 neorecall forecasting/AD부터 여러 구조 변경, reservoir·readout 진단, lookback 확장까지의 긴 실험사다. 초반 recall/self-attention의 작은 이득은 전체 그리드와 반복 seed에서 일관되지 않았다. Gate 수치 변화, 강한 제거 비용, reservoir 이용 흔적이 처음부터 없는 모델보다 좋은 성능을 뜻하지 않는 사례가 반복된다. 중복된 불필요 Neo 계산 때문에 잘못 산정한 에너지 설명은 후속 정정이 있어 과거 수치만 떼어 인용하면 안 된다.

중요한 기전 관찰은 reservoir 내부가 매우 느린 성분을 담아도 최종 희소 LIF readout이 그 프로파일을 잃게 할 수 있다는 점이다. Analog readout으로 전달을 복구해도 forecasting 개선은 따라오지 않았다. T축을 head 전에 평균내는 변경도 독립적인 성능 손실을 만들었다. 12개 변형 전체 비교에서는 Hippo 단독 h0가 가장 좋았다.

이후 핵심 발견은 당시 seq_len96 자체의 병목이다. Patch를 함께 키워 N/T를 고정한 L720은 L96 대비 평균 약4.83% 개선, 12/12 승이었다. L1440은 다시 악화하고 L720에서 Neo를 재시험해도 약+0.08%, 6/12 승으로 무효였다. 이는 현 v3의 L336/P8/T42와 다른 실험이다.

주의할 해석: test 내부 2-fold probe는 탐색적 진단이며 독립 test 일반화 점수로 쓸 수 없다. `[H,raw]` 선형 probe의 작은 추가 이득은 모든 비선형 함수가 무용하다는 수학적 상한 증명이 아니다. 현재 선택기의 weight나 ablation 하나만으로 연구 가설을 결론내리지 않는 근거로 기억한다.

## 13. reservoir_summary.md

[원문](reservoir_summary.md)

고정 recurrent reservoir/LSM의 시간 기억, E/I 균형·안정성·입력 scaling, spike/rate/membrane/trace/tap 특징, ridge 또는 학습 readout의 역할을 폭넓게 정리한다. Reservoir 내부의 느린 dynamics와 최종 출력의 정보 전달은 별개이며, 실제 과제 시간척도에 맞는 memory가 필요하다.

특히 window 내부 상태와 window 사이를 넘기는 장기 상태를 구분한다. 순서대로 cache를 만들고 train/val/test 경계·reset·초기 상태·teacher forcing을 명확히 해야 누출을 막을 수 있다. 고정 reservoir라고 전체 모델이 무조건 BPTT-free인 것은 아니며 학습 입력 경로의 gradient 여부를 별도로 봐야 한다.

Forecasting/AD의 retrieval, auxiliary loss, causal ablation, 에너지 회계, 통계·실행 절차가 함께 들어 있다. Analog 신호를 spike-only AC로 계산하거나 AD threshold를 test로 고른 결과를 엄격한 일반화로 설명하지 않는다. 이 문서의 cross-window 아이디어는 v3에서 승인된 forward-local reset을 임의로 바꾸라는 지시가 아니다.

## 14. project_layout.md

[원문](project_layout.md)

2026-09-09 정리 당시 모델·task별 코드, script, log/results, 공용 utility, 공유 dataset의 배치를 설명한다. v1/v2 tuning 결과가 분리되고 model_v1이 기준 코드 복제본으로 만들어졌다. 경로 이동과 결과 행 수 보존이 목적이며 새 학습 결과 보고가 아니다.

현재도 유지할 원칙은 task별 산출물·실행기 배치와 queue/lock 분리다. 과거의 파일 수/모델 목록은 시점 정보이므로 최신 v3 존재 여부를 판정하는 목록으로 쓰지 않는다. Canonical log 위치는 이후 운영 규칙을 따른다.

## 15. reorganization_manifest.json

[원문](reorganization_manifest.json)

산문이 아닌 이동·분할·복제의 증거다. 전체 1,852개 move의 source/destination은 각각 중복이 없고, 기록된 이동 파일 수 합은20,574다. 결과 분할은 v1 228행, v2 1,045행이며 이는 성공 학습 수가 아니다. Copy inventory는44개이고 Python cache도 포함한다.

현재 파일로 두 raw.txt의 SHA256, 행 수, 원래 행 인덱스의 연속성, 합친 원문의 hash를 모두 검산해 일치했다. Copy hash는 현재42/44 일치였으므로 이 manifest를 현재 파일 모두의 동일성 보장으로 읽으면 안 된다. [검산 결과](../f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/manifest_review.json)를 남겼으며, 파일 이동 자체를 다시 실행하지 않았다.

## 16. PROJECT_LOG.md — 루트 canonical 문서

[원문](PROJECT_LOG.md)

브랜치/base/태그, exact command, dataset split, seed/hyperparameter, metric/artifact, 실패·복구·제한을 append-only로 보존한다. 9/10 로컬 snapshot → Gaussian population Spikformer → ETT quick40/3-seed120 → population 삭제 대조96 → v1 patch H96/H720 각32 → v1 TCN48 → v2 4구조288 → v3 설계·수치 게이트·보정으로 발전했다.

역사적 결과의 요지는 다음과 같다. Attention 축·identity 변경은 반복 seed에서 안정적인 이득이 없었다. Gaussian 대 repeated scalar는 해당 짧은 예산에서 약3% 이득이 있었지만 독립적으로 최적화한 raw baseline에 대한 일반 우위는 아니다. v1 retrieval의 작은 H96 이득은 H720에서 악화했다. v2는 선택/empty read가 작동해도 이질성×선택의 일관된 성능 이득은 미입증이다. 네 backbone의72개씩 전체 표를 읽어 반올림 MSE macro를 재계산했다. 과거 checkpoint 전체 재평가는 이번에 하지 않았다.

운영 교훈도 중요하다. Dense tiny mass 때문에 진단 invariant가 실패했던 것을 단순 허용오차 완화가 아니라 실제 floor 수식으로 정정했다. TCN은 GPU당2작업이 오히려 처리량을 크게 낮췄지만 다른 구조는 달랐다. Queue의 running 표기만으로 프로세스 생존을 판단하면 안 된다. 학습 중 HEAD guard를 유지하기 위해 문서/분석을 다음 완료 commit에 모았던 전례가 있다.

최신 v3 기록은 수학 검토·R3/F1 승인·A/C 17 pass/1 not run·회상 생성기·발화율 보정이다. **v3 학습 완료 기록은 없다.** 이전 항목의 미결/실패는 이후 정정과 함께 읽어야 하며, 옛 O7·tau·eta·G11의 설명이 최신 상태와 섞이지 않도록 주의한다.

## 17. 루트 docs/environments/snn_recall-20260910-packages.txt

[원문](../../docs/environments/snn_recall-20260910-packages.txt)

129개 패키지 버전의 당시 inventory이며 설치 재현을 검증한 lockfile은 아니다. 핵심은 torch1.12.0+cu113, SpikingJelly0.0.0.0.14, cupy-cuda11x13.5.1, NumPy1.26.4, pandas2.3.1, scikit-learn1.7.1이다. CUDA13 계열 패키지 이름도 함께 있으므로 그 목록만으로 실제 torch CUDA runtime을 추정하면 안 된다.

이번 CPU 재검사는 snn_recall Python3.10.18/torch1.12.0+cu113로 수행했다. 별도 snn_jelly의 torch2.11.0+cu130에서 CPU forward/backward와 `torch.compile` attribute 존재도 확인했다. CUDA driver 제약과 CPU reference 가능 여부는 다른 문제다. 원본 spikeDE golden parity는 여전히 not run이다.

## 18. 다음 세션에서 유지할 해석

- 학습된 선택성, 의미 있는 기억 회상, 예측 향상, 효율 향상을 각각 입증한다.
- Full·mass-matched·oracle-trained·GRU가 서로 다른 질문에 답하도록 구현을 먼저 고정한다.
- Cap 이후 계수를 실제 수식과 지표의 공통 기준으로 쓴다. 확률 p만 보고 완전 배제나 M_eff를 주장하지 않는다.
- 알려진 결과로 수정한 과제·threshold·key·surrogate는 탐색적 변경으로 기록한다. 옛 결과를 새 사전등록의 검증 결과로 합치지 않는다.
- 지속 기억은 이 문서와 canonical log에 남긴다. 자동 갱신이나 다른 세션의 자동 확인이 설정된 것은 아니다.

## 추적 갱신 — 2026-09-21 23:54 KST

- `Population_fLIF_v3_prereg_KO.md` §2C가 추가됐다(문서 표기 날짜22일, 실제 관찰21일). D-P는 post-cap mass_matched/gradient 유지, D-Q는 실제 c의 M_eff와 .5 유지/집계 단위, D-R은 recall r2 min_gap 및 주3·5/stress8, D-S는 uniform-slot chance와 Full kernel mass 구분, D-T는 actual current 측정/안정성 범위 축소, D-U는 fixed eta/cap 배선, D-V는 G4 사유/G7b허용오차/G15 spike 검사 정정이다. 이전 상충 정의보다 이 append가 최신 계약이다. 학습 metric 구현 완료나 효과 입증을 뜻하지 않는다.
- `ASSESMENT.md` §9 및 canonical PROJECT_LOG에 추적 감사01 추가. 고정 사본 독립 재실행19 pass/0 fail/1 not run; A01/A04/A06/A11의 명시한 수치·배선 범위 확인. r2 8-key의 실제 재질의 coverage 부족, bound 초과 후보 선택, branch 최대값·cap_rate 의미, 보정 충돌/실패 증거 보존 문제는 남는다. 문서별 최초 요약/역사 판정은 덮어쓰지 않았다.

## 추적 갱신 — 2026-09-22 00:03 KST

- 사전등록 §2C의 D-W가 추가됐다: r2 8-key는 낮은 coverage stress로만 해석, capacity scaling 일반화 보류, recall-only MSE와 재등장 run 첫 사건 MSE 분리. 과제의 통계적 최대와 평균 run 예산을 구분해야 한다. §2C 날짜/독립 current probe ‘일치’ 표현도 담당 세션에서 정정했다.
- ASSESMENT §10의 감사자 재검사: Full 보정의 bound 거부, branch별 actual max, cap=False 실제 cap_rate0, exclusive 파일생성/실패 row 보존 확인. sparse 초기 main의 finite/bound 검사는 아직 누락됐으며 fault injection으로 재현했다. 실제 학습 폭주 증거는 아니다. A05 전체/학습 안정성은 OPEN.
- A07의 원본 부재 사유는 해소했다. pinned spikeDE fcd743b 원본 neuron+predictor를 기존 torch2 CPU에서 직접 실행: 24scalar조건 최대오차3.0184e-15/allspikes같음, 기존 reset port17spikes 확인, arctan5gradient차4.4409e-16. 원본 scalar 실행 대조 증거가 생겼지만 공식 runner G4/fullwrapper·GPU·학습 검사는 별도다. ‘원본 설치 불가능’으로 기억하지 않는다.


## 자동 감시 설정 — 2026-09-22 00:16 KST

사용자 요청으로 cron 10분 변경 감지와 기존 감사 세션 호출을 활성화했다. `ASSESSMENT_WATCH.md`는 운영/중지 방법, `ASSESSMENT_WATCH_PROMPT.md`는 3개 감사 항목과 append·snapshot·완료 ack 절차다. ASSESMENT/기억/canonical log 자체는 trigger에서 제외되지만 매 감사에서 읽는다. 이전 예약 미설정 설명은 역사적 상태다.


## 추적 갱신 — 2026-09-22 00:28 KST (예약 감사03)

- 사전등록 최신 계약은 D-W까지이며 이번 감사에서 변경하지 않았다. 새 M1/GRU/trainer/test와 첫 smoke checkpoint가 존재한다. 과거 ‘v3 학습 결과 없음’은 당시 상태다. 현재는 seed7/2epoch/r2 k3/512·128·128 기능 확인 결과만 독립 평가 재현했으며 정식 성능 판정은 보류한다.
- synthetic kind API는 bool에서 int8(0 copy/1 recall/2 recall-first)로 바뀌었다. evaluate의 사건별 MSE는 맞지만 selection_diagnostics는 kind를 무시해 copy933건을 섞고 batch×시점 평균을 취한다. 기존 M_eff0.22736 대 독립 recall-only sequence 평균0.12957. A02/A08 진단 OPEN.
- A08 analog live graph와 recall causal head를 소규모 CPU에서 확인했다. A12 oracle first-run uniform 설명 불일치(copy610건 nonuniform), ETT 빈 truth IndexError, GRU None metric 로깅 실패가 남는다. A10 calibration 요청 호환성/표준 CSV/평가 checkpoint provenance, A05 sparse finite/bound 연결도 OPEN. --no-test가 train 경로에 반영되지 않는다.
- ASSESMENT 추적 감사03과 `results/assessment/20260922-0017-kst-scheduled/`를 함께 읽는다. 캡처00:18:04 KST/HEAD0afda35, 종료 전 HEADc34fa1c 관찰(62개 캡처 파일 hash 동일). 이후 새 pilot 결과는 다음 주기의 범위다. 신규 진단 코드는 `scripts/audit_v3_pipeline.py`; optimizer step/학습은 수행하지 않았다.


## 추적 갱신 — 2026-09-22 00:33 KST (예약 감사04)

- Trigger의 코드 변경은 감사03과 같은 hash였다. 새 범위는 pilot-eta-001742의15epoch/seed7/2048·256·256 결과다. CPU에서 recall MSE .26989367 및 Full/recent/mass_matched/oracle 개입을 재현했다. 동일-checkpoint Full 대비3.75% 감소, oracle은27.47% 감소이나 재학습 대조·정식8seed 효과 판정은 아니다.
- A02 집계 오류 지속: 저장 M_eff .231467 대 독립 recall-only sequence 평균 .133093(Full kernel .130329). A10-HASH는 잠재 위험에서 실제 불일치 사례로 갱신: JSON parameter hash14a5650… 대 평가 best50d4af2…. 평가 MSE 자체는 재현된다. 기타 OPEN 상태 유지.
- ASSESMENT 감사04 및 results/assessment/20260921T153001Z-ed6b3b26/ 증거를 참조. 모델 소스/사전등록 변경 없음; 새 감사 스크립트 audit_v3_pilot.py만 작성. 기존 smoke와 같은 data_seed의 확장 표본이므로 독립 복제로 세지 않는다.


## 추적 갱신 — 2026-09-22 03:33 KST (예약 감사05)

- layers.py에 query/key frozen normalization 추가.03:30:45 사본 SHA001c4d82…에서 함수 통계/identity parity/인과성/new-schema roundtrip을 CPU로 확인. 캡처 당시 train/calibrate 연결은 아직 없었으므로 진행 중으로 분류. 새 학습 결과·성능 판단 근거 없음.
- A10-KEYNORM-CKPT: 새 key_mean/key_std buffer 때문에 기존 pilot strict load가 실패함을 재현. legacy identity migration 또는 과거 소스 보존 필요; 포괄 strict=False는 피할 것. 기존 OPEN은 유지.
- 새 주석의 gradient 폭주 원인/해결 단정은 미검증. Pascanu et al.(2013) 원문 §2를 확인하고 pre-clipping gradient/시간길이/validation의 통제 비교를 권고. 감사 중 calibrate/config/layers가 다시 바뀌어 다음 snapshot에서 검토. 증거 results/assessment/20260921T183001Z-0804948b/ 및 ASSESMENT 감사05.


## 추적 갱신 — 2026-09-22 03:45 KST (예약 감사06)

- 최신 사전등록 §2D D-X는 train Full 초기 궤적 θ 보정, D-Y는 고정 η0/.2/.5/1 주 경로 승격이다. Key_norm 기본은 none, frozen은 탐색 옵션. 결과 이후 개정된 탐색 계약으로 구분한다.
- A05 sparse finite/bound 실패 거부·row 보존·exit1을 독립 재확인(해당 범위 VERIFIED). Sparse branch health 거부는 아직 누락(죽은 branch/비율1000 accepted). A12-GRU None 제거 schema의 logger/verbose 통과 확인; 전체 학습 검증은 별도.
- A10-THETA: 보정5.56134335를 loader가 반환하지 않아6개 새 pilot은5.5. A02 oracle η.5 M_eff가 기존.52672에서 독립 recall-only.46655로 바뀌어0.5 판정에 영향. Legacy config key_norm 누락도 로드 실패.
- 12epoch/seed7 새6결과 재현: Full recall.27634, sparse학습η.27174, sparse고정.5 .41077, oracle-trained.5 .03621/1 .02968. oracle 사용 경로의 탐색 증거는 생겼지만 sparse 검색 성공은 미입증. Full 두 run은 같은 seed/hash로 독립 반복 아님. 학습η의 best/last hash 불일치 지속.
- D-X gradient 표의 재현 명령/seed/loss 누락, epoch max|grad| 기록 미연결; ‘어떤 길이에서도 안정’ 일반화 보류. ASSESMENT 감사06과 results/assessment/20260921T184001Z-e2dd33af/를 참조. 캡처03:40:36 KST/HEAD5512135; 이후 config/log 변경은 다음 주기.


## 추적 갱신 — 2026-09-22 03:53 KST (예약 감사07)

- GRU pilot12epoch/seed7의 실제 완료 best 평가를 CPU 재현: all.19439441/copy.01641275/recall.25310525/first.25632199, hash일치·causality/truth분리 확인. A12-GRU 완료 artifact 평가 범위 VERIFIED; 새 학습은 수행하지 않았다.
- run_id의 model/alpha/readout JSON 구분은 확인했지만 analog/spike의 log/checkpoint 경로는 여전히 충돌(FileExistsError). A10-PATH 부분 확인/잔여 OPEN.
- Recall 사용 모듈 parameter GRU4065 대 myModel490(selector 포함), 총 수134241/130666의 유사함은 미사용 forecasting head 때문. Parameter-matched라고 해석하지 않는다. 현재1seed에서는 GRU가 학습 sparse보다 낮은 recall MSE이나 정식 통계 우위/과제 완전 해결은 미입증. Oracle과 공정 순위 비교 금지.
- 사전등록 D-X/Y 유지. Canonical θ5.561 표기는 실제config5.5로 정정 필요. Copy gap의 readout 병목 해석은 가설. ASSESMENT 감사07/results/assessment/20260921T185001Z-86f72a84/ 참조.


## 추적 갱신 — 2026-09-22 10:15 KST (예약 감사08)

- 최신 사전등록 §2E D-Z/AA/AB 추가를 후속 문서 사본으로 확인: theta 전달/readout 경로/G4 범위, parameter 비교·readout 가설 표현 정정. 원 실험 사본10:10:40 KST/HEADd1fb43e는 따로 고정했다.
- A10-THETA 실제 main 설정에서123→5.56134335/scale1→8 확인, A10-PATH 같은 suite spike/analog 저장 경로 분리 VERIFIED. 기존1epoch wirecheck checkpoint2개 MSE/hash 재현. 성능 판정 아님. calibrated_fields는 config에 있고 JSON provenance에는 아직 theta와 함께 빠져 있음.
- A07/G4 공식 runner20pass/0fail/0notrun, pinned golden은 이전 감사 파일과 동일.24조건/360step/최대7.77e-16은 첫 spike 전 또는 spike-free 적분기 비교이며 전체뉴런 parity 아님. 원본 부재 NOT RUN 상태 해소. make_golden.py는 미완 scaffold; 상류 재생성은 이번에 not run.
- Best/last provenance와 A02/A05branch/legacy/ETT/test-off 등 기존 미수정 이슈 유지. 증거 ASSESMENT 감사08 및 results/assessment/20260922T011001Z-6c90ce78/.


## 추적 갱신 — 2026-09-22 10:21 KST (예약 감사09)

HEAD ecec8d397862e6367e556668d82e6dd11e2e36e4의 감지 내용은 감사08 사본+후속 문서와 hash가 같았다. 새 판단 근거 없음; 재검사 not run, 기존 상태 유지. Make_golden scaffold를 완료로 해석하지 않는다. 증거 results/assessment/20260922T012001Z-8b3e144d/inventory.json.


## 추적 갱신 — 2026-09-22 10:38 KST (예약 감사10)

- Snapshot10:30:36/HEAD73599f0,111개 파일. 후속 사전등록 §2F D-AC/AD·canonical10:31도 별도 보존/검토. A02 recall-only query→sequence 및 모든unit hit 배선 VERIFIED: pilot M_eff.133093031595, batch128/16 차0. 기본 첫4batch 표본 범위 주의.
- A05 sparse dead/imbalance 거부 VERIFIED. A10-CAL tau/cue 거부·require guard 부분확인, key_norm 변경은 여전히승인. A12 빈truth 및 no-test tail, A10 best/last/checkpoint hash 배선 scoped VERIFIED; 새학습not run.
- Legacy pilot 정상복원/MSE재현. 신규 **A10-RESTORE-STATS OPEN(P1)**: frozen norm_mean/std 제거 임시checkpoint가 승인되고 예측최대차.870653. 버전없는무조건optional허용 금지. 일반weight누락은거부.
- 새 diag3 sparse12epoch/seed7 recall.272620149439,M_eff.132756536374,kernel.130328893616,hit0. 8seed/CI·성능우위미확인. 중간 JSON을 완료로 세지 않음; sourcehash는종료시live라실행시점증거부족.
- A09: 표본 L2 norm 평균은 max|grad|가 아니며 canonical10:31의singleton기각/폭주소멸없음/eta원인단정은과도. 문헌기반통제진단권고유지. 추가 analog/ETT/Full/oracle 및 config/layers/ours의 감사중변경은다음주기. ASSESMENT 감사10과 results/assessment/20260922T013001Z-6c439d7a/ 참조.


## 추적 갱신 — 2026-09-22 10:45 KST (예약 감사11)

- Snapshot10:40:39/HEAD639d2293/143파일 + ETT CSV 고정. Prereg §2F 동일. A12-ORACLE 현재 kind 배선 VERIFIED(copy비균등1243→0/전체256sequence,recall질량오차0). 새csvchk2epoch 현재정책MSE재현.
- 과거12epoch oracle(spike/analog/drive)은 옛정책에서만 저장MSE차0. 새정책으로 같은checkpoint 평가시 recall .035348/.071974/.045209. 새정책재학습결과로세지않음. Driveoracle은새sourcehash기록에도옛정책만재현하므로 시작source/정책version고정필수(A10-PROVENANCE).
- Drive=소마전가지혼합,실제aux차0/gradient유한/causalprefix차0. Sparse recall spike.272620/analog.261620/drive.212159; drive Meff.134013 대kernel.130329로검색성공미입증. 1seed탐색;O7/8seedCI not run.
- A10-LOG 실제2행CSV확인,final+result.csv없음/동적nonfinite열누락위험잔여. Actualevaluated/checkpointhash일치,driveoracle last≠best확인. ETT첫3batch192창 MSE.911764재현(전체2785창아님),window_mean.856615. No-test결과test_skipped확인,감사자는해당test평가안함.
- A10-RESTORE-STATS/model동일로OPEN유지,A09과잉인과해석/A07재생성등도유지. ASSESMENT 감사11 및 results/assessment/20260922T014001Z-17a8da3d/ 참조.


## 추적 갱신 — 2026-09-22 10:55 KST (예약 감사12)

- Snapshot10:50:38/HEAD520bb568/139파일. A10-RESTORE-STATS 누락거부8조건+legacy정상복원 VERIFIED. Input/key frozen 및 fixedeta buffer누락거부,none/학습eta허용,weight누락거부.
- A09 absmax추가했지만np.mean집계유지:주입[1,9]→5,[2,10]→6. epoch최대아님;OPEN. *_nonfinite JSON전달과theta/calibrated_fields 실제JSON확인 VERIFIED.
- gradchk2epoch512/64/64,batch64,g11_every10:전체MSE.337799426411/recall.347767202408,hash/MSE재현. epoch8batch중1watch라최대집계문제안드러남. 성능통계not run.
- key_geometry 재생성코드/hash/정의미비(A09-KEY-GEOMETRY OPEN). 유닛별raw5/투영key4차원으로rank상한4;3.97/42를그자체붕괴증거로삼지말것. hit3.73e-5는pilot-eta이지diag3아님,동일표본비교필요.
- canonical10:47 과도gradient인과주장철회확인.10:40 readout원인단정/10:50 QKNormθ불필요·폭주제거/Gram해석과장잔여. TTR원문§3(bandwidth잔존),DeltaNet원문,entmaxPDF명제1실제확인;Hopfield초록만확인/PDF실패정리검증not run. ASSESMENT 감사12에직접링크와제한기록.


## 추적 갱신 — 2026-09-22 11:01 KST (예약 감사13)

HEAD41554c68cd97cbad522008297b98eda230b1e651은 감사12에서 검토한 key_geometry/문헌 내용을 commit한 상태. 감시 대상120개 파일 hash 동일, 감사자3문서 append 외 새 판단 근거 없음. 재검사/학습 not run, 감사12 VERIFIED 범위와 OPEN 유지. Inventory results/assessment/20260922T020001Z-8f8cd215/inventory.json.


## 우선순위 갱신 — 2026-09-22 11:15 KST (AUDIT-PRIORITY-01; 사용자 직접 요청)

채택 검증 순서: 고정η→QK L2정규화→causal key표현→entmax→별도Gram→별도Delta. Residual즉시삭제아님,η1이이미제거대조. 같은diag3checkpoint validation64/2004query:score/p hit.2715946,c hit0,p mass.256499,c mass.137037,kernel.134411,chance.161031. 작은η에서도lag2/5/10이상적정책은최근계수역전가능,전혀효과없다는명제기각. Cap/질량분리필요. 투영key4차원PR상한4,이번비중심PR1.17473. Hopfield원문PDF확보/Eq5·Thm4/5확인으로평균cosine만으로실패단정불가 재확인. 조건별채택기준/순위조정은ASSESMENT AUDIT-PRIORITY-01. Artifacts results/assessment/20260922-adoption-priority/. 학습/후보효능검증not run.


## 추적 갱신 — 2026-09-22 11:35 KST (감사14; 사용자 직접 요청)

HEAD4d67c5072b63bfeaf570be44c02c945d0f9aa8a8,11:33:02 snapshot123파일. 연구소스/결과변경없음;canonical11:31 우선순위수용·작은η/Hopfield/혼합checkpoint정정확인. 표본평균tokenGram은rank4초과가능하므로작업자보완수용,상한4는개별matrix에한정정정. 무작위64×42×4 예에서개별rank4,평균Gram rank42/PR36.3904;원3.97/35.67재현not run. 신규A02-REACHABILITY:‘단일정답η<1이면M_eff.5불가’반례,η.92 lag41 .50086126/lag2 .50704244. Cap구간경계η=1−b0/(B−bd),약.918–.920. 실제학습성능아님. 기존우선순위/OPEN유지,새학습not run. ASSESMENT감사14/results/assessment/20260922-manual-followup/.


## 추적 갱신 — 2026-09-22 11:49 KST (예약 감사15)

Snapshot11:40:55/HEAD2769c5c8100e367b965a47b20bd1924f0b0d18a1/148파일. Etagrid 고정η0/.2/.5/1 checkpoint CPU재평가 test256/8085query, MSE·M_eff 차0/hash일치. Recall .276339/.319387/.433504/.416683, M_eff .130329/.127182/.132871/.136899. η.5는patience10에맞는11epoch종료;η1도완료. A09 clipping전큰gradient(마지막표본absmax평균 .5:1.06e11/1:4.37e12),최대집계OPEN. A02-REACHABILITY §2G uniformoracle상한명제반례 .164497→.174429. 자유정책상한 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)]. 실제test분포η.5U .466613/.655uniform .557889;평균lag/정답수로최소η.70–.75단정불가. η0 learned/oracle비율1은선택성공증거아님. 우선순위유지,새oracle/학습η후속은snapshot후기록으로다음주기;8seedCI/새학습not run. ASSESMENT감사15,results/assessment/20260922T024001Z-acec6eac/.


## 추적 갱신 — 2026-09-22 11:54 KST (예약 감사16)

HEAD7520087c6d5578c1ec640350fa233717ce78d980/snapshot11:50:51/160파일,manifest불일치0/소스변경0. 새oracleη.5/1 및학습η3checkpoint test256/8085query CPU재현,전체MSE/M_eff 차0/hash일치. Recall .039580641913/.034965688339/.272620149439, M_eff .466552377101/1/.132756536374. 새kind정책재현확인;실행시작provenance보장은아님. Headroom .241372947820,공통oracle1분모G학습η .0154055653;독립8seedCI not run. 같은test uniform-slotchance .156583749693(기존.161과표본다름);fullmass .130329와구별. Split tensorhash세run동일,split간상이. 학습η는이전diag3수치재현이지새독립seed아님. ①탐색격자재현완료→②QK정규화후보유지,구현/효능not run. §2G상한정정미반영/A02-REACHABILITY및기존OPEN유지. ASSESMENT감사16/results/assessment/20260922T025001Z-31dd6781/.
