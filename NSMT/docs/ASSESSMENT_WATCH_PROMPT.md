# 예약된 f_lif_pop_v3 추적 감사

사용자가 승인한 주기적 감사 실행입니다. 예약 ID: `{{RUN_ID}}`.
작업 경로: `{{ROOT}}`. 이 감사 세션의 이전 목적을 유지합니다.

1. `docs/DOCS_REVIEW_MEMORY.md`, `docs/ASSESMENT.md`의 최신 append, 사전등록 최신 개정, canonical `docs/PROJECT_LOG.md`를 읽어 기존 정의와 열린 이슈를 복구하세요. 첨부된 변경 목록과 manifest부터 실제 현재 코드·결과를 대조하세요. 스케줄러의 해시는 변경 감지용이며 원본 사본이 아닙니다. 재현 검사 전에 대상 파일을 별도로 snapshot하고 hash를 기록하세요.
2. 세 검토 항목을 각각 판정하세요: (a) 아이디어 구현의 정확성, (b) 검증 과정·데이터 분리·대조군·통계·재현성의 적절성, (c) 새 실행 결과에 근거한 개선 방향. 개선 방향은 기존에 확인한 1차 문헌을 활용하고 새 사실·논문이 필요하면 웹/학술 원문을 실제 확인해 직접 링크하세요. 결과가 없으면 성능 판단을 보류하고 not run을 명시하세요.
3. 진행 중 파일의 미완성과 확정 오류를 구분하세요. 기존 감사 probe는 현재 API와 진단 정의를 먼저 확인한 뒤 필요한 것만 CPU 소규모로 실행하세요. 실패한 명령을 모델 실패로 오인하지 마세요. 재검사 근거 없이 FIXED-PENDING-REVIEW를 VERIFIED로 닫지 마세요.
4. `docs/ASSESMENT.md`에 날짜(KST), 관찰 commit/source hash, 대응 이슈 ID, 변경·실제 증거·판정·남은 조건을 **끝에 append**하세요. 오래된 본문을 수정하거나 삭제하지 마세요. 의미 있는 새 결과가 없으면 짧게 ‘새 판단 근거 없음’과 확인 범위를 기록하고 종료하세요. 필요한 문서 기억과 canonical PROJECT_LOG도 append하세요. 자신이 쓴 기록을 다음 변화로 오인해 반복 감사하지 마세요.
5. 이 작업은 감사입니다. 모델/학습 소스 수정, 학습 실행·GPU 점유, 진행 프로세스 중단, 환경 설치, git add/commit/tag/push/reset/switch는 하지 마세요. 기존 local changes·checkpoint·raw log를 보존하세요. 새 진단 코드/텍스트 증거와 감사 문서만 작성하세요. 감사 artifacts는 `f_lif_pop_v3/forecasting/results/assessment/`, raw logs는 같은 task의 `log/assessment/`에 두세요. 다른 세션의 대화 파일을 읽거나 다른 사람에게 메시지를 보내지 마세요. 감시 예약 자체를 새로 만들지 마세요.
6. 감사 append를 마친 다음 정확한 마커 `<!-- assessment-watch:{{RUN_ID}} -->`를 `docs/ASSESMENT.md` 끝에 추가하고 아래 명령으로 완료를 알리세요. marker 없이 ack하면 거부됩니다. 감사 중 새 변경은 다음 주기에 다시 감지합니다. 실행을 완료하지 못했다면 완료로 표시하지 말고 남은 문제를 기록하세요.

```bash
/usr/bin/python3 "{{ROOT}}/scripts/watch_f_lif_pop_v3.py" ack --run-id {{RUN_ID}}
```

짧은 한국어 결과를 사용자에게 알리고 이번 실행을 끝내세요. 다음 실행은 cron이 변경을 확인해 호출합니다. 이 turn에서 polling/sleep 루프를 만들 필요는 없습니다.
