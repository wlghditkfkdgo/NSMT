# f_lif_pop_v3 자동 추적 감사 운영

2026-09-22 KST 사용자 요청으로 활성화했다. 서버의 사용자 cron이 **매시00/10/20/30/40/50분**에 `scripts/watch_f_lif_pop_v3.py tick`을 실행한다. 실행 중인 감사 세션으로 `codex queue`를 통해 작업을 보낸다. ChatGPT 앱의 Scheduled 목록에 만든 예약은 아니다. CLI/IDE에는 해당 관리 화면이 없으므로 로컬 cron을 사용했다. [공식 예약 작업 문서](https://learn.chatgpt.com/docs/automations)

## 실행 내용

- `docs`와 `f_lif_pop_v3`의 코드·설정·텍스트 결과 및 Git HEAD 변경을 확인한다. 큰 checkpoint/데이터/캐시/이벤트 파일은 감시하지 않는다. 학습 지표 CSV, calibration JSON, 새 코드와 분석 결과는 포함한다.
- 변경이 없으면 모델을 호출하지 않는다. `ASSESMENT.md`, 문서 기억, canonical PROJECT_LOG, 감사 artifact/예약 운영 문서는 자기 반복을 막기 위해 trigger에서 제외한다. 예약 실행 시에는 이 문서들을 읽고 검토 맥락을 복구한다.
- 한 번에 감사 하나만 대기한다. 감사 중 변경은 완료 후 다음 주기에서 감지한다. 모델 호출은 기존 감사 thread `01a0c453-2d06-7ee2-bf44-dafb878d4a96`에 전달한다.
- 감사는 구현·검증·결과에 근거한 문헌 기반 개선 방향을 검토하고 **`docs/ASSESMENT.md` 끝에 append**한다. 모델/학습 코드는 직접 수정하지 않고 새 훈련을 실행하지 않는다. 상세 [예약 프롬프트](ASSESSMENT_WATCH_PROMPT.md).
- 완료 마커와 `ack`를 모두 기록해야 대기가 풀린다. 감사가 실패하거나 중단되면 status의 pending을 확인하고 해당 실행을 이어서 완료한다. 자동으로 완료했다고 처리하지 않는다. CLI 전달 실패는 다음 주기에 재시도한다. 전달 timeout은 중복 방지를 위해 대기로 유지하므로 상태 확인이 필요하다.

## 상태·중지·재개

아래 명령의 cwd는 NSMT다. 중지는 새 호출을 막는다. 이미 큐에 들어간 감사는 따로 완료하거나 취소해야 한다.

```bash
python3 scripts/watch_f_lif_pop_v3.py status
python3 scripts/watch_f_lif_pop_v3.py pause
python3 scripts/watch_f_lif_pop_v3.py resume
crontab -l
```

Cron 항목을 완전히 제거하려면 `crontab -e`에서 `BEGIN NSMT_F_LIF_POP_V3_ASSESSMENT_WATCH`와 `END NSMT_F_LIF_POP_V3_ASSESSMENT_WATCH` 사이의 전용 블록만 제거한다. 다른 예약을 지우지 않는다. 등록 전 crontab은 `scripts/queues/assessment_watch/crontab.before.txt`에 보관했다.

## 파일 및 운영 조건

- 감지 상태/lock/trigger/전달 응답: `scripts/queues/assessment_watch/` (Git 제외)
- cron 원시 출력: `f_lif_pop_v3/forecasting/log/assessment_watch/cron.stdout` (로컬)
- 설정/검사 증거: `f_lif_pop_v3/forecasting/results/assessment/automation-setup-20260922/`
- 보고서: `docs/ASSESMENT.md`, 실험의 canonical 기록: `docs/PROJECT_LOG.md` symlink

서버가 켜져 있고 cron·Codex daemon/로그인·네트워크가 동작해야 실제 감사가 실행된다. 예약은 제거하거나 pause할 때까지 유지된다. 파일 변화에 따라 호출하므로 의미 없는 상태 확인에 매번 모델을 사용하지 않는다. 감사 호출에는 기존 계정의 사용량이 적용된다.

설정 검증에서는 실제 queue 접수, crontab 등록 재조회, cron active를 확인했다. 무변경/변경/중복/ack/동시 변경/전달 실패/중지/timeout 동작은 임시 fixture와 mock 전달을 이용한9개 검사로 확인했다. 첫 실감사는 설정 시 큐에 넣었으며, 그 감사가 완료되었는지는 status의 `last_completed_run_id`와 보고서의 완료 마커로 확인한다.
