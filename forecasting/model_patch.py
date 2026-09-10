"""
model.py 수정 방법 (diff 형식으로 설명)
==========================================

기존 model.py 상단 import 블록 끝에 아래 두 줄을 추가:

    import patch_itransformer                          # x_mark=None 패치 (부작용 없음)
    from iTransformer_wrapper import load_iTransformer

그리고 파일 맨 아래 LOAD_MODEL 딕셔너리에 한 줄 추가:

    LOAD_MODEL = {
        'myModel'      : load_mymodel,
        'degree'       : load_mymodel_degree,
        'Spikformer'   : load_spikformer,
        'ab1'          : load_mymodel_ab1,
        'ab1_1'        : load_mymodel_ab1_1,
        'ab2'          : load_mymodel_ab2,
        'ab3'          : load_mymodel_ab3,
        'ab4'          : load_mymodel_ab4,
        'iTransformer' : load_iTransformer,   # ← 이 줄만 추가
    }

=======================================================
아래는 복사-붙여넣기용 완성 스니펫입니다.
model.py 기존 내용은 건드리지 않고 이 블록을 파일 끝에 붙여넣으면 됩니다.
=======================================================
"""

# ── 이 블록을 model.py 최상단 import 구역에 추가 ──────────────────────────
# import patch_itransformer                       # noqa: F401  (패치 적용)
# from iTransformer_wrapper import load_iTransformer
# ─────────────────────────────────────────────────────────────────────────

# ── LOAD_MODEL 딕셔너리 교체 (기존 항목 유지 + iTransformer 추가) ──────────
# LOAD_MODEL = {
#     'myModel'      : load_mymodel,
#     'degree'       : load_mymodel_degree,
#     'Spikformer'   : load_spikformer,
#     'ab1'          : load_mymodel_ab1,
#     'ab1_1'        : load_mymodel_ab1_1,
#     'ab2'          : load_mymodel_ab2,
#     'ab3'          : load_mymodel_ab3,
#     'ab4'          : load_mymodel_ab4,
#     'iTransformer' : load_iTransformer,         # ← 추가
# }
