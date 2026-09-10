"""
patch_itransformer.py
---------------------
iTransformer 원본 layers/Embed.py 의 DataEmbedding_inverted 는
x_mark 가 None 이면 에러가 납니다.

이 파일을 임포트하면 해당 클래스를 monkey-patch 하여
x_mark=None 일 때 시간 피처 임베딩을 건너뜁니다.

사용법 (iTransformer_wrapper.py 상단에서 자동 호출):
    import patch_itransformer   # noqa: F401
"""

import torch
import torch.nn as nn

try:
    from layers.Embed import DataEmbedding_inverted as _Orig

    _orig_forward = _Orig.forward

    def _patched_forward(self, x, x_mark):
        """
        x      : (B, L, N)
        x_mark : (B, L, time_feat_dim) or None
        """
        # value embedding: (B, N, d_model)
        x = self.value_embedding(x)          # → (B, N, d_model)

        if x_mark is not None:
            # 원본 로직: temporal embedding 도 더함
            # x_mark: (B, L, D_time) → inverted 후 (B, D_time, d_model) 로 embed
            x = x + self.temporal_embedding(x_mark.permute(0, 2, 1))

        x = self.dropout(x)
        return x

    # monkey-patch
    _Orig.forward = _patched_forward
    print("[patch_itransformer] DataEmbedding_inverted.forward patched (x_mark=None safe).")

except ImportError:
    print("[patch_itransformer] WARNING: layers.Embed not found — patch skipped.")
