"""
iTransformer_wrapper.py  (standalone, no external layers/ dependency)
----------------------------------------------------------------------
iTransformer (Yong Liu et al., ICLR 2024) 의 핵심 구성 요소를 인라인으로
포함하여, 기존 layers.py 와의 충돌 없이 동작합니다.

기존 train / test 루프 인터페이스 호환:
  - model(x)          : (B, L, C) → (B, pred_len, C)
  - model.pred_len    : int
  - model.train_mode  : str  (dummy)
  - functional.reset_net(model) → no-op (LIF 노드 없음)
  - model.init_testing()        → no-op
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1. Embedding
# ═══════════════════════════════════════════════════════════════════════

class DataEmbedding_inverted(nn.Module):
    """
    Inverted embedding: variate(N) 를 token 으로 취급.
    x_mark=None 허용 버전.
    """
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: (B, L, N)  →  (B, N, d_model)
        x = self.value_embedding(x.permute(0, 2, 1))   # (B, N, d_model)
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════
# 2. Attention
# ═══════════════════════════════════════════════════════════════════════

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.inner_attention = attention
        d_keys   = d_model // n_heads
        d_values = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, L, H, -1)
        values  = self.value_projection(values).view(B, L, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ═══════════════════════════════════════════════════════════════════════
# 3. Encoder
# ═══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention  = attention
        self.conv1      = nn.Conv1d(d_model, d_ff, 1)
        self.conv2      = nn.Conv1d(d_ff, d_model, 1)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm        = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# ═══════════════════════════════════════════════════════════════════════
# 4. iTransformer core model
# ═══════════════════════════════════════════════════════════════════════

class _iTransformerCore(nn.Module):
    """원본 iTransformer (paper: https://arxiv.org/abs/2310.06625)"""

    def __init__(self, seq_len, pred_len, c_in,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.1, factor=1, activation='gelu', use_norm=True):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, dropout, False),
                    d_model, n_heads),
                d_model, d_ff, dropout, activation
             ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc):
        """x_enc: (B, L, N)  →  (B, pred_len, N)"""
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc)            # (B, N, d_model)
        enc_out, _ = self.encoder(enc_out)             # (B, N, d_model)
        dec_out = self.projector(enc_out)              # (B, N, pred_len)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]  # (B, pred_len, N)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out


# ═══════════════════════════════════════════════════════════════════════
# 5. Config helper
# ═══════════════════════════════════════════════════════════════════════

class iTransformerConfig:
    """기존 args 에서 iTransformer 전용 필드를 읽어 오는 얇은 래퍼."""
    def __init__(self, args):
        self.seq_len    = args.seq_len
        self.pred_len   = args.pred_len
        self.c_in       = args.c_in
        self.d_model    = getattr(args, 'itrans_d_model',    512)
        self.n_heads    = getattr(args, 'itrans_n_heads',    8)
        self.e_layers   = getattr(args, 'itrans_e_layers',   3)
        self.d_ff       = getattr(args, 'itrans_d_ff',       512)
        self.dropout    = getattr(args, 'itrans_dropout',    0.1)
        self.factor     = getattr(args, 'itrans_factor',     1)
        self.activation = getattr(args, 'itrans_activation', 'gelu')
        self.use_norm   = getattr(args, 'itrans_use_norm',   True)


# ═══════════════════════════════════════════════════════════════════════
# 6. Wrapper  (기존 루프 인터페이스 호환)
# ═══════════════════════════════════════════════════════════════════════

class iTransformerWrapper(nn.Module):
    def __init__(self, cfg: iTransformerConfig):
        super().__init__()
        self._model     = _iTransformerCore(
            seq_len    = cfg.seq_len,
            pred_len   = cfg.pred_len,
            c_in       = cfg.c_in,
            d_model    = cfg.d_model,
            n_heads    = cfg.n_heads,
            e_layers   = cfg.e_layers,
            d_ff       = cfg.d_ff,
            dropout    = cfg.dropout,
            factor     = cfg.factor,
            activation = cfg.activation,
            use_norm   = cfg.use_norm,
        )
        self.pred_len   = cfg.pred_len
        self.seq_len    = cfg.seq_len
        self.train_mode = 'training'   # dummy

    def init_testing(self):
        """test.py 의 model.init_testing() 호출 대응 (no-op)."""
        self.train_mode = 'testing'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, C)  →  (B, pred_len, C)"""
        return self._model(x)


# ═══════════════════════════════════════════════════════════════════════
# 7. 팩토리 함수  (model.py LOAD_MODEL 에 등록)
# ═══════════════════════════════════════════════════════════════════════

def load_iTransformer(args, train=True):
    import os

    cfg   = iTransformerConfig(args)
    model = iTransformerWrapper(cfg)

    if not train:
        saved_path = os.path.join(
            args.save_result_path, "model_state", "best+model.pt"
        )
        state = torch.load(saved_path, map_location='cpu')
        model.load_state_dict(state)
        model = model.to(args.device)
        print(f"iTransformer loaded from {saved_path}")

    return model
