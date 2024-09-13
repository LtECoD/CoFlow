import math
import torch
import einops
import functools
import torch.nn as nn
import torch.nn.functional as F

import esm.utils.constants.esm3 as C
from esm.models.esm3 import EncodeInputs
from esm.layers.regression_head import RegressionHead
from esm.layers.blocks import (
    swiglu_ln_ffn,
    gelu_ln_ffn,
)
from esm.layers.rotary import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, attn_mask):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if attn_mask is not None:
            # # Where True, enable participation in attention.
            # mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            # mask_BHLL = mask_BLL.unsqueeze(1)
            mask_BHLL = attn_mask.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)


class FourierFeaturization(nn.Module):
    def __init__(self, d_time, d_model, max_positions=2000):
        super().__init__()
        assert d_time %2 == 0, "d_time needs to be even for this featurization, try again!"
        half_dim = d_time // 2

        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb.unsqueeze(0))    # 1, D/2
        
        self.proj = nn.Linear(d_time, d_model)

    def forward(self, t):
        h = t.float() @ self.emb      # B, D/2
        h = torch.cat([h.cos(), h.sin()], dim=-1)
        tx = self.proj(h)[:, None, :]       # B, 1, D/2
        return tx


class CoFlowEncodeInputs_simplified(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Sequence
        self.sequence_embed = nn.Embedding(64, d_model)
        # Structure
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)
    
    def forward(self, sequence_tokens, structure_tokens):
        sequence_embed = self.sequence_embed(sequence_tokens)
        structure_embed = self.structure_tokens_embed(structure_tokens)
        return sequence_embed + structure_embed
        

class CoFlowEncodeInputs(EncodeInputs):
    def __init__(self, d_model):
        super().__init__(d_model=d_model)

    def forward(self, sequence_tokens, structure_tokens):
        B, L = structure_tokens.size()
        device = structure_tokens.device

        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        ss8_tokens = defaults(None, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(None, C.SASA_PAD_TOKEN)
        average_plddt = defaults(None, 1).float()
        per_res_plddt = defaults(None, 0).float()
        function_tokens = torch.full(
            (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
        )
        residue_annotation_tokens = torch.full(
            (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
        )
        embed = super().forward(
            sequence_tokens=sequence_tokens,
            structure_tokens=structure_tokens,
            average_plddt=average_plddt,
            per_res_plddt=per_res_plddt,
            ss8_tokens=ss8_tokens,
            sasa_tokens=sasa_tokens,
            function_tokens=function_tokens,
            residue_annotation_tokens=residue_annotation_tokens,
        )
        return embed


class CoFlowOutputHeads(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.structure_head = RegressionHead(d_model, 4096)
        self.sequence_head = RegressionHead(d_model, 64)
        self.struc_fill_value = -1.e10

    def forward(self, x: torch.Tensor):
        structure_logits = self.structure_head(x)
        B, L, _ = structure_logits.size()
        device = structure_logits.device
        struc_logits_padding = torch.full(
            (B, L, 5), fill_value=self.struc_fill_value).to(device)
        structure_logits = torch.cat((structure_logits, struc_logits_padding), dim=-1)
        
        sequence_logits = self.sequence_head(x)
        sequence_logits = sequence_logits[:, :, :33]
        
        return structure_logits, sequence_logits


class CoFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_time: int|None,
        n_heads: int,
        bias: bool,
        expansion_ratio: float,
        residue_scaling_factor: float,
        qk_layernorm: bool,
        ffn_type: str,  # swiglu | gelu
    ):
        super().__init__()
        self.time_embed = FourierFeaturization(d_time=d_time, d_model=d_model) \
            if d_time is not None else None
        
        self.attn = MultiHeadAttention(
            d_model, n_heads, bias, qk_layernorm=qk_layernorm
        )

        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if self.time_embed is not None:
            tx = self.time_embed(t) 
            x = x + tx
        
        r1 = self.attn(x, attn_mask)
        x = x + r1 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x


class CoFlowTransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_time: int|None,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CoFlowTransformerBlock(
                    d_model=d_model,
                    d_time=d_time,
                    n_heads=n_heads,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, t=t, attn_mask=attn_mask)
        return self.norm(x), x

