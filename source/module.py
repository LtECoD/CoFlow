import math
import torch
import torch.nn as nn

import esm.utils.constants.esm3 as C
from esm.models.esm3 import EncodeInputs
from esm.layers.regression_head import RegressionHead
from esm.layers.blocks import (
    MultiHeadAttention,
    swiglu_ln_ffn,
    gelu_ln_ffn,
)


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
    def __init__(self, d_model, sequence_track, structure_track):
        super().__init__()
        self.structure_head = RegressionHead(d_model, 4096) if structure_track else None
        self.sequence_head = RegressionHead(d_model, 64) if sequence_track else None
        self.struc_fill_value = -1.e10

    def forward(self, x: torch.Tensor):
        structure_logits = self.structure_head(x) if self.structure_head else None
        sequence_logits = self.sequence_head(x) if self.sequence_head else None
        
        if structure_logits is not None:
            B, L, _ = structure_logits.size()
            device = structure_logits.device
            struc_logits_padding = torch.full(
                (B, L, 5), fill_value=self.struc_fill_value).to(device)
            structure_logits = torch.cat((structure_logits, struc_logits_padding), dim=-1)
        
        if sequence_logits is not None:
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
        sequence_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.time_embed is not None:
            tx = self.time_embed(t) 
            x = x + tx
        
        r1 = self.attn(x, sequence_id)
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
        sequence_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, t=t, sequence_id=sequence_id)
        return self.norm(x), x

