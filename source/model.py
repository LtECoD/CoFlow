import torch
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig, PreTrainedModel

import esm.utils.constants.esm3 as C

from diffusion import D3PM
from module import (
    CoDiffEncodeInputs,
    CoDiffTransformerStack,
    CoDiffOutputHeads,
)


class CoDiffConfig(PretrainedConfig):
    model_type = "codiffconfig"
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class CoDiffNetwork(PreTrainedModel):
    config_class = CoDiffConfig
    supports_gradient_checkpointing = False

    def __init__(self, conf, d3pm: D3PM):
        super(CoDiffNetwork, self).__init__(conf)

        self.encoder = CoDiffEncodeInputs(d_model=conf.d_model)
        self.transformer = CoDiffTransformerStack(
            d_model=conf.d_model,
            d_time=conf.d_time,
            n_heads=conf.n_heads,
            n_layers=conf.n_layers,
        )
        self.output_heads = CoDiffOutputHeads(
            d_model=conf.d_model,
            structure_track=conf.structure_track,
            sequence_track=conf.sequence_track,
        )
        self.diffusion = d3pm

        # self.sample = lambda **kwargs: self.gen_model.sample(
        #     denoise_func=self.denoise, **kwargs)

    def denoise(self, structure, sequence, t, mask=None):
        B, L = structure.size()
        device = structure.device
        if mask is None:
            mask = torch.ones(B, L).long().to(device)
        
        x = self.encoder(
            structure_tokens=structure,
            sequence_tokens=sequence,
        )
        x, embedding = self.transformer(
            x=x, t=t, sequence_id=mask
        )
        
        structure_logits, sequence_logits = self.output_heads(x)
        
        return structure_logits, sequence_logits

    def forward(self, structure, sequence, name, return_loss=True):
        B, L = structure.size()
        mask = (structure != C.STRUCTURE_PAD_TOKEN).long()
        
        noised_structure, noised_sequence, t = self.diffusion(
            structure=structure,
            sequence=sequence,
        )
        
        structure_logits, sequence_logits = self.denoise(
            structure=noised_structure,
            sequence=noised_sequence,
            t=t[:, None],
            mask=mask,
        )
        
        # loss
        if self.config.structure_track:
            structure_loss_items = self.diffusion.loss(
                pred_x0_logits=structure_logits,
                x_0=structure,
                x_t=noised_structure,
                t=t,
                track="structure",
                mask=mask,
            )
            struc_loss, struc_vb_loss, struc_ce_loss = structure_loss_items["loss"], \
                structure_loss_items["vb_loss"], structure_loss_items["ce_loss"]
            
            # acc
            struc_pred = torch.argmax(structure_logits, dim=-1)
            struc_mask = ((noised_structure == C.STRUCTURE_MASK_TOKEN) + mask) == 2
            struc_acc = torch.sum((struc_pred == structure) * struc_mask) / torch.sum(struc_mask)

        else:
            struc_loss, struc_vb_loss, struc_ce_loss = 0., 0., 0.
            struc_acc = 0.
        
        if self.config.sequence_track:
            sequence_loss_items = self.diffusion.loss(
                pred_x0_logits=sequence_logits,
                x_0=sequence,
                x_t=noised_sequence,
                t=t,
                track="sequence",
                mask=mask,
            )
            seq_loss, seq_vb_loss, seq_ce_loss = sequence_loss_items['loss'], \
                sequence_loss_items['vb_loss'], sequence_loss_items['ce_loss']
            
            # acc
            seq_pred = torch.argmax(sequence_logits, dim=-1)
            seq_mask = ((noised_sequence == C.SEQUENCE_MASK_TOKEN) + mask) == 2
            seq_acc = torch.sum((seq_pred == sequence) * seq_mask) / torch.sum(seq_mask)

        else:
            seq_loss, seq_vb_loss, seq_ce_loss = 0., 0., 0.
            seq_acc = 0.
        
        if self.config.structure_track and self.config.sequence_track:
            loss = struc_loss + seq_loss
        elif self.config.structure_track:
            loss = struc_loss
        elif self.config.sequence_track:
            loss =seq_loss
        else:
            raise ValueError

        return CoDiffOutput(
            loss=loss,
            struc_vb_loss=struc_vb_loss,
            struc_ce_loss=struc_ce_loss,
            struc_acc = struc_acc,
            seq_vb_loss=seq_vb_loss,
            seq_ce_loss=seq_ce_loss,
            seq_acc = seq_acc,
            t=t,
        )


@dataclass
class CoDiffOutput(ModelOutput):
    loss: torch.FloatTensor = None
    struc_vb_loss: torch.Tensor = None
    struc_ce_loss: torch.Tensor = None
    struc_acc: torch.Tensor = None
    seq_vb_loss: torch.Tensor = None
    seq_ce_loss: torch.Tensor = None
    seq_acc: torch.Tensor = None
    t: torch.FloatTensor = None
    

# if __name__ == "__main__":
#     from torch.optim import Adam
#     from torch.utils.data import DataLoader
#     from omegaconf import OmegaConf
#     from data import ProDataset
    
#     d3pm = D3PM(OmegaConf.create({"T": 1000, "hybrid_loss_coeff": 0.}))
#     model = CoDiffNetwork(
#         conf=CoDiffConfig(**OmegaConf.create({
#             "sequence_track": True,
#             "structure_track": True,
#             "d_time": 128,
#             "d_model": 256,
#             "n_layers": 8,
#             "n_heads": 4,
#         })),
#         d3pm=d3pm,
#     )
#     optim = Adam(model.parameters(), lr=0.0001)
#     dataset = ProDataset(
#         OmegaConf.create({
#             "struc_file": "./data/struc.txt",
#             "seq_file": "./data/seq.txt"    
#     }))
#     loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)
#     for data in loader:
#         name, structure, sequence = data['name'], data['structure'], data['sequence']
#         out = model(structure=structure, sequence=sequence, name=name)
#         loss = out.loss
#         loss.backward()
    
#         for name, param in model.encoder.structure_tokens_embed.named_parameters():
#             print(name, param.grad)
#             break
#         optim.step()
#         exit()