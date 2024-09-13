import torch
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig, PreTrainedModel

import esm.utils.constants.esm3 as C

from module import (
    CoFlowEncodeInputs_simplified,
    CoFlowEncodeInputs,
    CoFlowTransformerStack,
    CoFlowOutputHeads,
)
from flow import Flow


class CoFlowConfig(PretrainedConfig):
    model_type = "coflowconfig"
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class CoFlowModel(PreTrainedModel):
    config_class = CoFlowConfig
    supports_gradient_checkpointing = False

    def __init__(self, conf):        
        super(CoFlowModel, self).__init__(conf)
        
        encoder_class = CoFlowEncodeInputs_simplified \
            if getattr(conf, "simplified_encoder", False) else CoFlowEncodeInputs

        self.encoder = encoder_class(d_model=conf.d_model)
        self.transformer = CoFlowTransformerStack(
            d_model=conf.d_model,
            d_time=conf.d_time,
            n_heads=conf.n_heads,
            n_layers=conf.n_layers,
        )
        self.output_heads = CoFlowOutputHeads(
            d_model=conf.d_model,
            structure_track=conf.structure_track,
            sequence_track=conf.sequence_track,
        )
        
        self.flow = Flow(train_async=getattr(conf, "train_async", False), eps=conf.eps)
        self.sample = lambda **kwargs: self.flow.sample(
            denoise_func=self.denoise, **kwargs)

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
        mask = structure != C.STRUCTURE_PAD_TOKEN
        
        noised_structure, noised_sequence, t = self.flow(
            structure=structure,
            sequence=sequence,
        )
        structure_logits, sequence_logits = self.denoise(
            structure=noised_structure,
            sequence=noised_sequence,
            t=t[:, None],
            mask=mask.long(),
        )

        # loss
        if self.config.structure_track:
            struc_mask = (noised_structure == C.STRUCTURE_MASK_TOKEN) & mask
            struc_loss, struc_acc = self.flow.loss(
                pred_x0_logits=structure_logits,
                x_0=structure,
                mask=struc_mask.long(),
            )
        else:
            struc_loss = struc_acc = 0.

        if self.config.sequence_track:
            seq_mask = (noised_sequence == C.SEQUENCE_MASK_TOKEN) & mask
            seq_loss, seq_acc = self.flow.loss(
                pred_x0_logits=sequence_logits,
                x_0=sequence,
                mask=seq_mask.long(),
            )
        else:
            seq_loss = seq_acc = 0.
        
        if self.config.structure_track and self.config.sequence_track:
            loss = struc_loss + seq_loss
        elif self.config.structure_track:
            loss = struc_loss
        elif self.config.sequence_track:
            loss = seq_loss
        else:
            raise ValueError

        return CoFlowOutput(
            loss=loss,
            struc_loss=struc_loss,
            struc_acc=struc_acc,
            seq_loss=seq_loss,
            seq_acc=seq_acc,
            t=t,
        )


@dataclass
class CoFlowOutput(ModelOutput):
    loss: torch.FloatTensor = None
    struc_loss: torch.Tensor = None
    struc_acc: torch.Tensor = None
    seq_loss: torch.Tensor = None
    seq_acc: torch.Tensor = None
    t: torch.FloatTensor = None
    

# if __name__ == "__main__":
#     from torch.optim import Adam
#     from torch.utils.data import DataLoader
#     from omegaconf import OmegaConf
#     from data import ProDataset
    
#     model = CoFlowModel(
#         conf=CoFlowConfig(**OmegaConf.create({
#             "sequence_track": True,
#             "structure_track": True,
#             "d_time": 128,
#             "d_model": 256,
#             "n_layers": 8,
#             "n_heads": 4,
#             "eta": 5,
#             "eps": 1.e-10
#         })),
#     )
#     optim = Adam(model.parameters(), lr=0.0001)
#     dataset = ProDataset(
#         OmegaConf.create({
#             "struc_file": "./data/struc_example.txt",
#             "seq_file": "./data/seq_example.txt"    
#     }))
#     loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)
#     for data in loader:
#         name, structure, sequence = data['name'], data['structure'], data['sequence']
#         out = model(structure=structure, sequence=sequence, name=name)
#         loss = out.loss
#         loss.backward()
    
#         for name, param in model.output_heads.named_parameters():
#             print(name, param.grad)
        
#         optim.step()
#         exit()