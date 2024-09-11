import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import esm.utils.constants.esm3 as C


class Flow(nn.Module):
    def __init__(self, eps):
        super(Flow, self).__init__()
        self.struc_vocab_size = len(C.VQVAE_SPECIAL_TOKENS)+C.VQVAE_CODEBOOK_SIZE
        self.seq_vocab_size = len(C.SEQUENCE_VOCAB)
        self.eps = eps

    def forward(self, structure, sequence, pad_mask=None, t=None):
        device = structure.device
        B, L = structure.size()
        if t is None:
            t = torch.rand(size=(B,), device=device)        # B

        mask_index = torch.rand(B, L).to(device) < (1. - t[:, None])   # B, L
        if pad_mask is not None:
            mask_index = mask_index & pad_mask              # only mask non-pad tokens
        
        # mask
        noised_structure = torch.where(
            mask_index, C.STRUCTURE_MASK_TOKEN, structure
        ) if structure is not None else None
        noised_seq = torch.where(
            mask_index, C.SEQUENCE_MASK_TOKEN, sequence,
        ) if sequence is not None else None

        return noised_structure, noised_seq, t
    
    def loss(self, pred_x0_logits, x_0, mask):
        B, L = x_0.size()
        ce_loss = F.cross_entropy(
            input=pred_x0_logits.view(B*L, -1),
            target=x_0.view(-1),
            reduction="none",
        )
        ce_loss = ce_loss.view(B, L)

        if mask is None:
            mask = torch.ones(B, L).to(x_0.device)        
        loss = torch.sum(ce_loss * mask) / (torch.sum(mask) + self.eps)
        
        # acc
        pred = torch.argmax(pred_x0_logits, dim=-1)
        acc = torch.sum((pred == x_0) * mask) / (torch.sum(mask) + self.eps)
        
        return loss, acc

    @torch.no_grad()
    def sample(
        self,
        denoise_func,
        strategy:str,
        length=None,
        structure=None,
        sequence=None,
        temperature=1.,
        device="cpu",
    ):
        raise NotImplementedError
    #     """
    #     strategy:
    #         "0": sequence -> structure
    #         "1": structure -> sequence
    #         "2": seq_i, struc_i -> seq_{i+1}, struc_{i+1}
    #         "3": seq_i -> struc_i -> seq_{i+1} -> struc_{i+1}
    #         "4": struc_i -> seq_i -> struc_{i+1} -> seq_{i+1}
    #     """
    #     if (structure is None) and (sequence is None):
    #         assert length is not None
    #         structure = torch.ones(1, length).long() * C.STRUCTURE_MASK_TOKEN
    #         sequence = torch.ones(1, length).long() * C.SEQUENCE_MASK_TOKEN
    #     elif structure is None:
    #         structure = torch.ones_like(sequence) * C.STRUCTURE_MASK_TOKEN
    #     elif sequence is None:
    #         sequence = torch.ones_like(structure) * C.SEQUENCE_MASK_TOKEN
    #     else:
    #         assert structure.size() == sequence.size()
    #     structure, sequence = structure.to(device), sequence.to(device)

    #     start_time = list(range(self.conf.T, 0, -1))
    #     end_time = start_time[1:] + [0]

    #     if strategy == 0:
    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Generate Sequence")

    #         # sequence generation
    #         for idx, (t1, t0) in enumerate(iterator):
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             sequence = self._sample_next_x(
    #                 logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
    #                 track="sequence", temperature=temperature
    #             )
                
    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Generate Structure")
    #         for idx, (t1, t0) in enumerate(iterator):
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             structure = self._sample_next_x(
    #                 logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
    #                 track="structure", temperature=temperature
    #             )
    #         return structure.squeeze(0), sequence.squeeze(0)
    #     elif strategy == 1:
    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Generate Structure")
    #         for idx, (t1, t0) in enumerate(iterator):
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             structure = self._sample_next_x(
    #                 logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
    #                 track="structure", temperature=temperature
    #             )

    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Generate Sequence")

    #         # sequence generation
    #         for idx, (t1, t0) in enumerate(iterator):
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             sequence = self._sample_next_x(
    #                 logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
    #                 track="sequence", temperature=temperature
    #             )
    #         return structure.squeeze(0), sequence.squeeze(0)
    #     elif strategy == 2:
    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Co Generation")
    #         for idx, (t1, t0) in enumerate(iterator):                
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             structure = self._sample_next_x(
    #                 logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
    #                 track="structure", temperature=temperature
    #             )
    #             sequence = self._sample_next_x(
    #                 logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
    #                 track="sequence", temperature=temperature
    #             )
    #         return structure.squeeze(0), sequence.squeeze(0)
    #     elif strategy == 3:
    #         iterator = tqdm(zip(start_time, end_time))
    #         iterator.set_description("Co Generation")
    #         for idx, (t1, t0) in enumerate(iterator):                
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             sequence = self._sample_next_x(
    #                 logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
    #                 track="sequence", temperature=temperature
    #             )
                
    #             struc_logits, seq_logits = denoise_func(
    #                 structure=structure, sequence=sequence,
    #                 t=torch.LongTensor([t1]).to(device)) # 1, L, N
    #             structure = self._sample_next_x(
    #                 logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
    #                 track="structure", temperature=temperature
    #             )
                
    #             # print(idx)
    #             # print(sequence.tolist())
    #             # print(structure.tolist())

    #         return structure.squeeze(0), sequence.squeeze(0)


    # def _sample_next_x(
    #     self, logits, xt, t1, t0, device, track, temperature):        

    #     if track=="sequence":
    #         vocab_size = self.conf.seq_vocab_size
    #         mask_token_index = C.SEQUENCE_MASK_TOKEN
    #         # disable invalid tokens
    #         logits[:, :, :4] = float("-inf")
    #         logits[:, :, -4:] = float("-inf")
        
    #     elif track == "structure":
    #         vocab_size = self.conf.struc_vocab_size
    #         mask_token_index = C.STRUCTURE_MASK_TOKEN
    #         logits[:, :, 4096:] = float("-inf")

    #     _, L, N = logits.size()
    #     logits = logits / temperature
    #     if t0 == 0:
    #         prob = torch.softmax(logits, dim=-1)
    #         next_x = torch.where(
    #             xt==mask_token_index,
    #             torch.multinomial(input=prob.view(L, N), num_samples=1).view(1, L),
    #             xt
    #         )
    #         return next_x
    #     else:
    #         prob = torch.softmax(logits, dim=-1)

    #     q_onestep_mat = self.absorbing_q_onestep_mat(
    #         N=vocab_size,
    #         mask_index=mask_token_index,
    #         t=torch.LongTensor([t1]).to(device)
    #     )    # 1, N, N
    #     q_mat = self.absorbing_q_mat(
    #         N=vocab_size,
    #         mask_index=mask_token_index,
    #         t=torch.LongTensor([t0]).to(device),
    #     )    # 1, N, N
        
    #     fact1 = self._at(xt, q_onestep_mat.transpose(1, 2))
    #     fact2 = prob @ q_mat    
        
    #     next_logits = torch.log(fact1*fact2+self.eps)        
    #     next_prob = torch.softmax(next_logits, dim=-1)  # 1, L, N
    #     next_x = torch.multinomial(input=next_prob.view(L, N), num_samples=1).view(1, L)   
    #     return next_x


# if __name__ == "__main__":
#     device = "cuda:0"
#     flow = Flow(eps=1.e-10).to(device)

#     sequence = torch.randint(0, 33, size=(2, 8)).to(device)
#     sequence[0, 5:] = C.SEQUENCE_PAD_TOKEN
    
#     structure = torch.randint(0, 4101, size=(2, 8)).to(device)
#     structure[0, 5:] = C.STRUCTURE_PAD_TOKEN
    
#     pad_mask = torch.ones((2, 8)).bool().to(device)
#     pad_mask[0, 5:] = False
    
#     noised_struc, noiesed_seq, t = flow(structure, sequence, pad_mask=pad_mask, 
#                                         t=torch.Tensor([0.99, 0.99]).to(device)
#                                         )
    
#     print(t)
    
#     print(structure)
#     print(noised_struc)
    
#     print()
#     print(sequence)
#     print(noiesed_seq)

#     struc_logits = torch.randn(2, 8, 4101).to(device)
#     seq_logits = torch.randn(2, 8, 33).to(device)
    
#     loss_mask = (noised_struc == C.STRUCTURE_MASK_TOKEN) & pad_mask
#     struc_loss, struc_acc = flow.loss(struc_logits, structure, loss_mask.long())
#     seq_loss, seq_acc = flow.loss(seq_logits, sequence, loss_mask.long())
    
#     print(struc_loss, struc_acc)
#     print(seq_loss, seq_acc)