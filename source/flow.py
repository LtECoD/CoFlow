import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from torch.distributions.categorical import Categorical

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
        steps:int=100,
        eta: float=0.,
        structure=None,
        sequence=None,
        sequence_temp=1.,
        structure_temp=1.,
        device="cpu",
    ):
        """
        strategy:
            "0": sequence -> structure
            "1": structure -> sequence
            "2": seq_i, struc_i -> seq_{i+1}, struc_{i+1}, sync
            "3": seq_i, struc_i -> seq_{i+1}, struc_{i+1}, async

        """
        if (structure is None) and (sequence is None):
            assert length is not None
            structure = torch.ones(1, length).long() * C.STRUCTURE_MASK_TOKEN
            sequence = torch.ones(1, length).long() * C.SEQUENCE_MASK_TOKEN
        elif structure is None:
            structure = torch.ones_like(sequence) * C.STRUCTURE_MASK_TOKEN
        elif sequence is None:
            sequence = torch.ones_like(structure) * C.SEQUENCE_MASK_TOKEN
        else:
            assert structure.size() == sequence.size()    
        structure, sequence = structure.to(device), sequence.to(device)
        
        if strategy == 0:
            gen_func = partial(self.sample_sequential, seq_first=True)
        elif strategy == 1:
            gen_func = partial(self.sample_sequential, seq_first=False)
        elif strategy == 2:
            gen_func = partial(self.sample_parallel, sync=True)
        else:
            gen_func = partial(self.sample_parallel, sync=False)
        
        structure, sequence = gen_func(
            structure=structure,
            sequence=sequence,
            denoise_func=denoise_func,
            steps=steps,
            device=device,
            sequence_temp=sequence_temp,
            structure_temp=structure_temp,
            eta=eta
        )
        
        return structure.squeeze(0), sequence.squeeze(0)        

    def sample_sequential(self, structure, sequence, denoise_func, seq_first, **kwargs):
        if seq_first:
            sequence = self._sample_sequence(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
            structure = self._sample_structure(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs
            )
        else:
            structure = self._sample_structure(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs
            )
            sequence = self._sample_sequence(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
        return structure, sequence
    
    def sample_parallel(self, structure, sequence, denoise_func, sync, **kwargs):
        dt = 1. / kwargs['steps']
        t = torch.Tensor([[0]]).to(kwargs['device'])
        desc = "Sample Both " + "Sync" if sync else "Async"
        for idx in tqdm(range(kwargs['steps']), desc=desc):
            struc_logits, seq_logits = \
                denoise_func(structure=structure, sequence=sequence, t=t)
            struc_probs = torch.softmax(struc_logits/kwargs['structure_temp'], dim=-1)
            seq_probs = torch.softmax(seq_logits/kwargs['sequence_temp'], dim=-1)
            
            if sync:
                structure, sequence, t = self._sample_next_both_sync(
                    struc_probs=struc_probs, struc_xt=structure, seq_probs=seq_probs, seq_xt=sequence,
                    dt=dt, t=t, eta=kwargs['eta'])
            else:
                structure, sequence, t = self._sample_next_both_async(
                    struc_probs=struc_probs, struc_xt=structure, seq_probs=seq_probs, seq_xt=sequence,
                    dt=dt, t=t, eta=kwargs['eta']
                )
                
        return structure, sequence
    
    def _sample_structure(self, structure, sequence, denoise_func, **kwargs):
        # generate structure
        dt = 1. / kwargs['steps']
        t = torch.Tensor([[0]]).to(kwargs['device'])
        for idx in tqdm(range(kwargs['steps']), desc="Sample Structure"):
            struc_logits, _ = \
                denoise_func(structure=structure, sequence=sequence, t=t)
            struc_probs = torch.softmax(struc_logits/kwargs['structure_temp'], dim=-1)
            structure, t = self._sample_next_single(
                probs=struc_probs, xt=structure, mask_token=C.STRUCTURE_MASK_TOKEN,
                dt=dt, t=t, eta=kwargs['eta'])
        return structure 

    def _sample_sequence(self, structure, sequence, denoise_func, **kwargs):
        dt = 1. / kwargs['steps']
        t = torch.Tensor([[0]]).to(kwargs['device'])
        for idx in tqdm(range(kwargs['steps']), desc="Sample Sequence"):
            _, seq_logits = \
                denoise_func(structure=structure, sequence=sequence, t=t)
            seq_probs = torch.softmax(seq_logits/kwargs['sequence_temp'], dim=-1)
            sequence, t = self._sample_next_single(
                probs=seq_probs, xt=sequence, mask_token=C.SEQUENCE_MASK_TOKEN,
                dt=dt, t=t, eta=kwargs['eta'])
        return sequence

    def _sample_next_single(self, probs, xt, mask_token, dt, t, eta):
        """
        probs: 1, L, D
        xt: torch.LongTensor, 1, L
        x1: torch.LongTensor, 1, L
        mask_token: int
        dt: float,
        t: torch.FloatTensor, 1, 1
        eta: float
        """
        B, L = xt.size()
        device = xt.device

        will_unmask = torch.rand(B, L, device=device) < (dt * (1+eta*t) / (1.-t+self.eps))
        will_unmask = will_unmask & (xt == mask_token)
        
        will_mask = torch.rand(B, L, device=device) < (dt * eta)
        will_mask = will_mask & (xt != mask_token)
        
        x1 = Categorical(probs).sample()
        next_xt = torch.where(will_unmask, x1, xt)
        
        next_t = t + dt
        if next_t < 1.0:
            next_xt[will_mask] = mask_token
        return next_xt, next_t
      
    def _sample_next_both_sync(self, struc_probs, struc_xt, seq_probs, seq_xt, dt, t, eta):
        B, L = struc_xt.size()
        device = struc_xt.device
        
        assert not ((struc_xt == C.STRUCTURE_MASK_TOKEN) ^ (seq_xt == C.SEQUENCE_MASK_TOKEN)).any()
        mask = struc_xt == C.STRUCTURE_MASK_TOKEN
        
        will_unmask = torch.rand(B, L, device=device) < (dt * (1+eta*t) / (1.-t+self.eps))
        will_unmask = will_unmask & mask
        
        will_mask = torch.rand(B, L, device=device) < (dt * eta)
        will_mask = will_mask & ~mask
        
        struc_x1 = Categorical(struc_probs).sample()
        seq_x1 = Categorical(seq_probs).sample()
        next_struc_xt = torch.where(will_unmask, struc_x1, struc_xt)
        next_seq_xt = torch.where(will_unmask, seq_x1, seq_xt)
        
        next_t = t + dt
        if next_t < 1.0:
            next_struc_xt[will_mask] = C.STRUCTURE_MASK_TOKEN
            next_seq_xt[will_mask]= C.SEQUENCE_MASK_TOKEN
        
        return next_struc_xt, next_seq_xt, next_t
    
    def _sample_next_both_async(self, struc_probs, struc_xt, seq_probs, seq_xt, dt, t, eta):
        next_struc_xt, _ = self._sample_next_single(
            probs=struc_probs, xt=struc_xt, mask_token=C.STRUCTURE_MASK_TOKEN,
            dt=dt, t=t, eta=eta)
        next_seq_xt, next_t = self._sample_next_single(
            probs=seq_probs, xt=seq_xt, mask_token=C.SEQUENCE_MASK_TOKEN,
            dt=dt, t=t, eta=eta
        )
        return next_struc_xt, next_seq_xt, next_t


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