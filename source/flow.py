import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from torch.distributions.categorical import Categorical

import esm.utils.constants.esm3 as C


class Flow(nn.Module):
    def __init__(self, train_async, eps):
        super(Flow, self).__init__()
        self.struc_vocab_size = len(C.VQVAE_SPECIAL_TOKENS)+C.VQVAE_CODEBOOK_SIZE
        self.seq_vocab_size = len(C.SEQUENCE_VOCAB)
        self.train_async = train_async
        self.eps = eps

    def forward(self, structure, sequence, t=None):
        device = structure.device
        B, L = structure.size()
        if t is None:
            t = torch.rand(size=(B,), device=device)        # B
        pad_mask = structure != C.STRUCTURE_PAD_TOKEN
        assert not (pad_mask ^ (sequence != C.SEQUENCE_PAD_TOKEN)).any()

        if self.train_async:
            seq_mask_index = torch.rand(B, L).to(device) < (1. - t[:, None])
            struc_mask_index = torch.rand(B, L).to(device) < (1. - t[:, None])
            if pad_mask is not None:
                seq_mask_index = seq_mask_index & pad_mask
                struc_mask_index = struc_mask_index & pad_mask
        else:
            mask_index = torch.rand(B, L).to(device) < (1. - t[:, None])   # B, L
            if pad_mask is not None:
                mask_index = mask_index & pad_mask              # only mask non-pad tokens
            seq_mask_index = struc_mask_index = mask_index
        
        # mask
        noised_structure = torch.where(
            struc_mask_index, C.STRUCTURE_MASK_TOKEN, structure)
        noised_seq = torch.where(
            seq_mask_index, C.SEQUENCE_MASK_TOKEN, sequence)

        return noised_structure, noised_seq, t
    
    def loss(self, pred_x0_logits, x_0, mask):
        B, L = x_0.size()
        ce_loss = F.cross_entropy(
            input=pred_x0_logits.view(B*L, -1),
            target=x_0.view(-1),
            reduction="none",
        )
        ce_loss = ce_loss.view(B, L)
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
        purity: bool=False,
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
            "2": i>i+1
            "3": seq_i -> seq_{i+1}, struc_i -> struc_{i+1}

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
        structure_mask = structure == C.STRUCTURE_MASK_TOKEN
        sequence_mask = sequence == C.SEQUENCE_MASK_TOKEN
        
        if strategy == 0:
            gen_func = partial(self.sample_sequential, seq_first=True)
        elif strategy == 1:
            gen_func = partial(self.sample_sequential, seq_first=False)
        elif strategy == 2:
            gen_func = partial(self.sample_parallel, joint=True)
        elif strategy == 3:
            gen_func = partial(self.sample_parallel, joint=False)
        else:
            raise NotImplementedError

        structure, sequence = gen_func(
            structure=structure,
            sequence=sequence,
            denoise_func=denoise_func,
            steps=steps,
            device=device,
            sequence_temp=sequence_temp,
            structure_temp=structure_temp,
            eta=eta,
            purity=purity,
        )

        # estimate probs
        struc_logits, seq_logits = \
            denoise_func(structure=structure, sequence=sequence, t=torch.Tensor([[1]]).to(device))
        struc_probs = torch.softmax(struc_logits, dim=-1)
        seq_probs = torch.softmax(seq_logits, dim=-1)
        
        
        struc_probs[~structure_mask, :] = \
            F.one_hot(structure, num_classes=struc_probs.size(-1))[~structure_mask, :].float()
        seq_probs[~sequence_mask, :] = \
            F.one_hot(sequence, num_classes=seq_probs.size(-1))[~sequence_mask, :].float()

        return {
            "sequence": sequence.squeeze(0),
            "structure": structure.squeeze(0),
            "sequence_prob": seq_probs.squeeze(0),
            "structure_prob": struc_probs.squeeze(0),
        }

    def sample_sequential(self, structure, sequence, denoise_func, seq_first, **kwargs):
        if seq_first:
            sequence = self._sample_sequence(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
            structure = self._sample_structure(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
        else:
            structure = self._sample_structure(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
            sequence = self._sample_sequence(
                structure=structure, sequence=sequence, denoise_func=denoise_func, **kwargs)
        return structure, sequence

    def sample_parallel(self, structure, sequence, denoise_func, joint, **kwargs):        
        desc = "Sample Parallel " + ("Joint" if joint else "Peroidical")
        for idx in tqdm(range(kwargs['steps']), desc=desc):
            t = torch.Tensor([[idx/kwargs['steps']]]).to(kwargs['device'])
            
            if joint:
                struc_logits, seq_logits = \
                    denoise_func(structure=structure, sequence=sequence, t=t)
                struc_probs = torch.softmax(struc_logits/kwargs['structure_temp'], dim=-1)
                seq_probs = torch.softmax(seq_logits/kwargs['sequence_temp'], dim=-1)
            
                structure, sequence = self._sample_next_joint(
                    struc_probs=struc_probs, struc_xt=structure, seq_probs=seq_probs, seq_xt=sequence,
                    N=kwargs["steps"], step=idx, eta=kwargs['eta'], purity=kwargs['purity'])
            else:
                _, seq_logits = denoise_func(structure=structure, sequence=sequence, t=t)
                seq_probs = torch.softmax(seq_logits/kwargs['sequence_temp'], dim=-1)
                sequence = self._sample_next_single(
                    probs=seq_probs, xt=sequence, mask_token=C.SEQUENCE_MASK_TOKEN,
                    N=kwargs["steps"], step=idx, eta=kwargs['eta'], purity=kwargs['purity']
                )

                struc_logits, _ = denoise_func(structure=structure, sequence=sequence, t=t)
                struc_probs = torch.softmax(struc_logits/kwargs['structure_temp'], dim=-1)
                structure = self._sample_next_single(
                    probs=struc_probs, xt=structure, mask_token=C.STRUCTURE_MASK_TOKEN,
                    N=kwargs["steps"], step=idx, eta=kwargs['eta'], purity=kwargs['purity']
                )
 
        return structure, sequence
    
    def _sample_structure(self, structure, sequence, denoise_func, **kwargs):
        # generate structure
        for idx in tqdm(range(kwargs['steps']), desc="Sample Structure"):
            t = torch.Tensor([[idx/kwargs['steps']]]).to(kwargs['device'])
            struc_logits, _ = \
                denoise_func(structure=structure, sequence=sequence, t=t)
            struc_probs = torch.softmax(struc_logits/kwargs['structure_temp'], dim=-1)
            structure = self._sample_next_single(
                probs=struc_probs, xt=structure, mask_token=C.STRUCTURE_MASK_TOKEN,
                N=kwargs["steps"], step=idx, eta=kwargs['eta'], purity=kwargs['purity'])
        return structure 

    def _sample_sequence(self, structure, sequence, denoise_func, **kwargs):
        for idx in tqdm(range(kwargs['steps']), desc="Sample Sequence"):
            t = torch.Tensor([[idx/kwargs['steps']]]).to(kwargs['device'])
            _, seq_logits = \
                denoise_func(structure=structure, sequence=sequence, t=t)
            seq_probs = torch.softmax(seq_logits/kwargs['sequence_temp'], dim=-1)
            sequence = self._sample_next_single(
                probs=seq_probs, xt=sequence, mask_token=C.SEQUENCE_MASK_TOKEN,
                N=kwargs["steps"], step=idx, eta=kwargs['eta'], purity=kwargs['purity'])
        return sequence

    def _sample_next_single(self, probs, xt, mask_token, N, step, eta, purity=False):
        """
        probs: 1, L, D
        xt: torch.LongTensor, 1, L
        x1: torch.LongTensor, 1, L
        mask_token, N, step: int
        eta: float
        """
        B, L = xt.size()
        device = xt.device
        
        dt = 1. / N
        t = step / N

        if purity:
            will_unmask_num = math.ceil(L * dt * (1+eta*t) / (1.-t))
            will_unmask_num = min(L, will_unmask_num)
            will_unmask = build_single_mask_from_entropy(
                probs, k=will_unmask_num, mask=(xt==mask_token), largest=False)
        else:
            will_unmask = torch.rand(B, L, device=device) < (dt * (1+eta*t) / (1.-t))
        will_mask = torch.rand(B, L, device=device) < (dt * eta)

        will_unmask = will_unmask & (xt == mask_token)
        will_mask = will_mask & (xt != mask_token)
        
        x1 = Categorical(probs).sample()
        next_xt = torch.where(will_unmask, x1, xt)
        
        if (step + 1) < N:
            next_xt[will_mask] = mask_token
        return next_xt
      
    def _sample_next_joint(self, struc_probs, struc_xt, seq_probs, seq_xt, N, step, eta, purity):
        B, L = struc_xt.size()
        device = struc_xt.device
        
        dt = 1. / N
        t = step / N
                
        joint_mask = torch.cat(
            (struc_xt == C.STRUCTURE_MASK_TOKEN, seq_xt == C.SEQUENCE_MASK_TOKEN), dim=-1)      # B, 2L
        
        if purity:
            will_unmask_num = math.ceil(2*L * dt * (1+eta*t) / (1.-t))
            will_unmask_num = min(2*L, will_unmask_num)
            will_unmask = build_joint_mask_from_entropy(
                struc_probs=struc_probs, seq_probs=seq_probs, k=will_unmask_num,
                mask=joint_mask, largest=False)
        else:
            will_unmask = torch.rand(B, 2*L, device=device) < (dt * (1+eta*t) / (1.-t))
        will_mask = torch.rand(B, 2*L, device=device) < (dt * eta)
            
        will_unmask = will_unmask & joint_mask
        will_mask = will_mask & ~joint_mask
        
        struc_will_unmask, seq_will_unmask = will_unmask.split(L, dim=-1)
        struc_will_mask, seq_will_mask = will_mask.split(L, dim=-1)
        
        struc_x1 = Categorical(struc_probs).sample()
        seq_x1 = Categorical(seq_probs).sample()
        
        next_struc_xt = torch.where(struc_will_unmask, struc_x1, struc_xt)
        next_seq_xt = torch.where(seq_will_unmask, seq_x1, seq_xt)
        
        if (step + 1) < N:
            next_struc_xt[struc_will_mask] = C.STRUCTURE_MASK_TOKEN
            next_seq_xt[seq_will_mask]= C.SEQUENCE_MASK_TOKEN
        
        return next_struc_xt, next_seq_xt


def build_joint_mask_from_entropy(struc_probs, seq_probs, k, mask, eps=1e-10, largest=False):
    """
    For purity unmask
    """
    B, L, _ = seq_probs.size()
    device = struc_probs.device
    
    seq_entropy = torch.sum(
        -1. * seq_probs * torch.log(seq_probs+eps), dim=-1)
    struc_entropy = torch.sum(
        -1 * struc_probs * torch.log(struc_probs+eps), dim=-1)
    entropy = torch.cat((seq_entropy, struc_entropy), dim=-1)       # B, 2L
    
    entropy[~mask] = float("-inf") if largest else float("inf")
    
    values, indices = torch.topk(entropy, dim=-1, k=k, largest=largest) # B, K
    topk_mask = torch.zeros_like(entropy).bool()   # B, L
    topk_mask[torch.arange(B).to(device)[:, None], indices] = True
    return topk_mask


def build_single_mask_from_entropy(probs, k, mask, eps=1e-10, largest=False):
    """
    For purity unmask
    probs: B, L, D
    """
    B, L, _ = probs.size()
    device = probs.device
    
    log_probs = torch.log(probs+eps)
    entropy = torch.sum(-1. * probs * log_probs, dim=-1)
    entropy[~mask] = float("-inf") if largest else float("inf")

    values, indices = torch.topk(entropy, dim=-1, k=k, largest=largest) # B, K
    topk_mask = torch.zeros_like(entropy).bool()   # B, L
    topk_mask[torch.arange(B).to(device)[:, None], indices] = True
    return topk_mask


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