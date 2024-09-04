import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.generation import (
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper
)

import esm.utils.constants.esm3 as C


class D3PM(nn.Module):
    def __init__(self, conf):
        super(D3PM, self).__init__()
        setattr(conf, "struc_vocab_size",
                len(C.VQVAE_SPECIAL_TOKENS)+C.VQVAE_CODEBOOK_SIZE)
        setattr(conf, "seq_vocab_size", len(C.SEQUENCE_VOCAB))
        self.conf = conf
    
        self.eps = 1.e-6
        self._build_params()

    def _build_params(self):
        beta = 1. / (self.conf.T - torch.arange(self.conf.T+1) + 1)
        alpha = torch.cumprod(1.-beta, dim=-1)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
    
    def forward(self, structure, sequence, t=None):
        """the forward process"""
        device = structure.device
        B, L = structure.size()
        if t is None:
            t = torch.randint(1, self.conf.T+1, size=(B,), device=device)

        noised_structure = self.q_sample(
            x_0=structure, t=t, track="structure",
        ) if structure is not None else None

        noised_seq = self.q_sample(
            x_0=sequence, t=t, track="sequence",
        )
        return noised_structure, noised_seq, t
    
    def loss(self, pred_x0_logits, x_0, x_t, t, track, mask):        
        true_q_posterior_logits = self.q_posterior_logits(
            x_0, x_t, t, track)
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x0_logits, x_t, t, track)

        # vb loss
        vb_loss = self._vb(
            true_q_posterior_logits, pred_q_posterior_logits)
        # cross entropy loss
        B, L = x_0.size()
        ce_loss = F.cross_entropy(
            input=pred_x0_logits.view(B*L, -1),
            target=x_0.view(-1),
            reduction="none",
        )
        ce_loss = ce_loss.view(B, L)
            
        if mask is None:
            mask = torch.ones(B, L).to(x_0.device)
        reduce_fun = lambda x: torch.sum(mask * x) / torch.sum(mask)
        
        vb_loss_batch = reduce_fun(vb_loss)
        ce_loss_batch = reduce_fun(ce_loss)
        loss_batch = self.conf.hybrid_loss_coeff * vb_loss_batch + ce_loss_batch

        return {
            "loss": loss_batch,
            "vb_loss": vb_loss_batch.detach(),
            "ce_loss": ce_loss_batch.detach(),
        }

    def q_posterior_logits(self, x_0, x_t, t, track):
        """
        compute log(q(x_{t-1}|x_{t}, x_0))
        """
        if track == "structure":
            N = self.conf.struc_vocab_size
        elif track == "sequence":
            N = self.conf.seq_vocab_size
        else:
            raise ValueError(f"{track} is unknown")
        q_onestep = self.q_onestep(t, track)    # B, N, N
        q_onestep_transposed = q_onestep.transpose(1, 2)
        q_mat = self.q_mat(t-1, track)            # B, N, N

        if x_0.dim() == 2:      # token
            x_0_logits = torch.log(F.one_hot(x_0, N)+self.eps)
        else:
            x_0_logits = x_0.clone()      # B, L, N
        B, L, N = x_0_logits.size()

        fact1 = self._at(x_t, q_onestep_transposed)

        softmax_x_0 = torch.softmax(x_0_logits, dim=-1)
        fact2 = softmax_x_0 @ q_mat     # B, L, N
        out = torch.log(fact1+self.eps) + torch.log(fact2+self.eps)

        t_broadcast = t[:, None, None].repeat(1, L, N)
        bc = torch.where(t_broadcast==1, x_0_logits, out)
        return bc

    def _vb(self, dist1, dist2):
        """
        dist1: B, L, N
        dist2: B, L, N
        """
        vb_loss = torch.softmax(dist1+self.eps, dim=-1) * (
            torch.log_softmax(dist1+self.eps, dim=-1) - \
            torch.log_softmax(dist2+self.eps, dim=-1)
        )
        return vb_loss.sum(dim=-1)  # B, L  

    def _at(self, x, mat):
        """
        mat: B, N, N
        x: B, L
        """
        B = x.size(0)
        device = x.device
        return mat[torch.arange(B, device=device)[:, None], x, :]

    def q_sample(self, x_0, t, track):
        """
          forward process, x_0 is the clean input.
          x_0:   B L
          t: B
          return: B, L
        """
        B, L = x_0.size()
        q_mat = self.q_mat(t, track)
        probs = self._at(x_0, q_mat)  # B, L, N
        tokens = torch.multinomial(probs.view(B*L, -1), num_samples=1).view(B, L)
        return tokens

    def q_onestep(self, t, track):
        if track == "structure":
            return self.absorbing_q_onestep_mat(
                self.conf.struc_vocab_size,
                mask_index=C.STRUCTURE_MASK_TOKEN,
                t=t)
        elif track == "sequence":
            return self.absorbing_q_onestep_mat(
                self.conf.seq_vocab_size,
                mask_index=C.SEQUENCE_MASK_TOKEN,
                t=t)
        else:
            raise ValueError(f"{track} is unknown!")
    
    def q_mat(self, t, track):
        if track == "structure":
            return self.absorbing_q_mat(
                self.conf.struc_vocab_size,
                mask_index=C.STRUCTURE_MASK_TOKEN,
                t=t)
        elif track == "sequence":
            return self.absorbing_q_mat(
                self.conf.seq_vocab_size,
                mask_index=C.SEQUENCE_MASK_TOKEN,
                t=t)
        else:
            raise ValueError(f"{track} is unknown!")

    def absorbing_q_onestep_mat(self, N:int, mask_index:int, t: torch.LongTensor):
        device = t.device
        mat = torch.eye(N)[None, :, :].to(device) * (1. - self.beta[t]).view(-1, 1, 1)
        mat[:, :, mask_index] = mat[:, :, mask_index] + self.beta[t][:, None]
        return mat

    def absorbing_q_mat(self, N:int, mask_index:int, t: torch.LongTensor):
        device = t.device
        mat = torch.eye(N)[None, :, :].to(device) * self.alpha[t].view(-1, 1, 1)
        mat[:, :, mask_index] = mat[:, :, mask_index] + (1.-self.alpha[t])[:, None]
        return mat


# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     conf = OmegaConf.load("./config.yaml")

#     device = "cuda:0"
#     diffusion = D3PM(conf.diffusion).to(device)

#     t = torch.LongTensor([999]).to(device)
#     q_onestep_mat = diffusion.absorbing_q_onestep_mat(N=4, mask_index=3, t=t)
#     q_onestep_mat_transposed = q_onestep_mat.transpose(1, 2)
#     q_mat = diffusion.absorbing_q_mat(N=4, mask_index=3, t=t-1)

#     x_0 = torch.LongTensor([[0, 1]]).to(device)
#     x_t = torch.LongTensor([[0, 3]]).to(device)

#     fact1 = diffusion._at(x_t, q_onestep_mat_transposed)
#     fact2 = diffusion._at(x_0, q_mat)
    
#     pred_x_0 = torch.randn(1, 2, 4).to(device)
#     pred_x_0_softmaxed = torch.softmax(pred_x_0, dim=-1)
#     pred_fact2 = pred_x_0_softmaxed @ q_mat

#     # print(pred_x_0_softmaxed)
#     # print(q_mat)
#     print(fact1)
#     print(fact2)
#     print(pred_fact2)

#     print(fact1*fact2) 
#     print(fact1*pred_fact2)

#     # print(torch.softmax(fact1*fact2+1e-8, dim=-1)) 
#     # print(torch.softmax(fact1*pred_fact2+diffusion.eps, dim=-1))
