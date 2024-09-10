import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import esm.utils.constants.esm3 as C


class D3PM(nn.Module):
    def __init__(self, conf):
        super(D3PM, self).__init__()
        setattr(conf, "struc_vocab_size",
                len(C.VQVAE_SPECIAL_TOKENS)+C.VQVAE_CODEBOOK_SIZE)
        setattr(conf, "seq_vocab_size", len(C.SEQUENCE_VOCAB))
        self.conf = conf
    
        self.eps = 1.e-10
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
        softmax_x_0 = torch.softmax(x_0_logits, dim=-1)

        fact1 = self._at(x_t, q_onestep_transposed)
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

    @torch.no_grad()
    def reverse(
        self,
        denoise_func,
        strategy:str,
        length=None,
        structure=None,
        sequence=None,
        temperature=1.,
        device="cpu",
    ):
        """
        strategy:
            "0": sequence -> structure
            "1": structure -> sequence
            "2": seq_i, struc_i -> seq_{i+1}, struc_{i+1}
            "3": seq_i -> struc_i -> seq_{i+1} -> struc_{i+1}
            "4": struc_i -> seq_i -> struc_{i+1} -> seq_{i+1}
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

        start_time = list(range(self.conf.T, 0, -1))
        end_time = start_time[1:] + [0]

        if strategy == 0:
            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Generate Sequence")

            # sequence generation
            for idx, (t1, t0) in enumerate(iterator):
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                sequence = self._sample_next_x(
                    logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
                    track="sequence", temperature=temperature
                )
                
            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Generate Structure")
            for idx, (t1, t0) in enumerate(iterator):
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                structure = self._sample_next_x(
                    logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
                    track="structure", temperature=temperature
                )
            return structure.squeeze(0), sequence.squeeze(0)
        elif strategy == 1:
            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Generate Structure")
            for idx, (t1, t0) in enumerate(iterator):
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                structure = self._sample_next_x(
                    logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
                    track="structure", temperature=temperature
                )

            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Generate Sequence")

            # sequence generation
            for idx, (t1, t0) in enumerate(iterator):
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                sequence = self._sample_next_x(
                    logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
                    track="sequence", temperature=temperature
                )
            return structure.squeeze(0), sequence.squeeze(0)
        elif strategy == 2:
            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Co Generation")
            for idx, (t1, t0) in enumerate(iterator):                
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                structure = self._sample_next_x(
                    logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
                    track="structure", temperature=temperature
                )
                sequence = self._sample_next_x(
                    logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
                    track="sequence", temperature=temperature
                )
            return structure.squeeze(0), sequence.squeeze(0)
        elif strategy == 3:
            iterator = tqdm(zip(start_time, end_time))
            iterator.set_description("Co Generation")
            for idx, (t1, t0) in enumerate(iterator):                
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                sequence = self._sample_next_x(
                    logits=seq_logits, xt=sequence, t1=t1, t0=t0, device=device,
                    track="sequence", temperature=temperature
                )
                
                struc_logits, seq_logits = denoise_func(
                    structure=structure, sequence=sequence,
                    t=torch.LongTensor([t1]).to(device)) # 1, L, N
                structure = self._sample_next_x(
                    logits=struc_logits, xt=structure, t1=t1, t0=t0, device=device,
                    track="structure", temperature=temperature
                )
                
                # print(idx)
                # print(sequence.tolist())
                # print(structure.tolist())

            return structure.squeeze(0), sequence.squeeze(0)


    def _sample_next_x(
        self, logits, xt, t1, t0, device, track, temperature):        

        if track=="sequence":
            vocab_size = self.conf.seq_vocab_size
            mask_token_index = C.SEQUENCE_MASK_TOKEN
            # disable invalid tokens
            logits[:, :, :4] = float("-inf")
            logits[:, :, -4:] = float("-inf")
        
        elif track == "structure":
            vocab_size = self.conf.struc_vocab_size
            mask_token_index = C.STRUCTURE_MASK_TOKEN
            logits[:, :, 4096:] = float("-inf")

        _, L, N = logits.size()
        logits = logits / temperature
        if t0 == 0:
            prob = torch.softmax(logits, dim=-1)
            next_x = torch.where(
                xt==mask_token_index,
                torch.multinomial(input=prob.view(L, N), num_samples=1).view(1, L),
                xt
            )
            return next_x
        else:
            prob = torch.softmax(logits, dim=-1)

        q_onestep_mat = self.absorbing_q_onestep_mat(
            N=vocab_size,
            mask_index=mask_token_index,
            t=torch.LongTensor([t1]).to(device)
        )    # 1, N, N
        q_mat = self.absorbing_q_mat(
            N=vocab_size,
            mask_index=mask_token_index,
            t=torch.LongTensor([t0]).to(device),
        )    # 1, N, N
        
        fact1 = self._at(xt, q_onestep_mat.transpose(1, 2))
        fact2 = prob @ q_mat    
        
        next_logits = torch.log(fact1*fact2+self.eps)        
        next_prob = torch.softmax(next_logits, dim=-1)  # 1, L, N
        next_x = torch.multinomial(input=next_prob.view(L, N), num_samples=1).view(1, L)   
        return next_x


# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     device = "cuda:0"
#     diffusion = D3PM(OmegaConf.create({
#         "T": 1000,
#         "hybrid_loss_coeff": 0.
#     })).to(device)

#     t = torch.LongTensor([999]).to(device)
#     q_onestep_mat = diffusion.absorbing_q_onestep_mat(N=4, mask_index=3, t=t)
#     q_onestep_mat_transposed = q_onestep_mat.transpose(1, 2)
#     q_mat = diffusion.absorbing_q_mat(N=4, mask_index=3, t=t-1)
#     q_multistep_mat = diffusion.absorbing_q_multistep_mat(
#         N=4, mask_index=3, start=1, end=998, device=device)
    
#     print(q_mat)
#     print(q_multistep_mat)
    

    # x_0 = torch.LongTensor([[0, 1]]).to(device)
    # x_t = torch.LongTensor([[0, 3]]).to(device)

    # fact1 = diffusion._at(x_t, q_onestep_mat_transposed)
    # fact2 = diffusion._at(x_0, q_mat)
    
    # pred_x_0 = torch.randn(1, 2, 4).to(device)
    # pred_x_0_softmaxed = torch.softmax(pred_x_0, dim=-1)
    # pred_fact2 = pred_x_0_softmaxed @ q_mat

    # # print(pred_x_0_softmaxed)
    # # print(q_mat)
    # print(fact1)
    # print(fact2)
    # print(pred_fact2)

    # print(fact1*fact2) 
    # print(fact1*pred_fact2)

    # # print(torch.softmax(fact1*fact2+1e-8, dim=-1)) 
    # # print(torch.softmax(fact1*pred_fact2+diffusion.eps, dim=-1))
