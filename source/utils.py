import torch
import numpy as np
import torch.nn.functional as F
import esm.utils.constants.esm3 as C

from esm.sdk.api import ESMProtein
from esm.utils.decoding import decode_sequence, decode_structure
from esm.tokenization import StructureTokenizer, EsmSequenceTokenizer


def strip_protein(coordinates, res_delta_threshold=12, contact_threshold=8):
    """
    coordindate: L, 37, 3
    """
    ca_coord = coordinates[:, 1, :]     # L, 3
    distance = ca_coord[:, None, :] - ca_coord[None, :, :]  # L, L, 3
    distance = distance.numpy()
    distance = np.sqrt(np.sum(distance*distance, axis=-1))
    row_idx, col_idx = np.meshgrid(np.arange(len(ca_coord)), np.arange(len(ca_coord)))
    delta_index_mask = np.abs(row_idx - col_idx) <= res_delta_threshold
    distance[delta_index_mask] = float("inf")
    contact = distance < contact_threshold
    
    contact_num = np.sum(contact, axis=-1)
    res_num = len(contact_num)
    
    # the first non-zero element
    l = np.argmax(contact_num > 0)
    l = max(l-2, 0)
    
    # the last non-zero element
    r = res_num - np.argmax(contact_num[::-1]>0) - 1
    r = min(r+2, res_num-1)
    return l, r+1


@torch.no_grad()
def to_protein(
    structure, sequence, decoder,
    struc_tokenizer: StructureTokenizer,
    seq_tokenizer: EsmSequenceTokenizer,
    pad=True, allow_mask=False, strip=True
):  
    if pad:
        structure = F.pad(structure, pad=(1, 1), value=0)
        structure[0] = C.STRUCTURE_BOS_TOKEN
        structure[-1] = C.STRUCTURE_EOS_TOKEN
        
        sequence = F.pad(sequence, pad=(1, 1), value=0)
        sequence[0] = C.SEQUENCE_BOS_TOKEN
        sequence[-1] = C.SEQUENCE_EOS_TOKEN
    
    if torch.any(structure==C.STRUCTURE_PAD_TOKEN) or \
        torch.any(sequence==C.SEQUENCE_PAD_TOKEN):
            raise ValueError
    if not allow_mask:
        if torch.any(structure==C.STRUCTURE_MASK_TOKEN) or \
            torch.any(sequence==C.SEQUENCE_MASK_TOKEN):
                raise ValueError

    sequence = decode_sequence(sequence, seq_tokenizer)
    plddt, ptm = None, None
    coordinates, plddt, ptm = decode_structure(
        structure_tokens=structure,
        structure_decoder=decoder,
        structure_tokenizer=struc_tokenizer,
        sequence=sequence,
    )
    
    if strip:
        l, r = strip_protein(coordinates)
        sequence = sequence[l:r]
        coordinates = coordinates[l:r, ...]
        plddt = plddt[l: r]
    
    protein = ESMProtein(
        sequence=sequence,
        coordinates=coordinates,
        plddt=plddt,
        ptm=ptm,
    )

    return protein
