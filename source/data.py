import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

from esm.utils.constants.esm3 import (
    STRUCTURE_PAD_TOKEN,
    SEQUENCE_PAD_TOKEN,
    STRUCTURE_MASK_TOKEN,
    SEQUENCE_MASK_TOKEN,
    SEQUENCE_VOCAB,
)
seq_vocab = {aa: idx for idx, aa in enumerate(SEQUENCE_VOCAB)}


@dataclass
class ProteinToken:
    name: str
    structure: list
    sequence: list


class ProDataset(Dataset):
    def __init__(self, config):
        self.config = config
        struc_dict = {}
        if getattr(config, "struc_file", None) is not None:
            struc_lines = open(self.config.struc_file, "r").readlines()
            for l in struc_lines:
                items = l.strip().split(" ")
                struc_dict[items[0]] = list(map(int, items[1:]))
        
        seq_dict = {}
        if getattr(config, "seq_file", None) is not None:
            seq_lines = open(self.config.seq_file, "r").readlines()
            for l in seq_lines:
                items = l.strip().split(" ")
                seq_dict[items[0]] = [seq_vocab[aa] for aa in items[1]]
        assert len(struc_dict) + len(seq_dict) > 0
    
        self.data = []
        names = struc_dict.keys() if len(struc_dict) > 0 else seq_dict.keys()
        for name in names:
            struc_tokens = struc_dict.get(name, [])
            seq_tokens = seq_dict.get(name, [])
            if (len(struc_tokens) > 0) and (len(seq_tokens) > 0):
                assert len(struc_tokens) == len(seq_tokens)
            length = len(struc_tokens) or len(seq_tokens)
            
            self.data.append(ProteinToken(
                name=name,
                structure=struc_dict.get(name, [STRUCTURE_MASK_TOKEN] * length),
                sequence=seq_dict.get(name, [SEQUENCE_MASK_TOKEN] * length)
            ))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        protein = self.data[index]
        return protein

    def collate(self, features):
        struc_lens = [len(p.structure) if p.structure is not None else 0 for p in features]
        seq_lens = [len(p.sequence) if p.sequence is not None else 0 for p in features]        
        max_len = max(max(struc_lens), max(seq_lens))

        names, struc_tokens, seq_tokens = [], [], []
        for idx, protein in enumerate(features):
            names.append(protein.name)            
            struc_tokens.append(_pad_token(protein.structure, max_len, STRUCTURE_PAD_TOKEN))
            seq_tokens.append(_pad_token(protein.sequence, max_len, SEQUENCE_PAD_TOKEN))

        return {
            "name": names,
            "structure": torch.cat(struc_tokens, dim=0),
            "sequence": torch.cat(seq_tokens, dim=0)
        }


def _pad_token(token, max_len, pad_token):
    delta_length = max_len - len(token)
    return torch.cat((
        torch.LongTensor([token]),
        torch.LongTensor([[pad_token]*delta_length])
    ), dim=-1)


# just for test
# if __name__ == "__main__":
#     def test_dataloader():
#         for idx, data in enumerate(loader):
#             names, structure, sequence = data['names'], data['structure'],  data['sequence']
#             print(names)
#             print(structure)
#             print(sequence)
#             break

#     def length_statistics():
#         from collections import Counter
#         from tqdm import tqdm

#         lengths = []
#         for idx in tqdm(range(len(dataset))):            
#             protein = dataset[idx]
#             lengths.append(len(protein.structure))
            
#         lengths = Counter(lengths)
#         print(sorted(list(lengths.keys())))
    
#     from omegaconf import OmegaConf
#     from torch.utils.data import DataLoader
#     conf = OmegaConf.load("./config.yaml")
#     dataset = ProDataset(conf.data)

#     loader = DataLoader(
#         dataset=dataset,
#         collate_fn=dataset.collate,
#         batch_size=2,
#     )
    
#     length_statistics()
#     test_dataloader()