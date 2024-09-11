import os
import torch
import random
from torch.utils.data import IterableDataset

from esm.utils.constants.esm3 import (
    STRUCTURE_PAD_TOKEN,
    SEQUENCE_PAD_TOKEN,
    SEQUENCE_VOCAB,
)
seq_vocab = {aa: idx for idx, aa in enumerate(SEQUENCE_VOCAB)}


class ProDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.file_pairs = []
        for _prefix in self.config.prefix:
            self.file_pairs.append((
                os.path.join(self.config.dir, f"{_prefix}_struc.txt"),
                os.path.join(self.config.dir, f"{_prefix}_seq.txt")
            ))

    def __len__(self):
        return self.config.num

    def __iter__(self):
        for struc_fp, seq_fp in self.file_pairs:
            struc_dict = {}
            struc_lines = open(struc_fp, "r").readlines()
            for l in struc_lines:
                items = l.strip().split(" ")
                struc_dict[items[0]] = list(map(int, items[1:]))
    
            seq_dict = {}
            seq_lines = open(seq_fp, "r").readlines()
            for l in seq_lines:
                items = l.strip().split(" ")
                seq_dict[items[0]] = [seq_vocab[aa] for aa in items[1]]

            names = list(struc_dict.keys())
            # shuffle samples
            if self.config.shuffle:
                random.shuffle(names)
    
            for name in names:
                struc, seq = struc_dict[name], seq_dict[name]
                assert len(struc) == len(seq)
                length = len(struc)
                if (length < self.config.min_len) or (length > self.config.max_len):
                    continue
                yield name, struc, seq
    

def collate(features):
    names, struc_tokens, seq_tokens = zip(*features)
    names, struc_tokens, seq_tokens = \
        list(names), list(struc_tokens), list(seq_tokens)
    
    lens = [len(s) for s in struc_tokens]
    max_len = max(lens)

    for idx in range(len(lens)):
        struc_tokens[idx] = _pad_token(
            struc_tokens[idx], max_len, STRUCTURE_PAD_TOKEN)
        seq_tokens[idx] = _pad_token(
            seq_tokens[idx], max_len, SEQUENCE_PAD_TOKEN)

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
#             name, structure, sequence = data['name'], data['structure'],  data['sequence']
#             # print(name)
#             # print(structure)
#             # print(sequence)
#             # break
#         print(idx)

#     def length_statistics():
#         from collections import Counter
#         from tqdm import tqdm

#         lengths = []
#         for idx, (name, struc, seq) in enumerate(tqdm(dataset)):            
#             lengths.append(len(struc))
            
#         lengths = Counter(lengths)
#         print(sorted(list(lengths.keys())))
    
#     from omegaconf import OmegaConf
#     from torch.utils.data import DataLoader
#     trainset_conf = OmegaConf.create({
#         "dir": "./data",
#         "prefix": [1, 2],
#         "shuffle": True,
#         "min_len": 40,
#         "max_len": 256
#     })
#     dataset = ProDataset(trainset_conf)
#     loader = DataLoader(
#         dataset=dataset,
#         collate_fn=collate,
#         batch_size=4,
#     )
    
#     # length_statistics()
#     test_dataloader()