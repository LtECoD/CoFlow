## CoFlow

CoFlow is a discrete generative model for protein sequence and structure co-design, as described in our paper:
**Co-Design Protein Sequence and Structure in Discrete Space via Generative Flow**  
<!-- Authors: [Author Names]  
[Journal/Conference Name], [Year]  
[DOI or arXiv Link (if available)] -->

<!-- Our code is developed based on [ESM3](https://github.com/evolutionaryscale/esm). -->


## Dependencies

To run the source code, install the required dependencies:
```bash
# Install environment with dependencies.
conda env create -n coflow -f requirements.yaml

# Activate environment
conda activate coflow
```

CoFlow takes pre-trained structure VQ-VAE of ESM3 for tokenization. Therefore, running inference requires that you have access to the ESM3 weights. Make sure your huggingface account has access to "[EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/tree/main)". Then generate your huggingface access token (check the permissions to repositories) and run the following command to download the weights:
```bash
huggingface-cli login   
# input your huggingface access token
huggingface-cli download EvolutionaryScale/esm3-sm-open-v1
```
Downloaded files are typically located at "~/.cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1".


## Usage

### Inference
Note: Download the trained model weights from [here](https://doi.org/10.5281/zenodo.14842367), and extract them to the `checkpoint` directory. As a result, the `checkpoint` will include four files:

checkpoint \
├── config.json \
├── model-00001-of-00002.safetensors \
├── model-00002-of-00002.safetensors \
├── model.safetensors.index.json \
└── version 

See the notebook [example.ipynb](./example.ipynb) for information on how to use CoFlow, which includes examples of both unconditional and conditional generation.

<!-- To perform unconditional generation, run the following Python script:

```python
import sys
sys.path.append("./source")

from model import CoFlowModel
from utils import to_protein
from esm.tokenization import StructureTokenizer, EsmSequenceTokenizer
device="cuda:0"

model = CoFlowModel.from_pretrained(CHECKPOINT PATH).to(device)
out = model.sample(
    strategy=3,
    length=LENGTH,
    steps=400,
    eta=length*0.08,
    purity=False,
    sequence_temp=0.7,
    structure_temp=0.7,
    device=device,
)
structure, sequence = out['structure'], out['sequence']
protein, _, _ = to_protein(
    structure=structure,
    sequence=sequence,
    decoder=ESM3_structure_decoder_v0(device),
    struc_tokenizer=struc_tokenizer,
    seq_tokenizer=seq_tokenizer,
    strip=False,
)
protein.to_pdb(OUT_PDB_PATH)

```

For conditional generation, provide sequence, structure, or motif tokens as input:
```python
out = model.sample(
    sequence=SEQUENCE,
    structure=STRUCTURE,
    strategy=3,
    length=length,
    steps=400,
    eta=length*0.08,
    purity=False,
    sequence_temp=0.7,
    structure_temp=0.7,
    device=device,
)
```
Note: The sequence and structure parameters should be lists of indexed tokens. The tokenization process is detailed in [ESM3](https://github.com/evolutionaryscale/esm). -->


### Train

To train the model, you will need to pre-process dataset. Just run the following script:
```bash
python source/preprocess.py
```

Several parameters need to be specified in the script, including:

- `fp_txt`: Path to a text file containing PDB file paths
- `meta_fp`: Path to store metadata
- `txt_out`: Path to save processed data

You can also customize other parameters to control filtering granularity.

The processed dataset consists of two `.txt` files:

1. A sequence file, where each line corresponds to a protein sequence.
2. A structure token file, where each line represents a discrete protein structure, encoded using the VQVAE model in ESM3.

**Example:**
> **Sequence line:**
>>908/MGYP003390323908 MKLIITLLLFVSLLPAYAAIMDGNCRDSQGSFRGEIIFREARHTQVVVGIRDRADYLNRGLAITFPRLELSGHKVVAQYSHPHYAGIGSEASRLEFDGALIRLTTLVRNAPNGSFNLSVSCLLDVPRDRQELGRLVREMNTH
>
> **Structure line:**
>> 908/MGYP003390323908 1035 3954 305 3961 3961 2082 588 3101 588 3109 2439 3227 1763 852 1364 943 3799 3617 3106 177 3705 1220 3892 2520 3683 2945 2886 1805 3013 1862 194 1167 1487 2670 1191 3857 2302 163 3975 2293 1582 3211 322 3737 2446 560 1534 1177 697 794 1179 3994 3023 2983 2816 3148 1033 1395 2556 1712 3949 189 2536 2194 1451 1619 3509 1011 3332 872 1272 3660 3904 2463 3677 1419 767 2269 1399 1179 741 3378 1404 1993 82 1786 1204 795 3052 2452 496 3889 3331 2861 634 1057 978 1186 2781 2989 189 3166 1809 1547 2832 1367 276 4084 1076 2769 800 1480 1862 3721 3538 3362 1785 2081 3556 2557 2259 2756 1713 2331 2780 594 1169 1412 2776 3961 588 778 668 588 2587 1695 2048 1414 2425 2080 2103 969


CoFlow is trained with two datasets:
- [High confidence MGnify30 structures](https://github.com/facebookresearch/esm/blob/main/scripts/atlas/v0/highquality_clust30/tarballs.txt) 
- [Processed PDB](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A)


## License

The model and code are released under the [Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)

