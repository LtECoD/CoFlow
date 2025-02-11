## CoFlow

CoFlow is a discrete generative model for protein sequence and strucure co-design based on our paper:

**Co-Design Protein Sequence and Structure in Discrete Space via Generative Flow**  
<!-- Authors: [Author Names]  
[Journal/Conference Name], [Year]  
[DOI or arXiv Link (if available)] -->

Our code is developed based on [ESM3](https://github.com/evolutionaryscale/esm).


## Dependencies

To run the source code, please install the following dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Inference
You can download trained model weights from [here](https://doi.org/10.5281/zenodo.14842367) to run CoFlow. Then unzip the file into any directory.

Run the following python script to conduct unconditional generation: 

```python
import sys
sys.path.append(PATH TO THE SOURCE OF THIS REPO)

from model import CoFlowModel
from utils import to_protein
from esm.pretrained import ESM3_structure_decoder_v0

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

To conduct conditional generation, you are required to feed sequence/structure/motif tokens to the model. Just like this:
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
Note: paramters of *sequence* and *structure* are list of indexed tokens. 
Toknization procedure can be found in [ESM3](https://github.com/evolutionaryscale/esm).


### 1. Train

To run training, you will need to pre-process datasets. The processed dataset include two .txt files. One is the sequence file, and each line represents a protein sequence. 
Another file represents structure tokens processed by VQVAE encoder in ESM3, and each line represents a discrete protein structure. For example: 
> sequence line:
>>908/MGYP003390323908 MKLIITLLLFVSLLPAYAAIMDGNCRDSQGSFRGEIIFREARHTQVVVGIRDRADYLNRGLAITFPRLELSGHKVVAQYSHPHYAGIGSEASRLEFDGALIRLTTLVRNAPNGSFNLSVSCLLDVPRDRQELGRLVREMNTH
>
> structure line:
>> 908/MGYP003390323908 1035 3954 305 3961 3961 2082 588 3101 588 3109 2439 3227 1763 852 1364 943 3799 3617 3106 177 3705 1220 3892 2520 3683 2945 2886 1805 3013 1862 194 1167 1487 2670 1191 3857 2302 163 3975 2293 1582 3211 322 3737 2446 560 1534 1177 697 794 1179 3994 3023 2983 2816 3148 1033 1395 2556 1712 3949 189 2536 2194 1451 1619 3509 1011 3332 872 1272 3660 3904 2463 3677 1419 767 2269 1399 1179 741 3378 1404 1993 82 1786 1204 795 3052 2452 496 3889 3331 2861 634 1057 978 1186 2781 2989 189 3166 1809 1547 2832 1367 276 4084 1076 2769 800 1480 1862 3721 3538 3362 1785 2081 3556 2557 2259 2756 1713 2331 2780 594 1169 1412 2776 3961 588 778 668 588 2587 1695 2048 1414 2425 2080 2103 969

We train CoFlow with the High confidence MGnify30 structures in [ESMAtlas](https://github.com/facebookresearch/esm/tree/main/scripts/atlas). The finetune dataset is from [MultiFlow](https://github.com/jasonkyuyim/multiflow)


## License

Our model and code are released under [Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)

