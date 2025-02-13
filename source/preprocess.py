import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile
from biotite.structure import (
    annotate_sse,
    get_residue_count,
    gyration_radius,
)
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import warnings 
warnings.simplefilter("ignore")

from esm.utils.structure.protein_chain import ProteinChain
import esm.utils.constants.esm3 as C
from esm.pretrained import ESM3_structure_encoder_v0


def extract_meta_info(chain):
    res_num = get_residue_count(chain)
    if (res_num < min_len) or (res_num > max_len):
        return None

    sse = annotate_sse(chain)
    gy_radius = gyration_radius(chain)
        
    ca_coord = chain[chain.atom_name == 'CA'].coord    
    distance = ca_coord[:, None, :] - ca_coord[None, :, :]  # L, L, 3
    distance = np.sqrt(np.sum(distance*distance, axis=-1))
    row_idx, col_idx = np.meshgrid(np.arange(len(ca_coord)), np.arange(len(ca_coord)))
    delta_index_mask = np.abs(row_idx - col_idx) <= res_delta_threshold
    distance[delta_index_mask] = float("inf")
    contact_num = int(np.sum(distance < contact_threshold) / 2)
    
    meta = {}
    meta['length'] = res_num
    meta['contact_num'] = contact_num
    meta['coil_percent'] = (np.sum(sse == 'c') + np.sum(sse == '')) / res_num
    meta['alpha_percent'] = np.sum(sse == 'a') / res_num
    meta['beta_percent'] = np.sum(sse == 'b') / res_num
    meta['gyration_radius'] = gy_radius
    return meta


def _build_meta(fps):
    iterator = tqdm(fps)
    meta_dict = {}
    for idx, fp in enumerate(iterator):
        try:
            chain = PDBFile.read(fp).get_structure()[0]
            chain_meta = extract_meta_info(chain)
            if chain_meta is None:
                continue
            meta_dict[fp] = chain_meta
        except Exception as e:
            print(e)
            continue
    return meta_dict


def build_meta(fp_txt, meta_out):
    fps = open(fp_txt, "r").readlines()        
    meta_dict = _build_meta(fps)
    
    data_list = [{'pdb': key, **item} for key, item in meta_dict.items()]
    df = pd.json_normalize(data=data_list)
    df.to_csv(meta_out, sep="\t", float_format='%.2f')


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values='gyration_radius', 
        index='length',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.gyration_radius.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y


@torch.no_grad()
def filter(indexes, df):
    device = f"cuda:0"
    
    # encoder
    encoder = ESM3_structure_encoder_v0(device)
    iterator = tqdm(indexes)
    results = []
    for idx, _index in enumerate(iterator):
        fp = df.iloc[_index].pdb
        chain = ProteinChain.from_pdb(fp)
        sequence = chain.sequence
        
        # to encoder input
        coord, plddt, res_idx = chain.to_structure_encoder_inputs()
        coord = coord[..., :3, :]
        mask = torch.all(
            torch.all(torch.isfinite(coord) & (~torch.isnan(coord)), dim=-1),
            dim=-1,
        )
        coord = coord[mask, ...].unsqueeze(0)
        plddt = plddt[mask].unsqueeze(0)
        res_idx = res_idx[mask].unsqueeze(0)
        seq = "".join([r for m, r in zip(mask.squeeze(), chain.sequence) if m])
        z_q, struc_tokens = encoder.encode(
            coords=coord.to(device),
            residue_index=res_idx.to(device))
        z_q = z_q.cpu()

        strline = " ".join([
            fp, *[
                str(t) for t in struc_tokens.squeeze(0).cpu().numpy().tolist()]
        ]) + "\n"
        seqline = " ".join([fp, sequence]) + "\n"
        results.append((strline, seqline))

        iterator.set_description(f"Num={len(results)}")

    return results


def encode(meta_fp, txt_out_prefix):
    # 读取meta信息
    df = pd.read_csv(meta_fp, sep="\t")
    df = df[(df.length >= min_len) & (df.length <= max_len)]
    df = df[df.coil_percent <= max_coil]
    df = df[df.contact_num > df.length/2]
    
    prot_rog_low_pass = _rog_quantile_curve(
        df, 
        gy_radius_cutoff,
        np.arange(max_len+1),
    )
    
    row_rog_cutoffs = df.length.map(
        lambda x: prot_rog_low_pass[x-1])
    df = df[df.gyration_radius < row_rog_cutoffs]
    
    lines = filter(range(len(df)), df)
    strlines, seqlines = zip(*lines)
    
    structure_out = txt_out_prefix + "_struc.txt"
    sequence_out = txt_out_prefix + "_seq.txt"
    open(structure_out, "w").writelines(strlines)
    open(sequence_out, "w").writelines(seqlines)


if __name__ == "__main__":
    
    min_len = 32
    max_len = 512
    contact_threshold = 8.
    res_delta_threshold = 12

    max_coil = 0.5
    gy_radius_cutoff = 0.95

    fp_txt = "TEXT FILE THAT CONTAINS PATH OF PDB"
    meta_fp = "PATH TO STORE META INFORMATION"
    txt_out = "PATH TO STORE PROCESSED DATA"
    
    build_meta(fp_txt, meta_fp)
    encode(meta_fp, txt_out)