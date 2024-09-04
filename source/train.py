import os
import sys
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer

from diffusion import D3PM
from data import ProDataset
from model import CoDiffConfig, CoDiffNetwork


def metrics(eval_prediction):
    acc = lambda pred, target, mask: \
        np.sum(((pred==target)*mask).astype(np.int32)) / (np.sum(mask)+1.e-6)
        
        
    pred = eval_prediction.predictions
    struc_track = pred[0][0]
    seq_track = pred[1][0]
        
    if struc_track and seq_track:
        struc_vb = np.mean(pred[2])
        struc_ce = np.mean(pred[3])
        struc_pred, struc_gold, struc_mask = pred[4], pred[5], pred[6]
        
        seq_vb = np.mean(pred[7])
        seq_ce = np.mean(pred[8])
        seq_pred, seq_gold, seq_mask = pred[9], pred[10], pred[11]

        struc_acc = acc(struc_pred, struc_gold, struc_mask)
        seq_acc = acc(seq_pred, seq_gold, seq_mask)
        return {
            "struc_acc": round(struc_acc, 4),
            "struc_vb": round(struc_vb, 4),
            "struc_ce": round(struc_ce, 4),
            "seq_acc": round(seq_acc, 4),
            "seq_vb": round(seq_vb, 4),
            "seq_ce": round(seq_ce, 4),
        }
    elif struc_track:
        struc_vb = np.mean(pred[2])
        struc_ce = np.mean(pred[3])
        struc_pred, struc_gold, struc_mask = pred[4], pred[5], pred[6]
        struc_acc = acc(struc_pred, struc_gold, struc_mask)
        return {
            "struc_acc": round(struc_acc, 4),
            "struc_vb": round(struc_vb, 4),
            "struc_ce": round(struc_ce, 4),
        }
    elif seq_track:
        seq_vb = np.mean(pred[2])
        seq_ce = np.mean(pred[3])
        seq_pred, seq_gold, seq_mask = pred[4], pred[5], pred[6]
        seq_acc = acc(seq_pred, seq_gold, seq_mask)
        return {
            "seq_acc": round(seq_acc, 4),
            "seq_vb": round(seq_vb, 4),
            "seq_ce": round(seq_ce, 4),
        }
    

def build_trainer(model, trainset, valset, train_args, collate_fn):    
    config = TrainingArguments(
        **train_args,
        log_level="info",
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
    )
    trainer = Trainer(
        model=model,
        args=config,
        data_collator=collate_fn,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_metrics=metrics,
    )
    return trainer


def main():
    config_path = sys.argv[1]
    args = OmegaConf.load(config_path)
    
    data_args = args['data']
    diff_args = args['diffusion']
    model_args = args['model']
    train_args = args['train']
    train_args['ddp_find_unused_parameters'] = False

    # write configs to save dir
    os.makedirs(train_args['output_dir'], exist_ok=True)
    with open(os.path.join(train_args['output_dir'], "config.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(args))
    checkpoint = train_args.pop('checkpoint', None)

    d3pm = D3PM(conf=diff_args)
    model = CoDiffNetwork(CoDiffConfig(**model_args), d3pm=d3pm)
    dataset = ProDataset(data_args)
    trainset, valset = random_split(dataset, (0.96, 0.04))

    trainer = build_trainer(
        model, trainset, valset, train_args, collate_fn=dataset.collate)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(model)
        # json_string = json.dumps(train_args, indent=2, sort_keys=False) + "\n"
        json_string = OmegaConf.to_yaml(train_args)
        print(f"TrainConfig {json_string}")

    # import torch
    # torch.autograd.set_detect_anomaly(True)
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == '__main__':
    main()