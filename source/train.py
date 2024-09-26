import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from esm.utils.constants.esm3 import data_root
from transformers import TrainingArguments, Trainer

from data import ProDataset, collate
from model import CoFlowConfig, CoFlowModel


def metrics(eval_prediction):        
    pred = eval_prediction.predictions
    struc_loss = np.mean(pred[0])
    struc_acc = np.mean(pred[1])
    seq_loss = np.mean(pred[2])
    seq_acc = np.mean(pred[3])

    return {
        "struc_acc": np.round(struc_acc, 4),
        "struc_loss": np.round(struc_loss, 4),
        "seq_acc": np.round(seq_acc, 4),
        "seq_loss": np.round(seq_loss, 4),
    }


def build_trainer(model, trainset, valset, train_args, collate_fn):    
    config = TrainingArguments(
        **train_args,
        log_level="info",
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
        accelerator_config={
            "dispatch_batches": False,
        }
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


def build_data(conf):
    conf_dict = dict(**conf)
    train_prefix = conf_dict.pop("train_prefix")
    valid_prefix = conf_dict.pop("valid_prefix")
    train_num = conf_dict.pop("train_num")
    valid_num = conf_dict.pop("valid_num")
    
    train_conf = OmegaConf.create(conf_dict)
    train_conf.prefix = train_prefix
    train_conf.num = train_num
    trainset = ProDataset(train_conf)
    
    valid_conf = OmegaConf.create(conf_dict)
    valid_conf.prefix = valid_prefix
    valid_conf.num = valid_num
    valid_conf.shuffle = False
    validset = ProDataset(valid_conf)

    return trainset, validset


def main():
    config_path = sys.argv[1]
    args = OmegaConf.load(config_path)
    
    data_args = args['data']
    model_args = args['model']
    train_args = args['train']
    train_args['ddp_find_unused_parameters'] = False
    rank = int(os.environ.get("RANK", 0))

    # write configs to save dir
    os.makedirs(train_args['output_dir'], exist_ok=True)
    with open(os.path.join(train_args['output_dir'], "config.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(args))
    checkpoint = train_args.pop('checkpoint', None)

    if getattr(model_args, "pretrained", None) is not None:
        model = CoFlowModel.from_pretrained(model_args.pretrained)
    else:
        model = CoFlowModel(CoFlowConfig(**model_args))
        # load esm model
        if getattr(model_args, "finetune_esm", False):
            state_dict = torch.load(
                data_root() / "data/weights/esm3_sm_open_v1.pth")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False) 
            if rank == 0:
                print("Missing Keys: ", missing_keys)
                print("Unexpected Keys: ", unexpected_keys)
    
    trainset, validset = build_data(data_args)
    trainer = build_trainer(
        model, trainset, validset, train_args, collate_fn=collate)
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