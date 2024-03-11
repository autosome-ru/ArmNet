import math
import argparse
from pathlib import Path
import os, gc

from fastai.vision.all import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader

import numpy as np 
import pandas as pd

from config import ArmNetConfig
from model import ArmNet
from dataset import RNA_Dataset

from training_utils import parameter_count, loss, seed_everything, get_dataloaders
from training_utils import MAE, MAE_2A3, MAE_DMS, DeviceDataLoader


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type = str)
parser.add_argument("--forget_config", action = "store_true")
parser.add_argument("--out_dir_path", default = "./results", type = str)
parser.add_argument("--pretrained_model_weights", type = str)
parser.add_argument("--no_weights", action = "store_true")
parser.add_argument("--num_folds", type=int)
parser.add_argument("--fold", type=int)
parser.add_argument("--lr_max", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--pct_start", type=float)
parser.add_argument("--gradclip", type=float)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--batch_count", type=int)
parser.add_argument("--device", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--sgd_lr", type=float)
parser.add_argument("--sgd_num_epochs", type=int)
parser.add_argument("--sgd_batch_count", type=int)
parser.add_argument("--sgd_weight_decay", type=float)


args = parser.parse_args()
config = ArmNetConfig()

OUT_DIR_PATH = Path(args.out_dir_path)
os.makedirs(OUT_DIR_PATH, exist_ok=True)

if args.config_path is not None:
    config.load(args.config_path)
    
config.load_dict(vars(args))

if not args.forget_config:
    config.save(OUT_DIR_PATH / "config.json")
    
config.device = torch.device(f"cuda:{config.device}") ##########
BPPM_PATH = config.train_bppm_data_path
if BPPM_PATH is not None:
    BPPM_PATH = Path(BPPM_PATH).resolve()
    
MODEL_WEIGHTS_PATH = config.pretrained_model_weights
if MODEL_WEIGHTS_PATH is not None:
    MODEL_WEIGHTS_PATH = Path(MODEL_WEIGHTS_PATH).resolve()

seed_everything(config.seed)

df = pd.read_parquet(config.train_data_path)

###Make convenient saving and paths

save_model_cbk = SaveModelCallback(
    monitor='valid_loss',
    fname='model', 
    with_opt=True)

main_run_path  = OUT_DIR_PATH / 'main_run'
main_run_path.mkdir(parents=True, exist_ok=True)
logger = CSVLogger(fname = str(main_run_path / "loss.csv"))


print("Constructing training dataset.")
train_dataset = RNA_Dataset(
    df,
    mode = 'train', 
    fold = config.fold, 
    nfolds = config.num_folds,
    use_bppm = config.use_bppm,
    bppm_path = BPPM_PATH
)

print("Constructing validation dataset.")
val_dataset = RNA_Dataset(
    df,
    mode = 'eval', 
    fold = config.fold, 
    nfolds = config.num_folds,
    use_bppm = config.use_bppm,
    bppm_path = BPPM_PATH
)

print("Constructing dataloaders.")
data = get_dataloaders(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = config.batch_size,
    batch_count = config.batch_count,
    num_workers = config.num_workers,
    no_weights = config.no_weights,
    device = config.device
)

gc.collect()

model =  ArmNet(
     depth = config.num_encoder_layers, 
     num_convs = config.num_conv_layers,
     adj_ks = config.conv_2d_kernel_size,
     attn_kernel_size = config.conv_1d_kernel_size,
     dropout = config.dropout,
     conv_use_drop1d = config.conv_1d_use_dropout,
     use_bppm = config.use_bppm,
)

print("Parameter count: ", parameter_count(model))

if MODEL_WEIGHTS_PATH is not None:
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")['model']
    )

model = model.to(config.device)

if config.num_epochs > 0:
    learn = Learner(
        data, 
        model, 
        loss_func = loss,
        model_dir = main_run_path,
        cbs=[GradientClip(config.gradclip), 
             logger, 
             save_model_cbk],
        metrics=[MAE(),
                 MAE_DMS(),
                 MAE_2A3()]
    ).to_fp16() 
    
    print("Start learning cycle")
    
    learn.fit_one_cycle(
        config.num_epochs,
        lr_max = config.lr_max,
        wd = config.weight_decay,
        pct_start = config.pct_start
    )
    torch.save(
        learn.model.state_dict(),
        OUT_DIR_PATH / "model.pth"
    )
    gc.collect()

print("Constructing dataloaders for sgd run.")    
data = get_dataloaders(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = config.batch_size,
    batch_count = config.sgd_batch_count,
    num_workers = config.num_workers,
    no_weights = config.no_weights,
    device = config.device
)

save_model_cbk = SaveModelCallback(
    monitor='valid_loss',
    fname='model', 
    with_opt=True,
)

sgd_run_path = OUT_DIR_PATH / 'sgd_run'
sgd_run_path.mkdir(parents=True, exist_ok=True)
logger = CSVLogger(fname = str(sgd_run_path / "loss.csv"))

if config.sgd_num_epochs > 0:
    learn = Learner(
        data,
        model,
        model_dir = sgd_run_path,
        lr = config.sgd_lr,
        opt_func = partial(
                OptimWrapper, 
                opt=torch.optim.SGD
            ),
        loss_func = loss,
        cbs = [GradientClip(config.gradclip),
               save_model_cbk,
               logger],
        metrics=[MAE(), 
                 MAE_DMS(), 
                 MAE_2A3()]).to_fp16() 

    learn.fit(config.sgd_num_epochs, 
              wd=config.sgd_weight_decay)
