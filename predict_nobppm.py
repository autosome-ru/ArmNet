import math
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np 
import pandas as pd
from tqdm import tqdm

from config import ArmNetConfig
from model import ArmNet
from dataset import RNA_Dataset_Pred

from training_utils import parameter_count, seed_everything


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type = str)
parser.add_argument("--forget_config", action = "store_true")
parser.add_argument("-o", "--out_dir_path", default = "./predictions", type = str)
parser.add_argument("-d", "--data_path", type = str, required = True)
parser.add_argument("--pretrained_models_dir", type = str, required = True)

args = parser.parse_args()
config = ArmNetConfig()

OUT_PATH = Path(args.out_dir_path)
OUT_PATH.mkdir(parents=True, exist_ok=True)


if args.config_path is not None:
    config.load(args.config_path)
    
config.load_dict(vars(args))
config.use_bppm = False

if not args.forget_config:
    config.save(OUT_PATH / "config.json")
    
config.device = torch.device(f"cuda:{config.device}") 

    
MODEL_WEIGHTS_DIR = args.pretrained_models_dir
MODEL_WEIGHTS_DIR = Path(MODEL_WEIGHTS_DIR).resolve()

DATA_PATH = args.data_path
DATA_PATH = Path(DATA_PATH).resolve()

seed_everything(config.seed)

df = pd.read_csv(DATA_PATH, sep="\t")


ds = RNA_Dataset_Pred(df, bppm_path=None)
dl = DataLoader(dataset=ds,
               shuffle=False,
               batch_size = config.batch_size,
               num_workers = config.num_workers,
               persistent_workers=True,
               drop_last=False)

MODEL_WEIGHTS_PATH = MODEL_WEIGHTS_DIR / "model.pth"
config = ArmNetConfig()
model =  ArmNet(
     depth = config.num_encoder_layers, 
     num_convs = config.num_conv_layers,
     adj_ks = config.conv_2d_kernel_size,
     attn_kernel_size = config.conv_1d_kernel_size,
     dropout = config.dropout,
     conv_use_drop1d = config.conv_1d_use_dropout,
     use_bppm = False,
)
model.load_state_dict(
    torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
)
model = model.eval()

device = config.device
res = []
model.to(device)
for batch, ids in tqdm(dl):
    batch = {key: item.to(device) for key, item in batch.items()}
    with torch.inference_mode():
        pred = model(batch)
    res.append(pred.cpu()) 

out_res = torch.cat(res, 0)
out_res = out_res[:, 1:-1, :]
print(out_res.shape)

filename = OUT_PATH / f'dms_nobppm_{out_res.shape[0]}_{out_res.shape[1]}_float32.mmap'
fp = np.memmap(filename, dtype='float32', mode='w+', shape=out_res[:,:,0].shape)
fp[:] = out_res[:,:,0].numpy()[:]
fp.flush() 

filename = OUT_PATH / f'2a3_nobppm_{out_res.shape[0]}_{out_res.shape[1]}_float32.mmap'
fp = np.memmap(filename, dtype='float32', mode='w+', shape=out_res[:,:,1].shape)
fp[:] = out_res[:,:,1].numpy()[:]
fp.flush()

