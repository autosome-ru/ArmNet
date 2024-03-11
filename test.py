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
from dataset import RNA_Dataset_Test

from training_utils import parameter_count, seed_everything
from training_utils import DeviceDataLoader


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type = str)
parser.add_argument("--forget_config", action = "store_true")
parser.add_argument("--out_dir_path", default = "./predictions", type = str)
parser.add_argument("--pretrained_model_weights", type = str)

args = parser.parse_args()
config = ArmNetConfig()

OUT_DIR_PATH = Path(args.out_dir_path)
os.makedirs(OUT_DIR_PATH, exist_ok=True)

if args.config_path is not None:
    config.load(args.config_path)
    
config.load_dict(vars(args))

if not args.forget_config:
    config.save(OUT_DIR_PATH / "config.json")
    
config.device = torch.device(f"cuda:{config.device}") 
BPPM_PATH = config.test_bppm_data_path
if BPPM_PATH is not None:
    BPPM_PATH = Path(BPPM_PATH).resolve()
    
MODEL_WEIGHTS_PATH = config.pretrained_model_weights
if MODEL_WEIGHTS_PATH is not None:
    MODEL_WEIGHTS_PATH = Path(MODEL_WEIGHTS_PATH).resolve()

seed_everything(config.seed)

df = pd.read_csv(config.test_data_path).iloc[:2000]

print("Constructing test dataset.")
test_dataset = RNA_Dataset_Test(
    df,
    use_bppm = config.use_bppm,
    bppm_path = BPPM_PATH
)

test_dataloader = DeviceDataLoader(
    DataLoader(
        dataset = test_dataset, 
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        persistent_workers=True,
        shuffle=False,
        drop_last=False),
    device = config.device)

model =  ArmNet(
     depth = config.num_encoder_layers, 
     num_convs = config.num_conv_layers,
     adj_ks = config.conv_2d_kernel_size,
     attn_kernel_size = config.conv_1d_kernel_size,
     dropout = config.dropout,
     conv_use_drop1d = config.conv_1d_use_dropout,
     use_bppm = config.use_bppm,
)

model.load_state_dict(
    torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")['model']
)

print("Weights path: ", str(MODEL_WEIGHTS_PATH))
print("Parameter count: ", parameter_count(model))

ids,preds = [],[]
model = model.to(config.device)
model = model.eval()

for x,y in tqdm(test_dataloader):
    with torch.no_grad():
        p = model(x).clip(0,1)

    for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])

ids = torch.concat(ids)
preds = torch.concat(preds)

df = pd.DataFrame(
    {'id':ids.numpy(),
     'reactivity_DMS_MaP':preds[:,1].numpy(), 
     'reactivity_2A3_MaP':preds[:,0].numpy()
    }
)

print(df.head())

df.to_parquet(OUT_DIR_PATH / "predictions.parquet", index=False) 
