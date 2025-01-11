import argparse
from pathlib import Path
import os
import sys

ARNIE_CONFIG_PATH = 'path/to/arnie/arnie_config.txt'
os.environ['ARNIEFILE'] = ARNIE_CONFIG_PATH
ARNIE_PATH = 'path/to/arnie/src/'
sys.path.append(ARNIE_PATH)

import arnie
from arnie.bpps import bpps
import numpy as np 
import pandas as pd
import concurrent.futures
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out_dir", type = str)
parser.add_argument("-d", "--data_path", type = str)
args = parser.parse_args()

AIM_DIR = Path(args.out_dir)
AIM_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(args.data_path)

def calc_arnie(seq, seqid):
    res = bpps(seq, package="eternafold")
    outpath = AIM_DIR / f"{seqid}.npy"
    np.save(outpath, res)
    

    
df = pd.read_csv(DATA_PATH, sep="\t")
with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
    futures = {}
    for ind, seqid, seq in tqdm.tqdm(df[['id', 'sequence']].itertuples(), total=len(df)):
        ft = executor.submit(calc_arnie, seqid=seqid, seq=seq.upper())
        futures[ft] = seqid

    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            ft.result()
        except Exception as exc:
            print(exc)
            seqid = futures[ft]
            print(f"Error occured while processing {seqid}: {exc}")
    
