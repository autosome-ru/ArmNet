from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def to_train_format(
    df: pd.DataFrame,
    seq_data: pd.DataFrame,
    save_dir: Path,
) -> None:
    
    seqid_map = {}
    for i in seq_data.itertuples():
        for idx in range(i.id_min, i.id_max + 1):
            seqid_map[idx] = i.sequence_id
            
    df = df.copy()
    df["seq_id"] = df.id.apply(lambda x: seqid_map[x])
    L_max = seq_data.sequence.apply(len).max()
    
    pred_data = {i: np.full((2,L_max), fill_value=np.nan) for i in seq_data.sequence_id}
    coords_data = {i.sequence_id: (i.id_min, i.id_max) for i in seq_data.itertuples()}
    
    for data in tqdm(df.itertuples(), total = len(df)):
        seq_id = data.seq_id
        coords = coords_data[seq_id]
        pos_idx = data.id - coords[0]

        pred_data[seq_id][0][pos_idx] = data.reactivity_DMS_MaP
        pred_data[seq_id][1][pos_idx] = data.reactivity_2A3_MaP
    
    id2seq = {i.sequence_id: i.sequence for i in seq_data.itertuples()}
    
    new_df = defaultdict(list)
    
    for i in tqdm(pred_data): #DMS
        new_df["sequence_id"].append(i)
        new_df["sequence"].append(id2seq[i])
        new_df["experiment_type"].append("DMS_MaP")
        new_df["reactivity"].append(pred_data[i][0])
    for i in tqdm(pred_data): #2A3
        new_df["sequence_id"].append(i)
        new_df["sequence"].append(id2seq[i])
        new_df["experiment_type"].append("2A3_MaP")
        new_df["reactivity"].append(pred_data[i][1])
        
    new_df_pd = pd.DataFrame.from_dict(new_df)
    new_df_pd.to_parquet(save_dir / "predictions_plain.parquet", index = False)
    
    temp = np.stack(new_df["reactivity"])
    new_df_pd_2 = new_df_pd[[i for i in new_df_pd.columns if i not in ["reactivity", "reactivity_error"]]].copy()
    temp_df = pd.DataFrame(
        temp, 
        columns=[f"reactivity_{('000' + str(i+1))[-4:]}" for i in range(temp.shape[1])])
    
    new_df_pd_2 = pd.concat([new_df_pd_2, temp_df], axis=1)
    new_df_pd_2.to_parquet(save_dir / "predictions_wider.parquet", index = False)