import math
from pathlib import Path
from typing import Optional, ClassVar

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold


def _load_bppm(
    seq_id: str,
    Lmax: int,
    bppm_path: Path,
):
    path = bppm_path / f"{seq_id}.npy"
    mat = np.load(path)
    dif = Lmax - mat.shape[0]
    res = np.pad(mat, ((0, dif), (0, dif)))
    return res


class BPPFeatures:
    LMAX: ClassVar[int] = 206

    def __init__(self, index_path: str, mempath: str):
        self.index = self.read_index(index_path)
        self.storage = self.read_memmap(mempath, len(self.index))
        
    @classmethod
    def read_index(cls, index_path):
        with open(index_path) as inp:
            ids = [line.strip() for line in inp]
        index = {seqid: i for i, seqid in enumerate(ids)}
            
        return index
    
    @classmethod
    def read_memmap(cls, memmap_path, index_len):
        storage = np.memmap(memmap_path, 
                            dtype=np.float32,
                            mode='r', 
                            offset=0,
                            shape=(index_len, cls.LMAX, cls.LMAX),
                            order='C')
        return storage
    
    def __getitem__(self, seqid):
        ind = self.index[seqid]
        return self.storage[ind]


class RNA_Dataset(Dataset):
    def __init__(self, 
                 df,
                 mode: str='train', 
                 seed: int = 2023, 
                 fold: int = 0, 
                 nfolds: int = 4,
                 use_bppm: bool = False,
                 bppm_path: Optional[Path] = None
                ):
        self.seq_map = {'A':0,
                        'C':1,
                        'G':2,
                        'U':3,
                        "START": 4,
                        "END": 5,
                        "EMPTY": 6}

        assert mode in ('train', 'eval')
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        


        assert mode in ("train", "eval")
        
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        
        if mode == "eval":
            print("Keeping only clean data for validation")
            m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        self.sid = df_2A3['sequence_id'].values
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_error_0' in c]].values
        
        self.is_good =  ((df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0) )* 1
        self.sn_2A3 = df_2A3['SN_filter'].values 
        self.sn_DMS = df_DMS['SN_filter'].values
        
        sn = (df_2A3['signal_to_noise'].values + df_DMS['signal_to_noise'].values) / 2
        
        sn = torch.from_numpy(sn)
        self.weights = 0.5 * torch.clamp_min(torch.log(sn + 1.01),0.01)
        
        self.mode = mode
        self._use_bppm = use_bppm
        if use_bppm:
            if bppm_path is None:
                raise ValueError("If use_bppm is set True, bppm_path must be specified.")
            self.bppm_features = BPPFeatures(bppm_path / "index.ind", bppm_path / "joined.mmap")
        
    def __len__(self):
        return len(self.seq)  
    
    def _process_seq(self, rawseq):
        seq = [self.seq_map['START']]
        start_loc = 0
        seq.extend(self.seq_map[s] for s in rawseq)
        seq.append(self.seq_map['END'])
        end_loc = len(seq) - 1
        for i in range(len(seq), self.Lmax+2):
            seq.append(self.seq_map['EMPTY'])
            
        seq = np.array(seq)
        seq = torch.from_numpy(seq)
        
        return seq, start_loc, end_loc
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        real_seq_L = len(seq)
        
        lbord = 1
        rbord = self.Lmax  + 1 - real_seq_L
        
        seq_int, start_loc, end_loc = self._process_seq(seq)
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc+1:end_loc] = True # not including START and END
        conv_bpp_mask = torch.zeros(self.Lmax + 2, self.Lmax + 2, dtype=torch.bool)
        conv_bpp_mask[start_loc+1:end_loc, start_loc+1:end_loc] = True # not including START and END
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool) # START, seq, END
        forward_mask[start_loc:end_loc+1] = True # including START and END
        
        
        react = np.stack([self.react_2A3[idx][:real_seq_L],
                          self.react_DMS[idx][:real_seq_L]],
                         -1)
        react = np.pad(react, ((lbord,
                                rbord),
                               (0,0)), constant_values=np.nan)
        
        react = torch.from_numpy(react)
     
        
        X = {'seq_int': seq_int,
             'mask': mask, 
             'forward_mask': forward_mask,
             'conv_bpp_mask': conv_bpp_mask,
             'is_good': self.is_good[idx]}
        
        sid = self.sid[idx]
        
        if self._use_bppm:
            adj = self.bppm_features[sid][:real_seq_L, :real_seq_L]
            adj = np.pad(adj, ((lbord,rbord), (lbord, rbord)), constant_values=0)
            adj = torch.from_numpy(adj).float()
            X['adj'] = adj


        y = {'react': react.float(), 
             'mask': mask}
        
        
        return X, y
    
    
class RNA_Dataset_Test(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 use_bppm: bool = False,
                 bppm_path: Optional[Path] = None
                ):
        self.df = df
        self.seq_map = {'A':0,
                        'C':1,
                        'G':2,
                        'U':3,
                        "START": 4,
                        "END": 5,
                        "EMPTY": 6}
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.sid = df.sequence_id
        self._use_bppm = use_bppm
        self._bppm_path = bppm_path
        if use_bppm and bppm_path is None:
                raise ValueError("If use_bppm is set True, bppm_path must be specified.")
        
    def __len__(self):
        return len(self.df)
    
    def _process_seq(self, rawseq):
        seq = [self.seq_map['START']]
        start_loc = 0
        seq.extend(self.seq_map[s] for s in rawseq)
        seq.append(self.seq_map['END'])
        end_loc = len(seq) - 1
        for i in range(len(seq), self.Lmax+2):
            seq.append(self.seq_map['EMPTY'])
            
        seq = np.array(seq)
        seq = torch.from_numpy(seq)
        
        return seq, start_loc, end_loc
    
    def __getitem__(self, idx):
        id_min, id_max, seq = self.df.loc[idx, ['id_min','id_max','sequence']]
        L = len(seq)
        
        ids = np.arange(id_min,id_max+1)
        ids = np.pad(ids,(1,self.Lmax+1-L), constant_values=-1)
        
        
        seq_int, start_loc, end_loc = self._process_seq(seq)
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc+1:end_loc] = True # not including START and END
        
        conv_bpp_mask = torch.zeros(self.Lmax + 2, self.Lmax + 2, dtype=torch.bool)
        conv_bpp_mask[start_loc+1:end_loc, start_loc+1:end_loc] = True # not including START and END
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool) # START, seq, END
        forward_mask[start_loc:end_loc+1] = True # including START and END
      
       
        
        X = {'seq_int': seq_int, 
             'mask': mask, 
             "is_good":1,
             "forward_mask": forward_mask, 
             'conv_bpp_mask': conv_bpp_mask}
        
        sid = self.sid[idx]
        
        if self._use_bppm:
            adj = _load_bppm(self.sid[idx],
                             self.Lmax,
                             self._bppm_path)
            adj = np.pad(adj, ((1,1), (1, 1)), constant_values=0)
            X['adj'] = torch.from_numpy(adj).float()
        
        return X, {'ids':ids}