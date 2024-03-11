import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json

import torch

@dataclass
class ArmNetConfig:
    # DATA PATHS
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    train_bppm_data_path: Optional[str] = None
    test_bppm_data_path: Optional[str] = None
    pretrained_model_weights: Optional[str] = None
    
    # MODEL PARAMETERS
    hidden_dim: int = 192
    head_size: int = 32
    num_encoder_layers: int = 12
    num_conv_layers: Optional[int] = None
    conv_1d_kernel_size: int = 17
    conv_2d_kernel_size: int = 3
    dropout: float = 0.1
    conv_1d_use_dropout: bool = False 
    use_bppm: bool = False
    
    # TRIANING PARAMETERS
    no_weights: bool = False
    num_folds: int = 4
    fold: int = 0
    lr_max: float = 2.5e-3
    weight_decay: float = 0.05
    pct_start: float = 0.05
    gradclip: float = 1.0
    num_epochs: int = 200
    num_workers: int = 32
    batch_size: int = 128
    batch_count: int = 1791
    device: int = 0
    seed: int = 2023
    
    sgd_lr: float = 5e-5
    sgd_num_epochs: int = 25
    sgd_batch_count: int = 500
    sgd_weight_decay: float = 0.05
        
    
    def save(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.__dict__, file, indent=4)
            
    def load(self, config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
        for key, value in config.items():
            if key in self.__dict__:
                setattr(self, key, value)
                
    def load_dict(self, param_dict):
        for key, value in param_dict.items():
            if key in self.__dict__ and value is not None:
                setattr(self, key, value)
    
    