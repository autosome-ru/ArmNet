import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fastai.vision.all import *

import numpy as np 
from einops import rearrange

from typing import Optional

class DynamicPositionBias(nn.Module):
    '''
    Copyright (c) 2020 Phil Wang
    Licensed under The MIT License (https://github.com/lucidrains/x-transformers/blob/main/LICENSE)
    '''
    
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=192, dim_msa=16, out_dim=12):
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim,
                                    dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, 
                                    out_dim)
        self.dynpos = DynamicPositionBias(dim=in_dim//4, heads=out_dim, depth=2)

    def forward(self, seq_rep):
        L = seq_rep.shape[1]
        seq_rep=self.proj_down1(seq_rep)
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)
        outer_product = rearrange(outer_product, 'b i j m -> b m i j')
        
        pos_bias = self.dynpos(L, L).unsqueeze(0)

        return outer_product + pos_bias



class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = None,
                 dropout: float = 0.10, 
                 bias: bool = True,
                 temperature: float = 1,
                ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if num_heads == None:
            self.num_heads = 1
        else:
            self.num_heads = num_heads
        self.head_size = hidden_dim//self.num_heads
        self.dropout = dropout
        self.bias = bias
        self.temperature = temperature
        

        self.dynpos = DynamicPositionBias(dim = hidden_dim//4,
                                          heads = self.num_heads, 
                                          depth = 2)

        assert hidden_dim == self.head_size*self.num_heads, "hidden_dim must be divisible by num_heads"
        
        self.attn_dropout = nn.Dropout(dropout)
        self.weights = nn.Parameter(
            torch.empty(self.hidden_dim, 3 * self.hidden_dim) #Q, K, V of equal sizes in given order
        )
        self.out_w = nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim) #Q, K, V of equal sizes in given order
        )
        if self.bias:
            self.out_bias = nn.Parameter(
                torch.empty(1,1,self.hidden_dim) #Q, K, V of equal sizes in given order
            )
            torch.nn.init.constant_(self.out_bias, 0.)
            self.in_bias = nn.Parameter(
                torch.empty(1,1, 3*self.hidden_dim) #Q, K, V of equal sizes in given order
            )
            torch.nn.init.constant_(self.in_bias, 0.)
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)
      
    def forward(self, x, adj, mask = None):
        b, l, h = x.shape
        x = x @ self.weights + self.in_bias # b, l, 3*hidden
        Q, K, V = x.view(b, l, self.num_heads, -1).permute(0,2,1,3).chunk(3, dim=3) # b, a, l, head
        
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2,3)/self.temperature/norm)
        
        raw_attention = attention

        i, j = map(lambda t: t.shape[-2], (Q, K))
        attn_bias = self.dynpos(i, j).unsqueeze(0)
        attention = attention + attn_bias

        attention = attention + adj
        
        mask_value = -torch.finfo(attention.dtype).max
        if mask is not None:
            mask = mask.view(b,1,1,-1) 
            attention = attention.masked_fill(~mask, mask_value)
        
        attention = attention.softmax(dim = -1) # b, a, l, l
        attention = self.attn_dropout(attention)
        
        out = attention @ V  # b, a, l, head
        out = out.permute(0,2,1,3).flatten(2,3) # b, a, l, head -> b, l, (a, head) -> b, l, hidden
        if self.bias:
            out = out + self.out_bias
        
        return out, raw_attention
    

'''
Source of conformer implementation:
https://github.com/sooftware/conformer/blob/main/conformer/convolution.py
'''  
    
class DepthwiseConv1D(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels should be divisible by in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class Transpose(nn.Module):
    
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
    
class GLU(nn.Module):
    
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    
class ConvModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        use_drop1d: bool = False,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)), # B E S
            PointwiseConv1d(in_channels, 
                            in_channels * expansion_factor, 
                            stride=1, 
                            padding=0,
                            bias=True),
            GLU(dim=1),
            DepthwiseConv1D(in_channels, 
                            in_channels,
                            kernel_size,
                            stride=1, 
                            padding="same"),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1D(in_channels, 
                            in_channels,
                            stride=1,
                            padding=0, 
                            bias=True),
            nn.Dropout1d(p=dropout) if use_drop1d else nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor, mask = None) -> Tensor:
        outs = self.sequential(inputs).transpose(1, 2) # B S E
        if mask is not None:
            # mask shape is B S
            mask = mask.unsqueeze(2)
            outs = outs.masked_fill(~mask, 0)
        return outs

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = None,
                 ffn_size: int = None,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 attn_kernel_size: int = 17, 
                 attn_dropout: float = 0.10,
                 conv_dropout: float = 0.10,
                 ffn_dropout: float = 0.10,
                 post_attn_dropout: float = 0.10,
                 conv_use_drop1d: bool = False, 
                ):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if ffn_size is None:
            ffn_size = hidden_dim*4
        self.post_norm1 = nn.LayerNorm(hidden_dim)
        self.post_norm2 = nn.LayerNorm(hidden_dim)
        self.post_norm3 = nn.LayerNorm(hidden_dim)
        self.post_norm4 = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(hidden_dim=hidden_dim,
                                           num_heads=num_heads,
                                           dropout=attn_dropout,
                                           bias=True,
                                           temperature=temperature,
                                          )
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(ffn_dropout)
        )

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(ffn_dropout)
        )

        self.convmod = ConvModule(in_channels=hidden_dim,
                                  kernel_size= attn_kernel_size,
                                  dropout=conv_dropout,
                                  use_drop1d=conv_use_drop1d) 
        

    def forward(self, x, adj, mask = None):
        x_in = x
        
        x, raw_attn = self.mhsa(x, adj=adj, mask=mask)
        x = self.post_attn_dropout(x) + x_in

        x = self.post_norm1(x)
        x = self.ffn1(x) + x
        x = self.post_norm2(x)
        x = self.convmod(x, mask=mask) + x
        x = self.post_norm3(x)
        x = self.ffn2(x) + x
        x = self.post_norm4(x)

        return x, raw_attn
   
    
class SELayer2D(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)   
    
class ResConv2dSimple(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=7
                ):  
        super().__init__()
        self.conv = nn.Sequential(
            # b c w h
            nn.Conv2d(in_c,
                      out_c, 
                      kernel_size=kernel_size, 
                      padding="same", 
                      bias=False),
            # b w h c
            nn.BatchNorm2d(out_c), # maybe batchnorm
            SELayer2D(out_c),
            nn.GELU(),
            # b c e 
        )
        
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=1, 
                          bias=False)
            )

    def forward(self, x, bpp_mask = None):
        # b h s s
        h = self.conv(x)
        if bpp_mask is not None:
            bpp_mask = bpp_mask.unsqueeze(1) # b 1 s s 
            h = h.masked_fill(~bpp_mask, 0)

        x = self.res(x) + h
        return x


    
class AdjTransformerEncoder(nn.Module):
    def __init__(self,
                 dim: int  = 192,
                 head_size: int = 32,
                 dropout: float = 0.10,
                 dim_feedforward: int = 192 * 4,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 num_layers: int = 12,
                 num_adj_convs: int =3,
                 ks: int = 3,
                 attn_kernel_size: int = 17, 
                 conv_use_drop1d: bool = False, 
                 use_bppm: bool = False,
                ):
        super().__init__()
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        assert 0 <= num_adj_convs <= num_layers
        self.num_heads = num_heads
        
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim=dim,
                                     num_heads=num_heads,
                                     ffn_size=dim_feedforward,
                                     activation=activation,
                                     temperature=temperature,
                                     attn_kernel_size=attn_kernel_size,
                                     attn_dropout=dropout,
                                     conv_dropout=dropout,
                                     ffn_dropout=dropout,
                                     post_attn_dropout=dropout,
                                     conv_use_drop1d=conv_use_drop1d) 
             for i in range(num_layers)]
        )
        self.conv_layers = nn.ModuleList()
        for i in range(num_adj_convs):
            in_channels = 1 if i == 0 else num_heads * 2
            if not use_bppm:
                in_channels = num_heads * 2
            self.conv_layers.append(ResConv2dSimple(in_c = in_channels,
                                              out_c=num_heads,
                                              kernel_size=ks))
            
            
    def forward(self, x, adj, mask, bpp_mask):
        # adj B S S 
        
        for ind, mod in enumerate(self.layers):
            if ind < len(self.conv_layers):
                conv = self.conv_layers[ind]
                adj = conv(adj, bpp_mask=bpp_mask)
                x, raw_attn = mod(x, adj=adj, mask=mask) # B E S S
                raw_attn = raw_attn.masked_fill(~bpp_mask.unsqueeze(1), 0)
                if ind != len(self.conv_layers) - 1:
                    adj = torch.cat([adj, raw_attn], dim=1)
            else:
                x, raw_attn = mod(x, adj=adj, mask=mask) # B E S S
            

        return x

        
class ArmNet(nn.Module):
    def __init__(self,  
                 adj_ks: int = 3,
                 num_convs: Optional[int] = None,
                 dim=192, 
                 depth=12,
                 head_size=32,
                 attn_kernel_size: int = 17,
                 dropout: float = 0.1,
                 conv_use_drop1d: bool = False, 
                 use_bppm: bool = False,
                 ):
        super().__init__()
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        
        if num_convs is None:
            num_convs = depth
        
        assert 0 <= num_convs <= depth
        self.num_heads = num_heads
        
        self.emb = nn.Embedding(4+3,dim) # 4 nucleotides + 3 tokens
        
        self._use_bppm = use_bppm
        if not use_bppm:
            self.outer_product_mean = Outer_Product_Mean(
                in_dim=dim,
                dim_msa=16,
                out_dim=self.num_heads * 2 if num_convs != 0 else self.num_heads)
        
        self.transformer = AdjTransformerEncoder(
            num_layers=depth,
            num_adj_convs=num_convs,
            dim=dim,
            head_size=head_size,
            ks=adj_ks,
            attn_kernel_size=attn_kernel_size,
            dropout=dropout,
            conv_use_drop1d=conv_use_drop1d,
            use_bppm=use_bppm,
        )
        
        self.proj_out = nn.Sequential(nn.Linear(dim, dim),
                                      nn.GELU(),
                                      nn.Linear(dim, 2))
        
        self.is_good_embed = nn.Embedding(2, dim)
            
    def forward(self, x0):
        mask = x0['forward_mask']
        bpp_mask = x0['conv_bpp_mask']
      
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]  # B S
        bpp_mask = bpp_mask[:, :Lmax, :Lmax] # B S S
            
        
        e = self.emb(x0['seq_int'][:, :Lmax])
    
        x = e
        is_good = x0['is_good']
        e_is_good = self.is_good_embed(is_good) # B E
        e_is_good = e_is_good.unsqueeze(1) # B 1 E
        x = x + e_is_good
        
        if self._use_bppm:
            adj = x0['adj'] 
            adj = adj[:, :Lmax, :Lmax]
            adj = torch.log(adj+1e-5)
            adj = adj.unsqueeze(1) # B 1 S S
        else:
            adj = self.outer_product_mean(x)
        
        x = self.transformer(x, adj, mask=mask, bpp_mask=bpp_mask)
        
        x = self.proj_out(x)
   
        return x