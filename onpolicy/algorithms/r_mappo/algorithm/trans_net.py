import torch
import torch.nn as nn
import math

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.r_mappo.algorithm.frap_net import *

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # 形状: (1, max_len, d_model)

        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # x 的形状 (batch_size, seq_len, d_model)

        # 确保位置编码的维度与输入的 seq_len 匹配
        pe = self.pe[:, :seq_len]  # 取前 seq_len 个位置编码，形状为 (1, seq_len, d_model)

        # 将位置编码添加到输入 x 上
        x = x + pe.to(x.device)  # 将位置编码添加到输入 x
        return self.dropout(x)


class PositionalEncoding_Emb(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding_Emb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x): 
        self.pe = check(self.pe).to(**self.tpdv)     # # 1, N, x
        
        range_ = torch.arange(0, x.shape[0])   # #
        range_ = check(range_).to(**self.tpdv).long()

        x = x + self.pe(range_)
        
        return self.dropout(x)



class TransformerEncoderModel(nn.Module):
    def __init__(self, args, obs_dim, hidden_size, output_size, num_layers, num_heads, dropout, device=torch.device("cpu"), class_token=None):
        super(TransformerEncoderModel, self).__init__()
        
        self.args=args
        self._phase_embedding = ModelBody(1, args.hidden_layer_size, self.args.state_key, device=device, args=args).to(device)
        # hidden_size = args.hidden_layer_size * 7 * 8 # 7 features， 8 phases
        hidden_size = 64
        
        # define observation embeddings
        self._obs_embedding = nn.Linear(
            in_features=obs_dim,
            out_features=hidden_size,
            bias=False
        ).to(device)
        
        # define pos encoding
        self.ps_encoding = PositionalEncoding(
            d_model=hidden_size,
            dropout=0,
            max_len=30000,
            device=device
        )

        # define Transformer Encoder 
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))

        # define output layer
        # self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_layer = nn.Linear(hidden_size, args.trans_hidden)
        
        if class_token is not None:
            self.class_token = class_token
        else:
            self.class_token = None
        
        self.to(device)
        

    def forward(self, src, src_mask=None): # src.shape : T, B, x  T=1 timestep, B=agent number;; src_mask:16,8
        # num_agents = src.shape[0]
        # # phase_emb = self._phase_embedding(src.reshape(T_batch*num_agents, -1))
        # x = src.unsqueeze(0)
        #
        # x = x.permute(1, 0, 2)    # T, B, x  --> B, T, x
        # # x = self._obs_embedding(src)  # B, T, d_model
        # # x = x.permute(1, 0)        # T, B, x
        #
        # # if self.class_token is not None:
        # #     x = torch.cat((x, self.class_token.expand(T_batch, -1, -1)), dim=1)
        # #
        # x_pos = self.ps_encoding(x)   # T, B or B+1, x
        #
        # # add mask
        #
        # src_mask = src_mask.repeat(self.args.trans_heads, 1, 1)
        # # if src_mask is None:
        # #     max_len = src.shape[0]
        # #     src_mask = self.transformer_encoder.generate_square_subsequent_mask(max_len).to(x.device)
        # x_pos = x_pos.permute(1, 0, 2)        # B or B+1, T, x    ##### `(S, N, E)
        # x = self.transformer_encoder(x_pos)  ### Transformer input S (sequence length, here refers to the number of agents), N (batchsize, here is 1 moment), F (feature dimension)
        # #
        # if self.args.use_trans_hidden:
        #     x = self.output_layer(x)      # # B or B+1, T, x
        #
        #
        # x = x.permute(1, 0, 2)        # T, B or B+1, 64
        x = src.unsqueeze(1)
        x = self._obs_embedding(x)
        x = self.ps_encoding(x)
        # x = x + self.positional_encoding[:self.num_agents].unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        if self.args.use_trans_hidden:
            x = self.output_layer(x)

        return x.squeeze(1)
    
    
    def compute_score(self, src, src_mask=None): # src.shape : T, B, x  This refers to T=1 time step, B=agent number
        T_batch, num_agents = src.shape[0], src.shape[1]
        phase_emb = self._phase_embedding(src.reshape(T_batch*num_agents, -1))
        x = phase_emb.reshape(T_batch, num_agents, -1)
        # src = src.permute(1, 0, 2)    # T, B, x  --> B, T, x 
        # x = self._obs_embedding(src)  # B, T, d_model
        # x = x.permute(1, 0, 2)        # T, B, x
        
        # 
        x_pos = self.ps_encoding(x)   # T, B, x
        x_pos_ = x_pos.permute(1, 0, 2)  # B, T, d_model
        if src_mask is not None:
            src_mask = src_mask.repeat(self.args.trans_heads, 1, 1)

        attn = self.transformer_encoder(x_pos_, src_mask)  ### Transformer input S (sequence length, here refers to the number of agents), N (batchsize, here is 1 moment), F (feature dimension)
        #  
        # h, attn = self.transformer_encoder.layers[-1].self_attn(x_pos_, x_pos_, x_pos_, src_mask)    

        return attn


if __name__ == '__main__':
    # device = torch.device("cuda:0")
    obs = torch.rand(1, 32, 56)
    
    model = TransformerEncoderModel(obs_dim=56, hidden_size=32, output_size=64, num_layers=2, num_heads=4, dropout=0.1)
    x = model(obs)# 32.1.64
