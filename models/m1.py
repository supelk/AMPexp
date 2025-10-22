import torch
import torch.nn as nn
"""
Linear
"""
class Patch(nn.Module):
    def __init__(self, patch_len, stride, embedding, padding, d_model):
        super(Patch, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embedding = embedding
        if embedding:
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        if self.embedding:
            x = self.value_embedding(x)
        return x,n_vars

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.m1_type = configs.m1_type
        if self.m1_type == 'only Linear':
            self.block = nn.Linear(self.seq_len, self.pred_len)
        elif self.m1_type == 'with GELU':
            self.block = nn.Sequential(
                nn.Linear(in_features=configs.seq_len, out_features=configs.d_model),
                nn.GELU(),
                nn.Linear(in_features=configs.d_model, out_features=configs.pred_len),
            )
        elif configs.m1_type == 'patch':
            patch_len = 16
            p_model = 32
            stride = 8
            embedding = False
            padding = stride
            patch_num = (self.seq_len - self.patch_len) // stride + 1
            n_model = patch_num * 2
            self.patch = Patch(patch_len, stride, embedding, padding, d_model=self.d_model)
            self.inter_block = nn.Sequential(
                nn.Linear(in_features=patch_len, out_features=p_model),
                nn.GELU(),
                nn.Linear(in_features=p_model, out_features=patch_len),
            )
            self.intar_block = nn.Sequential(
                nn.Linear(in_features=patch_num, out_features=n_model),
                nn.GELU(),
                nn.Linear(in_features=n_model, out_features=patch_num),
            )
            self.channel_block = nn.Sequential(
                nn.Linear(self.channels,self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model,self.channels),
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.m1_type == 'patch':

            y,n_vars = self.patch(x)  # [bs*n,patch_n,patch_len/d_model]

            y = self.block(y)
            y = torch.reshape(y, (-1, n_vars, y.shape[-2], y.shape[-1]))
            y = y.permute(0, 1, 3, 2)

        return self.block(x)
