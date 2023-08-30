import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .prompt import Prompt
from .clustering import cluster_dpc_knn, merge_tokens
from finch import FINCH


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu",prompt_num=5):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.prompt_num = prompt_num
        d_keys = d_model // n_heads

        self.t_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.c_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.o_projection = nn.Linear(d_model,
                                          d_keys * n_heads)

    def forward(self, x, noise=None, attn_mask=None):
        if len(x)==2:
            x_t, x_c = x 
            x_o = x_t
        elif len(x)==3:
            x_t, x_c, x_o = x # [B,L,D], [B,C,D], [B,L,D]

        x_t = self.t_projection(x_t)
        x_c = self.c_projection(x_c)
        x_o = self.o_projection(x_o)
        
        if len(x)==2 and noise is not None: # synthetic signal (layer 1)
            x_o = torch.cat((x_o, noise), dim=1)
            x_t = torch.cat((x_t, noise), dim=1)

        # cross-attn (t,c,c)
        new_x, _, _, _, _ = self.attention(
            x_t, x_c, x_c,
            attn_mask=attn_mask
        )
        if len(x)==2 and noise is not None: # synthetic signal (layer 1)
            new_x = new_x[:,:-self.prompt_num,:]
            x_t = x_t[:,:-self.prompt_num,:]

        x1 = x_t + self.dropout(new_x)
        y1 = x1 = self.norm(x1)
        y1 = self.dropout(self.activation(self.conv1(y1.transpose(-1, 1))))
        y1 = self.dropout(self.conv2(y1).transpose(-1, 1))

        # cross-attn (c,t,t)
        new_x, _, _, _, _ = self.attention(
            x_c, x_t, x_t,
            attn_mask=attn_mask
        )
        x2 = x_c + self.dropout(new_x)
        y2 = x2 = self.norm(x2)
        y2 = self.dropout(self.activation(self.conv3(y2.transpose(-1, 1))))
        y2 = self.dropout(self.conv4(y2).transpose(-1, 1))

        # self-attn (t,t,t)
        new_x, attn_wo_soft, attn, mask, sigma = self.attention(
            x_o, x_o, x_o,
            attn_mask=attn_mask,
            use_prior=True
        )
        if len(x)==2 and noise is not None: # synthetic signal (layer 1)
            new_x = new_x[:,:-self.prompt_num,:]
            x_o = x_o[:,:-self.prompt_num,:]

        x3 = x_o + self.dropout(new_x)
        y3 = x3 = self.norm(x3)
        y3 = self.dropout(self.activation(self.conv5(y3.transpose(-1, 1))))
        y3 = self.dropout(self.conv6(y3).transpose(-1, 1))

        return (self.norm(x1 + y1), self.norm(x2 + y2), self.norm(x3 + y3)), attn_wo_soft, attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, cluster_t=1):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.cluster_t = cluster_t

    def forward(self, x, attn_mask=None, noise=None, finch_normal=None):
        attn_list = []
        series_list = []
        prior_list = []
        sigma_list = []

        for layer_i, attn_layer in enumerate(self.attn_layers):
            x, attn, series, prior, sigma = attn_layer(x, noise=noise, attn_mask=attn_mask)
            attn_list.append(attn)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

            if layer_i==2:
                if finch_normal is None: # store normal prototype
                    clustered_feats_lst = []
                    cluster_idx_lst = []
                    for branch in range(3):
                        feat_finch = x[branch]
                        B,L,D = feat_finch.shape
                        which_partition = 0
                        c, num_clust, req_c = FINCH(feat_finch.reshape(B*L, -1).detach().cpu().numpy(), verbose=False) # [BxL,D] -> [BxL,p]
                        c = c[:,which_partition] # -> [BxL]
                        c = torch.Tensor(c).to(feat_finch.device).to(dtype=torch.int64)
                        clustered_feats = merge_tokens(feat_finch.reshape(1, B*L,-1), c.reshape(1, -1), num_clust[which_partition]).squeeze(0) # [1,6000,D]
                        labels =  F.one_hot(c) 

                        clustered_feats_lst.append(clustered_feats)
                        cluster_idx_lst.append(c)

                    finch_out = {'clustered_feats': clustered_feats_lst,
                                'cluster_idx': cluster_idx_lst}
                else: # abnormal
                    logits_lst = []
                    labels_lst = []
                    for branch in range(3):
                        clustered_feats = finch_normal['clustered_feats'][branch]   # prototype of N
                        c = finch_normal['cluster_idx'][branch]                     # from N
                        labels =  F.one_hot(c)                              # from N

                        feat_finch = x[branch]               # from AN
                        B,L,D = feat_finch.shape        # from AN

                        similarity_matrix = torch.matmul(F.normalize(feat_finch.reshape(B*L,-1), dim=-1), F.normalize(clustered_feats, dim=-1).T)

                        pos = similarity_matrix[labels.bool()].unsqueeze(1)
                        neg = similarity_matrix[~labels.bool()].view(B*L,-1)

                        logits = torch.cat([pos, neg], dim=1) / self.cluster_t
                        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feat_finch.device)

                        logits_lst.append(logits)
                        labels_lst.append(labels)

                    finch_out = {'logits': logits_lst,
                                'labels': labels_lst}

        if self.norm is not None:
            x_t = self.norm(x[0])
            x_c = self.norm(x[1])
            x_o = self.norm(x[2])

        return (x_t, x_c, x_o), attn_list, series_list, prior_list, sigma_list, finch_out



class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, cluster_t=1, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, pool_size=10, prompt_num=10):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, win_size, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, prompt_num, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    prompt_num=prompt_num
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            cluster_t=cluster_t
        )

        self.projection_t = nn.Linear(d_model, c_out, bias=True)
        self.projection_c = nn.Linear(d_model, win_size, bias=True)
        self.projection_o = nn.Linear(d_model, c_out, bias=True)
        
        #! exp
        self.prompt = Prompt(ftr_dim=d_model, pool_size=pool_size, prompt_num=prompt_num, channel=enc_in)
   
        
    def forward(self, x, noise=None, feature=None, finch_normal=None):
        enc_out = self.embedding(x)
        enc_out, attn, series, prior, sigmas, finch_out = self.encoder(enc_out, noise=noise, finch_normal=finch_normal)

        enc_out1 = self.projection_t(enc_out[0])
        enc_out2 = self.projection_c(enc_out[1]).transpose(1,2)
        enc_out3 = self.projection_o(enc_out[-1]) 
        enc_out = (enc_out1 + enc_out2 + enc_out3) / 3.
        
        if feature is not None:
            prompt_out = self.prompt(feature)
        else:
            prompt_out = None

        if self.output_attention:
            return enc_out, attn, series, prior, sigmas, finch_out, prompt_out
        else:
            return enc_out, prompt_out  # [B, L, D]
