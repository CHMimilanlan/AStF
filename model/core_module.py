import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0.1, proj_do_rate=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None):
        b,f, c, p = x.shape
        x = rearrange(x, 'b f c p  -> b f (c p)', )
        
        B, N, C = x.shape  

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        x = rearrange(x, 'b f (c p) -> b f c p', c=c, p=p)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.k = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.v = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, y, z, mask=None):

        B, N, C = x.shape

        q = self.q(x)  
        k = self.k(y) 
        v = self.v(z)

        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = k.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = v.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = torch.matmul(q,k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  groups=in_channels, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        W = F.adaptive_avg_pool2d(x, 1).squeeze(-1).permute(0,2,1)
        b = F.adaptive_max_pool2d(x, 1).squeeze(-1).permute(0,2,1)
        return W,b


class GlobalExtractor(nn.Module):
    def __init__(self, style_channels):
        super().__init__()
        

    def compute_stats(self, x):
        mu = x.mean(dim=[2])     
        var = x.var(dim=[2], unbiased=False) + 1e-8  
        std = var.sqrt()
        
        skew = torch.mean(((x - mu.unsqueeze(-1).unsqueeze(-1)) / std.unsqueeze(-1).unsqueeze(-1) ** 3), dim=[2])
        kurt = torch.mean(((x - mu.unsqueeze(-1).unsqueeze(-1)) / std.unsqueeze(-1).unsqueeze(-1)) ** 4, dim=[2])
        return mu, var, skew, kurt 

    def forward(self, Fs):
        mu, var, skew, kurt = self.compute_stats(Fs) 
        W_mu, b_mu = self.dynamic_net_mu(Fs)
        W_var, b_var = self.dynamic_net_var(Fs)
        W_skew, b_skew = self.dynamic_net_skew(Fs)
        W_kurt, b_kurt = self.dynamic_net_kurt(Fs)

        combined = (W_mu * mu + b_mu + 
                          W_var * var + b_var +
                          W_skew * skew + b_skew + 
                          W_kurt * kurt + b_kurt)

        combined = combined.unsqueeze(-1).unsqueeze(-1)
        return combined  


class SimpleSDM(nn.Module):
    def __init__(self, channels, dim_emb):
        super().__init__()
        self.fq = nn.Linear(channels*dim_emb, channels*dim_emb)
        self.norm_content = nn.InstanceNorm2d(channels)

    def compute_stats(self, x):
        mu = x.mean(dim=[2])     
        var = x.var(dim=[2], unbiased=False) + 1e-8 
        std = var.sqrt()
        
        skew = torch.mean(((x - mu.unsqueeze(2)) / std.unsqueeze(2) ** 3), dim=[2])
        kurt = torch.mean(((x - mu.unsqueeze(2)) / std.unsqueeze(2)) ** 4, dim=[2])

        mu = mu.permute(0,2,1)
        var = var.permute(0,2,1)
        skew = skew.permute(0,2,1)
        kurt = kurt.permute(0,2,1)
        return mu, var, skew, kurt 

    def forward(self, Fc):
        b, frame, joint, dim = Fc.shape
        Q = self.fq(self.norm_content(Fc).reshape(b, frame, -1)) 
        Q = Q.reshape(b, frame, dim, joint).permute(0,2,1,3)
        mu_q, var_q, skew_q, kurt_q = self.compute_stats(Q)
        coef = torch.stack([mu_q, var_q, skew_q, kurt_q], dim=1)
        return coef


class CoreModule(nn.Module):
    def __init__(self, content_channels, style_channels):
        super().__init__()
        self.norm_content = nn.InstanceNorm2d(content_channels)
        self.norm_style = nn.InstanceNorm2d(style_channels)
        
        self.fq = nn.Linear(288, 288)
        self.fk = nn.Linear(288, 288)
        self.fv = nn.Linear(288, 288)

        self.cross_attention_mu = CrossAttention(48, 4, attn_do_rate=0.1, proj_do_rate=0.1)
        self.cross_attention_var = CrossAttention(48, 4, attn_do_rate=0.1, proj_do_rate=0.1)
        self.cross_attention_skew = CrossAttention(48, 4, attn_do_rate=0.1, proj_do_rate=0.1)
        self.cross_attention_kurt = CrossAttention(48, 4, attn_do_rate=0.1, proj_do_rate=0.1)

        self.attn = Attention(6*48, 6)

        self.dynamic_net_mu = DepthwiseSeparableConv(style_channels, style_channels)
        self.dynamic_net_var = DepthwiseSeparableConv(style_channels, style_channels)
        self.dynamic_net_skew = DepthwiseSeparableConv(style_channels, style_channels)
        self.dynamic_net_kurt = DepthwiseSeparableConv(style_channels, style_channels)

        self.fo = nn.Linear(288, 288)

    def compute_lambda_g(self, Q, K):
        Q = Q.flatten(2)
        K = K.flatten(2)
        cosine_sim = F.cosine_similarity(Q, K, dim=2) 
        lambda_g = (cosine_sim + 1) / 2 
        return lambda_g.unsqueeze(-1).unsqueeze(-1)  


    def compute_stats(self, x):
        mu = x.mean(dim=[2])
        var = x.var(dim=[2], unbiased=False) + 1e-8  
        std = var.sqrt()
        
        skew = torch.mean(((x - mu.unsqueeze(2)) / std.unsqueeze(2) ** 3), dim=[2])
        kurt = torch.mean(((x - mu.unsqueeze(2)) / std.unsqueeze(2)) ** 4, dim=[2])

        mu = mu.permute(0,2,1)
        var = var.permute(0,2,1)
        skew = skew.permute(0,2,1)
        kurt = kurt.permute(0,2,1)
        return mu, var, skew, kurt 

    def SDM_Process(self, Fc, Fs):
        b, dim, frame, joint = Fc.shape

        Q = self.fq(self.norm_content(Fc).permute(0,2,1,3).reshape(b, frame, -1)) # (B, C, H, W)
        K = self.fk(self.norm_content(Fs).permute(0,2,1,3).reshape(b, frame, -1))    # (B, C, H, W)
        V = self.fv(Fs.permute(0,2,1,3).reshape(Fs.shape[0], Fs.shape[2], -1))   # (B, C, H, W)
        Q = Q.reshape(b, frame, dim, joint).permute(0,2,1,3)
        K = K.reshape(b, frame, dim, joint).permute(0,2,1,3)
        V = V.reshape(b, frame, dim, joint).permute(0,2,1,3)

        mu_q, var_q, skew_q, kurt_q = self.compute_stats(Q)
        mu_k, var_k, skew_k, kurt_k = self.compute_stats(K)
        mu_v, var_v, skew_v, kurt_v = self.compute_stats(V)

        q_args = (mu_q, var_q, skew_q, kurt_q)
        k_args = (mu_k, var_k, skew_k, kurt_k)
        v_args = (mu_v, var_v, skew_v, kurt_v)

        return q_args, k_args, v_args, Q, K

    def HOSAttn_Process(self, q_args, k_args, v_args, Q, K):
        mu_q, var_q, skew_q, kurt_q = q_args
        mu_k, var_k, skew_k, kurt_k = k_args
        mu_v, var_v, skew_v, kurt_v = v_args
        o_mu = self.cross_attention_mu(mu_q, mu_k, mu_v).unsqueeze(-1).permute(0,2,3,1)
        o_var = self.cross_attention_var(var_q, var_k, var_v).unsqueeze(-1).permute(0,2,3,1)
        o_skew = self.cross_attention_skew(skew_q, skew_k, skew_v).unsqueeze(-1).permute(0,2,3,1)
        o_kurt = self.cross_attention_kurt(kurt_q, kurt_k, kurt_v).unsqueeze(-1).permute(0,2,3,1)

        combined = torch.cat([Q, o_mu,o_var,o_skew,o_kurt], dim=2).permute(0,2,1,3)

        combined = self.attn(combined)
        combined = combined[:,:200,:,:].permute(0,2,1,3)

        lambda_g = self.compute_lambda_g(Q, K)
        Fqk = lambda_g * combined + (1 - lambda_g) * Q 
        return Fqk


    def forward(self, Fc, Fs):
        b, dim, frame, joint = Fc.shape

        q_args, k_args, v_args, Q, K = self.SDM_Process(Fc, Fs)
        Fqk = self.HOSAttn_Process(q_args, k_args, v_args, Q, K)

        Fqk = Fqk.permute(0,2,1,3).reshape(b, frame, -1)
        O = self.fo(Fqk)       
        O = O.reshape(b, frame, dim ,joint)
        O = O.permute(0,2,1,3)

        return Fc + O