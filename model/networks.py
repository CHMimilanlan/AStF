from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from model.core_module import CoreModule, SimpleSDM

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..')
sys.path.append(models_dir)
# print(sys.path)

from model.transformer import TransformerEncoder, TransformerDecoder

num_bodypart = 6

LLeg_idx = [1, 2, 3, 4]
RLeg_idx = [5, 6, 7, 8]
Trunk_idx = [9, 10, 11, 12]
LArm_idx = [13, 14, 15, 16]
RArm_idx = [17, 18, 19, 20]
Root_idx = [0]

bodypart_idx = []
bodypart_idx.append(LLeg_idx)
bodypart_idx.append(RLeg_idx)
bodypart_idx.append(Trunk_idx)
bodypart_idx.append(LArm_idx)
bodypart_idx.append(RArm_idx)
bodypart_idx.append(Root_idx)

class StyleTransformer(nn.Module):
    def __init__(self, cfg, num_frame, dim_emb, num_heads, num_enc_blocks, num_dec_blocks):
        super().__init__()
        
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame+4, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_enc_blocks)]
        )

        self.encoder_IN = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(1)]
        )

        self.hsi = CoreModule(content_channels=48, style_channels=48)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_part=num_bodypart, num_frame=num_frame, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_dec_blocks)]
        )

        self.cnt_simplesdm = SimpleSDM(num_bodypart, dim_emb)
        self.sty_simplesdm = SimpleSDM(num_bodypart, dim_emb)

        self.d = cfg["joint_dims"]  # pos+rot (positions:3, rotations:4, root_trajectory:4)
        self.part_emb = nn.ModuleList([
            nn.Linear(len(bodypart_idx[i])*self.d, dim_emb)
            for i in range(num_bodypart-1)]) 
        self.part_emb.append(nn.Linear(self.d, int(dim_emb/2))) # Root pos+rot
        self.part_emb.append(nn.Linear(4, int(dim_emb/2))) # Root vel+rvel

        self.part_upsample = nn.ModuleList([
            nn.Linear(dim_emb, len(bodypart_idx[i])*self.d)
             for i in range(num_bodypart-1)])
        self.part_upsample.append(nn.Linear(dim_emb, self.d))
        self.part_upsample.append(nn.Linear(dim_emb, 4))

        self.dropout_cnt1 = nn.Dropout(p=0.1)
        self.dropout_sty1 = nn.Dropout(p=0.1)
        self.dropout_cnt2 = nn.Dropout(p=0.1)
        self.dropout_sty2 = nn.Dropout(p=0.1)


    def common_forward(self, cnt, sty, cnt_mask, sty_mask,):
        motion = [cnt, sty]
        motion_embs = []

        for i in range(2):
            # LLeg, RLeg, Trunk, LArm, RArm
            part = []
            for j in range(0, num_bodypart-1):
                part.append(motion[i][:, :self.d, :, bodypart_idx[j]])  
            Root_pos = motion[i][:, :self.d, :, Root_idx]
            part.append(Root_pos)
            traj = motion[i][:,self.d:, :, [0]]  
            part.append(traj)
            part_emb = []
            for j, p_ in enumerate(part):
                p = rearrange(p_, 'b c f p  -> b f (c p)', )  
                part_emb.append(self.part_emb[j](p).unsqueeze(2)) 
  
            part_emb[-2] = torch.cat((part_emb[-2], part_emb[-1]), axis=-1)
            del part_emb[-1]

            motion_emb = part_emb[0] 
            for j in range(1, len(part_emb)):
                motion_emb = torch.cat((motion_emb, part_emb[j]), axis=2)
            
            motion_embs.append(motion_emb)
  
        m_cnt = motion_embs[0]
        cnt_stat_coef = self.cnt_simplesdm(m_cnt)
        m_cnt_ = torch.cat((cnt_stat_coef, m_cnt), axis=1)

        # Generate mask for attention
        m_cnt_ = self.dropout_cnt1(m_cnt_)
        cnt_mask_ = torch.cat((cnt_mask[:,:,0,:].unsqueeze(2).repeat(1,1,4,1), cnt_mask), axis=2)
        cnt_mask_ = torch.cat((cnt_mask_[:,:,:,0].unsqueeze(3).repeat(1,1,1,4), cnt_mask_), axis=3)
        
        # Encode content motion
        for i, block in enumerate(self.encoder):
            m_cnt_ = block(m_cnt_, cnt_mask_, last_block=False)

        cnt_of_content_motion = m_cnt_[:,4:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_content_motion = block(cnt_of_content_motion, cnt_mask, last_block=True)

        ###############################################
        m_sty = motion_embs[1]
        sty_stat_coef = self.sty_simplesdm(m_sty)

        m_sty_ = torch.cat((sty_stat_coef, m_sty), axis=1)
        m_sty_ = self.dropout_sty1(m_sty_)

        # Generate mask for attention
        sty_mask_ = torch.cat((sty_mask[:,:,0,:].unsqueeze(2).repeat(1,1,4,1), sty_mask), axis=2)
        sty_mask_ = torch.cat((sty_mask_[:,:,:,0].unsqueeze(3).repeat(1,1,1,4), sty_mask_), axis=3)

        # Encode style motion
        for i, block in enumerate(self.encoder):
            m_sty_ = block(m_sty_, sty_mask_, last_block=False)

        # Last encoder block with IN
        cnt_of_style_motion = m_sty_[:,4:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_style_motion = block(cnt_of_style_motion, sty_mask, last_block=True)

        ###############################################
        cnt_of_content_motion = rearrange(cnt_of_content_motion, 'b f p c  -> b c f p', )
        cnt_of_style_motion = rearrange(cnt_of_style_motion, 'b f p c  -> b c f p', )
        cnt_of_content_motion = self.hsi(cnt_of_content_motion,cnt_of_style_motion)
        cnt_of_content_motion = rearrange(cnt_of_content_motion, 'b c f p  -> b f p c', )
        ###############################################

        # Generator
        m_gen = self.dropout_cnt2(cnt_of_content_motion)
        for i, block in enumerate(self.decoder):
            m_gen = block(m_gen, cnt_mask=cnt_mask)  # b f p c 

        # Part to joints                
        gen_part = []
        for i in range(m_gen.shape[2]):
            if i == 5:
                num_joint = 1 # root
            else:
                num_joint = 4
            x = self.part_upsample[i](m_gen[:,:,i,:])
            x = rearrange(x, 'b f (j c) -> b f j c', j=num_joint)
            gen_part.append(x)

        x = self.part_upsample[-1](m_gen[:,:,-1,:]).unsqueeze(2)
        x = x.expand(-1,-1, cnt.shape[3],-1)
        gen_part.append(x)

        gen_body = []
        gen_body.append(gen_part[5]) # 0
        gen_body.append(gen_part[0]) # 1,2,3,4
        gen_body.append(gen_part[1]) # 5,6,7,8
        gen_body.append(gen_part[2]) # 9,10,11,12
        gen_body.append(gen_part[3]) # 13,14,15,16
        gen_body.append(gen_part[4]) # 17,18,19,20
        gen_body = torch.cat(gen_body, axis=2)
        
        gen_body = torch.cat((gen_body, gen_part[-1]), axis=-1)

        gen_motion = rearrange(gen_body, 'b f j c -> b c f j', )
        align_sty_latent = rearrange(cnt_of_style_motion, ' b c f p -> b f p c')

        return gen_motion, align_sty_latent

    def align_forward(self, gen_motion, sty_mask):
        motion = [gen_motion]
        motion_embs = []
        for i in range(1):
            # LLeg, RLeg, Trunk, LArm, RArm
            part = []
            for j in range(0, num_bodypart-1):
                part.append(motion[i][:, :self.d, :, bodypart_idx[j]]) 
            Root_pos = motion[i][:, :self.d, :, Root_idx]  
            part.append(Root_pos)
            traj = motion[i][:,self.d:, :, [0]]  
            part.append(traj)
            part_emb = []
            for j, p_ in enumerate(part):
                p = rearrange(p_, 'b c f p  -> b f (c p)', )  
                part_emb.append(self.part_emb[j](p).unsqueeze(2))  
  
            part_emb[-2] = torch.cat((part_emb[-2], part_emb[-1]), axis=-1)
            del part_emb[-1]

            motion_emb = part_emb[0] 
            for j in range(1, len(part_emb)):
                motion_emb = torch.cat((motion_emb, part_emb[j]), axis=2)
            
            motion_embs.append(motion_emb)
  
        m_sty = motion_embs[0]
        sty_stat_coef = self.sty_simplesdm(m_sty)
        m_sty_ = torch.cat((sty_stat_coef, m_sty), axis=1)
        m_sty_ = self.dropout_sty1(m_sty_)

        sty_mask_ = torch.cat((sty_mask[:,:,0,:].unsqueeze(2).repeat(1,1,4,1), sty_mask), axis=2)
        sty_mask_ = torch.cat((sty_mask_[:,:,:,0].unsqueeze(3).repeat(1,1,1,4), sty_mask_), axis=3)

        for i, block in enumerate(self.encoder):
            m_sty_ = block(m_sty_, sty_mask_, last_block=False)

        cnt_of_style_motion = m_sty_[:,4:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_style_motion = block(cnt_of_style_motion, sty_mask, last_block=True)

        cnt_of_style_motion = rearrange(cnt_of_style_motion, 'b f p c  -> b c f p', )
        return cnt_of_style_motion

    def forward(self, args, align_flag=False):
        if not align_flag:
            cnt = args["cnt"]
            sty = args["sty"]
            cnt_mask = args["cnt_mask"]
            sty_mask = args["sty_mask"]
            gen_motion = self.common_forward(cnt, sty, cnt_mask, sty_mask)
            return gen_motion
        else:
            gen = args["gen"]
            cnt_mask = args["cnt_mask"]
            gen_latent = self.align_forward(gen, cnt_mask)
            return gen_latent


class MCRDiscriminator(nn.Module):
    def __init__(self, cfg, num_frame, dim_emb, num_heads, num_enc_blocks):
        super().__init__()

        self.simplesdm = SimpleSDM(num_bodypart+1, dim_emb)
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart+1, num_frame=num_frame+4, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_enc_blocks)]
        )

        self.d = cfg["joint_dims"]  # pos+rot (positions:3, rotations:4, root_trajectory:4)
        self.part_emb = nn.ModuleList([
            nn.Linear(len(bodypart_idx[i])*self.d, dim_emb)
            for i in range(num_bodypart-1)]) 
        self.part_emb.append(nn.Linear(self.d, dim_emb)) # pos+rot+traj
        self.part_emb.append(nn.Linear(4, dim_emb)) #traj

        self.dropout = nn.Dropout(p=0.1)
        num_style_cat = len(cfg["style_names"])


        self.head = nn.Sequential(
            nn.Linear(dim_emb*(num_bodypart+1), dim_emb),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_emb, num_style_cat)
        )

        self.dcr = MCRModule(7, 48)
        
        
    def forward(self, motion, style_label, mask):
        # LLeg, RLeg, Trunk, LArm, RArm
        part = []
        for j in range(0, num_bodypart-1):
            part.append(motion[:, :self.d, :, bodypart_idx[j]])
        Root_pos = motion[:, :self.d, :, Root_idx]
        part.append(Root_pos)
        traj = motion[:,self.d:, :, [0]]  # b,4,200
        part.append(traj)
        
        part_emb = []
        for j, p_ in enumerate(part):
            p = rearrange(p_, 'b d f p  -> b f (d p)', )
            part_emb.append(self.part_emb[j](p).unsqueeze(2))

        motion_emb = part_emb[0]
        for j in range(1, len(part)):
            motion_emb = torch.cat((motion_emb, part_emb[j]), axis=2)

        stat_coef = self.simplesdm(motion_emb)
        motion_emb = torch.cat((stat_coef, motion_emb), axis=1)

        # mask extension
        mask = torch.cat((mask[:,:,0,:].unsqueeze(2).repeat(1,1,4,1), mask), axis=2)
        mask = torch.cat((mask[:,:,:,-1].unsqueeze(3).repeat(1,1,1,4), mask), axis=3)
            
        # transformer encoder blocks
        motion_emb = self.dropout(motion_emb)
        for i, block in enumerate(self.encoder):
            motion_emb = block(motion_emb, mask)

        class_token = motion_emb[:,0,:,:]
        motion_seq_emb = motion_emb[:,4:,:,:]

        class_token = rearrange(class_token, 'b p c  -> b (p c)', )

        out = self.head(class_token) 

        out = out.view(out.shape[0], -1)
        idx = range(style_label.size(0))
        out = out[idx, style_label]
        return motion_seq_emb, class_token, out



class MCRModule(nn.Module):
    def __init__(self, in_channels, dim_emb):
        super().__init__()
        self.linear1 = nn.Linear(in_channels* dim_emb, in_channels* dim_emb)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.linear2 = nn.Linear(in_channels* dim_emb, in_channels* dim_emb)

    def forward(self, x):
        b, dim, frames, joints = x.shape
        x = x.permute(0,2,1,3).reshape(-1, frames, dim*joints)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = x.reshape(b, frames, dim ,joints)
        x = x.permute(0,2,1,3)
        return x