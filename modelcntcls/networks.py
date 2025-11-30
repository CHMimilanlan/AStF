from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from modelcntcls.transformer import TransformerEncoder, TransformerDecoder, TransformerModulator

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
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame+1, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_enc_blocks)]
        )
        self.encoder_IN = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(1)]
        )

        self.PSM = TransformerModulator(dim_emb, num_heads, num_bodypart)

        self.learnable_sty_token = nn.Parameter(torch.randn(num_bodypart, dim_emb)).cuda()
        
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


        self.proj1 = nn.Linear(dim_emb*(num_bodypart), 256)

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(128 // 4 ,128),

            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(32 // 4 ,32),

            nn.Linear(32, 6),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, cnt, cnt_mask):
        # Body-part and global translation embedding
        motion = [cnt]
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
  
        ######### Process of the content motion #########
        # Add learnabel style token
        m_cnt = motion_embs[0]
        learnable_sty_token = self.learnable_sty_token.unsqueeze(0).unsqueeze(0)
        learnable_sty_token = learnable_sty_token.repeat(m_cnt.shape[0], 1, 1, 1)
        m_cnt_ = torch.cat((learnable_sty_token, m_cnt), axis=1)

        # Generate mask for attention
        m_cnt_ = self.dropout_cnt1(m_cnt_)
        cnt_mask_ = torch.cat((cnt_mask[:,:,0,:].unsqueeze(2), cnt_mask), axis=2)
        cnt_mask_ = torch.cat((cnt_mask_[:,:,:,0].unsqueeze(3), cnt_mask_), axis=3)

        # Encode content motion
        for i, block in enumerate(self.encoder):
            m_cnt_ = block(m_cnt_, cnt_mask_, last_block=False)

            # Last encoder block with IN
        cnt_of_content_motion = m_cnt_[:,1:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_content_motion = block(cnt_of_content_motion, cnt_mask, last_block=True)

        # Pool content dynamics feater(Y^C) to generate C^C
        cnt_of_content_motion_ = rearrange(cnt_of_content_motion, 'b f p c -> b (p c) f')
        tm_pooling = nn.AvgPool1d(cnt_of_content_motion_.shape[-1])
        pool_cnt_of_content_motion = tm_pooling(cnt_of_content_motion_)
        pool_cnt_of_content_motion = rearrange(pool_cnt_of_content_motion.squeeze(-1), 'b (p c) -> b p c', p=num_bodypart)
        ###############################################

        p1 = torch.flatten(pool_cnt_of_content_motion, start_dim=1)
        p2 = self.proj1(p1)
        p3 = self.head(p2)
        output = self.sigmoid(p3)
        ###############################################

        return p2, output
    

