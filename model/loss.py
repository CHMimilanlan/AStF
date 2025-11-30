from einops import rearrange
import yaml
import torch
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

# Get data information
with open('xia_dataset.yml', "r") as fd:
    cfg = yaml.load(fd, Loader=yaml.Loader)


def generate_augmented_views(x, sty_mask,sty_valid_length):
    # batch, _, H, W = x.shape
    batch, feat, frame, joint = x.shape

    sty_valid_length = torch.tensor(sty_valid_length)
    # max_int = torch.div(sty_valid_length,2,rounding_mode='trunc') - (crop_size - overlap)

    x1 = []
    x2 = []
    sty_mask1 = []
    sty_mask2 = []
    valid_idx_list = []

    for i in range(batch):
        xtmp = x[i,...]
        mtmp = sty_mask[i,...]
        valid_idx = sty_valid_length[i]
        half_valid = torch.div(valid_idx,2, rounding_mode='trunc')
        valid_idx_list.append(half_valid)
        xtmp_1 = xtmp.clone()
        xtmp_1[:,half_valid:,:] = 0
        mtmp_1 = mtmp.clone()
        mtmp_1[:,:,half_valid:] = False
        xtmp_2 = xtmp.clone()
        mtmp_2 = mtmp.clone()
        if valid_idx % 2 == 0:
            xtmp_2[:,:half_valid,:] = xtmp_2[:,half_valid:valid_idx,:]
        else:
            xtmp_2[:,:half_valid,:] = xtmp_2[:,half_valid:valid_idx-1,:]
        xtmp_2[:,half_valid:,:] = 0
        mtmp_2[:,:,half_valid:] = False

        x1.append(xtmp_1)
        sty_mask1.append(mtmp_1)
        x2.append(xtmp_2)
        sty_mask2.append(mtmp_2)

    x1 = torch.stack(x1)
    x2 = torch.stack(x2)
    sty_mask1 = torch.stack(sty_mask1)
    sty_mask2 = torch.stack(sty_mask2)

    return x1, x2, sty_mask1, sty_mask2, valid_idx_list

def mcr_sim_loss(r, z, valid_idx_list):
    sim_losses = []
    for idx, s in enumerate(valid_idx_list):
        r_tmp = r[idx, :, :s, :]
        z_tmp = z[idx, :, :s, :]
        
        r_flat = r_tmp.flatten(1) 
        z_flat = z_tmp.flatten(1)
        cosine_sim = F.cosine_similarity(r_flat, z_flat, dim=1)  
        
        sim_loss = -cosine_sim.mean()  
        sim_losses.append(sim_loss)
    
    total_loss = torch.stack(sim_losses).mean()  
    return total_loss




def fill_clip(sty_clip, style_mask, sty_valid_length, max_fill=50):
    b,_,frame,_ = sty_clip.shape
    for i in range(b):
        valid_idx = sty_valid_length[i]
        fill_index = valid_idx.clone()
        if valid_idx < max_fill:
            fill_tmp = sty_clip[i,:,:valid_idx, :]
            mask_tmp = style_mask[i,:,:, :valid_idx]
            fillflag = 0 
            while fill_index < max_fill:
                if fillflag == 0:
                    sty_clip[i,:,fill_index:fill_index+valid_idx, :] = torch.flip(fill_tmp, dims=[1])
                    style_mask[i,:,:, fill_index:fill_index+valid_idx] = mask_tmp
                    
                    fill_index += valid_idx
                    fillflag = 1
                else:
                    sty_clip[i,:,fill_index:fill_index+valid_idx, :] = fill_tmp
                    style_mask[i,:,:, fill_index:fill_index+valid_idx] = mask_tmp

                    fill_index += valid_idx
                    fillflag = 0
        sty_valid_length[i] = fill_index
    return sty_clip, style_mask, sty_valid_length


def find_vaid_index(cnt_valid_length, valid_idx_list):
    gen_valid_idx_list = []
    for cnt, sty in zip(cnt_valid_length, valid_idx_list):
        if cnt < sty:
            gen_valid_idx_list.append(cnt)
        else:
            gen_valid_idx_list.append(sty)
    return gen_valid_idx_list


def mcr_loss(sty_clip, sty_label, sty_mask,cnt_clip,cnt_mask, model, sty_valid_length, cnt_valid_length):
    split_sty_clip = sty_clip.clone()
    split_sty_mask = sty_mask.clone()

    x1, x2, sty_mask1, sty_mask2, valid_idx_list = generate_augmented_views(split_sty_clip, split_sty_mask, sty_valid_length)

    features_x1,_, _ = model.MCRdiscriminator(x1, sty_label, sty_mask1)
    features_x2,_, _ = model.MCRdiscriminator(x2, sty_label, sty_mask2)

    features_x1 = features_x1.permute(0,3,1,2)
    features_x2 = features_x2.permute(0,3,1,2)
    r1 = model.MCRdiscriminator.dcr(features_x1)
    r2 = model.MCRdiscriminator.dcr(features_x2)

    z1 = features_x1
    z2 = features_x2

    loss_mcr = (1/3) * mcr_sim_loss(r1, z2.detach(),valid_idx_list) + (1/3) * mcr_sim_loss(r2, z1.detach(), valid_idx_list)
    sty_clip.requires_grad_()
    _ ,_, real_disc_out = model.MCRdiscriminator(sty_clip, sty_label, sty_mask)
    
    loss_real = adv_loss(real_disc_out, 1)
    loss_reg = r1_reg(real_disc_out, sty_clip)

    gen_args = {
            "cnt": cnt_clip,
            "sty": sty_clip,
            "cnt_mask": cnt_mask,
            "sty_mask": sty_mask,
    }
    gen, _ = model.generator(gen_args)
    features_gen, _, fake_disc_out = model.MCRdiscriminator(gen, sty_label, cnt_mask)

    features_gen = features_gen.permute(0,3,1,2)
    r_gen = model.MCRdiscriminator.mcr(features_gen)

    gen_valid_idx_list = find_vaid_index(cnt_valid_length, sty_valid_length)
    features_sty, _, _ = model.MCRdiscriminator(sty_clip, sty_label, sty_mask)
    features_sty = features_sty.permute(0,3,1,2)
    z_sty = features_sty


    loss_mcr += (1/3) * mcr_sim_loss(r_gen, z_sty.detach(),gen_valid_idx_list)

    loss_fake = adv_loss(fake_disc_out, 0)
    loss = loss_real + loss_fake + 1*loss_reg + loss_mcr

    loss_dict = OrderedDict([('D_loss', loss.item()),
                            ('D_real', loss_real.item()),
                            ('D_fake', loss_fake.item()),
                            ('loss_mcr', loss_mcr.item()),
                            ('D_reg', loss_reg.item())])

    return loss,loss_dict



def G_loss(args, model, cnt_clip, sty_clip, sty_clip2, sty_label, cnt_contact, cnt_mask, sty_mask, sty_mask2):
    posrot = cfg["joint_dims"]
    valid_token = cnt_mask[:,0,0,:]
    valid_token_sty = sty_mask[:,0,0,:]

    # adv loss
    gen_args = {
            "cnt": cnt_clip,
            "sty": sty_clip,
            "cnt_mask": cnt_mask,
            "sty_mask": sty_mask,
    }
    gen, sty_latent1  = model.generator(gen_args)
    _,_, fake_disc_out = model.MCRdiscriminator(gen, sty_label, cnt_mask)
    loss_adv = adv_loss(fake_disc_out, 1)

    # reconstruction loss
    recon_gen_args = {
            "cnt": cnt_clip,
            "sty": cnt_clip,
            "cnt_mask": cnt_mask,
            "sty_mask": cnt_mask,
    }
    gen_recon,_ = model.generator(recon_gen_args)
    gen_recon_valid = rearrange(gen_recon, 'b c f j -> b f c j ', )
    cnt_clip_valid = rearrange(cnt_clip, 'b c f j -> b f c j ', )
    loss_recon = torch.mean((gen_recon_valid[valid_token==True] - cnt_clip_valid[valid_token==True]).norm(dim=2))


    # cycle content consistency loss
    cyc_gen_args = {
            "cnt": gen,
            "sty": cnt_clip,
            "cnt_mask": cnt_mask,
            "sty_mask": cnt_mask,
    }
    gen_cycle, _ = model.generator(cyc_gen_args)
    gen_cycle_valid = rearrange(gen_cycle, 'b c f j -> b f c j ', )
    cnt_clip_valid = rearrange(cnt_clip, 'b c f j -> b f c j ', )
    loss_cycle_c = torch.mean((gen_cycle_valid[valid_token==True] - cnt_clip_valid[valid_token==True]).norm(dim=2))


    # cycle style consistency loss 
    cyc2_gen_args = {
            "cnt": sty_clip,
            "sty": gen,
            "cnt_mask": sty_mask,
            "sty_mask": cnt_mask,
    }
    gen_cycle2, _ = model.generator(cyc2_gen_args)
    gen_cycle2_valid = rearrange(gen_cycle2, 'b c f j -> b f c j ', )
    sty_clip_valid = rearrange(sty_clip, 'b c f j -> b f c j ', )
    loss_cycle_s = torch.mean((gen_cycle2_valid[valid_token_sty==True] - sty_clip_valid[valid_token_sty==True]).norm(dim=2))
    

    ############  physics-based loss #############
    # velocity regularization 
    vel_gen = gen[:,:,1:,:] - gen[:,:,:-1,:]
    pad = torch.zeros(vel_gen.shape[0], vel_gen.shape[1], 1, vel_gen.shape[3]).cuda()
    vel_gen = torch.cat((vel_gen, pad), axis=2)
    vel_gen_valid = rearrange(vel_gen[:,:posrot,:,:], 'b c f j -> b f c j ', )
    reg_vel = torch.mean((vel_gen_valid[valid_token==True]).norm(dim=2))

    # acceleration regularization
    acc_gen = vel_gen[:,:,1:,:] - vel_gen[:,:,:-1,:] 
    acc_gen = torch.cat((acc_gen, pad), axis=2)
    acc_gen_valid = rearrange(acc_gen[:,:posrot, :,:], 'b c f j -> b f c j ', )
    global_acc_valid = rearrange(vel_gen[:,posrot:,:,:], 'b c f j -> b f c j ', )
    reg_acc = torch.mean((acc_gen_valid[valid_token==True]).norm(dim=2)) + torch.mean((global_acc_valid[valid_token==True]).norm(dim=2)) 

    ## foot contact regularization
    gen_cycle_foot = gen_cycle[:,:3,:,(3,4,7,8)]
    gen_cycle_foot_vel = gen_cycle_foot[:,:,1:,:] - gen_cycle_foot[:,:,:-1,:]
    gen_cycle_foot_vel_sq = torch.norm(gen_cycle_foot_vel, dim=1)
    gen_cycle_foot_vel_sq = gen_cycle_foot_vel_sq[cnt_contact[:,1:,:] == 1]
    reg_contact = torch.sum(gen_cycle_foot_vel_sq)/len(gen_cycle_foot_vel_sq)
    ##############################################

    ########### style disentanglement loss #######
    with torch.no_grad():
        dis_args = {
            "cnt": cnt_clip,
            "sty": sty_clip2,
            "cnt_mask": cnt_mask,
            "sty_mask": sty_mask2,
        }
        gen2, _ = model.generator(dis_args)


    gen_valid = rearrange(gen, 'b c f j -> b f c j ', )
    gen2_valid = rearrange(gen2, 'b c f j -> b f c j ', )
    loss_sty_disentangle = torch.mean((gen_valid[valid_token==True] - gen2_valid[valid_token==True]).norm(dim=2))
    ##############################################

    align_args = {
        "gen": gen,
        "cnt_mask": cnt_mask,
    }
    gen_latent = model.generator(align_args, align_flag=True)
    gen_latent = gen_latent.permute(0,2,3,1)
    sty_align_loss = torch.mean((sty_latent1 - gen_latent).norm(dim=2))

  
    loss = args.lambda_adv*loss_adv + args.lambda_recon*loss_recon + args.lambda_cyc_c*loss_cycle_c + args.lambda_cyc_s*loss_cycle_s \
    + args.lambda_reg_vel*reg_vel + args.lambda_reg_acc*reg_acc + args.lambda_reg_contact*reg_contact \
    + args.lambda_sty_disentangle*loss_sty_disentangle + 1 * sty_align_loss

    
    loss_dict = OrderedDict([
                            ('G_adv', loss_adv.item()),
                            ('G_recon', loss_recon.item()),
                            ('G_cyc-c', loss_cycle_c.item()),
                            ('G_cyc-s', loss_cycle_s.item()),
                            ('G_reg_vel', reg_vel.item()),
                            ('G_reg_acc', reg_acc.item()),
                            ('G_reg_contact', reg_contact.item()),
                            ('G_loss_sty_disentangle', loss_sty_disentangle.item())
                            ])

    loss_dict_with_lamda = OrderedDict([
                            ('G_adv', args.lambda_adv*loss_adv.item()),
                            ('G_recon', args.lambda_recon*loss_recon.item()),
                            ('G_cyc-c', args.lambda_cyc_c*loss_cycle_c.item()),
                            ('G_cyc-s', args.lambda_cyc_s*loss_cycle_s.item()),
                            ('G_reg_vel', args.lambda_reg_vel*reg_vel.item()),
                            ('G_reg_acc', args.lambda_reg_acc*reg_acc.item()),
                            ('G_reg_contact', args.lambda_reg_contact*reg_contact.item()),
                            ('G_loss_sty_disentangle', args.lambda_sty_disentangle*loss_sty_disentangle.item()),
                            ('sty_align_loss', 1*sty_align_loss.item())
                            ])
    return loss, loss_dict, loss_dict_with_lamda


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    return loss

def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, 
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    
    return reg
