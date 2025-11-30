from data.xia_preprocess import generate_data
import torch
from tqdm import tqdm
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, quaternion_to_matrix, geodesic_distance
import numpy as np
from utils.save_bvh import SaveBVH
import os
from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')  



def normalize(x, mean, std):
    x = (x - mean) / std
    return x

def denormalize(x, mean, std):
    x = x * std + mean
    return x


def eval_all(bvh_files, content_full_namedict, eval_datapath, Xmean, Xstd, cfg, model, cntcls_model, stycls_model, args):
    sty_fid_list = []
    sty_acc_count = 0
    cnt_fid_list = []
    cnt_acc_count = 0
    geo_dist_list = []
    # base_dir = "bvh_outputs"
    # os.makedirs(base_dir, exist_ok=True)
    style_falied_txt = ""
    content_falied_txt = ""
    total_count = 0

    style_names = cfg["style_names"]
    style_name_to_idx = {name: i for i, name in enumerate(style_names)}

    content_names = cfg["content_names"]
    content_name_to_idx = {name: i for i, name in enumerate(content_names)}
    print("total length:", len(bvh_files))
    with torch.no_grad(): 
        for i, item in tqdm(enumerate(bvh_files)):
            filename = item.split('/')[-1]
            style, content_num, _ = filename.split('_')

            content = content_full_namedict[int(content_num) - 1]
            content_idx = content_name_to_idx[content]
            
            # content_dir = os.path.join(base_dir, filename.split('.')[0])
            # os.makedirs(content_dir, exist_ok=True)

            for i_ref, item_ref in enumerate(bvh_files):
                filename_ref = item_ref.split('/')[-1]
                style_ref, content_num_ref, _ = filename_ref.split('_')
                content_ref = content_full_namedict[int(content_num_ref) - 1]
                style_idx = style_name_to_idx[style_ref]

                cnt_path = eval_datapath+'/'+ filename
                sty_path = eval_datapath+'/'+ filename_ref

                cnt_clip_raw, _ = generate_data(cnt_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)
                sty_clip_raw, _ = generate_data(sty_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)

                cnt_clip = normalize(cnt_clip_raw, Xmean, Xstd)
                sty_clip = normalize(sty_clip_raw, Xmean, Xstd)

                cnt_clip = torch.tensor(cnt_clip, dtype=torch.float).unsqueeze(0).cuda()
                sty_clip = torch.tensor(sty_clip, dtype=torch.float).unsqueeze(0).cuda()

                # Generate temporal mask for the motion sequences & change nan to 0.0
                cnt_m  = cnt_clip[:,1,:,0]
                cnt_length  = sum(~torch.isnan(cnt_m[0])).cpu().numpy()
                cnt_mask = ~torch.isnan(cnt_m).unsqueeze(1).repeat(1, cnt_m.size(1), 1).unsqueeze(1)
                cnt_clip[torch.isnan(cnt_clip)] = 0.0
                sty_m  = sty_clip[:,1,:,0]
                sty_length  = sum(~torch.isnan(sty_m[0])).cpu().numpy()
                sty_mask = ~torch.isnan(sty_m).unsqueeze(1).repeat(1, sty_m.size(1), 1).unsqueeze(1)
                sty_clip[torch.isnan(sty_clip)] = 0.0

                gen_args = {
                        "cnt": cnt_clip,
                        "sty": sty_clip,
                        "cnt_mask": cnt_mask,
                        "sty_mask": sty_mask,
                }
                gen, _ = model.generator(gen_args)

                # Perform
                sty_feat, sty_output = stycls_model.generator(sty_clip, sty_mask)
                fk_feat_sty, fk_output_sty = stycls_model.generator(gen, cnt_mask)
                cnt_feat, cnt_output = cntcls_model.generator(cnt_clip, cnt_mask)
                fk_feat_cnt, fk_output_cnt = cntcls_model.generator(gen, cnt_mask)
                # get pred
                _, predicted_sty = torch.max(sty_output.data, 1)
                _, predicted_cnt = torch.max(cnt_output.data, 1)
                _, predicted_fk_sty = torch.max(fk_output_sty.data, 1)
                _, predicted_fk_cnt = torch.max(fk_output_cnt.data, 1)

                # dirpath = os.path.join()
                # bvh_Saver(args, cfg, [cnt_clip, sty_clip, gen], 
                #           [cnt_length, sty_length, cnt_length], 
                #           [f"{filename.split('.')[0]}", f"{filename_ref.split('.')[0]}", f"gen"],dirpath)
                # bvh_Saver(args, cfg, [gen], 
                #           [cnt_length], 
                #           [f"{filename.split('.')[0]}_{filename_ref.split('.')[0]}"],content_dir)

                # if predicted_sty == predicted_fk_sty:
                if style_idx == predicted_fk_sty:
                    sty_acc_count += 1
                if content_idx == predicted_fk_cnt:
                    cnt_acc_count += 1

                # if style_idx != predicted_fk_sty:
                #     print(f"{filename.split('.')[0]}_{filename_ref.split('.')[0]}")
                #     style_falied_txt += f"{filename.split('.')[0]}_{filename_ref.split('.')[0]} \n"
                # if content_idx != predicted_fk_cnt:
                #     print(f"{filename.split('.')[0]}_{filename_ref.split('.')[0]}")
                #     content_falied_txt += f"{filename.split('.')[0]}_{filename_ref.split('.')[0]} \n"
                # if predicted_cnt != predicted_fk_cnt:
                #     pass
                    # print(11)

                total_count += 1

                sty_feat = sty_feat.detach().cpu().numpy()
                fk_feat_sty = fk_feat_sty.detach().cpu().numpy()
                cnt_feat = cnt_feat.detach().cpu().numpy()
                fk_feat_cnt = fk_feat_cnt.detach().cpu().numpy()

                sty_gt_mu, sty_gt_cov = calculate_activation_statistics(sty_feat)
                sty_fk_mu, sty_fk_cov = calculate_activation_statistics(fk_feat_sty)
                cnt_gt_mu, cnt_gt_cov = calculate_activation_statistics(cnt_feat)
                cnt_fk_mu, cnt_fk_cov = calculate_activation_statistics(fk_feat_cnt)

                s_fid = calculate_frechet_distance(sty_gt_mu, sty_gt_cov, sty_fk_mu, sty_fk_cov)
                c_fid = calculate_frechet_distance(cnt_gt_mu, cnt_gt_cov, cnt_fk_mu, cnt_fk_cov)
                sty_fid_list.append(s_fid)
                cnt_fid_list.append(c_fid)

                # get geo dist
                cnt_rot = cnt_clip[0, 3:7,:cnt_length]
                gen_rot = gen[0, 3:7,:cnt_length]
                gen_rot = gen_rot.permute(1,2,0)
                cnt_rot = cnt_rot.permute(1,2,0)
                cnt_rot = quaternion_to_matrix(cnt_rot)
                gen_rot = quaternion_to_matrix(gen_rot)
                geo_dist = geodesic_distance(cnt_rot, gen_rot, reduction="none").mean([0,1])
                geo_dist_list.append(geo_dist.detach().cpu().numpy())

    styfid_mean = np.mean(np.asarray(sty_fid_list))
    sty_acc = sty_acc_count / total_count

    cntfid_mean = np.mean(np.asarray(cnt_fid_list))
    cnt_acc = cnt_acc_count / total_count
    mean_geo_dist = np.mean(np.asarray(geo_dist_list))
    with open("style_failed.txt","w") as f:
        f.write(style_falied_txt)        
    with open("content_failed.txt","w") as f:
        f.write(content_falied_txt)        


    return styfid_mean,sty_acc, cntfid_mean, cnt_acc, mean_geo_dist


def saver(args, cfg, clip, length, name):
    sb = SaveBVH(args)
    body = clip[0, :cfg["joint_dims"], :length,:].cpu().detach().numpy()
    traj = clip[0,cfg["joint_dims"]:,:length,:].cpu().detach().numpy()
    sb.save_output(body, traj, filename=f"{name}.bvh")



def bvh_Saver(args, cfg, clips, lengths, names, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    sb = SaveBVH(args)
    for clip, length, name in zip(clips, lengths, names):
        body = clip[0, :cfg["joint_dims"], :length,:].cpu().detach().numpy()
        traj = clip[0,cfg["joint_dims"]:,:length,:].cpu().detach().numpy()
        sb.save_output(body, traj, filename=f"{dir_path}/{name}.bvh")
