import torch
import numpy as np
import random
import yaml
from data.xia_preprocess import generate_data
from arguments import parse_args_test
from model.astf import ASTF
from utils.save_bvh import SaveBVH

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    args = parse_args_test()

    # Get data information
    with open('xia_dataset.yml', "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    # Get preprocess demo data
    cnt_path = args.demo_datapath+'/'+ args.cnt_clip + '.bvh'
    sty_path = args.demo_datapath+'/'+ args.sty_clip + '.bvh'

    cnt_clip, cnt_feet = generate_data(cnt_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)
    sty_clip, _ = generate_data(sty_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)

    data_dist = np.load(args.dist_datapath)
    Xmean = data_dist['Xmean']
    Xstd = data_dist['Xstd']

    def normalize(x, mean, std):
        x = (x - mean) / std
        return x
    
    cnt_clip = normalize(cnt_clip, Xmean, Xstd)
    sty_clip = normalize(sty_clip, Xmean, Xstd)

    cnt_clip = torch.tensor(cnt_clip, dtype=torch.float).unsqueeze(0).cuda()
    cnt_feet = torch.tensor(cnt_feet, dtype=torch.float).unsqueeze(0).cuda()
    sty_clip = torch.tensor(sty_clip, dtype=torch.float).unsqueeze(0).cuda()

    # Generate temporal mask for the motion sequences & change nan to 0.0
    cnt_m  = cnt_clip[:,1,:,0]
    cnt_length  = sum(~torch.isnan(cnt_m[0])).cpu().numpy()
    cnt_mask = ~torch.isnan(cnt_m).unsqueeze(1).repeat(1, cnt_m.size(1), 1).unsqueeze(1)
    cnt_clip[torch.isnan(cnt_clip)] = 0.0
    sty_m  = sty_clip[:,1,:,0]
    sty_mask = ~torch.isnan(sty_m).unsqueeze(1).repeat(1, sty_m.size(1), 1).unsqueeze(1)
    sty_clip[torch.isnan(sty_clip)] = 0.0
    
    # Load model
    model = ASTF(False, cfg, args)
    model = model.to(device)
    model.load_checkpoint()
    model.eval()
    
    # Perform style transfer
    gen_args = {
        "cnt": cnt_clip,
        "sty": sty_clip,
        "cnt_mask": cnt_mask,
        "sty_mask": sty_mask,
    }

    gen, _ = model.generator(gen_args)
    gen_traj = gen[0,cfg["joint_dims"]:,:cnt_length,:].cpu().detach().numpy()
    gen_body = gen[0, :cfg["joint_dims"], :cnt_length,:].cpu().detach().numpy()
    # # Save output
    save_bvh = SaveBVH(args)
    save_bvh.save_output(gen_body, gen_traj, color="orange",filename=f"03gen_{args.cnt_clip}_{args.sty_clip}")  


    # cnt_traj = cnt_clip[0,cfg["joint_dims"]:,:cnt_length,:].cpu().detach().numpy()
    # cnt_body = cnt_clip[0, :cfg["joint_dims"], :cnt_length,:].cpu().detach().numpy()
    # save_bvh = SaveBVH(args)
    # save_bvh.save_output(cnt_body, cnt_traj, color="blue", filename=f"01cnt")  


    sty_traj = sty_clip[0,cfg["joint_dims"]:,:cnt_length,:].cpu().detach().numpy()
    sty_body = sty_clip[0, :cfg["joint_dims"], :cnt_length,:].cpu().detach().numpy()
    save_bvh = SaveBVH(args)
    save_bvh.save_output(sty_body, sty_traj,  color= (1.0, 0.55, 0.0), filename=f"02sty")  
    save_bvh.save_split1(sty_body, sty_traj,  color= (1.0, 0.55, 0.0), filename=f"03sty")  



