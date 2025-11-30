import torch
import numpy as np
import random
import yaml
import os
import torch.nn as nn

from feeder.xia_loader import create_data_loader, InputFetcher
from arguments import parse_args_train
from torch.utils.tensorboard import SummaryWriter

from model.astf import ASTF
import model.loss as L

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_loss_dict(iter, loss_dict):
    if iter % 10 == 0:
        info = f"=========================={iter}==========================\n"
        keycount = 0
        for k in loss_dict.keys():
            info +=  f"{k} -- {round(loss_dict[k], 4)} | "
            keycount += 1
            if keycount % 4 == 0:
                info += " \n"
        print(info)


if __name__ == '__main__': 
    args = parse_args_train()
    tb_dir = os.path.join(args.save_path, "logs")
    tb_writer = SummaryWriter(tb_dir)

    # Get data information
    with open('xia_dataset.yml', "r") as fd:
        cfg = yaml.load(fd, Loader=yaml.Loader)

    # Set data loader
    content_loader = create_data_loader(args, cfg, mtype='content')
    style_loader = create_data_loader(args, cfg, mtype='style')
    fetcher = InputFetcher(args, content_loader, style_loader)

    # Set Model
    model = ASTF(True, cfg, args)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=np.arange(args.gpus).tolist())
    
    model.train()

    for iter in range(1, args.total_iters+1):
        data = next(fetcher)

        model.current_iter = iter
        cnt_clip = data['cnt_clip'].type(torch.float32)
        sty_clip = data['sty_clip'].type(torch.float32)
        sty_clip2 = data['sty_clip2'].type(torch.float32)
        cnt_label = data['cnt_label']
        sty_label = data['sty_label']
        cnt_contact = data['cnt_contact']

        # Generate temporal mask for the motion sequences & change nan to 0.0
        cnt_m  = cnt_clip[:,1,:,0]
        cnt_valid_length = []
        for i in range(cnt_m.shape[0]):
            indx = torch.nonzero(~torch.isnan(cnt_m[i]))[-1]
            cnt_valid_length.append(indx)
        cnt_mask = ~torch.isnan(cnt_m).unsqueeze(1).repeat(1, cnt_m.size(1), 1).unsqueeze(1)
        cnt_clip[torch.isnan(cnt_clip)] = 0.0
      
        sty_m  = sty_clip[:,1,:,0]
        sty_valid_length = []
        for i in range(sty_m.shape[0]):
            indx = torch.nonzero(~torch.isnan(sty_m[i]))[-1]
            sty_valid_length.append(indx)
        sty_mask = ~torch.isnan(sty_m).unsqueeze(1).repeat(1, sty_m.size(1), 1).unsqueeze(1)
        sty_clip[torch.isnan(sty_clip)] = 0.0
     
        sty_m2  = sty_clip2[:,1,:,0]
        sty_valid_length2 = []
        for i in range(sty_m2.shape[0]):
            indx = torch.nonzero(~torch.isnan(sty_m2[i]))[-1]
            sty_valid_length2.append(indx)

        sty_mask2 = ~torch.isnan(sty_m2).unsqueeze(1).repeat(1, sty_m2.size(1), 1).unsqueeze(1)
        sty_clip2[torch.isnan(sty_clip2)] = 0.0
        

        D_loss, D_loss_dict = L.mcr_loss(sty_clip, sty_label, sty_mask,cnt_clip,cnt_mask, model, sty_valid_length, cnt_valid_length)
        model.optimizer_MCR.zero_grad()
        D_loss.backward()
        model.optimizer_MCR.step()

        G_loss, G_loss_dict, G_loss_dict_with_lamda = L.G_loss(args, model, cnt_clip, sty_clip, sty_clip2, sty_label, cnt_contact, cnt_mask, sty_mask, sty_mask2)
        print_loss_dict(iter, G_loss_dict_with_lamda)
        model.optimizer_G.zero_grad()
        G_loss.backward()
        model.optimizer_G.step()

        if iter %10 == 0:
            print(iter, 'G_total: %f'%(G_loss), 'G_adv: %f'%(G_loss_dict['G_adv']), 'D_total: %f'%(D_loss), 'mcr_loss: %f'%(D_loss_dict['loss_mcr']),)

        tb_writer.add_scalars('G_total', 
                {"G_total": G_loss,
                }, iter)

        tb_writer.add_scalars('loss', 
                {
                "G_adv": G_loss_dict['G_adv'],
                "D_total": D_loss,
                }, iter)

        # Save checkpoints
        if iter % 10000 == 0 or iter==1:
            if torch.cuda.device_count() > 1:
                model.module.save_checkpoint()
            else:
                model.save_checkpoint()

