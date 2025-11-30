import torch
import numpy as np
import random
import yaml
import os
from data.xia_preprocess import generate_data
from arguments import parse_args_test
from model.astf import ASTF
from modelcntcls.clsastf import CNTCLSASTF
from modelstycls.clsastf import STYCLSASTF
from utils.eval_tools import *


def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def normalize(x, mean, std):
    x = (x - mean) / std
    return x
    
def denormalize(x, mean, std):
    x = x * std + mean
    return x

if __name__ == '__main__':
    eval_datapath = 'data/preprocessed_xia_test'
    args = parse_args_test()

    with open('xia_dataset.yml', "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Load model
    model = ASTF(False, cfg, args)
    model = model.to(device)
    model.load_checkpoint()
    model.eval()
    # Load cnt cls model
    cntcls_path = args.cntcls_path
    cntcls_model = CNTCLSASTF(False, cfg, args)
    cntcls_model = cntcls_model.to(device)
    cntcls_model.load_checkpoint(cntcls_path)
    cntcls_model.eval()

    # Load sty cls model
    stycls_path = args.stycls_path
    stycls_model = STYCLSASTF(False, cfg, args)
    stycls_model = stycls_model.to(device)
    stycls_model.load_checkpoint(stycls_path)
    stycls_model.eval()

    data_dist = np.load(args.dist_datapath)
    Xmean = data_dist['Xmean']
    Xstd = data_dist['Xstd']

    bvh_files = get_bvh_files(eval_datapath)
    content_full_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]]

    styfid,sty_acc, cntfid, cnt_acc, geo_dist = eval_all(bvh_files, content_full_namedict, eval_datapath, Xmean, Xstd, cfg, model, cntcls_model, stycls_model, args)
    print(f"styfid: {styfid} | sty_acc: {sty_acc}")
    print(f"cntfid: {cntfid} | cnt_acc: {cnt_acc}")
    print(f"geo_dist: {geo_dist}")
    content = f"styfid: {styfid} | sty_acc: {sty_acc} \n  cntfid: {cntfid} | cnt_acc: {cnt_acc} \n geo_dist: {geo_dist} \n"
    with open("eval_all.txt", "w") as f:
        f.write(content)
 