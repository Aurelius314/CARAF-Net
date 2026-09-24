from preprocess import getDataLoaders
import math
from torch.utils.tensorboard import SummaryWriter
import argparse
from train import *
import random
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(data_loader_dict, args, optim_config, cuda, one_subject, seed=3):
    set_seed(seed)
    if args.dataset_name == 'seed3':
        args.iteration = 7
    elif args.dataset_name == 'seed4':
        args.iteration = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    acc = trainCARAF(data_loader_dict, optim_config, cuda, args, args.iteration, one_subject)
    return acc

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='CARAF')
    parser.add_argument("--way", type=str, default='CARAF/seed4', help="name of current way")
    parser.add_argument("--index", type=str, default='0', help="tensorboard index")
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed4', help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument('--number_of_source', type=int, default=14,
                    help='Number of source subjects')
    
    args = parser.parse_args()
    args.source_subjects = args.subjects-1
    args.seed3_path = "{data_path}/SEED/"
    args.seed4_path = "{data_path}/SEED-IV/"
    if cuda:
        args.num_workers_train =2
        args.num_workers_test = 2
    else:
        args.num_workers_train = 0
        args.num_workers_test = 0
    if args.dataset_name == "seed3":
        args.path = args.seed3_path
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 32 
        args.epoch = 100 
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 64 
        args.epoch = 200 
    else:
        print("need to define the input dataset")
    args.epoch_preTraining = args.epoch
        
    cuda = torch.cuda.is_available()
    if cuda:
        print("GPU is available. Running on:", torch.cuda.get_device_name(0))
        torch.cuda.set_device(0) 
    else:
        print("No GPU detected. Running on CPU.")
    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    acc_list=[]
    for one_subject in range(0, 15):
        print(f"subject{one_subject}\n")
        source_loaders, test_loader,test_loader2 = getDataLoaders(one_subject, args)
        data_loader_dict = {"source_loader": source_loaders, "test_loader":test_loader,"test_loader2":test_loader2}
        acc = main(data_loader_dict, args, optim_config, cuda, one_subject)