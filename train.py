import argparse
import os, sys
import wandb
import torch
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.learning.train_part import train

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_to_int_list(s):
    s = s.strip().strip("'").strip('"')
    return list(map(int, s.strip().split()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='utils.model.varnet.VarNet')
    parser.add_argument('--restart_from_checkpoint', type=Path, default=None)
    parser.add_argument('--aug_on', type=str_to_bool, default=False)
    args, _ = parser.parse_known_args()

    if args.restart_from_checkpoint is not None:
        parser.add_argument('--continue_lr_scheduler', type=str_to_bool, default=True)
    if not args.model_name.endswith('CNN'):
        parser.add_argument('--task', type=str, default='brain')
        parser.add_argument('--acceleration', type=int, default=4)
        parser.add_argument('--input_key', type=str, default='kspace')
        parser.add_argument('--target_key', type=str, default='image_label')
        parser.add_argument('--max_key', type=str, default='max')
        parser.add_argument('--mask_type', type=str, default='fixed')
        parser.add_argument('--volume_sample_rate', type=float, default=1.0)
    elif args.model_name.endswith('PromptMR'):
        parser.add_argument('--num_cascades', type=int, default=1)
        parser.add_argument('--num_adj_slices', type=int, default=1)
        parser.add_argument('--n_feat0', type=int, default=12)
        parser.add_argument('--feature_dim', type=str_to_int_list, default=[16, 20, 24])
        parser.add_argument('--prompt_dim', type=str_to_int_list, default=[4, 8, 12])
        parser.add_argument('--sens_n_feat0', type=int, default=4)
        parser.add_argument('--sens_feature_dim', type=str_to_int_list, default=[6, 8, 10])
        parser.add_argument('--sens_prompt_dim', type=str_to_int_list, default=[2, 4, 6])
        parser.add_argument('--len_prompt', type=str_to_int_list, default=[2, 2, 2])
        parser.add_argument('--prompt_size', type=str_to_int_list, default=[64, 32, 16])
        parser.add_argument('--n_enc_cab', type=str_to_int_list, default=[2, 3, 3])
        parser.add_argument('--n_dec_cab', type=str_to_int_list, default=[2, 2, 3])
        parser.add_argument('--n_skip_cab', type=str_to_int_list, default=[1, 1, 1])
        parser.add_argument('--n_bottleneck_cab', type=int, default=3)
        parser.add_argument('--n_buffer', type=int, default=0)
        parser.add_argument('--n_history', type=int, default=0)
        parser.add_argument('--no_use_ca', type=str_to_bool, default=True)
        parser.add_argument('--learnable_prompt', type=str_to_bool, default=False)
        parser.add_argument('--adaptive_input', type=str_to_bool, default=False)
        parser.add_argument('--use_sens_adj', type=str_to_bool, default=False)
        parser.add_argument('--compute_sens_per_coil', type=str_to_bool, default=False)
    if args.aug_on:
        parser.add_argument('--aug_delay', type=int, default=0)
        parser.add_argument('--aug_strength', type=float, default=0.0)
        parser.add_argument('--aug_exp_decay', type=float, default=5.0) 
        parser.add_argument('--aug_weight_fliph', type=float, default=1.0)
        parser.add_argument('--aug_weight_flipv', type=float, default=1.0) 
        parser.add_argument('--aug_weight_rot90', type=float, default=1.0) 
        parser.add_argument('--aug_weight_translation', type=float, default=1.0)
        parser.add_argument('--aug_weight_rotation', type=float, default=1.0)  
        parser.add_argument('--aug_weight_shearing', type=float, default=1.0)
        parser.add_argument('--aug_weight_scaling', type=float, default=1.0)
        parser.add_argument('--aug_max_translation_x', type=float, default=0.125)
        parser.add_argument('--aug_max_translation_y', type=float, default=0.125)
        parser.add_argument('--aug_max_rotation', type=float, default=180.)
        parser.add_argument('--aug_max_shearing_x', type=float, default=15.0)
        parser.add_argument('--aug_max_shearing_y', type=float, default=15.0)
        parser.add_argument('--aug_max_scaling', type=float, default=0.25)

    parser.add_argument('--net_name', type=Path, default='test_varnet')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--data_path_train', type=Path, default='../Data/train/')
    parser.add_argument('--data_path_val', type=Path, default='../Data/val/')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--stop_epoch', type=int, default=-1)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0)
    parser.add_argument('--deterministic', type=str_to_bool, default=False)
    parser.add_argument('--seed', type=int, default=430)
    parser.add_argument('--save_artifact', type=str_to_bool, default=False)

    args, _ = parser.parse_known_args()

    try:
        train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
