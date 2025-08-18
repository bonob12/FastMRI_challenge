import argparse
import os, sys
import time
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.test_part import forward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=Path, default='test_varnet')
    parser.add_argument('--cnn_checkpoint', type=Path)
    parser.add_argument('--brain_acc4_checkpoint', type=Path)
    parser.add_argument('--brain_acc8_checkpoint', type=Path)
    parser.add_argument('--knee_acc4_checkpoint', type=Path)
    parser.add_argument('--knee_acc8_checkpoint', type=Path)
    parser.add_argument('--path_data', type=Path, default='../Data/leaderboard/')
    args = parser.parse_args()

    start_time = time.time()

    args.cnn_checkpoint = args.cnn_checkpoint / "mp_rank_00_model_states.pt"
    args.brain_acc4_checkpoint = args.brain_acc4_checkpoint / "mp_rank_00_model_states.pt"
    args.brain_acc8_checkpoint = args.brain_acc8_checkpoint / "mp_rank_00_model_states.pt"
    args.knee_acc4_checkpoint = args.knee_acc4_checkpoint / "mp_rank_00_model_states.pt"
    args.knee_acc8_checkpoint = args.knee_acc8_checkpoint / "mp_rank_00_model_states.pt"
    
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')