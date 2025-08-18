# FastMRI_challenge

This repository contains the implementation of Team EJ's 2025 FastMRI challenge

## Installation

``` bash
# Clone repository
git clone https://github.com/bonob12/FastMRI_challenge.git

# Password (read_only permisson for this repository)
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

# Enter repository
cd FastMRI_challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
sh requirements.sh
sh apt.sh
``` 

## Data Preparation

Validation dataset is not used. The training dataset must be separated into brain and knee with the following structure:

``` bash
root
├── Data
│   ├── leaderboard
│   │   └── ...
│   └── train
│       ├── brain
│       │   ├── image
│       │   │   ├── brain_acc4_1 ~ 100.h5
│       │   │   └── brain_acc8_1 ~ 100.h5
│       │   └── kspace
│       │       ├── brain_acc4_1 ~ 100.h5
│       │       └── brain_acc8_1 ~ 100.h5
│       └── knee
│           ├── image
│           │   ├── knee_acc4_1 ~ 100.h5
│           │   └── knee_acc8_1 ~ 100.h5
│           └── kspace
│               ├── knee_acc4_1 ~ 100.h5
│               └── knee_acc8_1 ~ 100.h5
├── FastMRI_challenge
│   └── ...
└── result
    └── ...
```

## Training

Parallel training on multiple GPUs is recommended. Using tmux is optional but convenient:

``` bash
# use tmux for convinience (not necessary)
tmux new -s train
tmux attach -t train

# detach
ctrl b + d

# attach
tmux attach -t train
```

**Train Models**

``` bash
# train cnn
sh scripts/train_cnn.sh

# train brain_acc4
sh scripts/train_brain_acc4_step1.sh ; sh scripts/train_brain_acc4_step2.sh ; sh scripts/train_brain_acc4_step3.sh

# train brain_acc8
sh scripts/train_brain_acc8_step1.sh ; sh scripts/train_brain_acc8_step2.sh ; sh scripts/train_brain_acc8_step3.sh

# train knee_acc4
sh scripts/train_knee_acc4_step1.sh ; sh scripts/train_knee_acc4_step2.sh ; sh scripts/train_knee_acc4_step3.sh

# train knee_acc8
sh scripts/train_knee_acc8_step1.sh ; sh scripts/train_knee_acc8_step2.sh ; sh scripts/train_knee_acc8_step3.sh
```

**Result Structure**

After training, results are saved in ../result:

``` bash
result
├── test_cnn
│   ├── checkpoints
│   │   └── step1
│   │       ├── eopch-1
│   │       │   ├── mp_rank_00_model_states.pt
│   │       │   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │       ...
│   │       └── epoch-5
│   │           └── ...
│   ├── loss_log
│   │   └── step1
│   │       └── loss_log.npy
│   └── wandb
│       └── ...
├── test_brain_acc4
│   ├── checkpoints
│   │   ├── step1
│   │   │   ├── eopch-1
│   │   │   │   ├── mp_rank_00_model_states.pt
│   │   │   │   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   │   ...
│   │   │   └── epoch-15
│   │   │       └── ...
│   │   ├── step2
│   │   │   ├── eopch-16
│   │   │   │   ├── mp_rank_00_model_states.pt
│   │   │   │   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   │   ...
│   │   │   └── epoch-30
│   │   │       └── ...
│   │   └── step3
│   │   │   ├── eopch-31
│   │   │   │   ├── mp_rank_00_model_states.pt
│   │   │   │   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   │   ...
│   │   │   └── epoch-45
│   │   │       └── ...
│   ├── loss_log
│   │   ├── step1
│   │   │   └── loss_log.npy
│   │   ├── step2
│   │   │   └── loss_log.npy
│   │   └── step3
│   │       └── loss_log.npy
│   └── wandb
│       └── ...
├── test_brain_acc8
│   └── ...
├── test_knee_acc4
│   └── ...
├── test_knee_acc8
│   └── ...
└── test_reconstruct
    └── ...
```

## Reconstruct & Evaluation

Reconstruct images from ../Data/leaderboard. Reconstructed images are saved in:

``` bash
../result/test_reconstruct/reconstructions_leaderboard
```

Can modify reconstruct.sh to use specific checkpoint paths:

``` bash
# Example
brain_acc4_checkpoint ../result/test_brain_acc4/checkpoints/step3/epoch-45
->
brain_acc4_checkpoint ../artifacts/brain_acc4
```

**Run reconstruction and evaluation**

``` bash
sh scripts/reconstruct.sh

sh scripts/leaderboard_eval.sh
```

