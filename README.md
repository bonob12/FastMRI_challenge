# FastMRI_challenge
---
This repository contains the implementation of Team EJ's 2025 FastMRI challenge

## Install
---
``` bash
# Clone repository
git clone https://github.com/bonob12/FastMRI_challenge.git

# password (read_only permisson for this repository)
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

cd FastMRI_challenge

# install venv
python -m venv venv
source venv/bin/activate

# install dependencies
sh requirements.sh
sh apt.sh
``` 

## Data preparement
---
validation dataset not used
train dataset needs to be seperated into brain / knee with following structure

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

## Train
---
Parallel training with multiple GPU recommended

``` bash
# use tmux for convinience (not necessary)
tmux new -s train
tmux attach -t train

# train cnn
sh train_cnn.sh

# train brain_acc4
sh train_brain_acc4_step1.sh ; sh train_brain_acc4_step2.sh ; sh train_brain_acc4_step3.sh

# train brain_acc8
sh train_brain_acc8_step1.sh ; sh train_brain_acc8_step2.sh ; sh train_brain_acc8_step3.sh

# train knee_acc4
sh train_knee_acc4_step1.sh ; sh train_knee_acc4_step2.sh ; sh train_knee_acc4_step3.sh

# train knee_acc8
sh train_knee_acc8_step1.sh ; sh train_knee_acc8_step2.sh ; sh train_knee_acc8_step3.sh

```

Results are saved in ../result folder with following structure

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
│   │   │   └── epoch-15
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

## Reconstruct & Evaluation
---
reconstruct images from ../Data/leaderboard
reconsturcted images saved at ../result/test_reconstruct/reconstructions
can modify reconstruct.sh (ex. brain_acc4_checkpoint ../result/test_brain_acc4/checkpoints/step3/epoch-45 -> ../artifacts/brain_acc4) to use submitted model_weights

``` bash
sh reconstruct.sh

sh leaderboard_eval.sh
```

