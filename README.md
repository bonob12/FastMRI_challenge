# FastMRI_challenge

This repository contains the implementation of Team EJ's 2025 FastMRI challenge

---

## 1. Reproducibility Guide

### Installation

``` bash
# Clone repository
git clone https://github.com/bonob12/FastMRI_challenge.git

# Enter repository
cd FastMRI_challenge

# Create virtual environment if needed
python -m venv venv
source venv/bin/activate

# Install dependencies if needed
sh scripts/requirements.sh
sh scripts/apt.sh
``` 

### Data Preparation

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

### Training

Parallel training on multiple GPUs is recommended. Using tmux is optional:

``` bash
# make session 'train'
tmux new -s train

# chek session list
tmux ls

# attach to session and do something...
tmux attach -t train
...

# detach from session
ctrl b + d

# detach and delete session
exit

```

**Train Models**

Weights & Biases (wandb) is used to save and visualize loss logs, so if wandb login is needed, use:

```bash
username: bono_b12
passowrd: 9db12a56848bf3208d13304186875ce79ffdb6c7
```

CNN training completes within 5 minutes
Each brain_acc4, brain_acc8, knee_acc4, and knee_acc8 experiments are trained for a total of 45 epochs (15 + 15 + 15), with each block of 15 epochs requiring roughly 2–3 days.

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

### Reconstruction & Evaluation

Reconstruct images from ../Data/leaderboard. Reconstructed images are saved in:

``` bash
../result/test_reconstruct/reconstructions_leaderboard
```

**Run reconstruction and evaluation**

``` bash
# reconstruct using checkpoints from ../result (retrained model weights to check reproducibility)
sh scripts/reconstruct_result.sh

# reconstruct using checkpoints from artifacts/ (submitted model weights)
sh scripts/reconstruct_artifacts.sh

# evaluate reconstruction
sh scripts/leaderboard_eval.sh
```

### Result

``` bash
Total SSIM:
acc4: 
acc8: 
```

---

## 2. Loss log analysis

**seed fix**

Although we used the existing seed_fix() function, we found that DeepSpeed still executed non-deterministic algorithms and memory optimizations. While this resulted in dramatic performance improvements of over 20–30%, it also introduced non-determinism. To address this, we added torch.use_deterministic_algorithms(True) and os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" to the seed_fix() function.

Furthermore, we fixed the seeds for potential sources of randomness, including worker_init_fn in data loading, mask generation, and the augmentor in the data processing stage.

In conclusion, inspection of the first 100–200 steps of the loss across all 12 cases (brain/knee, acc4/acc8, and steps 1–3) confirmed that the training was exactly reproducible.

**wandb loss log**

Although the run saves train_loss for each epoch as loss_log.npy in the ../result folder, using Weights & Biases (wandb) allows us to visualize not only the training loss but also step-wise loss, learning rate, and other metrics at a glance. The corresponding wandb workspace has been set to public, so running the experiment allows you to check the logs on the wandb web interface: https://wandb.ai/bono_b12-seoul-national-university/FastMRI_challenge

Detailed loss_log recorded via wandb has also been compiled into an Excel file and attached to the email, which can be used to verify reproducibility.

---

## 3. Model

Used Model: PromptMR+ (https://github.com/hellopipu/PromptMR-plus)

PromptMR+ is an advanced deep learning model designed for accelerated MRI reconstruction. Building upon the original PromptMR framework, PromptMR+ incorporates additional cascaded modules and adaptive prompt mechanisms to improve image quality and reconstruction speed across various acceleration factors.

**HyperParameters**

Initially we intended to apply different parameters for each task, but due to insufficient time for parameter sweeps we unified them into a single setting.

```bash
cascade: 8
num_adj_slices: 3
n_feat0: 48
feature_dim: 64 / 80 / 96
prompt_dim: 16 / 32 / 48
sens_n_feat0: 12
sens_feature_dim: 16 / 20 / 24
sens_prompt_dim:  4 / 8 / 12
len_prompt: 5 / 5 / 5
prompt_size: 64 / 32 / 16
n_enc_cab: 2 / 3 / 3
n_dec_cab: 2 / 2 / 3
n_skip_cab: 1 / 1 / 1
n_bottleneck_cab: 1 / 1 / 1
n_buffer: 4
n_history: 0
no_use_ca: true
learnable_prompt: false
adaptive_input: true
use_sens_adj: true
compute_sens_per_coil: true
```

---

## 4. Data Processing

* No validation

* MR augment(fliph, translation, shearing, scaling + flipv)

* Image domain cut (h: brain 384, knee 416)

* Mask generation (fixed, random_offset, random_spaced)

---

## 5. Implementation

* Optimizer: Deepspeed CPU Adam (ADAMW) 
    - weight decay: 1e-4

* SSIM loss with Image mask
    - same as leaderboard SSIM calculation

* Gradient accumulation
    - 2 steps

* Gradient checkpointing

* Learning schedule
    - 15 + 15 + 15 = 45 epochs for each (Total 180 Epochs)
    - 1~15 epoch: no augment, fixed mask, 1 warmup epoch, cosine annealing from 3e-4 to 0 (max_epoch: 20)
    - 16~30 epoch: MRAugment(fliph, translation, shear, scale), fixed mask, no warmup, cosine annealing from 2e-4 to 0 (max_epoch: 35)
    - 31~45 epoch: Brain – add flipv aug, fixed mask / Knee – no aug, random mask, cosine annealing form 1e-4 to 0 (max_epoch: 45)

The following schedule was applied because previously, the schedule was only planned up to 35 epochs. However, due to the allocation of an additional GPU, we had more capacity and were able to extend it to 45 epochs. At this time, we applied different data processing methods for the brain and knee. This was because during epochs 15 to 30, when we applied MRAugment, the brain was robust to transformations, while the knee was vulnerable. Therefore, we applied stronger transformations to the brain, and for the knee, instead of using MRAugment, we altered the mask pattern to achieve a weak augmentation effect.

On the other hand, we considered further training the model for several mask types and applying MoE for each mask, given that the mask patterns of the private dataset were unclear. However, we did not implement this, as it would involve using hidden models solely for the private dataset without any evaluation on the leaderboard, which could raise concerns regarding the competition’s intent and fairness.