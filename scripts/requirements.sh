pip install -r scripts/requirements.txt ; \
pip install wandb  ; \
pip install deepspeed  ; \
pip install mpi4py  ; \
pip uninstall torch ; \
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

