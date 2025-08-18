python train.py \
--net_name test_cnn \
--model_name utils.model.cnn.CNN \
--step 1 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--max_lr 1e-3 \
--deterministic False \
--seed 430 \
--save_artifact False