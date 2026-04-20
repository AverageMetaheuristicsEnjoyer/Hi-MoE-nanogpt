import time

# Launch: torchrun --standalone --nproc_per_node=4 train.py config/examples/train_nano_moe_himoe.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'nano-moe'
wandb_group = 'ep'
wandb_run_name = 'gpt2-1B-himoe-owt-' + time.strftime('%Y%m%d-%H%M%S') + '-ep-tutel'
out_dir = "nano_moe_1b_hi-moe_ep_tutel"

n_layer = 14
n_head = 8
n_embd = 512

moe_type = 'moge'
n_exp = 32
top_k = 2
n_groups = 4
use_router_scale = True
use_aux_loss = True
aux_loss_type = 'switch'
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 2.0
eval_capacity = 2.0
stride = 1
use_switch_tfm_init = True
switch_tfm_init_scale = 0.1
router_use_full_prec = True

use_himoe_penalty = True
himoe_tau = 0.01
himoe_beta = 0.9
himoe_temperature = 1.0
use_himoe_regularization = True
himoe_lambda1 = 0.1
himoe_lambda2 = 0.05

ep_backend = 'tutel'
ep_size = 4
compile = True

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 10 * 4

max_iters = 100000
lr_decay_iters = 100000

eval_interval = 1000
eval_iters = 200
log_interval = 10
dispatch_metrics_interval = 200
train_moe_metrics_interval = 200
wandb_step_log_interval = 200
eval_moe_metrics = True
eval_dispatch_metrics = True
plot_interval = 10000

weight_decay = 1e-1
