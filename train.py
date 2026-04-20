"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, MoEStats
from utils.expert_parallel import require_tutel, tutel_net
from utils.expert_tracker import ExpertActivationTracker
from collections import deque

# SpeedMonitor utilities
speed_monitor_window_size = 100
speed_monitor_start_times = deque([])
speed_monitor_device_interval_tokens = deque([])

def speed_monitor_batch_start(device_batch_num_tokens: int, record: bool = True):
    global speed_monitor_start_times, speed_monitor_device_interval_tokens
    if record:
        if len(speed_monitor_start_times) >= speed_monitor_window_size:
            speed_monitor_start_times.popleft()
            speed_monitor_device_interval_tokens.popleft()
        speed_monitor_start_times.append(time.monotonic())
        speed_monitor_device_interval_tokens.append(device_batch_num_tokens)

def speed_monitor_check():
    if speed_monitor_start_times:
        interval_seconds = time.monotonic() - speed_monitor_start_times[0]
        interval_tokens = sum(speed_monitor_device_interval_tokens)
        return interval_tokens / interval_seconds
    return 0.0

def speed_monitor_reset():
    global speed_monitor_start_times, speed_monitor_device_interval_tokens
    speed_monitor_start_times.clear()
    speed_monitor_device_interval_tokens.clear()

def peak_gpu_memory(reset: bool = False):
    if not torch.cuda.is_available():
        return None
    device = torch.device('cuda')
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if torch.distributed.is_initialized():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        torch.distributed.reduce(peak_mb_tensor, 0, torch.distributed.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak_mb

def _reduce_scalar(value: float, op: str = 'mean'):
    if not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if op == 'max':
        reduce_op = torch.distributed.ReduceOp.MAX
    elif op == 'sum':
        reduce_op = torch.distributed.ReduceOp.SUM
    else:
        reduce_op = torch.distributed.ReduceOp.SUM
    torch.distributed.all_reduce(tensor, op=reduce_op)
    if op == 'mean':
        tensor /= ddp_world_size
    return tensor.item()

def _reduce_moe_metrics(moe_stats: MoEStats | None):
    if moe_stats is None:
        return {}

    sums = torch.stack(
        [
            moe_stats.load_balance_sum.detach().to(dtype=torch.float32),
            moe_stats.himoe_intra_sum.detach().to(dtype=torch.float32),
            moe_stats.himoe_inter_sum.detach().to(dtype=torch.float32),
            moe_stats.router_z_loss_sum.detach().to(dtype=torch.float32),
            moe_stats.total_used.detach().to(dtype=torch.float32),
            moe_stats.capacity_std_sum.detach().to(dtype=torch.float32),
            moe_stats.expert_dispatch_cv_sum.detach().to(dtype=torch.float32),
            moe_stats.group_dispatch_cv_sum.detach().to(dtype=torch.float32),
            moe_stats.group_dispatch_max_frac_sum.detach().to(dtype=torch.float32),
            moe_stats.rank_dispatch_cv_sum.detach().to(dtype=torch.float32),
            moe_stats.rank_dispatch_max_frac_sum.detach().to(dtype=torch.float32),
        ],
        dim=0,
    )
    counts = torch.tensor(
        [
            moe_stats.load_balance_count,
            moe_stats.himoe_intra_count,
            moe_stats.himoe_inter_count,
            moe_stats.router_z_loss_count,
            moe_stats.total_attempted,
            moe_stats.capacity_std_count,
            moe_stats.expert_dispatch_cv_count,
            moe_stats.group_dispatch_cv_count,
            moe_stats.group_dispatch_max_frac_count,
            moe_stats.rank_dispatch_cv_count,
            moe_stats.rank_dispatch_max_frac_count,
        ],
        device=sums.device,
        dtype=torch.float32,
    )

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

    metrics = {}
    if counts[0] > 0:
        metrics['aux_loss'] = (sums[0] / counts[0]).item()
    if counts[1] > 0:
        metrics['intra_loss'] = (sums[1] / counts[1]).item()
    if counts[2] > 0:
        metrics['inter_loss'] = (sums[2] / counts[2]).item()
    if counts[3] > 0:
        metrics['zloss'] = (sums[3] / counts[3]).item()
    if counts[4] > 0:
        metrics['capacity_utilization'] = (sums[4] / counts[4]).item()
        metrics['drop_rate'] = (1.0 - sums[4] / counts[4]).item()
    if counts[5] > 0:
        metrics['load_std'] = (sums[5] / counts[5]).item()
    if counts[6] > 0:
        metrics['expert_dispatch_cv'] = (sums[6] / counts[6]).item()
    if counts[7] > 0:
        metrics['group_dispatch_cv'] = (sums[7] / counts[7]).item()
    if counts[8] > 0:
        metrics['group_dispatch_max_frac'] = (sums[8] / counts[8]).item()
    if counts[9] > 0:
        metrics['rank_dispatch_cv'] = (sums[9] / counts[9]).item()
    if counts[10] > 0:
        metrics['rank_dispatch_max_frac'] = (sums[10] / counts[10]).item()
    return metrics

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
plot_interval = 10000  # how often to save expert activation plots (0 to disable)
log_expert_balance = True
track_expert_activations = False
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'nano-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())
wandb_group = ''
run_id = ""
# keep step-level wandb logging disabled by default
wandb_step_log_interval = 0  # 0 disables step-level wandb logs

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
data_loader_cached_memmap = True

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# moe
moe_type = 'vanilla' # 'vanilla' or 'moge'
n_exp = 1 # if n_exp = 1 we just use regular MLP layers
top_k = 2
use_aux_loss = False
use_router_z_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = False
analysis_n_groups = 4  # conceptual device groups used for dispatch-balance metrics; set to 0 to disable
dispatch_metrics_interval = 0  # 0 disables expensive dispatch/rank-balance metrics during train
train_moe_metrics_interval = -1  # -1 follows log_interval, 0 disables, >0 logs every N train steps
eval_moe_metrics = False  # if True, compute moe/* from eval batches instead of reusing train-step values
eval_dispatch_metrics = False  # if True, include drop/group/expert/rank metrics during eval moe logging

# moge
n_groups = 4 # number of groups for MoGE
use_router_scale = True # use a learnable scale in MoGE router
aux_loss_type = 'switch'
deepseek_alpha = 1.0
deepseek_seq_aux = False

# Loss-Free MoE parameters
bias_update_rate = 0.001

# Hi-MoE parameters
use_himoe_penalty = False
use_himoe_regularization = False
himoe_tau = 0.01
himoe_beta = 0.9
himoe_temperature = 1.0
himoe_lambda1 = 1.0
himoe_lambda2 = 1.0
himoe_warmup_iters = 0
himoe_intra_source = 'raw'
himoe_intra_mode = 'global'
himoe_inter_mode = 'dense_group_mass'
himoe_entropy_mode = 'l2'

# ST-MoE routing
second_policy_train = 'random'
second_policy_eval = 'random'
second_threshold_train = 0.2
second_threshold_eval = 0.2

# momentum
use_momentum = False
momentum_type = "hb"
moe_gamma1 = 1.0
moe_gamma2 = 1.0
moe_mu = 0.7
moe_beta1 = 0.9
moe_beta2 = 0.999
c_norm_thresh = 1.0

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
ddp_find_unused_parameters = False
ep_backend = 'none'  # 'none' or 'tutel'
ep_size = 1
ep_overlap_degree = 1
ep_profile_nvtx = False

# default mfu
running_mfu = -1.0

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
# -----------------------------------------------------------------------------

if train_moe_metrics_interval < 0:
    train_moe_metrics_interval = log_interval
config['train_moe_metrics_interval'] = train_moe_metrics_interval

# various inits, derived attributes, I/O setup
distributed = int(os.environ.get('RANK', -1)) != -1
ep_enabled = ep_backend != 'none'
if ep_enabled:
    local_world_size_env = os.environ.get('LOCAL_WORLD_SIZE')
    if local_world_size_env and 'LOCAL_SIZE' not in os.environ:
        os.environ['LOCAL_SIZE'] = local_world_size_env
    require_tutel()

ddp = distributed and not ep_enabled # is this a ddp run?
if distributed:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank if ddp else 0
    model_seed_offset = ddp_rank
    if ddp:
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    model_seed_offset = 0
    ddp_world_size = 1
process_rank = ddp_rank if distributed else 0
if ep_enabled:
    if distributed and ep_size != ddp_world_size:
        raise SystemExit(
            f"ep_size={ep_size} must match WORLD_SIZE={ddp_world_size} in the current true-EP implementation."
        )
    if not distributed and ep_size != 1:
        raise SystemExit("ep_size > 1 requires torchrun / torch.distributed.")
    if moe_type not in {'vanilla', 'moge'}:
        raise SystemExit(f"ep_backend={ep_backend!r} only supports vanilla and moge/Hi-MoE in v1.")
    if use_momentum:
        raise SystemExit("Momentum/MARS MoE layers are not supported on the true EP path in v1.")
    if moe_type == 'moge' and n_groups != ep_size:
        raise SystemExit(
            f"True EP requires n_groups == ep_size for grouped/Hi-MoE (got n_groups={n_groups}, ep_size={ep_size})."
        )
    if n_exp % ep_size != 0:
        raise SystemExit(f"n_exp={n_exp} must be divisible by ep_size={ep_size}.")
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
if ddp:
    tokens_per_iter *= ddp_world_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process or ep_enabled:
    os.makedirs(out_dir, exist_ok=True)
if distributed and ep_enabled:
    if 'cuda' in device:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()
torch.manual_seed(1337 + model_seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + model_seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = None
val_data = None
data_offsets = np.arange(block_size + 1, dtype=np.int64)

if data_loader_cached_memmap:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    if split == 'train':
        data = train_data if train_data is not None else np.memmap(
            os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'
        )
    else:
        data = val_data if val_data is not None else np.memmap(
            os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r'
        )
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,), dtype=torch.int64).numpy()
    token_idx = ix[:, None] + data_offsets[None, :]
    block = np.asarray(data[token_idx], dtype=np.int64)
    x = torch.from_numpy(block[:, :-1])
    y = torch.from_numpy(block[:, 1:])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, moe_type=moe_type, n_exp=n_exp, top_k=top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, stride=stride,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec, n_groups=n_groups, use_router_scale=use_router_scale,
                  analysis_n_groups=analysis_n_groups,
                  bias_update_rate=bias_update_rate,
                  use_himoe_penalty=use_himoe_penalty, use_himoe_regularization=use_himoe_regularization, himoe_tau=himoe_tau, himoe_beta=himoe_beta,
                  himoe_temperature=himoe_temperature,
                  himoe_lambda1=himoe_lambda1, himoe_lambda2=himoe_lambda2, himoe_warmup_iters=himoe_warmup_iters,
                  himoe_intra_source=himoe_intra_source, himoe_intra_mode=himoe_intra_mode, himoe_inter_mode=himoe_inter_mode, himoe_entropy_mode=himoe_entropy_mode,
                  aux_loss_type=aux_loss_type, deepseek_alpha=deepseek_alpha, deepseek_seq_aux=deepseek_seq_aux,
                  second_policy_train=second_policy_train,
                  second_policy_eval=second_policy_eval,
                  second_threshold_train=second_threshold_train,
                  second_threshold_eval=second_threshold_eval,
                  use_momentum=use_momentum,
                  momentum_type=momentum_type,
                  moe_gamma1=moe_gamma1,
                  moe_gamma2=moe_gamma2,
                  moe_mu=moe_mu,
                  moe_beta1=moe_beta1,
                  moe_beta2=moe_beta2,
                  c_norm_thresh=c_norm_thresh,
                  ep_backend=ep_backend,
                  ep_size=ep_size,
                  ep_overlap_degree=ep_overlap_degree,
                  ep_profile_nvtx=ep_profile_nvtx,
                  )
print('\n\n')
print(model_args)
print('\n\n')
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_name = f'ckpt_rank{process_rank:04d}.pt' if ep_enabled else 'ckpt.pt'
    ckpt_path = os.path.join(out_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    if ep_enabled:
        for k, v in checkpoint_model_args.items():
            if k in model_args:
                model_args[k] = v
    else:
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    iter_num += 1
    best_val_loss = checkpoint['best_val_loss']
    running_mfu = checkpoint['mfu']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
if ep_enabled:
    for param in model.parameters():
        if not hasattr(param, 'skip_allreduce'):
            tutel_net.simple_broadcast(param.detach(), 0)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

if n_exp > 1 and (log_expert_balance or track_expert_activations):
    plots_dir = os.path.join(out_dir, "activation_plots")
    model_params = {"n_exp": n_exp}
    activation_tracker = ExpertActivationTracker(
        model_params=model_params,
        world_size=ddp_world_size if distributed else 1,
        output_dir=plots_dir
    )
else:
    activation_tracker = None

# # Track group (GPU) distribution for expert selections
# # Always use 4 analysis groups to enable direct comparison across all MoE types
# analysis_n_groups = 4
# if n_exp > 1 and n_exp >= analysis_n_groups:
#     effective_top_k = num_experts_per_tok if moe_type == 'lossfree' else (
#         n_groups * top_k if moe_type == 'moge' else top_k
#     )
#     group_dist_tracker = GroupDistributionTracker(
#         n_exp=n_exp,
#         n_groups=analysis_n_groups,
#         top_k=effective_top_k,
#         output_dir=os.path.join(out_dir, "group_distribution_plots"),
#     )
# else:
#     group_dist_tracker = None

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)
# if group_dist_tracker is not None:
#     group_dist_tracker.register_hook(model)

if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        find_unused_parameters=ddp_find_unused_parameters,
    )

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(
    before_val_hooks=None,
    collect_expert_counts=False,
    collect_moe_metrics=False,
    moe_metrics_split='val',
    collect_dispatch_metrics=False,
):
    out = {}
    moe_metrics = {}
    if collect_expert_counts and activation_tracker is not None:
        activation_tracker.reset()
    model.eval()
    for split in ['train', 'val']:
        if split == 'val' and before_val_hooks:
            for hook_fn in before_val_hooks:
                hook_fn()
        losses = torch.zeros(eval_iters)
        split_moe_stats = None
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if collect_expert_counts and collect_moe_metrics and split == moe_metrics_split:
                    _, loss, batch_moe_stats, expert_count_matrix = model(
                        X,
                        Y,
                        return_moe_stats=True,
                        return_expert_count_matrix=True,
                        collect_dispatch_metrics=collect_dispatch_metrics,
                    )
                    split_moe_stats = (
                        batch_moe_stats
                        if split_moe_stats is None
                        else split_moe_stats.merge(batch_moe_stats)
                    )
                    if activation_tracker is not None:
                        activation_tracker.update_count_matrix(expert_count_matrix)
                elif collect_moe_metrics and split == moe_metrics_split:
                    _, loss, batch_moe_stats = model(
                        X,
                        Y,
                        return_moe_stats=True,
                        collect_dispatch_metrics=collect_dispatch_metrics,
                    )
                    split_moe_stats = (
                        batch_moe_stats
                        if split_moe_stats is None
                        else split_moe_stats.merge(batch_moe_stats)
                    )
                elif collect_expert_counts:
                    _, loss, expert_count_matrix = model(X, Y, return_expert_count_matrix=True)
                    if activation_tracker is not None:
                        activation_tracker.update_count_matrix(expert_count_matrix)
                else:
                    _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        if collect_moe_metrics and split == moe_metrics_split:
            moe_metrics = _reduce_moe_metrics(split_moe_stats)
    model.train()
    return out, moe_metrics

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == "resume":
        wandb.init(project=wandb_project, id = run_id, resume = "allow")
        wandb_run_name = wandb.run.name
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, group=wandb_group or None, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
if wandb_log and master_process and init_from != 'resume':
    wandb.summary['model/total_params'] = raw_model.get_num_params()
    wandb.summary['model/active_params'] = raw_model.get_num_active_params()
latest_moe_metrics = {}

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Collective operation - must be called by ALL ranks
    if iter_num % eval_interval == 0:
        if iter_num < 3 or iter_num % 10 == 0:
            peak_mem = peak_gpu_memory()
        else:
            peak_mem = None

    run_eval = iter_num % eval_interval == 0 and (master_process or ep_enabled)
    if run_eval:
        # Reset group tracker right before val split so it only captures val data
        before_val_hooks = []
        # if group_dist_tracker is not None:
        #     before_val_hooks.append(group_dist_tracker.reset)
        losses, eval_moe_metrics_dict = estimate_loss(
            before_val_hooks=before_val_hooks,
            collect_expert_counts=(activation_tracker is not None),
            collect_moe_metrics=eval_moe_metrics,
            collect_dispatch_metrics=eval_dispatch_metrics,
        )
        speed_monitor_reset()
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        mean_variance = -1
        mean_cv = -1
        should_plot = False
        if activation_tracker is not None and master_process:
            mean_variance, mean_cv = activation_tracker.compute_metrics()
            should_plot = track_expert_activations and (plot_interval > 0) and (iter_num % plot_interval == 0)
            if should_plot:
                activation_tracker.save_plot(iter_num)

        # group_dist_metrics = {}
        # should_plot_groups = False
        # if group_dist_tracker is not None:
        #     group_dist_metrics = group_dist_tracker.compute_metrics()
        #     should_plot_groups = (plot_interval > 0) and (iter_num % plot_interval == 0)
        #     if should_plot_groups:
        #         group_dist_tracker.save_plot(iter_num)

        if wandb_log and master_process:
            log_dict = {
                "iter": iter_num,
                "tokens/cumulative": iter_num * tokens_per_iter,
                "tokens/per_iter": tokens_per_iter,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if activation_tracker is not None and mean_variance != -1:
                log_dict['expert_activations/mean_variance'] = mean_variance
            if activation_tracker is not None and mean_cv != -1:
                log_dict['expert_activations/cv'] = mean_cv
            if activation_tracker is not None and should_plot:
                heatmap_path = os.path.join(plots_dir, f"expert_activations_iter_{iter_num}.png")
                if os.path.exists(heatmap_path):
                    log_dict['expert_activations/heatmap'] = wandb.Image(heatmap_path)

            # if group_dist_metrics:
            #     if group_dist_metrics.get('mean_group_spread', -1) != -1:
            #         log_dict['gpu_distribution/mean_group_spread'] = group_dist_metrics['mean_group_spread']
            #     if group_dist_metrics.get('max_group_concentration', -1) != -1:
            #         log_dict['gpu_distribution/max_group_concentration'] = group_dist_metrics['max_group_concentration']
            #     for i, freq in enumerate(group_dist_metrics.get('group_freqs', [])):
            #         log_dict[f'gpu_distribution/group_{i}_freq'] = freq
            # if should_plot_groups:
            #     group_plot_path = os.path.join(out_dir, "group_distribution_plots", f"group_distribution_iter_{iter_num}.png")
            #     if os.path.exists(group_plot_path):
            #         log_dict['gpu_distribution/heatmap'] = wandb.Image(group_plot_path)

            for loss_key, loss_val in eval_moe_metrics_dict.items():
                log_dict[f'moe_eval/{loss_key}'] = loss_val

            tokens_per_sec = speed_monitor_check()
            if tokens_per_sec > 0:
                log_dict["throughput/device/tokens_per_second"] = tokens_per_sec

            if peak_mem:
                log_dict["System/Peak GPU Memory (MB)"] = peak_mem

            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0 and (master_process or ep_enabled):
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    "mfu": running_mfu,
                }
                ckpt_name = f'ckpt_rank{process_rank:04d}.pt' if ep_enabled else 'ckpt.pt'
                ckpt_path = os.path.join(out_dir, ckpt_name)
                if master_process:
                    print(f"saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)
                speed_monitor_reset()
            if ep_enabled and torch.distributed.is_initialized():
                if device_type == 'cuda':
                    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    torch.distributed.barrier()
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    step_t0 = time.perf_counter()
    first_batch = (iter_num == 0)
    collect_step_metrics = (iter_num % log_interval == 0)
    collect_train_moe_metrics = (
        train_moe_metrics_interval > 0 and (iter_num % train_moe_metrics_interval == 0)
    )
    collect_dispatch_metrics = (
        dispatch_metrics_interval > 0 and (iter_num % dispatch_metrics_interval == 0)
    )
    raw_model.set_training_step(iter_num)
    iter_moe_stats = None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if collect_train_moe_metrics:
                logits, loss, micro_moe_stats = model(
                    X,
                    Y,
                    return_moe_stats=True,
                    collect_dispatch_metrics=collect_dispatch_metrics,
                )
            else:
                logits, loss = model(X, Y)
                micro_moe_stats = None
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        if micro_moe_stats is not None:
            iter_moe_stats = micro_moe_stats if iter_moe_stats is None else iter_moe_stats.merge(micro_moe_stats)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        speed_monitor_batch_start(
            device_batch_num_tokens=batch_size * block_size,
            record=not first_batch
        )
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    if ep_enabled and torch.distributed.is_initialized():
        for param in raw_model.parameters():
            if getattr(param, 'grad', None) is not None and not hasattr(param, 'skip_allreduce'):
                tutel_net.simple_all_reduce(
                    param.grad,
                    op=torch.distributed.ReduceOp.AVG,
                    inplace=True,
                )
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Update MoE biases for loss-free balancing (Wang et al., 2024)
    # This must be called AFTER optimizer.step(), outside gradient flow
    if moe_type == 'lossfree':
        from model import update_moe_biases
        update_moe_biases(raw_model)

    # timing and logging
    dt = time.perf_counter() - step_t0

    # Collective operation - must be called by ALL ranks
    if iter_num % log_interval == 0:
        if iter_num < 3 or iter_num % 10 == 0:
            peak_mem_mb = peak_gpu_memory()
        else:
            peak_mem_mb = None
        latest_moe_metrics = _reduce_moe_metrics(iter_moe_stats)

        # Gather globally reduced values for logging/printing.
        loss_for_log = loss.detach()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_for_log, op=torch.distributed.ReduceOp.SUM)
            loss_for_log /= ddp_world_size
        lossf = loss_for_log.item() * gradient_accumulation_steps
        throughput_reduce = 'sum' if ddp else 'mean'
        tokens_per_sec = _reduce_scalar(speed_monitor_check(), op=throughput_reduce)
        step_time_ms = _reduce_scalar(dt * 1000.0, op='max')

    if iter_num % log_interval == 0 and master_process:
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        if peak_mem_mb:
            mem_str = f", mem {peak_mem_mb:.0f}MB"
        else:
            mem_str = ""
        tok_str = f", tok/s {tokens_per_sec:.0f}" if tokens_per_sec > 0 else ""
        moe_parts = []
        if 'drop_rate' in latest_moe_metrics:
            moe_parts.append(f"drop {latest_moe_metrics['drop_rate']:.3f}")
        if 'group_dispatch_cv' in latest_moe_metrics:
            moe_parts.append(f"group_cv {latest_moe_metrics['group_dispatch_cv']:.3f}")
        if 'expert_dispatch_cv' in latest_moe_metrics:
            moe_parts.append(f"expert_cv {latest_moe_metrics['expert_dispatch_cv']:.3f}")
        if 'rank_dispatch_cv' in latest_moe_metrics:
            moe_parts.append(f"rank_cv {latest_moe_metrics['rank_dispatch_cv']:.3f}")
        moe_str = f", {', '.join(moe_parts)}" if moe_parts else ""
        print(f"iter {iter_num}: loss {lossf:.4f}, time {step_time_ms:.2f}ms, mfu {running_mfu*100:.2f}%{tok_str}{mem_str}{moe_str}")

        if wandb_log and wandb_step_log_interval > 0 and (iter_num % wandb_step_log_interval == 0):
            step_log_dict = {
                "iter": iter_num,
                "tokens/cumulative": iter_num * tokens_per_iter,
                "tokens/per_iter": tokens_per_iter,
                "train/loss_step": lossf,
                "lr": lr,
                "mfu": running_mfu * 100,
                "system/step_time_ms": step_time_ms,
            }
            for loss_key, loss_val in latest_moe_metrics.items():
                step_log_dict[f'moe_train/{loss_key}'] = loss_val
            if tokens_per_sec > 0:
                step_log_dict["throughput/global/tokens_per_second"] = tokens_per_sec
                if ddp:
                    step_log_dict["throughput/device/tokens_per_second"] = tokens_per_sec / ddp_world_size
                else:
                    step_log_dict["throughput/device/tokens_per_second"] = tokens_per_sec
            if peak_mem_mb:
                step_log_dict["System/Peak GPU Memory (MB)"] = peak_mem_mb
            wandb.log(step_log_dict)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if wandb_log and master_process:
    import wandb
    wandb.finish()
if ddp:
    destroy_process_group()
