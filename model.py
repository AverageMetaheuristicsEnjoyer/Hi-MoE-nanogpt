"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.expert_parallel import (
    distributed_rank,
    distributed_world_size,
    require_tutel,
    tutel_moe,
)


class RouterOutput(NamedTuple):
    expert_indices: torch.Tensor
    slot_indices: torch.Tensor
    combine_weights: torch.Tensor
    selection_mask: torch.Tensor
    dispatch_mask: torch.Tensor
    used_capacity: torch.Tensor
    capacity: int
    load_balance_loss: torch.Tensor
    himoe_intra_loss: torch.Tensor
    himoe_inter_loss: torch.Tensor
    router_z_loss: torch.Tensor
    total_attempted: int


@dataclass
class MoEStats:
    load_balance_sum: torch.Tensor
    load_balance_count: int
    himoe_intra_sum: torch.Tensor
    himoe_intra_count: int
    himoe_inter_sum: torch.Tensor
    himoe_inter_count: int
    router_z_loss_sum: torch.Tensor
    router_z_loss_count: int
    total_used: torch.Tensor
    total_attempted: int
    capacity_std_sum: torch.Tensor
    capacity_std_count: int
    expert_dispatch_cv_sum: torch.Tensor
    expert_dispatch_cv_count: int
    group_dispatch_cv_sum: torch.Tensor
    group_dispatch_cv_count: int
    group_dispatch_max_frac_sum: torch.Tensor
    group_dispatch_max_frac_count: int
    rank_dispatch_cv_sum: torch.Tensor
    rank_dispatch_cv_count: int
    rank_dispatch_max_frac_sum: torch.Tensor
    rank_dispatch_max_frac_count: int

    @classmethod
    def zeros(cls, device):
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return cls(
            load_balance_sum=zero.clone(),
            load_balance_count=0,
            himoe_intra_sum=zero.clone(),
            himoe_intra_count=0,
            himoe_inter_sum=zero.clone(),
            himoe_inter_count=0,
            router_z_loss_sum=zero.clone(),
            router_z_loss_count=0,
            total_used=zero.clone(),
            total_attempted=0,
            capacity_std_sum=zero.clone(),
            capacity_std_count=0,
            expert_dispatch_cv_sum=zero.clone(),
            expert_dispatch_cv_count=0,
            group_dispatch_cv_sum=zero.clone(),
            group_dispatch_cv_count=0,
            group_dispatch_max_frac_sum=zero.clone(),
            group_dispatch_max_frac_count=0,
            rank_dispatch_cv_sum=zero.clone(),
            rank_dispatch_cv_count=0,
            rank_dispatch_max_frac_sum=zero.clone(),
            rank_dispatch_max_frac_count=0,
        )

    def merge(self, other: "MoEStats") -> "MoEStats":
        return MoEStats(
            load_balance_sum=self.load_balance_sum + other.load_balance_sum,
            load_balance_count=self.load_balance_count + other.load_balance_count,
            himoe_intra_sum=self.himoe_intra_sum + other.himoe_intra_sum,
            himoe_intra_count=self.himoe_intra_count + other.himoe_intra_count,
            himoe_inter_sum=self.himoe_inter_sum + other.himoe_inter_sum,
            himoe_inter_count=self.himoe_inter_count + other.himoe_inter_count,
            router_z_loss_sum=self.router_z_loss_sum + other.router_z_loss_sum,
            router_z_loss_count=self.router_z_loss_count + other.router_z_loss_count,
            total_used=self.total_used + other.total_used,
            total_attempted=self.total_attempted + other.total_attempted,
            capacity_std_sum=self.capacity_std_sum + other.capacity_std_sum,
            capacity_std_count=self.capacity_std_count + other.capacity_std_count,
            expert_dispatch_cv_sum=self.expert_dispatch_cv_sum + other.expert_dispatch_cv_sum,
            expert_dispatch_cv_count=self.expert_dispatch_cv_count + other.expert_dispatch_cv_count,
            group_dispatch_cv_sum=self.group_dispatch_cv_sum + other.group_dispatch_cv_sum,
            group_dispatch_cv_count=self.group_dispatch_cv_count + other.group_dispatch_cv_count,
            group_dispatch_max_frac_sum=self.group_dispatch_max_frac_sum + other.group_dispatch_max_frac_sum,
            group_dispatch_max_frac_count=self.group_dispatch_max_frac_count + other.group_dispatch_max_frac_count,
            rank_dispatch_cv_sum=self.rank_dispatch_cv_sum + other.rank_dispatch_cv_sum,
            rank_dispatch_cv_count=self.rank_dispatch_cv_count + other.rank_dispatch_cv_count,
            rank_dispatch_max_frac_sum=self.rank_dispatch_max_frac_sum + other.rank_dispatch_max_frac_sum,
            rank_dispatch_max_frac_count=self.rank_dispatch_max_frac_count + other.rank_dispatch_max_frac_count,
        )


def _coefficient_of_variation(loads: torch.Tensor):
    loads = loads.to(dtype=torch.float32)
    mean = loads.mean()
    if loads.numel() <= 1:
        return torch.zeros((), device=loads.device, dtype=torch.float32)
    return loads.std(unbiased=False) / mean.clamp_min(1e-10)


def _group_loads_from_expert_loads(expert_loads: torch.Tensor, n_groups: int):
    if n_groups <= 1 or expert_loads.numel() % n_groups != 0:
        return None
    experts_per_group = expert_loads.numel() // n_groups
    return expert_loads.view(n_groups, experts_per_group).sum(dim=-1)


def _route_to_slots(expert_indices, combine_weights, n_exp, capacity, active_mask=None):
    num_tokens, top_k = expert_indices.shape
    if active_mask is None:
        active_mask = torch.ones_like(expert_indices, dtype=torch.bool)
    else:
        active_mask = active_mask.to(dtype=torch.bool)

    route_major_experts = expert_indices.transpose(0, 1).reshape(-1)
    route_major_mask = active_mask.transpose(0, 1).reshape(-1)
    route_major_weights = combine_weights.transpose(0, 1).reshape(-1)

    order = torch.arange(route_major_experts.numel(), device=route_major_experts.device, dtype=torch.long)
    sort_key = route_major_experts.to(torch.long) * route_major_experts.numel() + order
    sort_order = torch.argsort(sort_key)

    sorted_experts = route_major_experts[sort_order]
    sorted_mask = route_major_mask[sort_order].to(torch.int64)
    sorted_cumsum = torch.cumsum(sorted_mask, dim=0)

    group_start = torch.ones_like(sorted_experts, dtype=torch.bool)
    group_start[1:] = sorted_experts[1:] != sorted_experts[:-1]

    prev_cumsum = torch.zeros_like(sorted_cumsum)
    prev_cumsum[1:] = sorted_cumsum[:-1]
    group_ids = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    group_bases = prev_cumsum[group_start]
    sorted_slots = sorted_cumsum - group_bases[group_ids] - 1

    route_major_slots = torch.empty_like(sorted_slots)
    route_major_slots[sort_order] = sorted_slots

    route_major_valid = route_major_mask & (route_major_slots >= 0) & (route_major_slots < capacity)
    route_major_slots = route_major_slots.masked_fill(~route_major_valid, -1)
    route_major_weights = route_major_weights * route_major_valid.to(route_major_weights.dtype)
    used_capacity = torch.bincount(route_major_experts[route_major_valid], minlength=n_exp)

    slot_indices = route_major_slots.view(top_k, num_tokens).transpose(0, 1).contiguous()
    combine_weights = route_major_weights.view(top_k, num_tokens).transpose(0, 1).contiguous()
    dispatch_mask = route_major_valid.view(top_k, num_tokens).transpose(0, 1).contiguous()
    return slot_indices, combine_weights, dispatch_mask, used_capacity


def _dispatch_to_experts(x_flat, router_output, n_exp, experts):
    num_tokens, _ = x_flat.shape
    _, top_k = router_output.expert_indices.shape
    capacity = router_output.capacity

    slot_offsets = router_output.expert_indices * capacity + router_output.slot_indices.clamp_min(0)
    token_indices = torch.arange(num_tokens, device=x_flat.device, dtype=torch.long).unsqueeze(1).expand(num_tokens, top_k)
    valid_assignments = router_output.dispatch_mask

    expert_batches = x_flat.new_zeros((n_exp * capacity, x_flat.size(-1)))
    if valid_assignments.any():
        flat_slots = slot_offsets[valid_assignments]
        flat_tokens = token_indices[valid_assignments]
        expert_batches.index_add_(0, flat_slots, x_flat[flat_tokens])
    else:
        flat_slots = slot_offsets.new_zeros((0,), dtype=torch.long)
        flat_tokens = slot_offsets.new_zeros((0,), dtype=torch.long)

    expert_batches = expert_batches.view(n_exp, capacity, x_flat.size(-1))
    expert_outputs = experts(expert_batches).view(n_exp * capacity, x_flat.size(-1))

    routed_output = expert_outputs.new_zeros((num_tokens, x_flat.size(-1)))
    if flat_slots.numel() > 0:
        weighted_outputs = expert_outputs[flat_slots] * router_output.combine_weights[valid_assignments].to(expert_outputs.dtype).unsqueeze(-1)
        routed_output.index_add_(0, flat_tokens, weighted_outputs)
    return routed_output


def _router_output_to_tutel_critical(router_output):
    expert_indices = router_output.expert_indices
    top_k = expert_indices.size(1)
    active_mask = router_output.dispatch_mask

    invalid_locations = torch.full_like(
        router_output.slot_indices,
        fill_value=int(router_output.capacity),
        dtype=torch.int32,
    )
    safe_indices = torch.where(
        active_mask,
        expert_indices.to(torch.int32),
        torch.full_like(expert_indices, fill_value=-1, dtype=torch.int32),
    )
    safe_locations = torch.where(
        active_mask,
        router_output.slot_indices.clamp_min(0).to(torch.int32),
        invalid_locations,
    )
    safe_gates = torch.where(
        active_mask,
        router_output.combine_weights,
        torch.zeros_like(router_output.combine_weights),
    )

    indices_s = [safe_indices[:, route_id].contiguous().view(-1) for route_id in range(top_k)]
    locations_s = [safe_locations[:, route_id].contiguous().view(-1) for route_id in range(top_k)]
    gates_s = [safe_gates[:, route_id].contiguous().view(-1) for route_id in range(top_k)]

    dispatch_count = torch.bincount(
        expert_indices[router_output.selection_mask].reshape(-1),
        minlength=router_output.used_capacity.numel(),
    ).to(torch.int32)
    return (
        int(router_output.used_capacity.numel()),
        indices_s,
        locations_s,
        gates_s,
        int(router_output.capacity),
        dispatch_count,
    )


def _compute_switch_aux_loss(global_probs_for_loss: torch.Tensor, indices: torch.Tensor, n_exp: int):
    with torch.no_grad():
        flat_indices = indices.reshape(-1)
        f_i = torch.bincount(flat_indices, minlength=n_exp).to(dtype=torch.float32)
        f_i /= max(indices.size(0), 1)
    p_i = torch.mean(global_probs_for_loss.float(), dim=0)
    return n_exp * (f_i * p_i).sum()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.router_type = config.moe_type
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None

        if self.router_type == 'moge':
            raise NotImplementedError("MoGERouter should be used for MoGE")

    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise
            benchmark_logit_bias = getattr(self, "benchmark_logit_bias", None)
            if benchmark_logit_bias is not None:
                logits = logits + benchmark_logit_bias.to(dtype=logits.dtype)

            zero = torch.zeros((), device=logits.device, dtype=torch.float32)

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            router_z_loss = self.compute_router_z_loss(logits) if self.use_router_z_loss else zero

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

            # normalize expert probabilities over selected experts only
            selected_probs = F.softmax(top_k_logits, dim=-1)
            expert_indices = top_k_indices.view(num_tokens, self.top_k)
            combine_weights = selected_probs.view(num_tokens, self.top_k)

            # compute auxiliary load balancing loss
            load_balance_loss = zero
            if self.use_aux_loss:
                load_balance_loss = self.compute_aux_loss(combine_weights, expert_indices)['load_balance']

            # compute expert capacity and assign compact routed slots
            exp_capacity = self.get_capacity(num_tokens)
            slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
                expert_indices=expert_indices,
                combine_weights=combine_weights,
                n_exp=self.n_exp,
                capacity=exp_capacity,
            )

            return RouterOutput(
                expert_indices=expert_indices,
                slot_indices=slot_indices,
                combine_weights=combine_weights,
                selection_mask=torch.ones_like(expert_indices, dtype=torch.bool),
                dispatch_mask=dispatch_mask,
                used_capacity=used_capacity,
                capacity=exp_capacity,
                load_balance_loss=load_balance_loss,
                himoe_intra_loss=zero,
                himoe_inter_loss=zero,
                router_z_loss=router_z_loss,
                total_attempted=self.top_k * num_tokens,
            )

    def compute_aux_loss(self, selected_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            tokens_per_expert = torch.bincount(flat_indices, minlength=self.n_exp).to(dtype=torch.float32)
            tokens_per_expert /= max(flat_indices.numel(), 1)

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.zeros(self.n_exp, device=selected_probs.device, dtype=torch.float32)
        prob_per_expert.scatter_add_(0, indices.reshape(-1), selected_probs.reshape(-1).float())
        prob_per_expert /= max(indices.size(0), 1)

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        load_balance_loss = self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

        return {'load_balance': load_balance_loss}

    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """

        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class STMoERouter(nn.Module):
    """ST-MoE top-n router with threshold-based secondary expert policy."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec
        self.second_policy_train = config.second_policy_train
        self.second_policy_eval = config.second_policy_eval
        self.second_threshold_train = config.second_threshold_train
        self.second_threshold_eval = config.second_threshold_eval

        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)

    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        return int(capacity)

    def compute_aux_loss(self, expert_probs, top1_indices):
        """Switch aux loss using top-1 assignments only (ST-MoE paper)."""
        with torch.no_grad():
            flat_indices = top1_indices.reshape(-1)
            tokens_per_expert = torch.bincount(flat_indices, minlength=self.n_exp).to(dtype=torch.float32)
            tokens_per_expert /= max(flat_indices.numel(), 1)
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))
        load_balance_loss = self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
        return load_balance_loss

    def compute_router_z_loss(self, logits):
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0
        return torch.mean(z_loss)

    @torch._dynamo.disable
    def _apply_threshold_policy(self, gate_values, policy, threshold):
        """Apply secondary expert threshold policy. Disabled for torch.compile compatibility."""
        if policy == 'all':
            return torch.ones_like(gate_values, dtype=torch.bool)
        elif policy == 'none':
            return torch.zeros_like(gate_values, dtype=torch.bool)
        elif policy == 'threshold':
            return gate_values > threshold
        elif policy == 'random':
            keep_prob = gate_values / max(threshold, 1e-9)
            keep_mask = torch.rand_like(gate_values) < keep_prob
            return keep_mask
        else:
            raise ValueError(f"Unknown secondary policy: {policy}")

    def forward(self, x):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            logits = self.w_g(x)  # [B, T, n_exp]
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            router_z_loss = self.compute_router_z_loss(logits) if self.use_router_z_loss else zero

            # Softmax over all experts -> raw gates
            raw_gates = F.softmax(logits, dim=-1)  # [B, T, n_exp]
            top_gates, top_indices = torch.topk(raw_gates, k=self.top_k, dim=-1)
            gates = top_gates / (top_gates.sum(dim=-1, keepdim=True) + 1e-9)

            # Aux loss using top-1 mask only
            load_balance_loss = self.compute_aux_loss(raw_gates, top_indices[..., 0]) if self.use_aux_loss else zero

            # Apply threshold policy to secondary experts (masks[1:])
            policy = self.second_policy_train if self.training else self.second_policy_eval
            threshold = self.second_threshold_train if self.training else self.second_threshold_eval
            active_mask = torch.ones_like(top_indices, dtype=torch.bool)
            for i in range(1, self.top_k):
                active_mask[..., i] = self._apply_threshold_policy(
                    gates[..., i], policy, threshold
                )

            # Compute capacity
            exp_capacity = self.get_capacity(num_tokens)
            expert_indices = top_indices.view(num_tokens, self.top_k)
            combine_weights = gates.view(num_tokens, self.top_k)
            slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
                expert_indices=expert_indices,
                combine_weights=combine_weights,
                n_exp=self.n_exp,
                capacity=exp_capacity,
                active_mask=active_mask.view(num_tokens, self.top_k),
            )

            return RouterOutput(
                expert_indices=expert_indices,
                slot_indices=slot_indices,
                combine_weights=combine_weights,
                selection_mask=active_mask.view(num_tokens, self.top_k),
                dispatch_mask=dispatch_mask,
                used_capacity=used_capacity,
                capacity=exp_capacity,
                load_balance_loss=load_balance_loss,
                himoe_intra_loss=zero,
                himoe_inter_loss=zero,
                router_z_loss=router_z_loss,
                total_attempted=self.top_k * num_tokens,
            )


class MoGERouter(Router):
    """
    A router that implements the Mixture of Grouped Experts (MoGE) logic from Pangu Pro MoE,
    with Hi-MoE optimizations (bias-corrected routing, intra/inter-group regularization).
    It partitions experts into groups and selects top_k experts from each group.
    """
    def __init__(self, config):
        # We call nn.Module's init directly to bypass the parent Router's init,
        # as we are overriding the logic completely.
        nn.Module.__init__(self)

        self.n_embd = config.n_embd
        self.n_exp = config.n_exp
        self.n_groups = config.n_groups
        assert self.n_exp % self.n_groups == 0, "Number of experts must be divisible by number of groups"
        self.n_exp_per_group = self.n_exp // self.n_groups
        self.top_k_per_group = config.top_k  # experts selected per group (from config)
        self.top_k = self.n_groups * self.top_k_per_group  # total experts selected

        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity

        self.router_use_full_prec = config.router_use_full_prec
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        self.use_himoe_penalty = config.use_himoe_penalty
        self.use_himoe_regularization = config.use_himoe_regularization
        self.tau = config.himoe_tau
        self.beta = config.himoe_beta
        self.temperature = config.himoe_temperature
        self.lambda1 = config.himoe_lambda1
        self.lambda2 = config.himoe_lambda2
        self.himoe_warmup_iters = getattr(config, 'himoe_warmup_iters', 0)
        self.himoe_intra_source = getattr(config, 'himoe_intra_source', 'raw')
        self.himoe_intra_mode = getattr(config, 'himoe_intra_mode', 'global')
        self.himoe_inter_mode = getattr(config, 'himoe_inter_mode', 'dense_group_mass')
        self.himoe_entropy_mode = getattr(config, 'himoe_entropy_mode', 'l2')
        if self.use_himoe_penalty:
            self.register_buffer("avg_logits", torch.zeros(self.n_exp))
        self.register_buffer("training_step", torch.tensor(0.0))

        # DeepSeek loss parameters
        self.aux_loss_type = getattr(config, 'aux_loss_type', 'switch')
        self.deepseek_alpha = getattr(config, 'deepseek_alpha', 1.0)
        self.deepseek_seq_aux = getattr(config, 'deepseek_seq_aux', False)

        self.w_g = nn.Linear(self.n_embd, self.n_exp, bias=False)
        self.use_router_scale = config.use_router_scale
        if self.use_router_scale:
            self.router_scale = nn.Parameter(torch.ones((1, self.n_exp)))

        if self.aux_loss_type == 'deepseek':
            self._aux_loss_fn = self._compute_deepseek_aux_loss
        else:
            self._aux_loss_fn = self._compute_switch_aux_loss

        valid_intra_sources = {'raw', 'selection'}
        valid_intra_modes = {'global', 'group_conditional'}
        valid_inter_modes = {'dense_group_mass', 'sparse_group_mass', 'selected_l2'}
        valid_entropy_modes = {'l2', 'shannon'}
        if self.himoe_intra_source not in valid_intra_sources:
            raise ValueError(f"Unknown himoe_intra_source: {self.himoe_intra_source}")
        if self.himoe_intra_mode not in valid_intra_modes:
            raise ValueError(f"Unknown himoe_intra_mode: {self.himoe_intra_mode}")
        if self.himoe_inter_mode not in valid_inter_modes:
            raise ValueError(f"Unknown himoe_inter_mode: {self.himoe_inter_mode}")
        if self.himoe_entropy_mode not in valid_entropy_modes:
            raise ValueError(f"Unknown himoe_entropy_mode: {self.himoe_entropy_mode}")

    @torch._dynamo.disable
    def _update_avg_logits_buffer(self, router_logits):
        """
        Update the avg_logits buffer using exponential moving average of raw logits.
        Per Hi-MoE paper: ḡ ← βḡ + (1-β)g(x) where g(x) is the raw gating logits.
        Decorated with @torch._dynamo.disable to prevent torch.compile graph fusion issues
        when the buffer is both read (for penalty) and written in the same forward pass.
        """
        with torch.no_grad():
            self.avg_logits.mul_(self.beta).add_(router_logits.mean(dim=0).detach(), alpha=1.0 - self.beta)

    def _compute_himoe_intra_core(self, pi_x: torch.Tensor):
        if self.himoe_intra_mode == 'group_conditional':
            grouped_probs = pi_x.view(-1, self.n_groups, self.n_exp_per_group)
            grouped_conditional = grouped_probs / (grouped_probs.sum(dim=-1, keepdim=True) + 1e-10)
            return torch.mean(grouped_conditional.pow(2).sum(dim=-1))

        return torch.mean(pi_x.pow(2).sum(dim=-1))

    def _compute_himoe_inter_core(
        self,
        global_probs_for_loss: torch.Tensor,
        indices: torch.Tensor,
        top_k_scores: torch.Tensor,
        pre_renorm_scores: torch.Tensor = None,
    ):
        if self.himoe_inter_mode == 'selected_l2':
            selected_scores = pre_renorm_scores if pre_renorm_scores is not None else top_k_scores
            return torch.mean(selected_scores.pow(2).sum(dim=-1))

        if self.himoe_inter_mode == 'sparse_group_mass':
            selected_scores = pre_renorm_scores if pre_renorm_scores is not None else top_k_scores
            group_ids = indices // self.n_exp_per_group
            sparse_group_mass = torch.zeros(
                indices.size(0),
                self.n_groups,
                device=selected_scores.device,
                dtype=selected_scores.dtype,
            )
            sparse_group_mass.scatter_add_(1, group_ids, selected_scores)
            return torch.mean(sparse_group_mass.pow(2).sum(dim=-1))

        # Default: dense group mass from full router probabilities.
        group_mass = global_probs_for_loss.view(-1, self.n_groups, self.n_exp_per_group).sum(dim=-1)
        return torch.mean(group_mass.pow(2).sum(dim=-1))

    def forward(self, x):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T
            x_flat = x.view(num_tokens, self.n_embd)

            router_logits = self.w_g(x_flat) # [B*T, n_exp]
            benchmark_logit_bias = getattr(self, "benchmark_logit_bias", None)
            if benchmark_logit_bias is not None:
                router_logits = router_logits + benchmark_logit_bias.to(dtype=router_logits.dtype)
            raw_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            zero = torch.zeros((), device=router_logits.device, dtype=torch.float32)

            if self.use_himoe_penalty:
                # Update EMA with raw logits BEFORE applying penalty (per Hi-MoE paper)
                if self.training:
                    self._update_avg_logits_buffer(router_logits)

                penalty = self.avg_logits.unsqueeze(0) * self.tau
                balanced_logits = (router_logits - penalty) / self.temperature

                selection_logits = balanced_logits
            else:
                selection_logits = router_logits


            global_probs_for_selection = F.softmax(selection_logits, dim=-1, dtype=torch.float)
            grouped_weights = global_probs_for_selection.view(num_tokens, self.n_groups, self.n_exp_per_group)

            # Use topk instead of max to support top_k_per_group > 1 (like GroupsGate)
            topk_weights_per_group, topk_indices_in_group = torch.topk(
                grouped_weights, k=self.top_k_per_group, dim=-1
            )  # [num_tokens, n_groups, top_k_per_group]

            # Compute global expert indices
            group_offsets = torch.arange(0, self.n_exp, self.n_exp_per_group, device=x.device, dtype=torch.long)
            # Shape: [1, n_groups, 1] for broadcasting with [num_tokens, n_groups, top_k_per_group]
            global_top_k_indices = topk_indices_in_group + group_offsets.unsqueeze(0).unsqueeze(-1)
            top_k_indices = global_top_k_indices.view(num_tokens, -1)  # [num_tokens, n_groups * top_k_per_group]

            # Flatten and renormalize scores to sum to 1 (key difference from previous implementation)
            top_k_scores = topk_weights_per_group.view(num_tokens, -1)
            pre_renorm_scores = top_k_scores  # raw selected probs for R_inter (paper Eq. 6)
            top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-10)

            router_z_loss = self.compute_router_z_loss(router_logits) if self.use_router_z_loss else zero

            if self.use_aux_loss:
                # Raw probs for standard aux loss; bias-corrected probs for Hi-MoE regularizers
                aux_loss = self.compute_aux_loss(raw_probs, top_k_indices, top_k_scores, pre_renorm_scores, global_probs_for_selection)
            else:
                aux_loss = {'load_balance': zero}

            exp_capacity = self.get_capacity(num_tokens)

            selected_weights = top_k_scores.permute(1, 0).clone()  # [top_k, B*T]
            if self.use_router_scale:
                selected_scales = torch.gather(self.router_scale.squeeze(0), -1, top_k_indices.view(-1)).view(num_tokens, self.top_k).permute(1,0)
                selected_weights *= selected_scales

            expert_indices = top_k_indices.view(num_tokens, self.top_k)
            combine_weights = selected_weights.permute(1, 0)
            slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
                expert_indices=expert_indices,
                combine_weights=combine_weights,
                n_exp=self.n_exp,
                capacity=exp_capacity,
            )

            return RouterOutput(
                expert_indices=expert_indices,
                slot_indices=slot_indices,
                combine_weights=combine_weights,
                selection_mask=torch.ones_like(expert_indices, dtype=torch.bool),
                dispatch_mask=dispatch_mask,
                used_capacity=used_capacity,
                capacity=exp_capacity,
                load_balance_loss=aux_loss.get('load_balance', zero),
                himoe_intra_loss=aux_loss.get('himoe_intra', zero),
                himoe_inter_loss=aux_loss.get('himoe_inter', zero),
                router_z_loss=router_z_loss,
                total_attempted=self.top_k * num_tokens,
            )

    def compute_aux_loss(self, global_probs_for_loss: torch.Tensor, indices: torch.Tensor, top_k_scores: torch.Tensor, pre_renorm_scores: torch.Tensor = None, selection_probs: torch.Tensor = None):
        """
        Computes auxiliary loss using pre-assigned method to avoid runtime branching.
        selection_probs: bias-corrected π(x) used for Hi-MoE regularizers (paper Eq. 7).
        """
        return self._aux_loss_fn(global_probs_for_loss, indices, top_k_scores, pre_renorm_scores, selection_probs)

    def _compute_switch_aux_loss(self, global_probs_for_loss: torch.Tensor, indices: torch.Tensor, top_k_scores: torch.Tensor, pre_renorm_scores: torch.Tensor = None, selection_probs: torch.Tensor = None):
        """
        Computes standard Switch Transformer auxiliary loss with optional Hi-MoE regularization.
        """
        # Standard Switch Transformer load balancing loss
        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            f_i = torch.bincount(flat_indices, minlength=self.n_exp).to(dtype=torch.float32)
            f_i /= max(indices.size(0), 1)

        P_i = torch.mean(global_probs_for_loss.float(), dim=0)
        load_balance_loss = self.n_exp * (f_i * P_i).sum()

        loss_dict = {'load_balance': load_balance_loss}

        if self.use_himoe_regularization:
            if self.himoe_entropy_mode == 'shannon':
                # Shannon entropy mode: additive chain rule H(π) = H(group) + E[H(expert|group)]
                # decouples the two objectives. Condition for group balance: λ₂ > λ₁.
                pi_x = global_probs_for_loss  # raw probs [B*T, N]

                # R_intra: +λ₁ H(π) — add to loss → minimize H(π) → concentrate routing
                intra_loss = self.lambda1 * torch.mean((-pi_x * torch.log(pi_x + 1e-10)).sum(dim=-1))

                if self.himoe_warmup_iters > 0:
                    warmup_frac = (
                        self.training_step.to(device=intra_loss.device, dtype=intra_loss.dtype)
                        / float(self.himoe_warmup_iters)
                    ).clamp(max=1.0)
                    intra_loss = intra_loss * warmup_frac

                # R_inter: -λ₂ H(group) — subtract from loss → maximize H(group) → balance groups
                group_mass = pi_x.view(-1, self.n_groups, self.n_exp_per_group).sum(dim=-1)
                inter_loss = -self.lambda2 * torch.mean((-group_mass * torch.log(group_mass + 1e-10)).sum(dim=-1))
            else:
                # L2 / Tsallis-2 mode (original Hi-MoE formulation).
                if self.himoe_intra_source == 'selection' and selection_probs is not None:
                    pi_x = selection_probs
                else:
                    pi_x = global_probs_for_loss
                intra_loss = -self.lambda1 * self._compute_himoe_intra_core(pi_x)

                if self.himoe_warmup_iters > 0:
                    warmup_frac = (
                        self.training_step.to(device=intra_loss.device, dtype=intra_loss.dtype)
                        / float(self.himoe_warmup_iters)
                    ).clamp(max=1.0)
                    intra_loss = intra_loss * warmup_frac

                inter_loss = self.lambda2 * self._compute_himoe_inter_core(
                    global_probs_for_loss=global_probs_for_loss,
                    indices=indices,
                    top_k_scores=top_k_scores,
                    pre_renorm_scores=pre_renorm_scores,
                )

            loss_dict['himoe_intra'] = intra_loss
            loss_dict['himoe_inter'] = inter_loss

        return loss_dict

    def _compute_deepseek_aux_loss(self, global_probs_for_loss: torch.Tensor, indices: torch.Tensor, top_k_scores: torch.Tensor, pre_renorm_scores: torch.Tensor = None, selection_probs: torch.Tensor = None):
        """
        Computes DeepSeek-style auxiliary loss with optional sequence-level balancing.

        Based on DeepSeek-MoE: https://arxiv.org/abs/2401.06066
        """
        if self.deepseek_seq_aux:
            # Sequence-level auxiliary loss (requires reshaping back to [batch, seq_len, ...])
            # For now, we assume num_tokens = batch_size * seq_len, but we don't have seq_len info here
            # So we'll implement the global version with DeepSeek's formulation
            # TODO: To properly implement seq_aux, we'd need to pass batch_size and seq_len
            pass

        # DeepSeek's standard auxiliary loss (equivalent to Switch when alpha=1)
        # L_aux = alpha * sum(P_i * f_i * num_experts)
        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            f_i = torch.bincount(flat_indices, minlength=self.n_exp).to(dtype=torch.float32)
            f_i /= max(indices.size(0), 1)

        # Average router probability for each expert
        P_i = torch.mean(global_probs_for_loss.float(), dim=0)

        # DeepSeek formulation: ce * P_i where ce = f_i * n_routed_experts
        fi = f_i * self.n_exp
        aux_loss = (P_i * fi).sum() * self.deepseek_alpha

        return aux_loss


class LossFreeMoERouter(nn.Module):
    """
    Loss-Free load balancing router (Wang et al., 2024).
    Maintains balance without auxiliary loss using bias terms updated outside gradient flow.

    Paper: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_exp = config.n_exp
        self.top_k = config.top_k

        # Expert centroids (learnable parameters)
        self.expert_centroids = nn.Parameter(torch.empty(self.n_exp, self.n_embd))
        nn.init.kaiming_uniform_(self.expert_centroids, a=math.sqrt(5))

        # Bias is a BUFFER (not parameter) - no gradients, just state
        self.register_buffer('expert_bias', torch.zeros(self.n_exp))

        # Update rate from paper (Table 3, Fig 4)
        self.bias_update_rate = config.bias_update_rate  # default: 0.001

        # Capacity and training params
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity

        # Store expert counts for bias update
        self.last_expert_counts = None

    def get_capacity(self, num_tokens):
        """Compute expert capacity based on number of tokens"""
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * num_tokens / self.n_exp)
        return max(capacity, self.min_capacity)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            RouterOutput with compact expert-slot routing metadata.
        """
        B, T, _ = x.size()
        num_tokens = B * T
        x_flat = x.view(-1, x.size(-1))  # [B*T, n_embd]
        zero = torch.zeros((), device=x_flat.device, dtype=torch.float32)

        # 1. Compute affinity scores with SIGMOID (Paper Appendix C)
        scores = torch.sigmoid(x_flat @ self.expert_centroids.T)  # [B*T, n_exp]

        # 2. Add bias for SELECTION only (Paper Eq. 3)
        scores_for_selection = scores + self.expert_bias  # [B*T, n_exp]

        # 3. Top-K selection using biased scores
        _, top_k_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1, sorted=False)

        # 4. Gating weights from ORIGINAL scores (NO bias!) - Paper Eq. 3
        topk_weight = scores.gather(-1, top_k_indices)  # [B*T, top_k]

        # 5. Normalize gating weights to sum to 1
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 6. Count expert assignments for bias update (Paper Algorithm 1, step 3)
        if self.training:
            with torch.no_grad():
                # Count tokens assigned to each expert
                flat_indices = top_k_indices.view(-1)
                expert_counts = torch.bincount(flat_indices, minlength=self.n_exp).float()
                self.last_expert_counts = expert_counts

        # 7. Compute capacities (similar to Router class)
        exp_capacity = self.get_capacity(num_tokens)
        expert_indices = top_k_indices.view(num_tokens, self.top_k)
        slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
            expert_indices=expert_indices,
            combine_weights=topk_weight,
            n_exp=self.n_exp,
            capacity=exp_capacity,
        )

        return RouterOutput(
            expert_indices=expert_indices,
            slot_indices=slot_indices,
            combine_weights=combine_weights,
            selection_mask=torch.ones_like(expert_indices, dtype=torch.bool),
            dispatch_mask=dispatch_mask,
            used_capacity=used_capacity,
            capacity=exp_capacity,
            load_balance_loss=zero,
            himoe_intra_loss=zero,
            himoe_inter_loss=zero,
            router_z_loss=zero,
            total_attempted=self.top_k * num_tokens,
        )


    @torch.no_grad()
    def update_bias(self):
        """
        Update bias terms after optimizer step (Paper Algorithm 1, step 6).
        Call this AFTER optimizer.step() in training loop.
        """
        if self.last_expert_counts is None:
            return

        # Expected load (Paper Algorithm 1, step 4)
        expected_load = self.last_expert_counts.sum() / self.n_exp

        # Error: expected - actual (Paper Algorithm 1, step 5)
        error = expected_load - self.last_expert_counts

        # Update with sign (Paper Table 3: sign is better than proportional)
        self.expert_bias += self.bias_update_rate * torch.sign(error)


def update_moe_biases(model):
    """
    Update all MoE router biases after optimizer step (Paper Algorithm 1, step 6).
    Call this AFTER optimizer.step() in training loop.
    """
    for module in model.modules():
        if isinstance(module, LossFreeMoERouter):
            module.update_bias()


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x


class TutelMLPExpertModule(nn.Module):
    def __init__(
        self,
        model_dim,
        num_experts_per_device,
        sharded_count,
        hidden_size_per_expert=None,
        dropout=0.0,
        bias=True,
        use_switch_tfm_init=False,
        switch_tfm_init_scale=1.0,
        **_,
    ):
        super().__init__()
        if sharded_count != 1:
            raise ValueError("TutelMLPExpertModule only supports sharded_count == 1 in this integration.")
        hidden_size = hidden_size_per_expert or (4 * model_dim)
        self.bias = bool(bias)
        self.c_fc = nn.Parameter(torch.empty(num_experts_per_device, model_dim, hidden_size))
        self.c_proj = nn.Parameter(torch.empty(num_experts_per_device, hidden_size, model_dim))
        self.fc_bias = nn.Parameter(torch.empty(num_experts_per_device, 1, hidden_size)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(num_experts_per_device, 1, model_dim)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters(use_switch_tfm_init, switch_tfm_init_scale)

    def _reset_parameters(self, use_switch_tfm_init: bool, switch_tfm_init_scale: float) -> None:
        with torch.no_grad():
            if use_switch_tfm_init:
                fc_fan_in = self.c_fc.shape[-2]
                fc_std = math.sqrt(switch_tfm_init_scale / fc_fan_in)
                torch.nn.init.trunc_normal_(self.c_fc, mean=0.0, std=fc_std, a=-2 * fc_std, b=2 * fc_std)

                proj_fan_in = self.c_proj.shape[-2]
                proj_std = math.sqrt(switch_tfm_init_scale / proj_fan_in)
                torch.nn.init.trunc_normal_(self.c_proj, mean=0.0, std=proj_std, a=-2 * proj_std, b=2 * proj_std)
            else:
                torch.nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
            if self.fc_bias is not None:
                torch.nn.init.zeros_(self.fc_bias)
                torch.nn.init.zeros_(self.proj_bias)

    def forward(self, x, _ctx):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x = x + self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x = x + self.proj_bias
        x = self.dropout(x)
        return x


class TutelVanillaGate(nn.Module):
    def __init__(
        self,
        model_dim,
        num_global_experts,
        k=1,
        router_use_full_prec=False,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_noisy_top_k=False,
        min_capacity=4,
        gate_noise=0.0,
        capacity_factor=1.0,
        **_,
    ):
        super().__init__()
        self.num_global_experts = int(num_global_experts)
        self.top_k = min(int(k), self.num_global_experts)
        self.router_use_full_prec = bool(router_use_full_prec)
        self.use_aux_loss = bool(use_aux_loss)
        self.use_router_z_loss = bool(use_router_z_loss)
        self.use_noisy_top_k = bool(use_noisy_top_k)
        self.min_capacity = int(min_capacity)
        self.gate_noise = float(gate_noise)
        self.capacity_factor = float(capacity_factor)
        weight_dtype = torch.float32 if self.router_use_full_prec else None
        self.wg = nn.Linear(model_dim, self.num_global_experts, bias=False, dtype=weight_dtype)
        self.w_noise = nn.Linear(model_dim, self.num_global_experts, bias=False, dtype=weight_dtype) if self.use_noisy_top_k else None

    def forward(self, x):
        wg = self.wg.float() if self.router_use_full_prec else self.wg
        logits = wg(x.to(dtype=wg.weight.dtype))
        if self.use_noisy_top_k:
            wn = self.w_noise.float() if self.router_use_full_prec else self.w_noise
            noise = F.softplus(wn(x.to(dtype=wn.weight.dtype)))
            logits = logits + noise * torch.randn_like(noise)
        return logits

    def _compute_router_z_loss(self, logits: torch.Tensor):
        return torch.mean(torch.logsumexp(logits, dim=-1) ** 2.0)

    def _get_capacity(self, num_tokens: int, capacity_factor: float):
        capacity = math.floor(self.top_k * capacity_factor * num_tokens / self.num_global_experts)
        capacity += capacity % 2
        return max(capacity, self.min_capacity)

    def route(
        self,
        x,
        training,
        top_k,
        capacity_factor,
        batch_prioritized_routing,
        normalize_gate,
        alignment,
        group,
        inequivalent_tokens,
        is_gshard_loss,
    ):
        del top_k, batch_prioritized_routing, normalize_gate, alignment, group, inequivalent_tokens, is_gshard_loss, training
        logits = self.forward(x)
        zero = torch.zeros((), device=logits.device, dtype=torch.float32)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        selected_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        load_balance_loss = _compute_switch_aux_loss(probs, topk_indices, self.num_global_experts) if self.use_aux_loss else zero
        router_z_loss = self._compute_router_z_loss(logits) if self.use_router_z_loss else zero
        capacity = self._get_capacity(int(x.size(0)), capacity_factor)
        slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
            expert_indices=topk_indices,
            combine_weights=selected_probs,
            n_exp=self.num_global_experts,
            capacity=capacity,
        )
        router_output = RouterOutput(
            expert_indices=topk_indices,
            slot_indices=slot_indices,
            combine_weights=combine_weights,
            selection_mask=torch.ones_like(topk_indices, dtype=torch.bool),
            dispatch_mask=dispatch_mask,
            used_capacity=used_capacity,
            capacity=capacity,
            load_balance_loss=load_balance_loss,
            himoe_intra_loss=zero,
            himoe_inter_loss=zero,
            router_z_loss=router_z_loss,
            total_attempted=self.top_k * int(x.size(0)),
        )
        return x.dtype, _router_output_to_tutel_critical(router_output), {
            'l_aux': load_balance_loss,
            'router_output': router_output,
        }


class TutelGroupedGate(nn.Module):
    def __init__(
        self,
        model_dim,
        num_global_experts,
        k=1,
        num_groups=1,
        router_use_full_prec=False,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_router_scale=False,
        use_himoe_penalty=False,
        use_himoe_regularization=False,
        himoe_tau=0.01,
        himoe_beta=0.9,
        himoe_temperature=1.0,
        himoe_lambda1=1.0,
        himoe_lambda2=1.0,
        himoe_warmup_iters=0,
        himoe_intra_source='raw',
        himoe_intra_mode='global',
        himoe_inter_mode='dense_group_mass',
        himoe_entropy_mode='l2',
        min_capacity=4,
        gate_noise=0.0,
        capacity_factor=1.0,
        **_,
    ):
        super().__init__()
        self.num_global_experts = int(num_global_experts)
        self.n_groups = int(num_groups)
        self.n_exp_per_group = self.num_global_experts // self.n_groups
        self.top_k_per_group = int(k)
        self.top_k = self.n_groups * self.top_k_per_group
        self.router_use_full_prec = bool(router_use_full_prec)
        self.use_aux_loss = bool(use_aux_loss)
        self.use_router_z_loss = bool(use_router_z_loss)
        self.use_router_scale = bool(use_router_scale)
        self.use_himoe_penalty = bool(use_himoe_penalty)
        self.use_himoe_regularization = bool(use_himoe_regularization)
        self.tau = float(himoe_tau)
        self.beta = float(himoe_beta)
        self.temperature = float(himoe_temperature)
        self.lambda1 = float(himoe_lambda1)
        self.lambda2 = float(himoe_lambda2)
        self.himoe_warmup_iters = int(himoe_warmup_iters)
        self.himoe_intra_source = himoe_intra_source
        self.himoe_intra_mode = himoe_intra_mode
        self.himoe_inter_mode = himoe_inter_mode
        self.himoe_entropy_mode = himoe_entropy_mode
        self.min_capacity = int(min_capacity)
        self.gate_noise = float(gate_noise)
        self.capacity_factor = float(capacity_factor)

        weight_dtype = torch.float32 if self.router_use_full_prec else None
        self.wg = nn.Linear(model_dim, self.num_global_experts, bias=False, dtype=weight_dtype)
        if self.use_router_scale:
            self.router_scale = nn.Parameter(torch.ones((1, self.num_global_experts)))
        if self.use_himoe_penalty:
            self.register_buffer("avg_logits", torch.zeros(self.num_global_experts))
        self.register_buffer("training_step", torch.tensor(0.0))

    def forward(self, x):
        wg = self.wg.float() if self.router_use_full_prec else self.wg
        return wg(x.to(dtype=wg.weight.dtype))

    def _compute_router_z_loss(self, logits: torch.Tensor):
        return torch.mean(torch.logsumexp(logits, dim=-1) ** 2.0)

    def _get_capacity(self, num_tokens: int, capacity_factor: float):
        capacity = math.floor(self.top_k * capacity_factor * num_tokens / self.num_global_experts)
        capacity += capacity % 2
        return max(capacity, self.min_capacity)

    @torch._dynamo.disable
    def _update_avg_logits_buffer(self, router_logits):
        with torch.no_grad():
            self.avg_logits.mul_(self.beta).add_(router_logits.mean(dim=0).detach(), alpha=1.0 - self.beta)

    def _compute_himoe_intra_core(self, pi_x: torch.Tensor):
        if self.himoe_intra_mode == 'group_conditional':
            grouped_probs = pi_x.view(-1, self.n_groups, self.n_exp_per_group)
            grouped_conditional = grouped_probs / (grouped_probs.sum(dim=-1, keepdim=True) + 1e-10)
            return torch.mean(grouped_conditional.pow(2).sum(dim=-1))
        return torch.mean(pi_x.pow(2).sum(dim=-1))

    def _compute_himoe_inter_core(
        self,
        global_probs_for_loss: torch.Tensor,
        indices: torch.Tensor,
        top_k_scores: torch.Tensor,
        pre_renorm_scores: torch.Tensor = None,
    ):
        if self.himoe_inter_mode == 'selected_l2':
            selected_scores = pre_renorm_scores if pre_renorm_scores is not None else top_k_scores
            return torch.mean(selected_scores.pow(2).sum(dim=-1))

        if self.himoe_inter_mode == 'sparse_group_mass':
            selected_scores = pre_renorm_scores if pre_renorm_scores is not None else top_k_scores
            group_ids = indices // self.n_exp_per_group
            sparse_group_mass = torch.zeros(
                indices.size(0),
                self.n_groups,
                device=selected_scores.device,
                dtype=selected_scores.dtype,
            )
            sparse_group_mass.scatter_add_(1, group_ids, selected_scores)
            return torch.mean(sparse_group_mass.pow(2).sum(dim=-1))

        group_mass = global_probs_for_loss.view(-1, self.n_groups, self.n_exp_per_group).sum(dim=-1)
        return torch.mean(group_mass.pow(2).sum(dim=-1))

    def route(
        self,
        x,
        training,
        top_k,
        capacity_factor,
        batch_prioritized_routing,
        normalize_gate,
        alignment,
        group,
        inequivalent_tokens,
        is_gshard_loss,
    ):
        del top_k, batch_prioritized_routing, normalize_gate, alignment, group, inequivalent_tokens, is_gshard_loss
        router_logits = self.forward(x)
        zero = torch.zeros((), device=router_logits.device, dtype=torch.float32)
        raw_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.use_himoe_penalty:
            if training:
                self._update_avg_logits_buffer(router_logits)
            penalty = self.avg_logits.unsqueeze(0) * self.tau
            selection_logits = (router_logits - penalty) / self.temperature
        else:
            selection_logits = router_logits

        selection_probs = F.softmax(selection_logits, dim=-1, dtype=torch.float)
        grouped_weights = selection_probs.view(-1, self.n_groups, self.n_exp_per_group)
        topk_weights_per_group, topk_indices_in_group = torch.topk(
            grouped_weights, k=self.top_k_per_group, dim=-1
        )
        group_offsets = torch.arange(
            0, self.num_global_experts, self.n_exp_per_group, device=selection_probs.device, dtype=torch.long
        )
        expert_indices = (topk_indices_in_group + group_offsets.unsqueeze(0).unsqueeze(-1)).view(-1, self.top_k)

        top_k_scores = topk_weights_per_group.view(-1, self.top_k)
        pre_renorm_scores = top_k_scores
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-10)

        if self.use_router_scale:
            selected_scales = torch.gather(self.router_scale.squeeze(0), -1, expert_indices.reshape(-1)).view_as(top_k_scores)
            top_k_scores = top_k_scores * selected_scales

        load_balance_loss = _compute_switch_aux_loss(raw_probs, expert_indices, self.num_global_experts) if self.use_aux_loss else zero
        router_z_loss = self._compute_router_z_loss(router_logits) if self.use_router_z_loss else zero

        if self.use_himoe_regularization:
            if self.himoe_entropy_mode == 'shannon':
                pi_x = raw_probs
                intra_loss = self.lambda1 * torch.mean((-pi_x * torch.log(pi_x + 1e-10)).sum(dim=-1))
                group_mass = pi_x.view(-1, self.n_groups, self.n_exp_per_group).sum(dim=-1)
                inter_loss = -self.lambda2 * torch.mean((-group_mass * torch.log(group_mass + 1e-10)).sum(dim=-1))
            else:
                pi_x = selection_probs if self.himoe_intra_source == 'selection' else raw_probs
                intra_loss = -self.lambda1 * self._compute_himoe_intra_core(pi_x)
                inter_loss = self.lambda2 * self._compute_himoe_inter_core(
                    global_probs_for_loss=raw_probs,
                    indices=expert_indices,
                    top_k_scores=top_k_scores,
                    pre_renorm_scores=pre_renorm_scores,
                )
            if self.himoe_warmup_iters > 0:
                warmup_frac = (
                    self.training_step.to(device=intra_loss.device, dtype=intra_loss.dtype) / float(self.himoe_warmup_iters)
                ).clamp(max=1.0)
                intra_loss = intra_loss * warmup_frac
        else:
            intra_loss = zero
            inter_loss = zero

        capacity = self._get_capacity(int(x.size(0)), capacity_factor)
        slot_indices, combine_weights, dispatch_mask, used_capacity = _route_to_slots(
            expert_indices=expert_indices,
            combine_weights=top_k_scores,
            n_exp=self.num_global_experts,
            capacity=capacity,
        )
        router_output = RouterOutput(
            expert_indices=expert_indices,
            slot_indices=slot_indices,
            combine_weights=combine_weights,
            selection_mask=torch.ones_like(expert_indices, dtype=torch.bool),
            dispatch_mask=dispatch_mask,
            used_capacity=used_capacity,
            capacity=capacity,
            load_balance_loss=load_balance_loss,
            himoe_intra_loss=intra_loss,
            himoe_inter_loss=inter_loss,
            router_z_loss=router_z_loss,
            total_attempted=self.top_k * int(x.size(0)),
        )
        return x.dtype, _router_output_to_tutel_critical(router_output), {
            'l_aux': load_balance_loss,
            'router_output': router_output,
        }


class EPMoELayerBase(nn.Module):
    def __init__(self, config, gate_cls):
        super().__init__()
        require_tutel()
        self.config = config
        self.ep_backend = config.ep_backend
        self.ep_group = None
        self.layer_name = "ep_moe"

        self.ep_world_size = max(int(getattr(config, "ep_size", 1)), 1)
        actual_world_size = distributed_world_size(group=self.ep_group)
        if actual_world_size > 1:
            if self.ep_world_size != actual_world_size:
                raise ValueError(
                    f"ep_size={self.ep_world_size} must match WORLD_SIZE={actual_world_size} for v1 true EP."
                )
        elif self.ep_world_size != 1:
            raise ValueError("ep_size > 1 requires torch.distributed to be initialized.")

        if config.n_exp % self.ep_world_size != 0:
            raise ValueError(
                f"n_exp={config.n_exp} must be divisible by ep_size={self.ep_world_size}."
            )
        self.experts_per_rank = config.n_exp // self.ep_world_size
        self.rank = distributed_rank(group=self.ep_group)
        self.last_rank_route_loads = None
        self.last_phase_metrics = None

        gate_kwargs = {
            'type': 'custom',
            'module': gate_cls,
            'k': config.top_k,
            'num_groups': getattr(config, 'n_groups', 1),
            'router_use_full_prec': config.router_use_full_prec,
            'use_aux_loss': config.use_aux_loss,
            'use_router_z_loss': config.use_router_z_loss,
            'use_noisy_top_k': getattr(config, 'use_noisy_top_k', False),
            'use_router_scale': getattr(config, 'use_router_scale', False),
            'use_himoe_penalty': getattr(config, 'use_himoe_penalty', False),
            'use_himoe_regularization': getattr(config, 'use_himoe_regularization', False),
            'himoe_tau': getattr(config, 'himoe_tau', 0.01),
            'himoe_beta': getattr(config, 'himoe_beta', 0.9),
            'himoe_temperature': getattr(config, 'himoe_temperature', 1.0),
            'himoe_lambda1': getattr(config, 'himoe_lambda1', 1.0),
            'himoe_lambda2': getattr(config, 'himoe_lambda2', 1.0),
            'himoe_warmup_iters': getattr(config, 'himoe_warmup_iters', 0),
            'himoe_intra_source': getattr(config, 'himoe_intra_source', 'raw'),
            'himoe_intra_mode': getattr(config, 'himoe_intra_mode', 'global'),
            'himoe_inter_mode': getattr(config, 'himoe_inter_mode', 'dense_group_mass'),
            'himoe_entropy_mode': getattr(config, 'himoe_entropy_mode', 'l2'),
            'min_capacity': config.min_capacity,
            'capacity_factor': config.train_capacity,
        }
        self._moe_layer = tutel_moe.moe_layer(
            gate_type=gate_kwargs,
            experts={
                'type': 'custom',
                'module': TutelMLPExpertModule,
                'num_experts_per_device': self.experts_per_rank,
                'hidden_size_per_expert': 4 * config.n_embd,
                'dropout': config.dropout,
                'bias': config.bias,
                'use_switch_tfm_init': config.use_switch_tfm_init,
                'switch_tfm_init_scale': config.switch_tfm_init_scale,
            },
            model_dim=config.n_embd,
            scan_expert_func=self._mark_tutel_expert_parameter,
            group=self.ep_group,
        )
        self.router = self._moe_layer.gates[0]

    @staticmethod
    def _mark_tutel_expert_parameter(_name: str, param: torch.Tensor) -> None:
        setattr(param, "_ep_local_expert", True)
        setattr(param, "_tutel_expert", True)
        setattr(param, "skip_allreduce", True)

    def set_layer_name(self, name: str) -> None:
        self.layer_name = name

    def _distributed_forward(self, x: torch.Tensor, collect_dispatch_metrics: bool = False):
        capacity_factor = self.config.train_capacity if self.training else self.config.eval_capacity
        routed_output = self._moe_layer(x, capacity_factor=capacity_factor)
        route_metadata = getattr(self._moe_layer, "route_metadata", {})
        router_output = route_metadata.get("router_output")

        if collect_dispatch_metrics and router_output is not None:
            rank_route_loads = router_output.used_capacity.view(self.ep_world_size, self.experts_per_rank).sum(dim=-1).to(dtype=torch.float32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(rank_route_loads, op=torch.distributed.ReduceOp.SUM)
            self.last_rank_route_loads = rank_route_loads
        else:
            self.last_rank_route_loads = None
        self.last_phase_metrics = None
        return routed_output, router_output


class EPMOELayer(EPMoELayerBase):
    def __init__(self, config):
        super().__init__(config, TutelVanillaGate)

    def forward(self, x: torch.Tensor, collect_dispatch_metrics: bool = False):
        return self._distributed_forward(x, collect_dispatch_metrics=collect_dispatch_metrics)


class EPMoGELayer(EPMoELayerBase):
    def __init__(self, config):
        super().__init__(config, TutelGroupedGate)

    def forward(self, x: torch.Tensor, collect_dispatch_metrics: bool = False):
        return self._distributed_forward(x, collect_dispatch_metrics=collect_dispatch_metrics)

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        router_output = self.router(x)
        x_flat = x.view(num_tokens, n_embd)
        output = _dispatch_to_experts(x_flat, router_output, self.router.n_exp, self.experts)
        return output.view(B, T, n_embd), router_output

class MoGELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = MoGERouter(config)
        self.experts = MLPExperts(config)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()
        num_tokens = B * T

        router_output = self.router(x)
        x_flat = x.view(num_tokens, n_embd)
        routed_output = _dispatch_to_experts(x_flat, router_output, self.router.n_exp, self.experts)
        return routed_output.view(B, T, n_embd), router_output

class STMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = STMoERouter(config)
        self.experts = MLPExperts(config)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()
        num_tokens = B * T

        router_output = self.router(x)
        x_flat = x.view(num_tokens, n_embd)
        routed_output = _dispatch_to_experts(x_flat, router_output, self.router.n_exp, self.experts)
        return routed_output.view(B, T, n_embd), router_output

class MomentumLayer(MOELayer):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.moe_gamma2
        self.mu = config.moe_mu
        self.ln = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, momentum: torch.Tensor):
        moe_out, router_output = super().forward(self.ln(x))
        momentum = -moe_out + self.mu * momentum
        out = self.gamma * momentum
        out = x + out

        return out, momentum, router_output

class MarsLayer(MOELayer):
    def __init__(self, config):
        super().__init__(config)
        self.gamma1 = config.moe_gamma1
        self.gamma2 = config.moe_gamma2
        self.beta1 = config.moe_beta1
        self.beta2 = config.moe_beta2
        self.c_norm_thresh = config.c_norm_thresh
        self.ln = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, momentum: torch.Tensor):
        moe_out, router_output = super().forward(self.ln(x))

        m, v, moe_out_prev = momentum

        outps_diff = -moe_out - (-moe_out_prev)
        c = -moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * outps_diff

        with torch.no_grad():
            c_norm = torch.linalg.matrix_norm(c, dim = (-2, -1), ord = "fro")
            scaling_facs = torch.maximum(
                torch.tensor(1.0, device = c.device),
                c_norm / self.c_norm_thresh,
            )
        c_t = c / scaling_facs.view(-1, 1, 1)

        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2

        out = self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))
        out = x + out

        return out, (m_t, v_t, moe_out.detach()), router_output


class LossFreeMoELayer(nn.Module):
    """
    Loss-Free MoE layer combining LossFreeMoERouter with MLPExperts.
    Implements the Wang et al., 2024 loss-free load balancing strategy.
    """
    def __init__(self, config):
        super().__init__()
        self.router = LossFreeMoERouter(config)
        self.experts = MLPExperts(config)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()
        num_tokens = B * T

        router_output = self.router(x)
        x_flat = x.view(num_tokens, n_embd)
        routed_output = _dispatch_to_experts(x_flat, router_output, self.router.n_exp, self.experts)
        return routed_output.view(B, T, n_embd), router_output


class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.use_momentum = config.use_momentum
        self.is_moge_layer = (use_moe and config.moe_type == 'moge')
        if use_moe:
            if self.use_momentum:
                if config.momentum_type == "hb":
                    self.mlp = MomentumLayer(config)
                elif config.momentum_type == "mars":
                    self.mlp = MarsLayer(config)
                else:
                    raise ValueError("Incorrect momentum type")
            else:
                self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
                if config.moe_type == 'vanilla':
                    if config.ep_backend != 'none':
                        self.mlp = EPMOELayer(config)
                    else:
                        self.mlp = MOELayer(config)
                elif config.moe_type == 'moge':
                    if config.ep_backend != 'none':
                        self.mlp = EPMoGELayer(config)
                    else:
                        self.mlp = MoGELayer(config)
                elif config.moe_type == 'lossfree':
                    self.mlp = LossFreeMoELayer(config)
                elif config.moe_type == 'stmoe':
                    self.mlp = STMoELayer(config)
                else:
                    raise ValueError(f"Unknown MoE type: {config.moe_type}")
        else:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

    def forward(self, x, momentum, collect_dispatch_metrics: bool = False):
        block_moe_output = None
        x = x + self.attn(self.ln_1(x))
        if isinstance(self.mlp, (MomentumLayer, MarsLayer)):
            x, momentum, block_moe_output = self.mlp(x, momentum)
        elif isinstance(self.mlp, EPMoELayerBase):
            mlp_out, block_moe_output = self.mlp(
                self.ln_2(x),
                collect_dispatch_metrics=collect_dispatch_metrics,
            )
            x = x + mlp_out
        elif self.is_moge_layer:
            mlp_out, block_moe_output = self.mlp(self.ln_2(x))
            x = x + mlp_out
        else:
            mlp_out = self.mlp(self.ln_2(x))
            if isinstance(mlp_out, tuple):
                mlp_out, block_moe_output = mlp_out
            x = x + mlp_out
        return x, momentum, block_moe_output

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # MoE-related configs
    moe_type: str = 'vanilla' # 'vanilla', 'moge', or 'lossfree'
    n_exp: int = 1 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router

    # Loss-Free MoE configs (Wang et al., 2024)
    bias_update_rate: float = 0.001  # Bias update rate for loss-free balancing (paper verified optimal)

    # MoGE-related configs
    n_groups: int = 4 # Number of groups for MoGE
    use_router_scale: bool = True # Use a learnable scale in MoGE router
    analysis_n_groups: int = 4  # Conceptual device groups used for dispatch-balance metrics
    ep_backend: str = 'none'  # 'none' or 'tutel'
    ep_size: int = 1
    ep_overlap_degree: int = 1
    ep_profile_nvtx: bool = False

    # SRoME / PIS configs
    use_himoe_penalty: bool = False
    use_himoe_regularization: bool = False
    himoe_tau: float = 0.01
    himoe_beta: float = 0.9
    himoe_temperature: float = 1.0
    himoe_lambda1: float = 1.0
    himoe_lambda2: float = 1.0
    himoe_warmup_iters: int = 0  # Linear warmup steps for R_intra (0 = no warmup)
    himoe_intra_source: str = 'raw'  # 'raw' or 'selection'
    himoe_intra_mode: str = 'global'  # 'global' or 'group_conditional'
    himoe_inter_mode: str = 'dense_group_mass'  # 'dense_group_mass', 'sparse_group_mass', 'selected_l2'
    himoe_entropy_mode: str = 'l2'  # 'l2' (Tsallis-2, original) or 'shannon' (additive chain rule)

    # DeepSeek loss configs
    aux_loss_type: str = 'switch'  # 'switch' or 'deepseek'
    deepseek_alpha: float = 1.0  # Loss weight multiplier for DeepSeek loss
    deepseek_seq_aux: bool = False  # Enable sequence-level auxiliary loss

    # ST-MoE routing configs
    second_policy_train: str = 'random'    # 'all', 'none', 'threshold', 'random'
    second_policy_eval: str = 'random'     # 'all', 'none', 'threshold', 'random'
    second_threshold_train: float = 0.2
    second_threshold_eval: float = 0.2

    # Momentum-related configs
    use_momentum: bool = False
    momentum_type: str = "hb"
    moe_gamma1: float = 1.0
    moe_gamma2: float = 1.0
    moe_mu: float = 0.7
    moe_beta1: float = 0.9
    moe_beta2: float = 0.999
    c_norm_thresh: float = 1.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for i in range(config.n_layer):
                # TODO: how to implement this?
                # should we change below to i + 1 ?
                use_moe = (i % config.stride) == 0
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # optionall use switch transformer special init scheme for experts
        # See pg. 10 here: https://arxiv.org/abs/2101.03961
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        for i, block in enumerate(self.transformer.h):
            if hasattr(block, "mlp") and hasattr(block.mlp, "set_layer_name"):
                block.mlp.set_layer_name(f"layer_{i:02d}")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = 0
        ep_scale = max(int(getattr(self.config, "ep_size", 1)), 1)
        for p in self.parameters():
            multiplier = ep_scale if getattr(p, "_ep_local_expert", False) else 1
            n_params += p.numel() * multiplier
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_active_params(self, non_embedding=True):
        """
        Return the number of parameters active per forward pass.
        For MoE layers, expert params are scaled by top_k / n_exp since only
        top_k experts are activated per token. Non-expert params count fully.
        """
        cfg = self.config
        if cfg.n_exp == 1:
            return self.get_num_params(non_embedding)

        ep_scale = max(int(getattr(cfg, "ep_size", 1)), 1)
        expert_params = 0
        for m in self.modules():
            if isinstance(m, (MLPExperts, TutelMLPExpertModule)):
                shard_multiplier = ep_scale if any(getattr(p, "_ep_local_expert", False) for p in m.parameters()) else 1
                expert_params += sum(p.numel() for p in m.parameters()) * shard_multiplier
        total_params = self.get_num_params(non_embedding)
        if cfg.moe_type == 'moge':
            effective_top_k = cfg.n_groups * cfg.top_k
        else:
            effective_top_k = cfg.top_k
        active_expert_params = int(expert_params * effective_top_k / cfg.n_exp)
        return total_params - expert_params + active_expert_params

    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim]
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_training_step(self, step):
        step_value = float(step)
        for module in self.modules():
            if isinstance(module, (MoGERouter, TutelGroupedGate)):
                module.training_step.fill_(step_value)

    def forward(
        self,
        idx,
        targets=None,
        return_moe_stats=False,
        return_expert_count_matrix=False,
        collect_dispatch_metrics: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        zero = torch.zeros((), device=device, dtype=torch.float32)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.config.use_momentum:
            if self.config.moe_type == 'moge':
                momentum = None
            elif self.config.momentum_type == "hb":
                momentum = torch.zeros_like(x)
            elif self.config.momentum_type == "mars":
                momentum = (
                    torch.zeros_like(x),
                    torch.zeros_like(x),
                    torch.zeros_like(x),
                )
            else:
                raise ValueError("Incorrect momentum type")
        else:
            momentum = None

        load_balance_sum = zero.clone()
        himoe_intra_sum = zero.clone()
        himoe_inter_sum = zero.clone()
        router_z_loss_sum = zero.clone()
        total_used = zero.clone()
        total_attempted = 0
        capacity_std_sum = zero.clone()
        capacity_std_count = 0
        expert_dispatch_cv_sum = zero.clone()
        expert_dispatch_cv_count = 0
        group_dispatch_cv_sum = zero.clone()
        group_dispatch_cv_count = 0
        group_dispatch_max_frac_sum = zero.clone()
        group_dispatch_max_frac_count = 0
        rank_dispatch_cv_sum = zero.clone()
        rank_dispatch_cv_count = 0
        rank_dispatch_max_frac_sum = zero.clone()
        rank_dispatch_max_frac_count = 0
        expert_count_matrix = None
        if return_expert_count_matrix and self.config.n_exp > 1:
            expert_count_matrix = torch.zeros(
                (len(self.transformer.h), self.config.n_exp),
                device=device,
                dtype=torch.float32,
            )

        for block_idx, block in enumerate(self.transformer.h):
            x, momentum, block_moe_output = block(
                x,
                momentum,
                collect_dispatch_metrics=collect_dispatch_metrics,
            )
            if block_moe_output is not None:
                load_balance_sum = load_balance_sum + block_moe_output.load_balance_loss
                himoe_intra_sum = himoe_intra_sum + block_moe_output.himoe_intra_loss
                himoe_inter_sum = himoe_inter_sum + block_moe_output.himoe_inter_loss
                router_z_loss_sum = router_z_loss_sum + block_moe_output.router_z_loss
                if collect_dispatch_metrics:
                    total_used = total_used + block_moe_output.used_capacity.float().sum()
                    total_attempted += block_moe_output.total_attempted
                    capacity_std_sum = capacity_std_sum + block_moe_output.used_capacity.float().std()
                    capacity_std_count += 1
                    expert_dispatch_cv_sum = expert_dispatch_cv_sum + _coefficient_of_variation(block_moe_output.used_capacity)
                    expert_dispatch_cv_count += 1
                    group_loads = _group_loads_from_expert_loads(
                        block_moe_output.used_capacity,
                        getattr(self.config, "analysis_n_groups", 0),
                    )
                    if group_loads is not None:
                        group_loads = group_loads.to(dtype=torch.float32)
                        group_dispatch_cv_sum = group_dispatch_cv_sum + _coefficient_of_variation(group_loads)
                        group_dispatch_cv_count += 1
                        group_total = group_loads.sum().clamp_min(1e-10)
                        group_dispatch_max_frac_sum = group_dispatch_max_frac_sum + (group_loads.max() / group_total)
                        group_dispatch_max_frac_count += 1
                    rank_loads = getattr(block.mlp, "last_rank_route_loads", None)
                    if rank_loads is not None:
                        rank_loads = rank_loads.to(device=device, dtype=torch.float32)
                        rank_dispatch_cv_sum = rank_dispatch_cv_sum + _coefficient_of_variation(rank_loads)
                        rank_dispatch_cv_count += 1
                        rank_total = rank_loads.sum().clamp_min(1e-10)
                        rank_dispatch_max_frac_sum = rank_dispatch_max_frac_sum + (rank_loads.max() / rank_total)
                        rank_dispatch_max_frac_count += 1
                if expert_count_matrix is not None:
                    selected_experts = block_moe_output.expert_indices[block_moe_output.selection_mask]
                    if selected_experts.numel() > 0:
                        expert_count_matrix[block_idx] = torch.bincount(
                            selected_experts.reshape(-1),
                            minlength=self.config.n_exp,
                        ).to(dtype=expert_count_matrix.dtype)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # add the auxiliary load balancing loss and router z loss to the main loss
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                # apply global weight only to load balance loss
                loss += self.config.aux_loss_weight * load_balance_sum

                # Hi-MoE losses already weighted by lambda
                if self.config.use_himoe_regularization:
                    loss += himoe_intra_sum
                    loss += himoe_inter_sum

            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                loss += self.config.router_z_loss_weight * router_z_loss_sum
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_moe_stats:
            moe_stats = MoEStats(
                load_balance_sum=load_balance_sum,
                load_balance_count=int(self.config.n_exp > 1 and self.config.use_aux_loss),
                himoe_intra_sum=himoe_intra_sum,
                himoe_intra_count=int(self.config.n_exp > 1 and self.config.use_himoe_regularization),
                himoe_inter_sum=himoe_inter_sum,
                himoe_inter_count=int(self.config.n_exp > 1 and self.config.use_himoe_regularization),
                router_z_loss_sum=router_z_loss_sum,
                router_z_loss_count=int(self.config.n_exp > 1 and self.config.use_router_z_loss),
                total_used=total_used,
                total_attempted=total_attempted,
                capacity_std_sum=capacity_std_sum,
                capacity_std_count=capacity_std_count,
                expert_dispatch_cv_sum=expert_dispatch_cv_sum,
                expert_dispatch_cv_count=expert_dispatch_cv_count,
                group_dispatch_cv_sum=group_dispatch_cv_sum,
                group_dispatch_cv_count=group_dispatch_cv_count,
                group_dispatch_max_frac_sum=group_dispatch_max_frac_sum,
                group_dispatch_max_frac_count=group_dispatch_max_frac_count,
                rank_dispatch_cv_sum=rank_dispatch_cv_sum,
                rank_dispatch_cv_count=rank_dispatch_cv_count,
                rank_dispatch_max_frac_sum=rank_dispatch_max_frac_sum,
                rank_dispatch_max_frac_count=rank_dispatch_max_frac_count,
            )
            if expert_count_matrix is not None:
                return logits, loss, moe_stats, expert_count_matrix
            return logits, loss, moe_stats
        if expert_count_matrix is not None:
            return logits, loss, expert_count_matrix
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert not 'moe' in model_type, "Pretrained checkpoints not available for MoE"
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
        nodecay_params = [p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def shared_parameters(self):
        return [
            param
            for param in self.parameters()
            if param.requires_grad and not getattr(param, "_ep_local_expert", False)
        ]

    def local_expert_parameters(self):
        return [
            param
            for param in self.parameters()
            if param.requires_grad and getattr(param, "_ep_local_expert", False)
        ]

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 1_513e12 # H100 GPU bfloat16
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
