from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

vendor_root = Path(__file__).resolve().parents[1] / "third_party" / "tutel"
if vendor_root.exists() and str(vendor_root) not in sys.path:  # pragma: no cover - depends on local environment
    # Prefer the repo-local patched Tutel checkout over any pip-installed Tutel.
    sys.path.insert(0, str(vendor_root))

try:  # pragma: no cover - depends on local environment
    from tutel import net as tutel_net
    from tutel import moe as tutel_moe
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    tutel_net = None
    tutel_moe = None

def tutel_available() -> bool:
    return tutel_net is not None and tutel_moe is not None


def require_tutel() -> None:
    if not tutel_available():
        raise RuntimeError(
            "ep_backend='tutel' was requested, but Tutel is not installed. "
            "Install Tutel before enabling true expert parallelism, or keep the "
            "patched checkout at third_party/tutel available."
        )


def distributed_world_size(group=None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    return 1


def distributed_rank(group=None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=group)
    return 0


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def synchronize_if_cuda(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def active_autocast_dtype(device_type: str, fallback: torch.dtype) -> torch.dtype:
    try:
        if device_type == "cuda":
            enabled = torch.is_autocast_enabled()
        elif device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            enabled = torch.is_autocast_cpu_enabled()
        else:
            enabled = False
    except TypeError:
        enabled = torch.is_autocast_enabled()

    if not enabled:
        return fallback

    if hasattr(torch, "get_autocast_dtype"):
        try:
            return torch.get_autocast_dtype(device_type)
        except TypeError:
            pass

    if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
        return torch.get_autocast_gpu_dtype()
    if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
        return torch.get_autocast_cpu_dtype()
    return fallback


def mark_local_expert_parameters(module: torch.nn.Module) -> None:
    for param in module.parameters():
        setattr(param, "_ep_local_expert", True)
        setattr(param, "_tutel_expert", True)
        setattr(param, "skip_allreduce", True)


def sync_shared_parameters(parameters, src: int = 0, group=None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for param in parameters:
        dist.broadcast(param.data, src=src, group=group)


def _all_reduce_gradient_bucket(
    grads: list[torch.Tensor],
    group=None,
    scale: float = 1.0,
) -> None:
    if not grads:
        return

    flat = torch.cat([grad.contiguous().view(-1) for grad in grads], dim=0)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    if scale != 1.0:
        flat.mul_(scale)

    offset = 0
    for grad in grads:
        numel = grad.numel()
        grad.copy_(flat[offset : offset + numel].view_as(grad))
        offset += numel


def all_reduce_shared_gradients(
    parameters,
    group=None,
    average: bool = True,
    bucket_size_mb: int = 64,
) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size(group=group)
    scale = (1.0 / world_size) if average else 1.0
    bucket_size_bytes = max(int(bucket_size_mb * 1024 * 1024), 1)

    grads_by_type: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad
        grads_by_type.setdefault((grad.device, grad.dtype), []).append(grad)

    for grads in grads_by_type.values():
        bucket: list[torch.Tensor] = []
        bucket_bytes = 0
        for grad in grads:
            grad_bytes = grad.numel() * grad.element_size()
            if bucket and bucket_bytes + grad_bytes > bucket_size_bytes:
                _all_reduce_gradient_bucket(bucket, group=group, scale=scale)
                bucket = []
                bucket_bytes = 0
            bucket.append(grad)
            bucket_bytes += grad_bytes
        _all_reduce_gradient_bucket(bucket, group=group, scale=scale)


def gather_scalar_per_rank(value: int | float, device: torch.device, group=None) -> torch.Tensor:
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)


def exchange_splits(send_splits: list[int], device: torch.device, group=None) -> list[int]:
    if not (dist.is_available() and dist.is_initialized()) or len(send_splits) == 1:
        return list(send_splits)
    send = torch.tensor(send_splits, device=device, dtype=torch.int64)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return [int(x) for x in recv.cpu().tolist()]


def _all_to_all_variable_torch_impl(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group=None,
) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or len(send_splits) == 1:
        return tensor

    if tensor.dim() == 0:
        tensor = tensor.view(1)

    rest_shape = tuple(tensor.shape[1:])
    input_chunks = []
    start = 0
    for split in send_splits:
        input_chunks.append(tensor.narrow(0, start, split).contiguous())
        start += split

    output_chunks = [
        torch.empty((split, *rest_shape), device=tensor.device, dtype=tensor.dtype)
        for split in recv_splits
    ]
    dist.all_to_all(output_chunks, input_chunks, group=group)
    if not output_chunks:
        return torch.empty((0, *rest_shape), device=tensor.device, dtype=tensor.dtype)
    return torch.cat(output_chunks, dim=0)


class _AllToAllVariableTorchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, send_splits: list[int], recv_splits: list[int], group):
        ctx.send_splits = list(send_splits)
        ctx.recv_splits = list(recv_splits)
        ctx.group = group
        return _all_to_all_variable_torch_impl(tensor, send_splits, recv_splits, group=group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_variable_torch_impl(
            grad_output,
            ctx.recv_splits,
            ctx.send_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all_variable_torch(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group=None,
) -> torch.Tensor:
    if tensor.requires_grad and tensor.is_floating_point():
        return _AllToAllVariableTorchAutograd.apply(tensor, send_splits, recv_splits, group)
    return _all_to_all_variable_torch_impl(tensor, send_splits, recv_splits, group=group)


def all_to_all_variable(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    backend: str,
    group=None,
) -> torch.Tensor:
    del backend
    return _all_to_all_variable_torch(tensor, send_splits, recv_splits, group=group)


def _route_rank_splits(
    router_output,
    experts_per_rank: int,
    world_size: int,
    respect_capacity: bool,
) -> list[int]:
    valid_mask = router_output.dispatch_mask if respect_capacity else router_output.selection_mask
    route_experts = router_output.expert_indices[valid_mask]
    if route_experts.numel() == 0:
        return [0 for _ in range(world_size)]
    route_ranks = torch.div(route_experts, experts_per_rank, rounding_mode="floor")
    return torch.bincount(route_ranks, minlength=world_size).tolist()


def _build_tutel_critical_data(router_output, n_exp: int, respect_capacity: bool):
    expert_indices = router_output.expert_indices
    top_k = expert_indices.size(1)
    active_mask = router_output.dispatch_mask if respect_capacity else router_output.selection_mask

    # Tutel's fast dispatcher consumes one tensor per top-k route.
    # Invalid routes are represented out of range so Tutel skips them entirely.
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

    indices_s = [
        safe_indices[:, route_id].contiguous().view(-1)
        for route_id in range(top_k)
    ]
    locations_s = [
        safe_locations[:, route_id].contiguous().view(-1)
        for route_id in range(top_k)
    ]
    gates_s = [
        safe_gates[:, route_id].contiguous().view(-1)
        for route_id in range(top_k)
    ]

    selected_mask = router_output.selection_mask.to(torch.bool)
    dispatch_count = torch.bincount(
        expert_indices[selected_mask].reshape(-1),
        minlength=n_exp,
    ).to(torch.int32)

    return (
        n_exp,
        indices_s,
        locations_s,
        gates_s,
        int(router_output.capacity),
        dispatch_count,
    )


def _global_local_expert_counts(
    used_capacity: torch.Tensor,
    experts_per_rank: int,
    rank: int,
    group=None,
) -> torch.Tensor:
    global_counts = used_capacity.detach().to(dtype=torch.long).clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=group)
    start = rank * experts_per_rank
    return global_counts.narrow(0, start, experts_per_rank).contiguous()


def _distributed_expert_forward_tutel(
    x_flat: torch.Tensor,
    router_output,
    local_experts: torch.nn.Module,
    experts_per_rank: int,
    device_type: str,
    nvtx_enabled: bool,
    layer_name: str,
    respect_capacity: bool,
    collect_dispatch_metrics: bool,
    group=None,
) -> tuple[torch.Tensor, dict]:
    require_tutel()

    world_size = distributed_world_size(group=group)
    rank = distributed_rank(group=group)
    n_exp = int(router_output.used_capacity.numel())
    measure_phase_timing = nvtx_enabled

    crit = _build_tutel_critical_data(router_output, n_exp=n_exp, respect_capacity=respect_capacity)
    send_splits = None
    recv_splits = None
    if collect_dispatch_metrics:
        send_splits = _route_rank_splits(
            router_output,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            respect_capacity=respect_capacity,
        )
        recv_splits = exchange_splits(send_splits, x_flat.device, group=group)

    with nvtx_range(f"{layer_name}/dispatch", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        encoded = tutel_moe.fast_encode(x_flat.contiguous(), crit)
        dispatched = tutel_net.all_to_all(encoded, 1, 0, group=group)
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            dispatch_ms = (time.perf_counter() - t0) * 1000.0
        else:
            dispatch_ms = 0.0

    with nvtx_range(f"{layer_name}/expert_compute", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        processed = local_experts(dispatched)
        local_counts = None
        if collect_dispatch_metrics:
            local_counts = _global_local_expert_counts(
                router_output.used_capacity,
                experts_per_rank=experts_per_rank,
                rank=rank,
                group=group,
            )
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            expert_ms = (time.perf_counter() - t0) * 1000.0
        else:
            expert_ms = 0.0

    with nvtx_range(f"{layer_name}/combine", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        gathered = tutel_net.all_to_all(processed, 0, 1, group=group)
        combined = tutel_moe.fast_decode(gathered, crit)
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            combine_ms = (time.perf_counter() - t0) * 1000.0
        else:
            combine_ms = 0.0

    if collect_dispatch_metrics:
        recv_total = int(sum(recv_splits))
        rank_route_loads = gather_scalar_per_rank(recv_total, device=x_flat.device, group=group)
        routes_sent = int(sum(send_splits))
        send_splits_list = [int(x) for x in send_splits]
        recv_splits_list = [int(x) for x in recv_splits]
    else:
        recv_total = None
        rank_route_loads = None
        routes_sent = None
        send_splits_list = None
        recv_splits_list = None
    metrics = {
        "dispatch_ms": dispatch_ms,
        "expert_ms": expert_ms,
        "combine_ms": combine_ms,
        "routes_sent": routes_sent,
        "routes_received": recv_total,
        "local_expert_counts": local_counts,
        "send_splits": send_splits_list,
        "recv_splits": recv_splits_list,
        "rank_route_loads": rank_route_loads,
        "rank": rank,
    }
    return combined, metrics


def distributed_expert_forward(
    x_flat: torch.Tensor,
    router_output,
    local_experts: torch.nn.Module,
    experts_per_rank: int,
    backend: str,
    device_type: str,
    nvtx_enabled: bool,
    layer_name: str,
    respect_capacity: bool,
    collect_dispatch_metrics: bool = False,
    group=None,
) -> tuple[torch.Tensor, dict]:
    if backend == "tutel":
        return _distributed_expert_forward_tutel(
            x_flat=x_flat,
            router_output=router_output,
            local_experts=local_experts,
            experts_per_rank=experts_per_rank,
            device_type=device_type,
            nvtx_enabled=nvtx_enabled,
            layer_name=layer_name,
            respect_capacity=respect_capacity,
            collect_dispatch_metrics=collect_dispatch_metrics,
            group=group,
        )

    world_size = distributed_world_size(group=group)
    rank = distributed_rank(group=group)
    num_tokens, model_dim = x_flat.shape
    _, top_k = router_output.expert_indices.shape
    expert_output_dtype = active_autocast_dtype(device_type, x_flat.dtype)
    measure_phase_timing = nvtx_enabled

    token_indices = torch.arange(num_tokens, device=x_flat.device, dtype=torch.long).unsqueeze(1)
    token_indices = token_indices.expand(num_tokens, top_k)

    valid_mask = router_output.dispatch_mask if respect_capacity else router_output.selection_mask
    route_experts = router_output.expert_indices[valid_mask]
    route_tokens = token_indices[valid_mask]
    route_weights = router_output.combine_weights[valid_mask]

    route_ranks = torch.div(route_experts, experts_per_rank, rounding_mode="floor")
    local_expert_ids = route_experts.remainder(experts_per_rank)

    with nvtx_range(f"{layer_name}/dispatch", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        if route_ranks.numel() > 0:
            order = torch.argsort(route_ranks)
            route_ranks = route_ranks[order]
            local_expert_ids = local_expert_ids[order]
            route_tokens = route_tokens[order]
            route_weights = route_weights[order]
            send_hidden = x_flat[route_tokens]
        else:
            send_hidden = x_flat.new_empty((0, model_dim))

        send_splits = torch.bincount(route_ranks, minlength=world_size).tolist()
        recv_splits = exchange_splits(send_splits, x_flat.device, group=group)

        recv_hidden = all_to_all_variable(send_hidden, send_splits, recv_splits, backend=backend, group=group)
        recv_expert_ids = _all_to_all_variable_torch(local_expert_ids, send_splits, recv_splits, group=group)
        recv_token_ids = _all_to_all_variable_torch(route_tokens, send_splits, recv_splits, group=group)
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            dispatch_ms = (time.perf_counter() - t0) * 1000.0
        else:
            dispatch_ms = 0.0

    recv_total = int(sum(recv_splits))
    recv_counts_tensor = torch.tensor(recv_splits, device=x_flat.device, dtype=torch.long)
    if recv_total > 0:
        recv_sources = torch.repeat_interleave(
            torch.arange(world_size, device=x_flat.device, dtype=torch.long),
            recv_counts_tensor,
        )
    else:
        recv_sources = torch.empty((0,), device=x_flat.device, dtype=torch.long)

    with nvtx_range(f"{layer_name}/expert_compute", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        if recv_total > 0:
            local_counts = torch.bincount(recv_expert_ids, minlength=experts_per_rank)
            max_count = int(local_counts.max().item())
            expert_inputs = recv_hidden.new_zeros((experts_per_rank, max_count, model_dim))
            route_order_by_expert: list[torch.Tensor] = []
            for local_expert in range(experts_per_rank):
                expert_routes = torch.nonzero(recv_expert_ids == local_expert, as_tuple=False).flatten()
                route_order_by_expert.append(expert_routes)
                if expert_routes.numel() > 0:
                    expert_inputs[local_expert, : expert_routes.numel()] = recv_hidden[expert_routes]

            expert_outputs = local_experts(expert_inputs)
            expert_output_dtype = expert_outputs.dtype
            processed = expert_outputs.new_empty((recv_total, model_dim))
            for local_expert, expert_routes in enumerate(route_order_by_expert):
                if expert_routes.numel() > 0:
                    processed[expert_routes] = expert_outputs[local_expert, : expert_routes.numel()]
        else:
            local_counts = torch.zeros(experts_per_rank, device=x_flat.device, dtype=torch.long)
            processed = recv_hidden.new_empty((0, model_dim), dtype=expert_output_dtype)
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            expert_ms = (time.perf_counter() - t0) * 1000.0
        else:
            expert_ms = 0.0

    with nvtx_range(f"{layer_name}/combine", nvtx_enabled):
        t0 = time.perf_counter() if measure_phase_timing else None
        if recv_total > 0:
            back_order = torch.argsort(recv_sources)
            send_back = processed[back_order]
            send_back_tokens = recv_token_ids[back_order]
        else:
            send_back = processed
            send_back_tokens = recv_token_ids

        result_back = all_to_all_variable(send_back, recv_splits, send_splits, backend=backend, group=group)
        result_back_tokens = _all_to_all_variable_torch(send_back_tokens, recv_splits, send_splits, group=group)

        combined = result_back.new_zeros((num_tokens, model_dim))
        if result_back_tokens.numel() > 0:
            weighted = result_back * route_weights.to(result_back.dtype).unsqueeze(-1)
            combined.index_add_(0, result_back_tokens, weighted)
        if measure_phase_timing:
            synchronize_if_cuda(device_type)
            combine_ms = (time.perf_counter() - t0) * 1000.0
        else:
            combine_ms = 0.0

    rank_route_loads = None
    if collect_dispatch_metrics:
        rank_route_loads = gather_scalar_per_rank(recv_total, device=x_flat.device, group=group)
    metrics = {
        "dispatch_ms": dispatch_ms,
        "expert_ms": expert_ms,
        "combine_ms": combine_ms,
        "routes_sent": int(sum(send_splits)) if collect_dispatch_metrics else None,
        "routes_received": recv_total if collect_dispatch_metrics else None,
        "local_expert_counts": local_counts if collect_dispatch_metrics else None,
        "send_splits": [int(x) for x in send_splits] if collect_dispatch_metrics else None,
        "recv_splits": [int(x) for x in recv_splits] if collect_dispatch_metrics else None,
        "rank_route_loads": rank_route_loads,
        "rank": rank,
    }
    return combined, metrics
