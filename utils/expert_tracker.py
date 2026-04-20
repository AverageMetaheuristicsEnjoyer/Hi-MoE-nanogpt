import os
import torch
import torch._dynamo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ExpertActivationTracker:
    def __init__(self, model_params, world_size, output_dir, annot=False):
        self.model_params = model_params
        self.world_size = world_size
        self.output_dir = output_dir
        self.activation_counts = defaultdict(list)
        self.aggregated_count_matrix = None
        self.annot = annot
        os.makedirs(self.output_dir, exist_ok=True)

    def update_count_matrix(self, count_matrix: torch.Tensor):
        if count_matrix is None:
            return
        count_matrix = count_matrix.detach().to(dtype=torch.float32)
        if self.aggregated_count_matrix is None:
            self.aggregated_count_matrix = count_matrix
            return
        if self.aggregated_count_matrix.device != count_matrix.device:
            count_matrix = count_matrix.to(self.aggregated_count_matrix.device)
        self.aggregated_count_matrix = self.aggregated_count_matrix + count_matrix

    def _get_activation_hook(self, layer_name, total_experts):
        @torch._dynamo.disable
        def hook(module, input, output):
            gate_top_k_idx = output[0]
            if gate_top_k_idx is None or gate_top_k_idx.numel() == 0:
                return
            if gate_top_k_idx.numel() > 0:
                counts = torch.bincount(gate_top_k_idx.flatten(), minlength=total_experts)
                self.activation_counts[layer_name].append(counts.cpu())
        return hook

    def register_hook(self, model):
        print("Registering expert activation hooks...")
        unwrapped_model = model.module if hasattr(model, "module") else model
        num_experts = self.model_params.get("n_exp", 8)
        total_experts = num_experts
        layer_count = 0
        if hasattr(unwrapped_model, 'transformer') and hasattr(unwrapped_model.transformer, 'h'):
            for i, block in enumerate(unwrapped_model.transformer.h):
                if hasattr(block, "mlp") and hasattr(block.mlp, "router"):
                    router = block.mlp.router
                    layer_name = f"layer_{i}"
                    router.register_forward_hook(
                        self._get_activation_hook(layer_name, total_experts)
                    )
                    print(f"  - Hook registered for router in block {i} (layer_{i})")
                    layer_count += 1
        if layer_count == 0:
            print("  WARNING: No router modules found! Check model architecture.")
        print(f"  Total routers found: {layer_count}")

    def reset(self):
        self.activation_counts.clear()
        self.aggregated_count_matrix = None
        print("Expert activation counts reset")

    def _build_activation_matrix(self):
        """Build the activation frequency matrix from collected counts.

        Returns:
            tuple: (activation_matrix, layer_indices) or (None, None) if no data
        """
        if self.aggregated_count_matrix is not None:
            layer_totals = self.aggregated_count_matrix.sum(dim=1)
            layer_indices = torch.nonzero(layer_totals > 0, as_tuple=False).flatten().tolist()
            if not layer_indices:
                return None, None
            counts = self.aggregated_count_matrix[layer_indices]
            frequencies = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
            return frequencies.detach().cpu().numpy(), layer_indices

        if len(self.activation_counts) == 0:
            return None, None

        layer_indices = sorted([int(k.split('_')[1]) for k in self.activation_counts.keys()])
        num_layers = len(layer_indices)
        num_experts = self.model_params.get("n_exp", 8)
        total_experts = num_experts

        activation_matrix = np.zeros((num_layers, total_experts))

        for matrix_row, layer_idx in enumerate(layer_indices):
            layer_name = f"layer_{layer_idx}"
            if layer_name in self.activation_counts:
                total_counts_for_layer = torch.stack(self.activation_counts[layer_name]).sum(dim=0)
                total_activations_in_layer = total_counts_for_layer.sum()
                if total_activations_in_layer > 0:
                    frequencies = total_counts_for_layer / total_activations_in_layer
                    activation_matrix[matrix_row, :] = frequencies.numpy()

        if np.sum(activation_matrix) == 0:
            return None, None

        return activation_matrix, layer_indices

    def compute_metrics(self):
        """Compute mean variance and coefficient of variation metrics.

        Returns:
            tuple: (mean_variance, mean_cv) or (-1, -1) if no data
        """
        activation_matrix, _ = self._build_activation_matrix()
        if activation_matrix is None:
            return -1, -1

        layer_variances = []
        layer_cvs = []
        for i in range(activation_matrix.shape[0]):
            layer_frequencies = activation_matrix[i, :]
            if np.sum(layer_frequencies) > 0:
                variance = np.var(layer_frequencies)
                layer_variances.append(variance)
                mean_freq = np.mean(layer_frequencies)
                if mean_freq > 0:
                    cv = np.std(layer_frequencies) / mean_freq
                    layer_cvs.append(cv)

        mean_variance = -1
        mean_cv = -1
        if layer_variances:
            mean_variance = np.mean(layer_variances)
            print(f"Mean Variance of Expert Frequencies across layers: {mean_variance:.8f}")
        if layer_cvs:
            mean_cv = np.mean(layer_cvs)
            print(f"Mean Coefficient of Variation (CV) across layers: {mean_cv:.8f}")

        return mean_variance, mean_cv

    def save_plot(self, iter_no):
        """Generate and save the expert activation heatmap plot.

        Args:
            iter_no: Current iteration number for the plot title and filename
        """
        print(f"Processing expert activations for iteration {iter_no}...")

        activation_matrix, layer_indices = self._build_activation_matrix()
        if activation_matrix is None:
            print("No activation data recorded. Skipping plot generation.")
            return

        num_layers = len(layer_indices)

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            activation_matrix,
            annot=self.annot,
            fmt=".3f",
            cmap="viridis",
            linewidths=.5,
            ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_xlabel("Expert Index", fontsize=12)
        ax.set_ylabel("MoE Layer Index", fontsize=12)

        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(layer_indices)

        ax.set_title(f"Expert Activation Frequency - Iteration {iter_no}", fontsize=14, pad=20)
        output_filename = os.path.join(self.output_dir, f"expert_activations_iter_{iter_no}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Expert activation plot saved to: {output_filename}")

    def plot_and_save(self, iter_no):
        """Compute metrics and save plot. Kept for backward compatibility.

        Args:
            iter_no: Current iteration number

        Returns:
            tuple: (mean_variance, mean_cv)
        """
        mean_variance, mean_cv = self.compute_metrics()
        self.save_plot(iter_no)
        return mean_variance, mean_cv


class GroupDistributionTracker:
    """Tracks how selected experts distribute across groups (conceptual GPUs).

    For vanilla/lossfree MoE with global top-k, this reveals whether tokens
    tend to concentrate experts on a few groups or spread them evenly —
    the key difference from groups/hi-moe which enforce per-group selection.

    Expert-to-group mapping: group_id = expert_id // (n_exp // n_groups)
    E.g., with n_exp=8, n_groups=4: experts 0-1→group0, 2-3→group1, etc.
    """

    def __init__(self, n_exp, n_groups, top_k, output_dir):
        self.n_exp = n_exp
        self.n_groups = n_groups
        self.top_k = top_k
        self.n_exp_per_group = n_exp // n_groups
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Accumulated stats per layer: group selection counts and group spread
        self.group_counts = defaultdict(list)   # layer -> list of [n_groups] tensors
        self.group_spreads = defaultdict(list)   # layer -> list of scalar tensors (mean spread)
        self.max_group_fracs = defaultdict(list) # layer -> list of scalar tensors (mean max-group frac)

    def _get_hook(self, layer_name):
        @torch._dynamo.disable
        def hook(module, input, output):
            top_k_indices = output[0]  # [num_tokens, top_k] or [B, T, top_k]
            if top_k_indices is None or top_k_indices.numel() == 0:
                return

            flat_indices = top_k_indices.reshape(-1, self.top_k)  # [num_tokens, top_k]
            group_ids = flat_indices // self.n_exp_per_group       # [num_tokens, top_k]

            # Per-group selection count across all tokens
            group_onehot = torch.nn.functional.one_hot(group_ids, num_classes=self.n_groups)  # [num_tokens, top_k, n_groups]
            per_token_group_counts = group_onehot.sum(dim=1)  # [num_tokens, n_groups]

            # Total selections per group (summed over tokens)
            total_group_counts = per_token_group_counts.sum(dim=0)  # [n_groups]
            self.group_counts[layer_name].append(total_group_counts.cpu())

            # Group spread: how many unique groups each token uses
            unique_groups_per_token = (per_token_group_counts > 0).sum(dim=1).float()  # [num_tokens]
            self.group_spreads[layer_name].append(unique_groups_per_token.mean().cpu())

            # Max-group concentration: fraction of top-k from the most-selected group
            max_group_frac = per_token_group_counts.float().max(dim=1).values / self.top_k  # [num_tokens]
            self.max_group_fracs[layer_name].append(max_group_frac.mean().cpu())

        return hook

    def register_hook(self, model):
        print("Registering group distribution hooks...")
        unwrapped_model = model.module if hasattr(model, "module") else model
        layer_count = 0
        if hasattr(unwrapped_model, 'transformer') and hasattr(unwrapped_model.transformer, 'h'):
            for i, block in enumerate(unwrapped_model.transformer.h):
                if hasattr(block, "mlp") and hasattr(block.mlp, "router"):
                    router = block.mlp.router
                    layer_name = f"layer_{i}"
                    router.register_forward_hook(self._get_hook(layer_name))
                    layer_count += 1
        print(f"  Group distribution hooks registered for {layer_count} routers "
              f"(n_groups={self.n_groups}, n_exp_per_group={self.n_exp_per_group})")

    def reset(self):
        self.group_counts.clear()
        self.group_spreads.clear()
        self.max_group_fracs.clear()

    def _build_group_freq_matrix(self):
        """Build group selection frequency matrix: layers x groups."""
        if len(self.group_counts) == 0:
            return None, None

        layer_indices = sorted([int(k.split('_')[1]) for k in self.group_counts.keys()])
        num_layers = len(layer_indices)

        matrix = np.zeros((num_layers, self.n_groups))
        for row, layer_idx in enumerate(layer_indices):
            layer_name = f"layer_{layer_idx}"
            if layer_name in self.group_counts:
                total = torch.stack(self.group_counts[layer_name]).sum(dim=0).float()
                if total.sum() > 0:
                    matrix[row, :] = (total / total.sum()).numpy()

        if np.sum(matrix) == 0:
            return None, None
        return matrix, layer_indices

    def compute_metrics(self):
        """Compute group distribution metrics.

        Returns:
            dict with keys: mean_group_spread, max_group_concentration,
                            group_freqs (list of per-group frequencies),
                            or empty dict if no data.
        """
        if len(self.group_counts) == 0:
            return {}

        # Mean group spread across layers
        all_spreads = []
        for layer_name in self.group_spreads:
            layer_spread = torch.stack(self.group_spreads[layer_name]).mean().item()
            all_spreads.append(layer_spread)
        mean_spread = np.mean(all_spreads) if all_spreads else -1

        # Mean max-group concentration across layers
        all_max_fracs = []
        for layer_name in self.max_group_fracs:
            layer_frac = torch.stack(self.max_group_fracs[layer_name]).mean().item()
            all_max_fracs.append(layer_frac)
        mean_max_conc = np.mean(all_max_fracs) if all_max_fracs else -1

        # Per-group selection frequencies (averaged across layers)
        matrix, _ = self._build_group_freq_matrix()
        group_freqs = matrix.mean(axis=0).tolist() if matrix is not None else []

        metrics = {
            'mean_group_spread': mean_spread,
            'max_group_concentration': mean_max_conc,
            'group_freqs': group_freqs,
        }

        print(f"Group distribution: spread={mean_spread:.3f}/{self.n_groups}, "
              f"max_conc={mean_max_conc:.3f}, "
              f"group_freqs={[f'{f:.3f}' for f in group_freqs]}")

        return metrics

    def save_plot(self, iter_no):
        """Generate group distribution heatmap: layers x groups."""
        matrix, layer_indices = self._build_group_freq_matrix()
        if matrix is None:
            return

        num_layers = len(layer_indices)
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=.5,
            ax=ax,
            annot_kws={"size": 10},
            vmin=0, vmax=max(0.5, matrix.max()),
        )
        ax.set_xlabel("Group (GPU)", fontsize=12)
        ax.set_ylabel("MoE Layer Index", fontsize=12)
        ax.set_xticks([i + 0.5 for i in range(self.n_groups)])
        ax.set_xticklabels([f"G{i}" for i in range(self.n_groups)])
        ax.set_yticks([i + 0.5 for i in range(num_layers)])
        ax.set_yticklabels(layer_indices)
        ax.set_title(f"Group (GPU) Selection Frequency - Iteration {iter_no}", fontsize=13, pad=15)

        output_filename = os.path.join(self.output_dir, f"group_distribution_iter_{iter_no}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Group distribution plot saved to: {output_filename}")
