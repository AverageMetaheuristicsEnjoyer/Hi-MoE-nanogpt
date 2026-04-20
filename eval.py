"""
Evaluate a trained nanoMoE model on standard LLM benchmarks using lm-evaluation-harness.

Usage:
    python eval.py --checkpoint out/ckpt.pt --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu
    python eval.py --checkpoint out/ckpt.pt --tasks arc_easy --limit 10   # quick test
    python eval.py --checkpoint out/ckpt.pt --tasks arc_easy --batch_size 16 --device cuda:0
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext

import numpy as np

import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

import lm_eval
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from model import GPT, GPTConfig
from utils.expert_tracker import ExpertActivationTracker


class TiktokenWrapper:
    """Minimal wrapper around tiktoken to expose the interface lm-eval expects."""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")
        self.eos_token_id = self.encoding.eot_token  # 50256
        self.pad_token_id = self.eos_token_id
        self.vocab_size = self.encoding.n_vocab  # 50257

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)


@register_model("nanomoe")
class NanoMoELM(TemplateLM):
    """lm-eval wrapper for nanoMoE GPT models.

    Subclasses TemplateLM to get automatic tokenization, batching,
    progress bars, and caching in loglikelihood().
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
    ):
        super().__init__()
        self.tokenizer = TiktokenWrapper()
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self.backend = "causal"

        # dtype / autocast setup
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        device_type = "cuda" if "cuda" in device else "cpu"
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = checkpoint["model_args"]
        config = GPTConfig(**model_args)
        self._max_length = config.block_size

        model = GPT(config)
        state_dict = checkpoint["model"]
        # handle torch.compile prefix
        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod.") :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)
        self.model = model

    # ── TemplateLM required properties ──────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    # ── Tokenization (required by TemplateLM) ───────────────────────

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    # ── Model helpers ───────────────────────────────────────────────

    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning full logits (B, T, vocab_size).

        We run the transformer backbone + lm_head directly instead of
        calling forward(targets=...) because eval needs full sequence
        logits instead of the training loss path.
        """
        with torch.no_grad(), self.ctx:
            model = self.model
            b, t = input_ids.size()
            pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)

            tok_emb = model.transformer.wte(input_ids)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)

            if model.config.use_momentum:
                if model.config.moe_type == 'moge':
                    momentum = None
                elif model.config.momentum_type == "hb":
                    momentum = torch.zeros_like(x)
                elif model.config.momentum_type == "mars":
                    momentum = (torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x))
                else:
                    momentum = None
            else:
                momentum = None

            for block in model.transformer.h:
                x, momentum, _ = block(x, momentum)
            x = model.transformer.ln_f(x)
            logits = model.lm_head(x)
        return logits

    # ── _loglikelihood_tokens (core method for TemplateLM) ──────────

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Score pre-tokenized context/continuation pairs in batches.

        Each request is ((context_str, continuation_str), context_enc, continuation_enc).
        We left-pad sequences to batch them together for a single forward pass.
        """
        # build (index, full_tokens, cont_len) list
        indexed = []
        for i, (_, ctx_enc, cont_enc) in enumerate(requests):
            full = ctx_enc + cont_enc
            # truncate context from the left if too long
            if len(full) > self._max_length:
                full = full[-self._max_length :]
            cont_len = min(len(cont_enc), len(full))
            indexed.append((i, full, cont_len))

        results: list[tuple[float, bool] | None] = [None] * len(requests)

        # process in batches
        chunks = [
            indexed[i : i + self.batch_size]
            for i in range(0, len(indexed), self.batch_size)
        ]
        for chunk in tqdm(chunks, desc="loglikelihood", disable=disable_tqdm):
            max_len = min(max(len(full) for _, full, _ in chunk), self._max_length)

            # left-pad with eot_token_id
            batch_ids = []
            for _, full, _ in chunk:
                padding = [self.eot_token_id] * (max_len - len(full))
                batch_ids.append(padding + full)

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self._device)
            logits = self._model_logits(input_ids)  # (B, T, V)

            for j, (orig_idx, full, cont_len) in enumerate(chunk):
                pad_len = max_len - len(full)
                # continuation occupies the last cont_len tokens
                # logits[t] predicts token[t+1], so for cont tokens at positions [T-cont_len, T),
                # we need logits at positions [T-cont_len-1, T-1)
                cont_start = max_len - cont_len
                shift_logits = logits[j, cont_start - 1 : max_len - 1, :].float()
                shift_targets = input_ids[j, cont_start:max_len]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
                total = token_log_probs.sum().item()

                greedy = (shift_logits.argmax(dim=-1) == shift_targets).all().item()
                results[orig_idx] = (total, bool(greedy))

        return results

    # ── loglikelihood_rolling ───────────────────────────────────────

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", disable=disable_tqdm):
            (text,) = req.args
            tokens = self.tok_encode(text)

            if not tokens:
                results.append(0.0)
                continue

            # prepend EOT token as context
            all_tokens = [self.eot_token_id] + tokens
            total_log_prob = 0.0

            # sliding window
            stride = self._max_length
            for start in range(0, len(tokens), stride):
                end = min(start + self._max_length, len(tokens))
                window_start = max(0, end - self._max_length)
                window = all_tokens[window_start : end + 1]

                input_ids = torch.tensor([window], dtype=torch.long, device=self._device)
                logits = self._model_logits(input_ids)

                score_from = start - window_start
                shift_logits = logits[0, score_from : len(window) - 1, :].float()
                shift_targets = input_ids[0, score_from + 1 : len(window)]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
                total_log_prob += token_log_probs.sum().item()

            results.append(total_log_prob)

        return results

    # ── generate_until ──────────────────────────────────────────────

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        results = []
        for req in tqdm(requests, desc="generate_until", disable=disable_tqdm):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_k_val = gen_kwargs.get("top_k", None)

            ctx_tokens = self.tok_encode(context)
            if len(ctx_tokens) > self._max_length - 1:
                ctx_tokens = ctx_tokens[-(self._max_length - 1) :]

            input_ids = torch.tensor([ctx_tokens], dtype=torch.long, device=self._device)
            with torch.no_grad(), self.ctx:
                t = temperature if temperature > 0 else 1e-10
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=t,
                    top_k=top_k_val,
                )
            gen_tokens = out[0, len(ctx_tokens) :].tolist()
            gen_text = self.tok_decode(gen_tokens)

            for stop in until:
                idx = gen_text.find(stop)
                if idx != -1:
                    gen_text = gen_text[:idx]

            results.append(gen_text)

        return results


# ── Output formatting ───────────────────────────────────────────────


def format_results_table(task_rows: list[dict]) -> str:
    """Format per-task results (with CV) as a markdown table.

    task_rows: list of dicts with keys: task, metric, value, stderr, cv
    """
    has_cv = any(r.get("cv") is not None for r in task_rows)

    if has_cv:
        lines = ["| Task | Metric | Value | Stderr | CV |",
                 "|------|--------|------:|-------:|---:|"]
    else:
        lines = ["| Task | Metric | Value | Stderr |",
                 "|------|--------|------:|-------:|"]

    scores = []
    cvs = []
    for r in task_rows:
        stderr_str = f"{r['stderr']:.4f}" if isinstance(r["stderr"], float) else str(r["stderr"])
        if has_cv:
            cv_str = f"{r['cv']:.4f}" if r.get("cv") is not None else "N/A"
            lines.append(f"| {r['task']} | {r['metric']} | {r['value']:.4f} | {stderr_str} | {cv_str} |")
        else:
            lines.append(f"| {r['task']} | {r['metric']} | {r['value']:.4f} | {stderr_str} |")
        scores.append(r["value"])
        if r.get("cv") is not None:
            cvs.append(r["cv"])

    if scores:
        avg = sum(scores) / len(scores)
        if has_cv and cvs:
            avg_cv = sum(cvs) / len(cvs)
            lines.append(f"| **Average** | | **{avg:.4f}** | | **{avg_cv:.4f}** |")
        elif has_cv:
            lines.append(f"| **Average** | | **{avg:.4f}** | | |")
        else:
            lines.append(f"| **Average** | | **{avg:.4f}** | |")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def format_activation_table(activation_matrix: np.ndarray, layer_indices: list[int]) -> str:
    """Format per-layer expert activation frequencies (%) as a markdown table.

    activation_matrix: (num_moe_layers, num_experts) with values in [0, 1]
    layer_indices: actual transformer block indices for each MoE layer row
    """
    num_experts = activation_matrix.shape[1]
    # header
    expert_cols = " | ".join(f"E{j}" for j in range(num_experts))
    lines = [f"| Layer | {expert_cols} | CV |"]
    lines.append("|" + "---:|" * (num_experts + 2))

    for i, layer_idx in enumerate(layer_indices):
        freqs = activation_matrix[i] * 100  # convert to %
        mean_freq = np.mean(freqs)
        cv = np.std(freqs) / mean_freq if mean_freq > 0 else 0.0
        freq_strs = " | ".join(f"{f:.2f}" for f in freqs)
        lines.append(f"| {layer_idx} | {freq_strs} | {cv:.4f} |")

    return "\n".join(lines)


def _extract_primary_metric(task_data: dict) -> tuple[str, float, float | str]:
    """Extract the primary accuracy metric from a task result dict.

    Returns (metric_name, value, stderr).
    """
    for metric_key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if metric_key in task_data:
            value = task_data[metric_key]
            stderr_key = metric_key.replace(",none", "_stderr,none")
            if stderr_key not in task_data:
                stderr_key = metric_key + "_stderr"
            stderr = task_data.get(stderr_key, "N/A")
            metric_name = metric_key.split(",")[0]
            return metric_name, value, stderr
    return "acc", 0.0, "N/A"


def main():
    parser = argparse.ArgumentParser(description="Evaluate nanoMoE model on LLM benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt.pt")
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,hellaswag,winogrande,piqa",
        help="Comma-separated list of tasks",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for testing)")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results JSON to this path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    lm = NanoMoELM(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )

    # Set up expert activation tracker (only useful for MoE models with n_exp > 1)
    model_params = {
        "n_exp": lm.model.config.n_exp,
        "moe_type": lm.model.config.moe_type,
    }
    is_moe = lm.model.config.n_exp > 1
    tracker = None
    if is_moe:
        tracker = ExpertActivationTracker(
            model_params=model_params,
            world_size=1,
            output_dir=os.path.join(os.path.dirname(args.checkpoint), "eval_plots"),
        )
        tracker.register_hook(lm.model)

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"Evaluating on: {task_list}")
    print(f"num_fewshot={args.num_fewshot}, limit={args.limit}")

    # Evaluate each task independently so we can measure per-task expert CV
    task_rows = []
    all_results = {}

    for task_name in task_list:
        print(f"\n--- Evaluating {task_name} ---")

        if tracker:
            tracker.reset()

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        # Compute expert CV and activation frequencies for this task
        cv = None
        activations = None
        if tracker:
            _, cv = tracker.compute_metrics()
            if cv == -1:
                cv = None
            act_matrix, layer_indices = tracker._build_activation_matrix()
            if act_matrix is not None:
                activations = {
                    "layer_indices": layer_indices,
                    "frequencies_pct": (act_matrix * 100).tolist(),
                }
                print(f"\n  Expert Activation Frequencies (%) for {task_name}:")
                print(format_activation_table(act_matrix, layer_indices))

        task_data = results.get("results", {}).get(task_name, {})
        all_results[task_name] = task_data

        metric_name, value, stderr = _extract_primary_metric(task_data)
        task_rows.append({
            "task": task_name,
            "metric": metric_name,
            "value": value,
            "stderr": stderr,
            "cv": cv,
            "activations": activations,
        })
        cv_str = f", CV={cv:.4f}" if cv is not None else ""
        print(f"  {task_name}: {metric_name}={value:.4f}{cv_str}")

    table = format_results_table(task_rows)
    print("\n" + table + "\n")

    if args.output:
        save_data = {
            "results": all_results,
            "task_rows": task_rows,
            "config": vars(args),
        }
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
