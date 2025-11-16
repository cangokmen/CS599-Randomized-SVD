import os
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class EvalMetrics:
    split: str
    loss: float
    perplexity: float
    bits_per_token: float
    tokens_total: int
    batches: int
    avg_ms_per_batch: float
    tokens_per_sec: float
    peak_mem_mb: float
    device: str
    dtype: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@torch.inference_mode()
def evaluate_perplexity_memmap(
    model: torch.nn.Module,
    data_dir: str,
    split: str = "val",
    block_size: int = 1024,
    batch_size: int = 12,
    eval_iters: int = 200,
    device: Optional[torch.device] = None,
    amp: bool = True,
    notes: Optional[str] = None,
) -> EvalMetrics:
        assert split in ("train", "val")
        device = device or next(model.parameters()).device
        device_type = device.type
        is_cuda = (device_type == "cuda" and torch.cuda.is_available())
        model.eval()

        def get_batch():
            bin_path = os.path.join(data_dir, f"{split}.bin")
            data = np.memmap(bin_path, dtype=np.uint16, mode='r')
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            if device_type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            return x, y

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        if device_type == "cuda":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
        else:
            amp_dtype = None

        autocast_ctx = (
            torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
            if (amp and device_type != "cpu" and amp_dtype is not None)
            else torch.cuda.amp.autocast(enabled=False)
        )

        times_ms = []
        total_tokens = 0
        loss_sum = 0.0

        warmup = min(5, max(1, eval_iters // 20))
        for _ in range(warmup):
            x, y = get_batch()
            with autocast_ctx:
                _, _loss = model(x, y)

        for _ in range(eval_iters):
            x, y = get_batch()
            t0 = time.perf_counter()
            with autocast_ctx:
                _, loss = model(x, y)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)
            loss_sum += float(loss.item())
            total_tokens += batch_size * block_size

        mean_loss = loss_sum / eval_iters
        perplexity = math.exp(mean_loss)
        bits_per_token = mean_loss / math.log(2.0)
        avg_ms = sum(times_ms) / eval_iters
        tokens_per_sec = total_tokens / (sum(times_ms) / 1000.0 + 1e-12)
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            if is_cuda else 0.0
        )
        param_dtype = str(next(model.parameters()).dtype).replace("torch.", "")

        return EvalMetrics(
            split=split,
            loss=mean_loss,
            perplexity=perplexity,
            bits_per_token=bits_per_token,
            tokens_total=total_tokens,
            batches=eval_iters,
            avg_ms_per_batch=avg_ms,
            tokens_per_sec=tokens_per_sec,
            peak_mem_mb=peak_mem_mb,
            device=str(device),
            dtype=param_dtype,
            notes=notes,
        )