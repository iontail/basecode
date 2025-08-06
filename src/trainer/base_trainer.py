from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import math
import os
import torch
from torch import nn
from contextlib import nullcontext
from tqdm.auto import tqdm

MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    size = 0
    for p in model.parameters(): size += p.nelement() * p.element_size()
    for b in model.buffers():    size += b.nelement() * b.element_size()
    return size

# -------------------- EMA (선택) --------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.shadow = {}
        self.decay = decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (n, p) in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {}
        for (n, p) in model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for (n, p) in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n].data)
        self._backup = {}

# -------------------- EarlyStopping --------------------
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.count = 0

    def better(self, curr: float, best: float) -> bool:
        if self.mode == "min":
            return curr < best - self.min_delta
        else:
            return curr > best + self.min_delta

    def step(self, curr: float) -> bool:
        if self.best is None:
            self.best = curr
            return False
        if self.better(curr, self.best):
            self.best = curr
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count > self.patience

# -------------------- Trainer --------------------
class Trainer(ABC):
    def __init__(self, model: nn.Module):
        self.model = model
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.ema: Optional[EMA] = None
        self.best_metric: Optional[float] = None
        self.best_ckpt_path: Optional[str] = None
        self.mode: str = "min"  # or "max" for metric comparison

    # ----- 반드시 서브클래스가 구현 -----
    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def get_optimizer(self, lr: float):
        ...

    @abstractmethod
    def get_scheduler(self, optimizer):
        ...

    @abstractmethod
    def validate(self, **kwargs) -> Dict[str, float]:
        """검증 단계. 반환 예: {'val_loss': 0.123, 'acc': 0.9}"""
        ...

    # ----- 환경/정밀도 -----
    def set_seed(self, seed: int):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # 성능 우선

    def select_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def setup_precision(self, precision: str = "fp32"):
        """
        precision: 'fp32' | 'amp' | 'bf16'
        Returns: (autocast_context, scaler_or_None)
        """
        if precision == "amp":
            self.scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = torch.cuda.amp.autocast
        elif precision == "bf16":
            self.scaler = None
            autocast_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.scaler = None
            autocast_ctx = nullcontext
        return autocast_ctx

    def maybe_compile_model(self, use_compile: bool, **kwargs):
        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, **kwargs)

    # ----- 보고/로그 -----
    def param_count(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def log_model_footprint(self):
        size_mb = model_size_b(self.model) / MiB
        n_all = self.param_count(False)
        n_trn = self.param_count(True)
        print(f"Model: {size_mb:.2f} MiB | params: {n_all:,} (trainable: {n_trn:,})")

    # ----- 스케줄러 -----
    def step_scheduler(self, scheduler, metric: Optional[float] = None):
        if scheduler is None: 
            return
        # ReduceLROnPlateau는 metric 필요
        if scheduler.__class__.__name__.lower().startswith("reducelronplateau"):
            if metric is not None:
                scheduler.step(metric)
        else:
            scheduler.step()

    # ----- 체크포인트 -----
    def save_checkpoint(self, path: str, optimizer, scheduler=None, extra: Optional[Dict]=None):
        state = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "extra": extra or {},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: str, optimizer=None, scheduler=None, strict: bool = True):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if optimizer and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        return ckpt.get("extra", {})

    def is_better(self, curr: float, best: Optional[float], mode: str = "min") -> bool:
        if best is None:
            return True
        return (curr < best) if mode == "min" else (curr > best)

    # ----- 메인 루프 -----
    def train(
        self,
        num_epochs: int,
        device: torch.device,
        lr: float = 1e-3,
        precision: str = "fp32",
        accum_steps: int = 1,
        grad_clip: Optional[float] = None,
        scheduler_mode_metric: str = "val_loss",
        early_stop_patience: Optional[int] = None,
        early_stop_delta: float = 0.0,
        metric_mode: str = "min",
        ema_decay: Optional[float] = None,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        resume_path: Optional[str] = None,
        use_compile: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.mode = metric_mode
        self.model.to(device)
        self.maybe_compile_model(use_compile, **(compile_kwargs or {}))

        # 리포트
        self.log_model_footprint()

        # 옵티마/스케줄러
        opt = self.get_optimizer(lr)
        sch = self.get_scheduler(opt)

        # AMP/bf16
        autocast_ctx = self.setup_precision(precision)

        # EMA
        if ema_decay:
            self.ema = EMA(self.model, decay=ema_decay)

        # 재개
        if resume_path:
            self.load_checkpoint(resume_path, optimizer=opt, scheduler=sch, strict=False)
            print(f"Resumed from {resume_path}")

        # 얼리 스톱
        stopper = EarlyStopping(early_stop_patience, early_stop_delta, mode=metric_mode) if early_stop_patience else None

        self.model.train()
        global_step = 0
        best_metric_name = scheduler_mode_metric
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            pbar = tqdm(range(1), desc=f"Epoch {epoch}/{num_epochs}")
            running_loss = 0.0

            for _ in pbar:
                with autocast_ctx():
                    loss = self.get_train_loss(**kwargs) / accum_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulation
                if (global_step + 1) % accum_steps == 0:
                    if grad_clip:
                        if self.scaler:
                            self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                    if self.scaler:
                        self.scaler.step(opt)
                        self.scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                    # EMA 업데이트
                    if self.ema:
                        self.ema.update(self.model)

                running_loss += loss.item() * accum_steps
                global_step += 1
                pbar.set_postfix({"loss": f"{(running_loss):.4f}"})

            # ---- Epoch 끝: 검증 → 스케줄러/체크포인트 ----
            val_stats = self.validate(**kwargs)  # {'val_loss':..., 'acc':...}
            val_metric = val_stats.get(best_metric_name)
            self.step_scheduler(sch, metric=val_metric)

            # 베스트 저장/얼리 스탑
            if save_best and val_metric is not None:
                if self.is_better(val_metric, self.best_metric, mode=metric_mode):
                    self.best_metric = val_metric
                    self.best_ckpt_path = os.path.join(save_dir, "best.pt")
                    self.save_checkpoint(self.best_ckpt_path, opt, sch, extra={"epoch": epoch, "val": val_stats})
                    print(f"[Best] {best_metric_name}={val_metric:.4f} → saved: {self.best_ckpt_path}")

            # 마지막 에폭 저장(선택)
            last_ckpt = os.path.join(save_dir, "last.pt")
            self.save_checkpoint(last_ckpt, opt, sch, extra={"epoch": epoch, "val": val_stats})

            if stopper and val_metric is not None:
                if stopper.step(val_metric):
                    print(f"Early stopping at epoch {epoch}. Best {best_metric_name}={stopper.best:.4f}")
                    break

        # EMA 가중치로 최종 평가가 필요하면 아래 활용
        # if self.ema:
        #     self.ema.apply_to(self.model)
        #     _ = self.validate(**kwargs)
        #     self.ema.restore(self.model)

        self.model.eval()