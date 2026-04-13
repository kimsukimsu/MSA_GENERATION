"""
Training script for the MSAFlow SFM Decoder.

Matches the paper's setup (Section 6.8.2):
  • 129M parameter SFM decoder (DiT, 12 blocks, hidden 768, 12 heads)
  • 7 epochs on OpenFold-derived LMDB (4M MSAs)
  • 4 × H200 GPUs via Accelerate
  • LR warmup 5000 steps, peak LR 1e-5, weight decay 0.1
  • Batch size: 2560 max query-sequence tokens
  • 32 sequences sampled per MSA (Neff-weighted)

Usage:
    accelerate launch --num_processes 4 msaflow/training/train_decoder.py \
        --lmdb_path /data/msaflow.lmdb \
        --output_dir /runs/decoder \
        --config msaflow/configs/decoder.yaml
"""

import os
import math
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from msaflow.models.sfm_decoder import SFMDecoder, sfm_loss
from msaflow.data.dataset import MSADecoderDataset, decoder_collate_fn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
):
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, _lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Simple EMA wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name] = self.shadow[name].to(param.device)
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
    )
    set_seed(cfg.training.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SFMDecoder(
        vocab_size=cfg.model.vocab_size,
        msa_dim=cfg.model.msa_dim,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        max_seq_len=cfg.model.max_seq_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info("SFM decoder  ----- %.1fM params  depth=%d  hidden=%d  vocab=%d",
                     n_params / 1e6, cfg.model.depth, cfg.model.hidden_size, cfg.model.vocab_size)

    ema = EMA(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = MSADecoderDataset(
        lmdb_path=cfg.data.lmdb_path,
        n_seqs_per_msa=cfg.data.n_seqs_per_msa,
        max_seq_len=cfg.data.max_seq_len,
    )
    if accelerator.is_main_process:
        logger.info("Dataset loaded  ----- %d entries", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=decoder_collate_fn,
        drop_last=True,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = len(loader) * cfg.training.epochs // cfg.training.grad_accumulation
    scheduler = get_lr_schedule(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, global_step = 0, 0

    ckpt_path = output_dir / "latest.pt"
    if ckpt_path.exists() and cfg.training.resume:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if accelerator.is_main_process:
            logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)

    # ── Prepare with Accelerate ───────────────────────────────────────────────
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    # ── Optional wandb ────────────────────────────────────────────────────────
    use_wandb = cfg.training.get("use_wandb", False)
    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.init(
            project=cfg.training.wandb_project,
            name=cfg.training.get("wandb_run_name", None),
            config=OmegaConf.to_container(cfg),
        )
        wandb.watch(
            accelerator.unwrap_model(model),
            log="gradients",
            log_freq=cfg.training.log_every,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Training started  ----- total_steps=%d  epochs=%d  lr=%.2e",
                     total_steps, cfg.training.epochs, cfg.training.lr)
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            with accelerator.accumulate(model):
                msa_emb = batch["msa_emb"]      # (B_flat, L, 128)
                tokens  = batch["tokens"]       # (B_flat, L)
                weights = batch["weights"]      # (B_flat,)

                loss = sfm_loss(
                    accelerator.unwrap_model(model),
                    tokens,
                    msa_emb,
                    weights=weights,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if ema is not None:
                        ema.update(accelerator.unwrap_model(model))

            epoch_loss += loss.detach().float()

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "epoch %d | step %d | loss %.4f | lr %.2e",
                    epoch, global_step, loss.item(), lr,
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                    }, step=global_step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            ckpt = {
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
            torch.save(ckpt, output_dir / "latest.pt")
            torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")
            avg_loss = epoch_loss.item() / len(loader)
            logger.info(
                "Epoch %d complete | avg loss %.4f",
                epoch,
                avg_loss,
            )
            logger.info("-" * 60)
            if use_wandb:
                import wandb
                wandb.log({
                    "epoch/avg_loss": avg_loss,
                    "epoch/epoch_num": epoch,
                    "epoch/learning_rate": scheduler.get_last_lr()[0],
                }, step=global_step)

    if accelerator.is_main_process:
        logger.info("Training complete.")
        if ema is not None:
            # Save EMA weights as the final model
            unwrapped = accelerator.unwrap_model(model)
            ema.copy_to(unwrapped)
            torch.save({"model": unwrapped.state_dict()}, output_dir / "decoder_ema_final.pt")
        if use_wandb:
            import wandb
            wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to decoder.yaml config")
    parser.add_argument("--lmdb_path",  default=None,   help="Override data.lmdb_path")
    parser.add_argument("--output_dir", default=None,   help="Override training.output_dir")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.lmdb_path:
        cfg.data.lmdb_path = args.lmdb_path
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
