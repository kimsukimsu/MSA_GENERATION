"""
Training script for the MSAFlow Latent Flow Matching encoder.

Matches the paper's setup (Section 6.8.2):
  • 130M parameter latent FM (DiT, 12 blocks, hidden 768, 12 heads)
  • 15 epochs on OpenFold-derived LMDB (4M MSAs)
  • 4 × H200 GPUs via Accelerate
  • LR warmup 3000 steps, peak LR 2.6e-4, weight decay 0.1
  • Batch size: 32768 max total sequence length

Usage:
    accelerate launch --num_processes 4 msaflow/training/train_latent_fm.py \
        --config msaflow/configs/latent_fm.yaml \
        --lmdb_path /data/msaflow.lmdb \
        --output_dir /runs/latent_fm
"""

import math
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from msaflow.models.latent_fm import LatentFMEncoder, rectified_flow_loss
from msaflow.data.dataset import LatentFMDataset, latent_collate_fn
from msaflow.training.train_decoder import EMA, get_lr_schedule

logger = logging.getLogger(__name__)


def train(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
    )
    set_seed(cfg.training.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = LatentFMEncoder(
        msa_dim=cfg.model.msa_dim,
        esm_dim=cfg.model.esm_dim,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        max_seq_len=cfg.model.max_seq_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info("Latent FM encoder: %.1fM parameters", n_params / 1e6)

    ema = EMA(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = LatentFMDataset(
        lmdb_path=cfg.data.lmdb_path,
        max_seq_len=cfg.data.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=latent_collate_fn,
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

    # ── Prepare ───────────────────────────────────────────────────────────────
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    if accelerator.is_main_process and cfg.training.get("use_wandb", False):
        import wandb
        wandb.init(project=cfg.training.wandb_project, config=OmegaConf.to_container(cfg))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            with accelerator.accumulate(model):
                z1      = batch["msa_emb"]   # (B, L, 128)
                esm_emb = batch["esm_emb"]   # (B, L, 1280)

                loss = rectified_flow_loss(
                    accelerator.unwrap_model(model),
                    z1,
                    esm_emb,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if ema is not None:
                        ema.update(accelerator.unwrap_model(model))

            epoch_loss += loss.detach().float()

            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "epoch %d | step %d | loss %.4f | lr %.2e",
                    epoch, global_step, loss.item(), lr,
                )
                if cfg.training.get("use_wandb", False):
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)

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
            logger.info(
                "Epoch %d complete | avg loss %.4f",
                epoch,
                epoch_loss.item() / len(loader),
            )

    if accelerator.is_main_process:
        logger.info("Training complete.")
        if ema is not None:
            unwrapped = accelerator.unwrap_model(model)
            ema.copy_to(unwrapped)
            torch.save({"model": unwrapped.state_dict()}, output_dir / "latent_fm_ema_final.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--lmdb_path",  default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.lmdb_path:
        cfg.data.lmdb_path = args.lmdb_path
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
