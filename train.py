import os

from codon.utils.parsing import parse_train_args
args = parse_train_args()
from codon.datasets import AFDBDataset, seq_collate
from codon.wrapper import PMPNNWrapper


import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np

torch.set_float32_matmul_precision('medium')

if args.wandb:
    wandb.init(
        entity='coarse-graining-mit',
        settings=wandb.Settings(start_method="fork"),
        project="codon",
        name=args.run_name,
        config=args,
    )

full_ds = AFDBDataset(args)

train_len = int(len(full_ds) * 0.95)
if len(full_ds) < 30:
    train_ds = val_ds = full_ds
else:
    train_ds, val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])
print('train, val lens', len(train_ds), len(val_ds))
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=seq_collate,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=seq_collate,
    shuffle=False,
)

model = PMPNNWrapper(args)

if args.overfit:
    val_loader = train_loader

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=args.val_batches or 1.0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    gradient_clip_val=args.grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=1,
            save_last=True,
            every_n_epochs=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    check_val_every_n_epoch=args.val_epoch_freq,
    val_check_interval=args.val_check_interval,
    logger=False
)
torch.manual_seed(1)
np.random.seed(1)

if args.validate:
    trainer.validate(model, val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
