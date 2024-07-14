from codon.utils.parsing import parse_train_args
# sample command: python likelihood_eval.py --data_csv /data/scratch/diaoc/codon/data/shen2022_codon_sequences.csv --batch_size 1 --high_plddt --taxon_condition --num_foldability_batches 3
args = parse_train_args()
from codon.datasets import multi_seq_collate, Shen2022Dataset
from codon.wrapper import PMPNNWrapper

import torch
from torch.utils.data import random_split
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')


full_ds = Shen2022Dataset(args)

data_loader = torch.utils.data.DataLoader(
    full_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=multi_seq_collate,
    shuffle=False,
)

model = PMPNNWrapper.load_from_checkpoint(checkpoint_path='/data/rsg/nlp/hstark/codon/workdir/taxCond20000/epoch=7-step=639028-copy.ckpt',
                                          map_location=None)

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=args.val_batches or 1.0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    gradient_clip_val=args.grad_clip,

    accumulate_grad_batches=args.accumulate_grad,
    check_val_every_n_epoch=args.val_epoch_freq,
    val_check_interval=args.val_check_interval,
    logger=False
)
pl.seed_everything(1)

trainer.test(ckpt_path='/data/rsg/nlp/hstark/codon/workdir/taxCond20000/epoch=7-step=639028-copy.ckpt', model=model, dataloaders=data_loader,
             verbose=True)
