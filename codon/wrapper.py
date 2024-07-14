import csv
from collections import defaultdict
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch, time, os
import wandb
import math
from matplotlib import pyplot as plt
from openfold.np.residue_constants import restype_order_with_x, restypes_with_x, unk_restype_index

from codon.utils.codon_const import codon_order, codon_types, codon_to_res, res_to_codon, unk_codon_index
from codon.utils.foldability_utils import run_foldability
from codon.utils.logging import get_logger
from codon.utils.pmpnn import ProteinMPNN

logger = get_logger(__name__)


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


class Wrapper(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        out = self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()
        return out

    def test_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=True)

    def on_validation_epoch_end(self):
        self.print_log(prefix='val', save=True)

    def on_test_epoch_end(self):
        self.print_log(prefix='test', save=True)

    def on_before_optimizer_step(self, optimizer):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]),

        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
        )
        return optimizer


class PMPNNWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        self.K = len(restype_order_with_x) if args.train_aa else len(codon_order)
        self.model = ProteinMPNN(args, vocab=self.K, node_features=args.hidden_dim,
                                 edge_features=args.hidden_dim,
                                 hidden_dim=args.hidden_dim, num_encoder_layers=args.num_encoder_layers,
                                 num_decoder_layers=args.num_decoder_layers,
                                 k_neighbors=args.num_neighbors, dropout=args.dropout, ca_only=False)
        self.val_dict = defaultdict(list)

    def general_step(self, batch, stage):
        self.iter_step += 1
        self.stage = stage
        start = time.time()

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        codons = batch['codons']  # (B, L, 65)
        seq = batch['seq']  # (B, L, 21)
        B, L, _, _ = atom37.shape

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        log_probs = self.model.forward_train(
            X=bb_pos,
            S=seq if self.args.train_aa else codons,
            taxon_id=batch['taxon_id'],
            mask=mask,
            chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
            residue_idx=batch['pmpnn_res_idx'],
            chain_encoding_all=batch['pmpnn_chain_encoding'],
        )  # (B, L, self.K)

        train_target = batch['seq'] if self.args.train_aa else batch['codons']
        loss = torch.nn.functional.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1),
                                                 reduction='none')  # (B * L)
        loss = loss.view(B, L)
        loss = loss * mask.float()  # (B, L)
        loss = loss.sum() / mask.sum()

        self.log('loss', loss)
        self.log('forward_dur', time.time() - start)
        self.log('dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        seq = batch['seq']  # (B, L, 21)
        codons = batch['codons']  # (B, L, 65)
        B, L, _, _ = atom37.shape

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        pred_dict = self.model.sample(
            X=bb_pos,
            randn=torch.randn(B, L, device=self.device),
            S_true=seq if self.args.train_aa else codons,
            taxon_id=batch['taxon_id'],
            chain_mask=torch.ones([B, L], device=self.device, dtype=torch.long),
            chain_encoding_all=batch['pmpnn_chain_encoding'],
            residue_idx=batch['pmpnn_res_idx'],
            mask=mask,
            temperature=self.args.sampling_temp,
            omit_AAs_np=np.zeros(self.K).astype(np.float32),
            bias_AAs_np=np.zeros(self.K),
            chain_M_pos=torch.ones([B, L], device=self.device, dtype=torch.long),
            # 1.0 for the bits that need to be predicted
            bias_by_res=torch.zeros([B, L, self.K], dtype=torch.float32, device=self.device)
        )
        probs = pred_dict['probs']  # (B, L, self.K)

        if self.args.train_aa:
            pred_res = probs.argmax(-1)
            pred_codons = torch.stack([torch.tensor(
                [codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in seq], device=self.device) for seq in
                                    pred_res], dim=0)
        else:
            pred_codons = probs.argmax(-1)
            pred_res = torch.stack([torch.tensor(
                [restype_order_with_x.get(codon_to_res[codon_types[i.long().item()]], unk_restype_index) for i in seq], device=self.device) for seq
                in pred_codons], dim=0)

        codon_recovery = (pred_codons == codons).float() * mask
        codon_recovery = codon_recovery.sum() / mask.sum()
        res_recovery = (pred_res == batch['seq']).float() * mask
        res_recovery = res_recovery.sum() / mask.sum()

        self.val_dict['pred_codons'].append(pred_codons)
        self.val_dict['pred_res'].append(pred_res)
        self.val_dict['codons'].append(codons)
        self.val_dict['seq'].append(batch['seq'])
        self.val_dict['mask'].append(mask)
        self.log('codon_recovery', codon_recovery)
        self.log('res_recovery', res_recovery)

        if batch_idx < self.args.num_foldability_batches:
            atom37s = [atom37_e[mask_e.bool()].cpu().numpy() for atom37_e, mask_e in zip(atom37, mask)]
            pred_seqs = [seq_e[mask_e.bool()] for seq_e, mask_e in zip(pred_res, mask)]
            fold_results = run_foldability(atom37s, pred_seqs, device=self.device)
            self.log('tm_score', np.array(fold_results['tm_score']).mean())
            self.log('rmsd', np.array(fold_results['rmsd']).mean())
        else:
            self.log('tm_score', np.nan)
            self.log('rmsd', np.nan)
        return out


    def on_validation_epoch_end(self):
        max_len = max([seq.shape[1] for seq in self.val_dict['seq']])
        for k in self.val_dict:
            for i in range(len(self.val_dict[k])):
                L = self.val_dict[k][i].shape[1]
                if L < max_len:
                    self.val_dict[k][i] = torch.cat([self.val_dict[k][i], torch.zeros(self.args.batch_size, max_len - L,
                                                                                      *self.val_dict[k][i].shape[2:],
                                                                                      device=self.device)],
                                                    dim=1)
            self.val_dict[k] = torch.cat(self.val_dict[k], dim=0)

        codons = self.val_dict['codons']
        pred_codons = self.val_dict['pred_codons']
        pred_res = self.val_dict['pred_res']
        mask = self.val_dict['mask']
        B, L = codons.shape
        seq = self.val_dict['seq']

        prob_c_given_a = torch.zeros(self.K, len(restype_order_with_x), device=self.device)
        for i in range(self.K):
            for j in range(len(restype_order_with_x)):
                num_aa = ((seq == j) * mask).sum().float()
                if num_aa != 0:
                    prob_c_given_a[i, j] = ((codons == i) * (seq == j) * mask).sum().float() / num_aa

        codons_from_res = torch.stack([torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in s], device=self.device) for s in pred_res])
        codons_from_oracle_res = torch.stack([torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in s], device=self.device) for s in seq])

        naive_codon_recovery = ((codons_from_res == codons).float() * mask).sum() / mask.sum()
        oracle_codon_recovery = ((codons_from_oracle_res == codons).float() * mask).sum() / mask.sum()

        per_aa_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_naive_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_aa_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_oracle_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        for i in range(len(restype_order_with_x)):
            id_mask = (seq == i) * mask
            per_aa_codon_recovery[i] = (codons == pred_codons)[id_mask.bool()].float().sum() / id_mask.sum()
            per_aa_aa_recovery[i] = (seq == pred_res)[id_mask.bool()].float().sum() / id_mask.sum()
            per_aa_naive_codon_recovery[i] = (codons_from_res == codons)[id_mask.bool()].float().sum() / id_mask.sum()
            per_aa_oracle_codon_recovery[i] = (codons_from_oracle_res == codons)[
                                                  id_mask.bool()].float().sum() / id_mask.sum()

        x = np.arange(len(restypes_with_x))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2 - width, per_aa_codon_recovery.cpu(), width, label='Codon Recovery')
        rects2 = ax.bar(x - width / 2, per_aa_naive_codon_recovery.cpu(), width, label='Naive Codon Recovery')
        rects3 = ax.bar(x + width / 2, per_aa_oracle_codon_recovery.cpu(), width, label='Oracle Codon Recovery')
        rects4 = ax.bar(x + width / 2 + width, per_aa_aa_recovery.cpu(), width, label='Amino Acid Recovery')

        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Recovery Rate')
        ax.set_title('Recovery Rates Per Amino Acid')
        ax.set_xticks(x)
        ax.set_xticklabels(restypes_with_x)
        ax.legend()
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f'{os.environ["MODEL_DIR"]}/recovery_rates_per_aa_epoch{self.current_epoch}_iter{self.iter_step}.png')
        if self.args.wandb:
            wandb.log({'recovery_rates_per_aa': wandb.Image(
                f'{os.environ["MODEL_DIR"]}/recovery_rates_per_aa_epoch{self.current_epoch}_iter{self.iter_step}.png')})
        for k in self.val_dict:
            self.val_dict[k] = []
        self.print_log(prefix='val', save=True, extra_logs={'codon_from_res_recovery': naive_codon_recovery.item(),
                                                            'codon_from_oracle_res_recovery':
                                                                oracle_codon_recovery.item()})
    
    def test_step(self, batch, batch_idx):
        self.iter_step += 1
        self.stage = 'test'

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        B, L, _, _ = atom37.shape
        seq = batch['seq']  # (B, L, 21)
        wildtype_codons = batch['wildtype_codons']  # (B, L, 65)
        mut_codons = batch['mut_codons'] # (B, L, 65)
        mut_position = torch.ceil((batch['mut_position'] + 1) / 3) - 1 
        mask_seq = torch.nn.functional.one_hot(mut_position.long(), num_classes=L) # (B, L) 
        B, L, _, _ = atom37.shape

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        chain_M = torch.ones([B, L], device=self.device, dtype=torch.long)

        decoding_order = torch.arange(L).repeat(B, 1) # (B, L)
        decoding_order[mask_seq == 1] = L - 1
        decoding_order[:, L - 1] = mut_position
        
        log_probs = self.model.forward_inference(
            X=bb_pos,
            randn=torch.randn(B, L, device=self.device),
            S=seq if self.args.train_aa else wildtype_codons,
            taxon_id=batch['taxon_id'],
            mask=mask,
            chain_M=chain_M,
            residue_idx=batch['pmpnn_res_idx'],
            chain_encoding_all=batch['pmpnn_chain_encoding'],
            use_input_decoding_order=True, # only for option 2
            decoding_order=decoding_order, # only for option 2
        ) # (B, L, 65)
        mask_logits = mask_seq.unsqueeze(-1).repeat(1, 1, self.K) # (B, L, 65)
        mut_log_probs = log_probs[mask_logits==1].view(B, self.K) # (B, 65)
        self.log('mut_log_prob', mut_log_probs.detach().cpu().numpy())
        self.log('output_seq', torch.argmax(log_probs, dim=-1).tolist())

        train_target = batch['seq'] if self.args.train_aa else batch['wildtype_codons']
        loss = torch.nn.functional.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1),
                                                 reduction='none')  # (B * L)
        loss = loss.view(B, L)
        loss = loss * mask.float()  # (B, L)
        loss = loss.sum() / mask.sum()

        return loss


    
    def on_test_epoch_end(self):
        np.save('test_mut_log_prob', np.concatenate(self._log['test_mut_log_prob'], axis=0))
        with open("test_output_seq.csv", "w") as f:
            wr = csv.writer(f)
            wr.writerows(self._log['test_output_seq'])
