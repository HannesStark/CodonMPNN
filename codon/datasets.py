import os

import numpy as np
import pandas as pd
import torch
from openfold.np.residue_constants import (
    restype_order_with_x,
    restypes_with_x,
    aatype_to_str_sequence,
)
from torch.utils.data import default_collate
from tqdm import tqdm

from codon.utils.codon_const import (
    unk_codon,
    codon_order,
    codon_to_res,
    unk_codon_index,
)
from codon.utils.data_utils import parse_mmcif
from codon.utils.pmpnn import get_weird_pmpnn_stuff

from typing import Union
from pathlib import Path
import sqlite3


class AFDBDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        df = pd.read_csv(args.data_csv)
        df = df[(df["seq"].str.len() / 3) < args.max_seq_len]
        if args.high_plddt:
            df_high = pd.read_csv(
                "/data/rsg/nlp/ujp/codon/data/bigquery_plddt_geq90.csv"
            )
            df = df[
                df["af_id"].isin(df_high["entryId"].apply(lambda x: x.split("-")[1]))
            ]

        self.df = df
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        df_row = self.df.iloc[idx]
        af_id = df_row["af_id"]
        taxon_id = df_row[f"{self.args.num_taxon_ids}_grouping"]
        dna_seq = df_row["seq"]
        cif_path = os.path.join(self.args.afdb_dir, df_row["fpath"])

        if len(dna_seq) % 3 != 0:
            print(
                f"Skipping protein because len dna_seq {len(dna_seq)} % 3 != 0 for {af_id} at {cif_path}"
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))

        prots = parse_mmcif(cif_path)
        prot = prots[0]

        # convert dna seq to codon seq
        codon_seq = [dna_seq[i : i + 3] for i in range(0, len(dna_seq), 3)]
        # remove the last codon which corresponds to the stop codon and is not present in the protein sequence
        codon_seq = codon_seq[:-1]
        codons = [
            codon_order.get(a, unk_codon_index) for a in codon_seq
        ]  # convert to indices
        codons = np.array(codons, dtype=np.int32)

        if len(codons) != len(prot["seq"]):
            print(
                f'Skipping protein because len codons {len(codons)} != len protein {len(prot["seq"])} for {af_id} at '
                f"{cif_path}"
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))

        seq_str = aatype_to_str_sequence(prot["seq"])
        codon_str = "".join(
            [codon_to_res.get(codon, codon_to_res[unk_codon]) for codon in codon_seq]
        )
        if seq_str != codon_str:
            mask = np.array(list(seq_str)) == np.array(list(codon_str))
            prot["seq"] = prot["seq"][mask]
            prot["atom37"] = prot["atom37"][mask]
            prot["atom_mask"] = prot["atom_mask"][mask]
            codons = codons[mask]
            if mask.sum() < 0.8 * len(codon_str):
                print(
                    f"Skipping protein because less than 80% of the codon and prot sequence matched up for {af_id} at "
                    f"{cif_path}"
                )
                return self.__getitem__(np.random.randint(len(self.df) - 1))

        # only keep residues that have all 3 backbone atoms
        bb_mask = prot["atom_mask"][:, :3].all(-1)
        prot["atom37"] = torch.from_numpy(prot["atom37"][bb_mask]).float()
        prot["atom_mask"] = torch.from_numpy(prot["atom_mask"][bb_mask]).long()
        prot["seq"] = torch.from_numpy(prot["seq"][bb_mask]).long()
        codons = torch.from_numpy(codons[bb_mask]).long()

        residue_idx, chain_encoding = get_weird_pmpnn_stuff(
            chain_idx=torch.zeros(len(codons), dtype=torch.long)
        )

        prot.update(
            {
                "codons": codons,
                "taxon_id": taxon_id,
                "af_id": af_id,
                "pmpnn_res_idx": residue_idx,
                "pmpnn_chain_encoding": chain_encoding,
            }
        )
        return prot
    
class Shen2022Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        df = pd.read_csv(args.data_csv)
        df = df[(df["wildtype_seq"].str.len() / 3) < args.max_seq_len]
        df = df[(df["mut_seq"].str.len() / 3) < args.max_seq_len]
        # if args.high_plddt:
        #     df_high = pd.read_csv(
        #         "/data/rsg/nlp/ujp/codon/data/bigquery_plddt_geq90.csv"
        #     )
        #     df = df[
        #         df["af_id"].isin(df_high["entryId"].apply(lambda x: x.split("-")[1]))
        #     ]

        self.df = df
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        df_row = self.df.iloc[idx]
        # af_id = df_row["af_id"]
        taxon_id = df_row["taxon_id"]
        wildtype_seq = df_row["wildtype_seq"]
        mut_seq = df_row["mut_seq"]
        # dna_seq = df_row["seq"]
        gene = df_row["gene"] 
        mut_position = df_row["position"]
        cif_path = os.path.join(os.path.join("/data/scratch/diaoc/codon/data/shen2022_structs", gene), "pdb/best.cif")

        if len(wildtype_seq) % 3 != 0:
            print(
                f"Skipping protein because len wildtype_seq {len(wildtype_seq)} % 3 != 0"
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))
        if len(mut_seq) % 3 != 0:
            print(
                f"Skipping protein because len mut_seq {len(mut_seq)} % 3 != 0"
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))

        prots = parse_mmcif(cif_path)
        prot = prots[0]

        # convert wildtype dna seq to codon seq
        wildtype_codon_seq = [wildtype_seq[i : i + 3] for i in range(0, len(wildtype_seq), 3)]
        # remove the last codon which corresponds to the stop codon and is not present in the protein sequence
        wildtype_codon_seq = wildtype_codon_seq[:-1]
        wildtype_codons = [
            codon_order.get(a, unk_codon_index) for a in wildtype_codon_seq
        ]  # convert to indices
        wildtype_codons = np.array(wildtype_codons, dtype=np.int32)

         # convert mutant dna seq to codon seq
        mut_codon_seq = [mut_seq[i : i + 3] for i in range(0, len(mut_seq), 3)]
        # remove the last codon which corresponds to the stop codon and is not present in the protein sequence
        mut_codon_seq = mut_codon_seq[:-1]
        mut_codons = [
            codon_order.get(a, unk_codon_index) for a in mut_codon_seq
        ]  # convert to indices
        mut_codons = np.array(mut_codons, dtype=np.int32)

        if len(wildtype_codons) != len(prot["seq"]):
            print(
                f'Skipping protein because len wildtype_codons {len(wildtype_codons)} != len protein {len(prot["seq"])}'
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))
        if len(mut_codons) != len(prot["seq"]):
            print(
                f'Skipping protein because len mut_codons {len(mut_codons)} != len protein {len(prot["seq"])}'
            )
            return self.__getitem__(np.random.randint(len(self.df) - 1))

        seq_str = aatype_to_str_sequence(prot["seq"])
        wildtype_codon_str = "".join(
            [codon_to_res.get(codon, codon_to_res[unk_codon]) for codon in wildtype_codon_seq]
        )
        mut_codon_str = "".join(
            [codon_to_res.get(codon, codon_to_res[unk_codon]) for codon in mut_codon_seq]
        )
        if seq_str != wildtype_codon_str:
            mask = np.array(list(seq_str)) == np.array(list(wildtype_codon_str))
            prot["seq"] = prot["seq"][mask]
            prot["atom37"] = prot["atom37"][mask]
            prot["atom_mask"] = prot["atom_mask"][mask]
            wildtype_codons = wildtype_codons[mask]
            if mask.sum() < 0.8 * len(wildtype_codon_str):
                print(
                    f"Skipping protein because less than 80% of the codon and prot sequence matched up"
                )
                return self.__getitem__(np.random.randint(len(self.df) - 1))
        if seq_str != mut_codon_str:
            mask = np.array(list(seq_str)) == np.array(list(mut_codon_str))
            prot["seq"] = prot["seq"][mask]
            prot["atom37"] = prot["atom37"][mask]
            prot["atom_mask"] = prot["atom_mask"][mask]
            mut_codons = mut_codons[mask]
            if mask.sum() < 0.8 * len(mut_codon_str):
                print(
                    f"Skipping protein because less than 80% of the codon and prot sequence matched up"
                )
                return self.__getitem__(np.random.randint(len(self.df) - 1))

        # only keep residues that have all 3 backbone atoms
        bb_mask = prot["atom_mask"][:, :3].all(-1)
        prot["atom37"] = torch.from_numpy(prot["atom37"][bb_mask]).float()
        prot["atom_mask"] = torch.from_numpy(prot["atom_mask"][bb_mask]).long()
        prot["seq"] = torch.from_numpy(prot["seq"][bb_mask]).long()
        wildtype_codons = torch.from_numpy(wildtype_codons[bb_mask]).long()
        mut_codons = torch.from_numpy(mut_codons[bb_mask]).long()

        residue_idx, chain_encoding = get_weird_pmpnn_stuff(
            chain_idx=torch.zeros(len(wildtype_codons), dtype=torch.long)
        )

        prot.update(
            {
                "wildtype_codons": wildtype_codons,
                "mut_codons": mut_codons,
                "mut_position": mut_position,
                "taxon_id": taxon_id,
                "pmpnn_res_idx": residue_idx,
                "pmpnn_chain_encoding": chain_encoding,
            }
        )
        return prot

def multi_seq_collate(batch):
    seq_len_keys = [
        "atom37",
        "seq",
        "atom_mask",
        "wildtype_codons",
        "mut_codons",
        "pmpnn_res_idx",
        "pmpnn_chain_encoding",
    ]
    max_L = max([len(v) for v in [item[seq_len_keys[0]] for item in batch]])

    seq_len_batch = {}
    for key in seq_len_keys:
        elems = [item[key] for item in batch]
        assert max([len(elem) for elem in elems]) == max_L
        mask = torch.zeros((len(elems), max_L), dtype=torch.int16)
        elem_tensor = []
        for i, elem in enumerate(elems):
            L = len(elem)
            elem = torch.cat(
                [elem, torch.zeros(max_L - L, *elem.shape[1:], dtype=elem.dtype)], dim=0
            )
            elem_tensor.append(elem)
            mask[i, :L] = 1
        seq_len_batch[key] = torch.stack(elem_tensor, dim=0)
        seq_len_batch["mask"] = mask

    # remove all seq_len_keys from batch and put it through default collate
    for item in batch:
        for key in seq_len_keys:
            del item[key]
    batch = default_collate(batch)

    batch.update(seq_len_batch)
    return batch

def seq_collate(batch):
    # batch is a list. It should be a list of dictionaries.
    # This collate pads all elements in the seq_len_keys to maximum length and puts all other keys through
    # default_collate.
    seq_len_keys = [
        "atom37",
        "seq",
        "atom_mask",
        "codons",
        "pmpnn_res_idx",
        "pmpnn_chain_encoding",
    ]
    max_L = max([len(v) for v in [item[seq_len_keys[0]] for item in batch]])

    seq_len_batch = {}
    for key in seq_len_keys:
        elems = [item[key] for item in batch]
        assert max([len(elem) for elem in elems]) == max_L
        mask = torch.zeros((len(elems), max_L), dtype=torch.int16)
        elem_tensor = []
        for i, elem in enumerate(elems):
            L = len(elem)
            elem = torch.cat(
                [elem, torch.zeros(max_L - L, *elem.shape[1:], dtype=elem.dtype)], dim=0
            )
            elem_tensor.append(elem)
            mask[i, :L] = 1
        seq_len_batch[key] = torch.stack(elem_tensor, dim=0)
        seq_len_batch["mask"] = mask

    # remove all seq_len_keys from batch and put it through default collate
    for item in batch:
        for key in seq_len_keys:
            del item[key]
    batch = default_collate(batch)

    batch.update(seq_len_batch)
    return batch


class CodonSqliteDataset(torch.utils.data.Dataset):
    """
    Example usage:

    with open("/data/rsg/nlp/ujp/codon/data/cds/uniref/ids.tsv", "r") as inf:
        next(inf) # skip header
        ids = [tuple(line.strip().split("\t")) for line in inf]

    # subset ids to make splits

    d = CodonSqliteDataset("/data/rsg/nlp/ujp/codon/data/cds/uniref/uniref.db", ids)

    # d[idx] -> (nt, aa, tax_id)

    """

    def __init__(self, db_path: Union[str, Path], ids=[]):
        self.db_path = db_path
        self.ids = ids
        self.conn = sqlite3.connect(db_path, isolation_level="DEFERRED")
        self.cursor = self.conn.cursor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        (emblcds_id, uniprot_id) = self.ids[idx]
        query = f"SELECT nt, aa, tax_id FROM dataset WHERE emblcds = '{emblcds_id}' AND uniprot = '{uniprot_id}';"
        self.cursor.execute(query)
        nt, aa, tax_id = self.cursor.fetchone()
        return nt, aa, tax_id
