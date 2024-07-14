from argparse import ArgumentParser
import subprocess, os

def parse_train_args():
    parser = ArgumentParser()
    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--val_check_interval", type=int, default=1.0)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--num_foldability_batches", type=int, default=0)
    group.add_argument("--train_aa", action='store_true')

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    group.add_argument("--workdir", type=str, default="workdir")

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--lr", type=float, default=1e-3)

    ## Training data
    group = parser.add_argument_group("Training data settings")
    group.add_argument('--afdb_dir', type=str, default='afdb_small')
    group.add_argument('--data_csv', type=str, default='afdb_small/dataset_small.csv')
    group.add_argument('--max_seq_len', type=int, default=750)
    group.add_argument('--num_taxon_ids', type=int, default=100)
    group.add_argument("--high_plddt", action="store_true")


    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hidden_dim', type=int, default=128)
    group.add_argument("--taxon_condition", action="store_true")
    group.add_argument('--num_encoder_layers', type=int, default=3)
    group.add_argument('--num_decoder_layers', type=int, default=3)
    group.add_argument('--num_neighbors', type=int, default=48)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--backbone_noise', type=float, default=0.02)

    ## Inference settings
    group = parser.add_argument_group("Inference settings")
    group.add_argument('--sampling_temp', type=float, default=0.1)

    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join(args.workdir, args.run_name)
    os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    os.environ["TORCH_HOME"] = "/data/rsg/nlp/hstark/torch_cache" if os.getcwd() != '/Users/hstark/projects/codon' else "/Users/hstark/projects/torch_cache"

    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            print("There were uncommited changes. Commit before running")
            exit()
    args.commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    return args
