import argparse
import shutil
import tempfile
import logging
import concurrent

from pathlib import Path

import numpy as np
import omegaconf
import torch
import wandb

from tqdm.auto import tqdm

from hackrl.core import nest
from hackrl.eval import load_model_flags_and_step
from hackrl.experiment import make_ttyrec_envpool, compute_kickstarting_loss


def log_forgetting(model, flags, forgetting_dataset, n_batches=2000, log_to_wandb=False):
    model.eval()

    global FORGETTING_ENVPOOL, FORGETTING_HIDDEN_STATE
    tp2 = concurrent.futures.ThreadPoolExecutor(max_workers=flags.ttyrec_cpus)
    FORGETTING_HIDDEN_STATE = []
    for _ in range(flags.ttyrec_envpool_size):
        hs = nest.map(
            lambda x: x.to(flags.device),
            model.initial_state(batch_size=flags.ttyrec_batch_size),
        )
        FORGETTING_HIDDEN_STATE.append(hs)
    FORGETTING_ENVPOOL = make_ttyrec_envpool(tp2, forgetting_dataset, flags)

    forgetting_losses = []
    for i in tqdm(range(n_batches)):
        kick_data = FORGETTING_ENVPOOL.result()
        idx = FORGETTING_ENVPOOL.idx
        kick_predictions, FORGETTING_HIDDEN_STATE[idx] = model(
            kick_data, FORGETTING_HIDDEN_STATE[idx]
        )
        FORGETTING_HIDDEN_STATE[idx] = nest.map(
            lambda t: t.detach(), FORGETTING_HIDDEN_STATE[idx]
        )

        forgetting_loss = compute_kickstarting_loss(
            kick_predictions["policy_logits"],
            kick_predictions["kick_policy_logits"],
        )
        forgetting_losses.append(forgetting_loss.item())
        
        # Only call step when you are done with ttyrec_data - it may get overwritten
        FORGETTING_ENVPOOL.step()

        if log_to_wandb:
            wandb.log({"forgetting_losses": forgetting_loss.item()})

    tp2.shutdown()

    forgetting_losses = np.array(forgetting_losses)

    return {
        "eval/mean_forgetting_loss": np.mean(forgetting_losses),
        "eval/std_forgetting_loss": np.std(forgetting_losses),
        "eval/median_forgetting_loss": np.median(forgetting_losses),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--teacher_path", type=Path, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=100_000_000)
    parser.add_argument("--n_batches", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--forgetting_dataset", type=str, default="bc1")
    parser.add_argument("--dbfilename", type=str, default="/ttyrecs/ttyrecs.db")
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--exp_kind", type=str, default="eval")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]

    checkpoint_dir = Path(checkpoint_dir)

    def tryint(i):
        try:
            int(i)
            return True
        except Exception as e:
            return False

    # create dictionary containing checkpoint_step as key and checkpoint_path as value
    checkpoints = list(checkpoint_dir.iterdir())
    checkpoints = list(filter(lambda path: path.name.startswith("checkpoint_"), checkpoints))
    checkpoints = list(filter(lambda path: tryint(path.name.split('_')[1][1:]), checkpoints))
    checkpoints = list(filter(lambda path: int(path.name.split('_')[1][1:]) % variant["checkpoint_step"] == 0, checkpoints))
    checkpoints = sorted(checkpoints, key=lambda path: int(path.name.split('_')[1][1:]))
    checkpoints = {int(path.name.split('_')[1][1:]): path for path in checkpoints}

    # add checkpoint.tar
    if (checkpoint_dir / "checkpoint.tar").exists():
        checkpoint_tar = checkpoint_dir / "checkpoint.tar"
        load_data = torch.load(checkpoint_tar, map_location=torch.device(variant["device"]))
        step = load_data["learner_state"]["global_stats"]["steps_done"]["value"]
        checkpoints[step] = checkpoint_tar

    # sort checkpoints, we need to process them from oldest due to wandb
    for step, checkpoint in sorted(checkpoints.items()):
        print(f"Evaluating checkpoint {checkpoint}")

        model, flags, step = load_model_flags_and_step(checkpoint, variant["teacher_path"], variant["device"])

        if wandb.run is None and log_to_wandb:
            config = omegaconf.OmegaConf.to_container(flags)
            config.update(variant)

            wandb.init(
                project=flags.project,
                config=config,
                group=config["group"],
                entity=flags.entity,
                name=name,
            )

        flags.dbfilename = variant["dbfilename"]
        flags.ttyrec_batch_size = variant["batch_size"]
        results = log_forgetting(model, flags, variant["forgetting_dataset"], variant["n_batches"], log_to_wandb)
        results["global/env_train_steps"] = step

        if log_to_wandb:
            wandb.log(results)


if __name__ == "__main__":
    tempdir = tempfile.mkdtemp()
    tempfile.tempdir = tempdir

    try:
        args = vars(parse_args())
        main(variant=args)
    finally:
        logging.info(f"Removing all temporary files in {tempfile.tempdir}")
        shutil.rmtree(tempfile.tempdir)
