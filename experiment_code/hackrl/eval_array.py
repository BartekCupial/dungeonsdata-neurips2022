import argparse
import shutil
import tempfile
import logging

from pathlib import Path

import moolib
import omegaconf
import torch
import wandb

import hackrl.core
import hackrl.environment
import hackrl.models

from hackrl.eval import load_model_flags_and_step, evaluate_model


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--checkpoint_step", type=int, default=100_000_000)
    parser.add_argument("--results_path", type=Path, default="data.json")
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--exp_kind", type=str, default="eval")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]

    kwargs = dict(
        device=variant["device"],
        rollouts=variant["rollouts"],
        batch_size=variant["batch_size"],
        num_actor_batches=variant["num_actor_batches"],
        score_target=variant["score_target"],
        log_to_wandb=log_to_wandb,
    )

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

    ENVS = None

    # sort checkpoints, we need to process them from oldest due to wandb
    for step, checkpoint in sorted(checkpoints.items()):
        print(f"Evaluating checkpoint {checkpoint}")

        model, flags, step = load_model_flags_and_step(checkpoint, variant["device"])

        if ENVS is None:
            ENVS = moolib.EnvPool(
                lambda: hackrl.environment.create_env(flags),
                num_processes=variant["num_actor_cpus"],
                batch_size=variant["batch_size"],
                num_batches=variant["num_actor_batches"],
            )

        if wandb.run is None:
            config = omegaconf.OmegaConf.to_container(flags)
            config.update(variant)

            wandb.init(
                project=flags.project,
                config=config,
                group=config["group"],
                entity=flags.entity,
                name=name,
            )

        results, _ = evaluate_model(ENVS, model, action=None, **kwargs)
        results["global/env_train_steps"] = step

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
