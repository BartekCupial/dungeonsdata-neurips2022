import argparse
import os
import tempfile
import shutil

from pathlib import Path

import numpy as np
import wandb
import torch

from hackrl.eval import evaluate_folder

os.environ["MOOLIB_ALLOW_FORK"] = "1"


def delete_temp_files(directory):
    i = 0
    for path in Path(directory).iterdir():
        if path.name.startswith("nle"):
            try:
                shutil.rmtree(path)
                i += 1
            except Exception as e:
                print(f"Error deleting dir: {path}\n{str(e)}")
    print(f"Deleted {i} dirs")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation_array")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--checkpoint_step", type=int, default=100_000_000)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--group", type=str)
    parser.add_argument("--exp_tags", type=str)
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]

    kwargs = dict(
        device=variant["device"], 
        rollouts=variant["rollouts"],
        batch_size=variant["batch_size"],
        num_actor_cpus=variant["num_actor_cpus"],
        num_actor_batches=variant["num_actor_batches"],
        score_target=variant["score_target"],
        log_to_wandb=False,
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
    checkpoint_tar = checkpoint_dir / "checkpoint.tar"
    load_data = torch.load(checkpoint_tar, map_location=torch.device(variant["device"]))
    step = load_data["learner_state"]["global_stats"]["steps_done"]["value"]
    checkpoints[step] = checkpoint_tar

    tempdir = tempfile.TemporaryDirectory()

    # sort checkpoints, we need to process them from oldest due to wandb
    summary_results = dict()
    for e, (step, checkpoint) in enumerate(sorted(checkpoints.items())):
        print(f"Evaluating checkpoint {step}")

        results = evaluate_folder(
            pbar_idx=e, 
            path=checkpoint, 
            **kwargs,
        )

        returns = results["returns"]
        steps = results["steps"]
        scores = results["scores"]
        times = results["times"]
        summary_results[step] = {
            "eval/mean_episode_return": np.mean(returns),
            "eval/std_episode_return": np.std(returns),
            "eval/median_episode_return": np.median(returns),
            "eval/mean_episode_steps": np.mean(steps),
            "eval/std_episode_steps": np.std(steps),
            "eval/median_episode_steps": np.median(steps),

            "eval/mean_episode_scores": np.mean(scores),
            "eval/std_episode_scores": np.std(scores),
            "eval/median_episode_scores": np.median(scores),
            "eval/mean_episode_times": np.mean(times),
            "eval/std_episode_times": np.std(times),
            "eval/median_episode_times": np.median(times),
        }
        delete_temp_files(Path(tempdir.name).parent)
        
    if log_to_wandb:
        wandb.init(
            project="nle",
            config=variant,
            group=variant["group"],
            entity="gmum",
            name=name,
        )
        
        for step, results in summary_results.items():
            wandb.log(results, step=step)


if __name__ == "__main__":
    args = vars(parse_args())
    main(variant=vars(args))
