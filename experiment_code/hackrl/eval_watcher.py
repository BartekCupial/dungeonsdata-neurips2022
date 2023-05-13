import time
import argparse
import re

from pathlib import Path
from functools import partial

import moolib
import wandb
import omegaconf

import hackrl.environment

from hackrl.eval import evaluate_model, load_model_flags_and_step

ENVS = None


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--checkpoint_step", type=int, default=100_000_000)
    parser.add_argument("--max_step", type=int, default=2_000_000_000)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--score_target", type=float, default=5000)

    return parser.parse_known_args(args=args)[0]


def dummy_log():
    if wandb.run is not None:
        wandb.log({"test/dummy": time.time()})


def scan_checkpoints(checkpoint_dir, processed, checkpoint_step):
    pattern = r"checkpoint_v(\d+)"
    match_function = partial(match_pattern, pattern=pattern, step=checkpoint_step)

    def tryint(i):
        try:
            int(i)
            return True
        except Exception as e:
            return False

    files = list(checkpoint_dir.iterdir())
    files = list(filter(lambda p: p not in processed, files))
    files = list(filter(match_function, files))
    files = list(filter(lambda p: tryint(p.name.split('_')[1][1:]), files))
    files = sorted(files, key=lambda path: int(path.name.split('_')[1][1:]))

    if len(files) == 0:
        return None
    else:
        return files[0]


def match_pattern(file, pattern, step):
    rmatch = re.search(pattern, file.name)
    if rmatch:
        checkpoint_number = int(rmatch.group(1))
        if checkpoint_number % step == 0:
            return True
    return False


def main(variant):
    global ENVS
   
    print("Watcher listening...")
    try: 
        processed = []
        while True:
            time.sleep(0.25)
            
            checkpoint_path = scan_checkpoints(variant["checkpoint_dir"], processed, variant["checkpoint_step"])

            dummy_log()

            if checkpoint_path:
                processed.append(checkpoint_path)

                model, flags, step = load_model_flags_and_step(checkpoint_path, variant["device"])

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
                        project="nle",
                        config=config,
                        group=config["group"],
                        entity="gmum",
                        name="eval_watcher",
                    )

                kwargs = dict(
                    device=variant["device"],
                    rollouts=variant["rollouts"],
                    batch_size=variant["batch_size"],
                    num_actor_batches=variant["num_actor_batches"],
                    score_target=variant["score_target"],
                )

                print(f"Evaluating checkpoint {checkpoint_path}")
                results, _ = evaluate_model(ENVS, model, action=None, **kwargs)
                results["global/env_train_steps"] = step

                wandb.log(results)

                if int(checkpoint_path.name.split('_')[1][1:]) == variant["max_step"]:
                    print("Max step reached, training finished. Shutdown.")
                    break

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = vars(parse_args())
    main(variant=args)