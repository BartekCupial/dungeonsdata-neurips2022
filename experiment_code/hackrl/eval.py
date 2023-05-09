import argparse

from collections import deque
from pathlib import Path

import moolib
import numpy as np
import omegaconf
import torch
import tqdm
import wandb
import pandas as pd

from nle import nethack

import hackrl.core
import hackrl.environment
import hackrl.models
from hackrl.utils.temp_files import delete_temp_files
from hackrl.core import nest
import matplotlib.pyplot as plt

ENVS = None


def load_model_flags_and_step(path, device):
    load_data = torch.load(path, map_location=torch.device(device))
    flags = omegaconf.OmegaConf.create(load_data["flags"])
    flags.device = device
    model = hackrl.models.create_model(flags, device)
    step = load_data["learner_state"]["global_stats"]["steps_done"]["value"]

    if flags.use_kickstarting:
        print("Kickstarting")
        # remove teacher weights
        student_params = dict(filter(lambda x: x[0].startswith("student"), load_data["learner_state"]["model"].items()))
        # modify keys
        student_params = dict(map(lambda x: (x[0].removeprefix("student."), x[1]), student_params.items()))
        model.load_state_dict(student_params)
        return model, flags, step

    model.load_state_dict(load_data["learner_state"]["model"])
    return model, flags, step


def generate_envpool_rollouts(
    model, 
    flags, 
    rollouts=1024, 
    batch_size=512,
    num_actor_cpus = 20,
    num_actor_batches = 2,
    pbar_idx=0, 
    score_target=10000,
    savedir=None,
    save_ttyrec_every=0,
    log_to_wandb=False,
):
    global ENVS
    # NB: We do NOT want to generate the first N rollouts from B batch
    # of envs since this will bias short episodes.
    # Instead lets just allocate some episodes to each env
    split = rollouts // (batch_size * num_actor_batches)
    flags.batch_size = batch_size
    device = flags.device

    ENVS = moolib.EnvPool(
        lambda: hackrl.environment.create_env(flags, savedir=savedir, save_ttyrec_every=save_ttyrec_every),
        num_processes=num_actor_cpus,
        batch_size=batch_size,
        num_batches=num_actor_batches,
    )

    rollouts_left = (
        torch.ones(
            (
                num_actor_batches,
                batch_size,
            )
        )
        .long()
        .to(device)
        * split
    )
    current_reward = torch.zeros(
        (
            num_actor_batches,
            batch_size,
        )
    ).to(device)
    timesteps = torch.zeros(
        (
            num_actor_batches,
            batch_size,
        )
    ).to(device)

    returns = []
    scores = []
    times = []
    lens = []
    results = [None, None]
    grand_pbar = tqdm.tqdm(position=0, leave=True)
    pbar = tqdm.tqdm(
        total=batch_size * num_actor_batches * split, position=pbar_idx + 1, leave=True
    )

    action = torch.zeros((num_actor_batches, batch_size)).long().to(device)
    hs = [model.initial_state(batch_size) for _ in range(num_actor_batches)]
    hs = nest.map(lambda x: x.to(device), hs)

    bl_scores = [deque(maxlen=2), deque(maxlen=2)]
    bl_times = [deque(maxlen=2), deque(maxlen=2)]

    totals = torch.sum(rollouts_left).item()
    subtotals = [torch.sum(rollouts_left[i]).item() for i in range(num_actor_batches)]
    while totals > 0:
        grand_pbar.update(1)
        for i in range(num_actor_batches):
            if subtotals[i] == 0:
                continue
            if results[i] is None:
                results[i] = ENVS.step(i, action[i])
            outputs = results[i].result()

            env_outputs = nest.map(lambda t: t.to(flags.device, copy=True), outputs)
            env_outputs["prev_action"] = action[i]
            current_reward += env_outputs["reward"]

            bl_scores[i].append(env_outputs["blstats"][:, nethack.NLE_BL_SCORE])
            bl_times[i].append(env_outputs["blstats"][:, nethack.NLE_BL_TIME])

            env_outputs["timesteps"] = timesteps[i]
            env_outputs["max_scores"] = (torch.ones_like(env_outputs["timesteps"]) * score_target).float()
            env_outputs["mask"] = torch.ones_like(env_outputs["timesteps"]).to(torch.bool)
            env_outputs["scores"] = current_reward[i]

            done_and_valid = env_outputs["done"].int() * rollouts_left[i].bool().int()
            finished = torch.sum(done_and_valid).item()
            totals -= finished
            subtotals[i] -= finished

            for j in np.argwhere(done_and_valid.cpu().numpy()):
                returns.append(current_reward[i][j[0]].item())
                lens.append(int(env_outputs["timesteps"][j[0]]))
                scores.append(bl_scores[i][-2][j[0]].item())
                times.append(bl_times[i][-2][j[0]].item())
                # if log_to_wandb:
                #     wandb.log(
                #         {
                #             "episode_return": returns[-1],
                #             "episode_len": lens[-1],
                #             "episode_score": scores[-1],
                #             "episode_time": times[-1],
                #         },
                #         step=lens[-1]
                #     )

            current_reward[i] *= 1 - env_outputs["done"].int()
            timesteps[i] += 1
            timesteps[i] *= 1 - env_outputs["done"].int()
            rollouts_left[i] -= done_and_valid
            if finished:
                pbar.update(finished)

            env_outputs = nest.map(lambda x: x.unsqueeze(0), env_outputs)
            with torch.no_grad():
                outputs, hs[i] = model(env_outputs, hs[i])
            action[i] = outputs["action"].reshape(-1)
            results[i] = ENVS.step(i, action[i])

    results = {
        "returns": returns,
        "steps": lens,
        "scores": scores,
        "times": times,
    }
    return results


def evaluate_folder(path, device, **kwargs):
    model, flags, step = load_model_flags_and_step(path, device)
    returns = generate_envpool_rollouts(
        model=model, 
        flags=flags, 
        **kwargs,
    )
    return returns, flags, step


def log(results, step=None):
    returns = results["returns"]
    steps = results["steps"]
    scores = results["scores"]
    times = results["times"]

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    ax1.scatter(steps, returns)
    ax2.scatter(times, scores)
    wandb.log({"scatter_plot": wandb.Image(fig)})

    df = pd.DataFrame({"returns": returns, "steps": steps, "scores": scores, "times": times})
    df.to_csv("rollout_stats.csv", index=None)
    
    table = wandb.Table(dataframe=df)
    wandb.log({"frame": table})
    data = {
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
    wandb.log(data, step=step)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--checkpoint_dir", type=Path)
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
        num_actor_cpus=variant["num_actor_cpus"],
        num_actor_batches=variant["num_actor_batches"],
        score_target=variant["score_target"],
        log_to_wandb=log_to_wandb,
    )

    print(f"Evaluating checkpoint {checkpoint_dir}")

    results, flags, step = evaluate_folder(
        pbar_idx=0, 
        path=checkpoint_dir, 
        **kwargs
    )

    config = omegaconf.OmegaConf.to_container(flags)
    config.update(variant)

    if log_to_wandb:
        wandb.init(
            project="nle",
            config=config,
            group=config["group"],
            entity="gmum",
            name=name,
        )
        log(results, step)

    delete_temp_files()


if __name__ == "__main__":
    args = vars(parse_args())
    main(variant=vars(args))