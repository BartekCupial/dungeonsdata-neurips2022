import argparse
import shutil
import tempfile
import logging
import os
import timeit
import time
import multiprocessing

from collections import defaultdict
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

from collections import deque
from pathlib import Path

import moolib
import numpy as np
import omegaconf
import torch
import json
import tqdm
import wandb
import pandas as pd

import nle  # noqa: F401
from nle import nethack

import hackrl.core
import hackrl.environment
import hackrl.models
from hackrl.core import nest
import matplotlib.pyplot as plt

from hackrl.eval import load_model_flags_and_step


def go_back(num_lines):
    print("\033[%dA" % num_lines)


def results_to_dict(results):
    returns = results["return"]
    steps = results["steps"]
    blscore = results["blscore"]
    bltime = results["bltime"]

    return {
        "eval/mean_episode_return": np.mean(returns),
        "eval/std_episode_return": np.std(returns),
        "eval/median_episode_return": np.median(returns),
        "eval/mean_episode_steps": np.mean(steps),
        "eval/std_episode_steps": np.std(steps),
        "eval/median_episode_steps": np.median(steps),
        "eval/mean_episode_blscore": np.mean(blscore),
        "eval/std_episode_blscore": np.std(blscore),
        "eval/median_episode_blscore": np.median(blscore),
        "eval/mean_episode_bltime": np.mean(bltime),
        "eval/std_episode_bltime": np.std(bltime),
        "eval/median_episode_bltime": np.median(bltime),
    }


def single_rollout(
    model,
    flags,
    score_target=10000,
    savedir=None,
    save_ttyrec_every=0,
    env="challenge",
    gameloaddir=None,
    render=False,
    render_mode="human",
    print_frames_separately=True,
):
    flags.reward_win = 1000
    flags.reward_lose = 1
    device = flags.device

    env = hackrl.environment.create_env(
        flags, 
        savedir=savedir, 
        save_ttyrec_every=save_ttyrec_every, 
        gameloaddir=gameloaddir,
    )


    action = torch.tensor(0)
    hs = model.initial_state(1)
    hs = nest.map(lambda x: x.to(device), hs)

    obs = env.reset()
    reward = 0.0
    current_reward = torch.tensor(0.0)
    done = False

    bl_score = 0
    bl_time = 0

    start_time = timeit.default_timer()
    timesteps = 0
    while True:
        if render:
            print("-" * 8 + " " * 71)
            print(f"Previous reward: {str(reward):64s}")
            act_str = repr(env.actions[action]) if action is not None else ""
            print(f"Previous action: {str(act_str):64s}")
            print("-" * 8)
            env.render(render_mode)
            print("-" * 8)
            print(obs["blstats"])
            if not print_frames_separately:
                go_back(num_lines=33)

        env_outputs = nest.map(lambda t: torch.from_numpy(t).unsqueeze(0), obs)
        env_outputs["reward"] = torch.tensor(reward)
        env_outputs["done"] = torch.tensor(done)
        
        current_reward += env_outputs["reward"]

        bl_score = env_outputs["blstats"][:, nethack.NLE_BL_SCORE].item()
        bl_time = env_outputs["blstats"][:, nethack.NLE_BL_TIME].item()

        env_outputs["timesteps"] = torch.tensor(timesteps)
        env_outputs["max_scores"] = torch.tensor(score_target)
        env_outputs["mask"] = torch.tensor(True)
        env_outputs["scores"] = current_reward
        env_outputs["prev_action"] = action

        timesteps += 1

        env_outputs = nest.map(lambda x: x.unsqueeze(0).to(flags.device), env_outputs)
        with torch.no_grad():
            outputs, hs = model(env_outputs, hs)
        action = outputs["action"].reshape(-1)

        obs, reward, done, info = env.step(action)

        if done:
            break

    env.close()

    time_delta = timeit.default_timer() - start_time
    sps = timesteps / time_delta
    print(f"Finished after: {timesteps} steps and {time_delta} seconds. SPS: {sps}")

    returns = {
        "return": current_reward.item(),
        "steps": timesteps,
        "blscore": bl_score,
        "bltime": bl_time,
    }
    print(f"Agent got {returns['return']} additional reward in {returns['steps']} timesteps, blscore: {returns['blscore']}, bltime: {returns['bltime']}")

    return returns


# def evaluate_save(path, device, gameloaddir, **kwargs):
#     model, flags, step = load_model_flags_and_step(path, device)

#     if isinstance(gameloaddir, list) and len(gameloaddir) > 1:
#         print(f"Evaluate {len(gameloaddir)} saves.")
#         returns = []
#         for gamedir in tqdm.tqdm(gameloaddir):
#             single_returns = single_rollout(
#                 model=model,
#                 flags=flags,
#                 gameloaddir=gamedir,
#                 **kwargs
#             )
#             returns.append(single_returns)
#     else:
#         print("Evaluate one save.")
#         returns = single_rollout(
#             model=model,
#             flags=flags,
#             gameloaddir=gameloaddir,
#             **kwargs
#         )
#         returns = [returns]

#     return results_to_dict(returns), flags, step

def single_evaluation(path, device, **kwargs):
    model, flags, step = load_model_flags_and_step(path, device)

    returns = single_rollout(
        model=model,
        flags=flags,
        **kwargs
    )

    return returns, flags, step


def multiple_evaluations(path, device, gameloaddir, **kwargs):
    import ray
    ray.init(ignore_reinit_error=True, _temp_dir="/net/ascratch/people/plgbartekcupial/tmp")

    refs = []

    @ray.remote(num_gpus=0)
    def remote_evaluation(gameloaddir=gameloaddir, **kwargs):
        q = Queue()

        def sim():
            q.put(single_evaluation(path, device, gameloaddir=gameloaddir, **kwargs))

        try:
            p = Process(target=sim, daemon=False)
            p.start()
            return q.get()
        finally:
            p.terminate()
            p.join()

    for gamepath in gameloaddir:
        refs.append(remote_evaluation.remote(gameloaddir=gamepath, **kwargs))

    all_res = defaultdict(list)
    count = 0
    for handle in refs:
        ref, refs = ray.wait(refs, num_returns=1, timeout=None)
        single_res = ray.get(ref[0])
        results, flags, step = single_res

        count += 1
        for k, v in results.items():
            all_res[k].append(v)

        text = []
        text.append(f'count                         : {count}')
        print('\n'.join(text) + '\n')
    
    print('DONE!')
    ray.shutdown()

    return all_res, flags, step


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--gameloaddir", type=str, default=None, help="can be a single path or a list of paths")
    parser.add_argument("--results_path", type=Path, default="data.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    parser.add_argument("--env", type=str, default="challenge")
    # render
    parser.add_argument("--render", type=bool, default=False, help="Disables env.render().")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "full", "ansi"],
        help="Render mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "--print_frames_separately",
        "-p",
        type=bool, 
        default=True,
        help="Don't overwrite frames, print them all.",
    )
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--exp_kind", type=str, default="eval")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]

    kwargs = dict(
        path=checkpoint_dir,
        device=variant["device"],
        score_target=variant["score_target"],
        env=variant["env"],
        gameloaddir=variant["gameloaddir"],
        render=variant["render"],
        render_mode=variant["render_mode"],
        print_frames_separately=variant["print_frames_separately"],
    )

    print(f"Evaluating checkpoint {checkpoint_dir}")

    if isinstance(variant["gameloaddir"], list):
        results, flags, step = multiple_evaluations(**kwargs)
    else:
        results, flags, step = single_evaluation(**kwargs)

    results = results_to_dict(results)
    print(json.dumps(results, indent=4))

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

        results["global/env_train_steps"] = step
        wandb.log(results, step=step)

    with open(variant["results_path"], "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    tempdir = tempfile.mkdtemp()
    tempfile.tempdir = tempdir

    try:
        args = vars(parse_args())
        main(variant=args)
    finally:
        logging.info(f"Removing all temporary files in {tempfile.tempdir}")
        shutil.rmtree(tempfile.tempdir)
