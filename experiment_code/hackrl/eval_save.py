import argparse
import shutil
import tempfile
import logging
import os
import timeit
import time
import multiprocessing

from collections import defaultdict, namedtuple
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
from hackrl.utils.tasks_rewards import (
    GoldScore,
    StaircasePetScore,
    ScoutScore,
    StaircaseScore,
    EatingScore,
    SokobanfillpitScore,
    SokobansolvedlevelsScore,
)


BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


def go_back(num_lines):
    print("\033[%dA" % num_lines)


def results_to_dict(results):
    return {f"eval/mean_episode_{k}": np.mean(v) for k, v in results.items()}


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
    model.eval()

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

    starting_blstats = BLStats(*obs["blstats"])

    gold_score = GoldScore()
    eating_score = EatingScore()
    scout_score = ScoutScore()
    staircase_score = StaircaseScore()
    staircasepet_score = StaircasePetScore()
    sokobanfillpit_score = SokobanfillpitScore()
    sokobansolvedlevel_score = SokobansolvedlevelsScore()

    start_time = timeit.default_timer()
    timesteps = 0
    while True:
        blstats = BLStats(*obs["blstats"])

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

        last_observation = tuple(a.copy() for a in env.last_observation)
        obs, reward, done, info = env.step(action)
        observation = tuple(a.copy() for a in env.last_observation)
        end_status = info["end_status"]

        gold_score.reward(env.env, last_observation, observation, end_status)
        eating_score.reward(env.env, last_observation, observation, end_status)
        scout_score.reward(env.env, last_observation, observation, end_status)
        staircase_score.reward(env.env, last_observation, observation, end_status)
        staircasepet_score.reward(env.env, last_observation, observation, end_status)
        sokobanfillpit_score.reward(env.env, last_observation, observation, end_status)
        sokobansolvedlevel_score.reward(env.env, last_observation, observation, end_status)

        if done:
            break

        yield 1

    env.close()

    time_delta = timeit.default_timer() - start_time
    sps = timesteps / time_delta

    returns = {
        "score": blstats.score,
        "agent_score": blstats.score - starting_blstats.score,
        "turns": blstats.time,
        "agent_turns": blstats.time - starting_blstats.time,
        "steps": timesteps,
        "dlvl": blstats.depth,
        "max_hitpoints": blstats.max_hitpoints,
        "max_energy": blstats.max_energy,
        "armor_class": blstats.armor_class,
        "experience_level": blstats.experience_level,
        "experience_points": blstats.experience_points,
        "time": time_delta,
        "sps": sps,
        "gold_score": gold_score.score,
        "eating_score": eating_score.score,
        "scout_score": scout_score.score,
        "staircase_score": staircase_score.score,
        "staircasepet_score": staircasepet_score.score,
        "sokobanfillpit_score": sokobanfillpit_score.score,
        "sokobansolvedlevel_score": sokobansolvedlevel_score.score,
    }

    yield returns


def single_evaluation(path, device, **kwargs):
    model, flags, step = load_model_flags_and_step(path, device)

    start_time = time.time()
    pbar = tqdm.tqdm()
    for returns in single_rollout(model=model, flags=flags, **kwargs):
        pbar.update()
    wall_time = time.time() - start_time

    return returns, flags, step, 1, wall_time


def multiple_evaluations(path, device, gameloaddir, **kwargs):
    model, flags, step = load_model_flags_and_step(path, device)

    start_time = time.time()
    count = 0
    all_res = defaultdict(list)

    pbar = tqdm.tqdm()
    master_pbar = tqdm.tqdm(total=len(gameloaddir))

    for gamepath in gameloaddir:
        if gamepath:
            master_pbar.set_description(f"Current save: {Path(gamepath).name}")
        try: 
            for returns in single_rollout(model=model, flags=flags, gameloaddir=gamepath, **kwargs):
                pbar.update()
            count += 1

            for k, v in returns.items():
                all_res[k].append(v)
        except Exception as e:
            print(e)
        master_pbar.update(1)
    wall_time = time.time() - start_time

    return all_res, flags, step, count, wall_time


def ray_evaluations(path, device, gameloaddir, **kwargs):
    import ray
    ray.init(ignore_reinit_error=True)
    # ray.init(ignore_reinit_error=True, _temp_dir="/net/ascratch/people/plgbartekcupial/tmp")

    assert device == "cpu", "eay_evaluations with cuda is slower than multiple_evaluations"

    model, flags, step = load_model_flags_and_step(path, device)
    model_object_id = ray.put(model)

    @ray.remote
    class SharedQueueActor:
        def __init__(self):
            self.queue = []

        def put(self, item):
            self.queue.append(item)

        def get(self):
            if self.queue:
                return self.queue.pop(0)
            return None

    @ray.remote(num_gpus=0)
    def remote_evaluation(queue_actor, gameloaddir=gameloaddir):
        model = ray.get(model_object_id)

        try:
            for i in single_rollout(model=model, flags=flags, gameloaddir=gameloaddir, **kwargs):
                queue_actor.put.remote(i)
        except Exception as e:
            queue_actor.put.remote({})

    shared_queue = SharedQueueActor.remote()

    futures = [remote_evaluation.remote(shared_queue, gameloaddir=gamepath) for gamepath in gameloaddir]

    pbar = tqdm.tqdm()
    master_pbar = tqdm.tqdm(total=len(gameloaddir))

    start_time = time.time()
    count = 0
    all_res = defaultdict(list)

    while True: 
        result = ray.get(shared_queue.get.remote())

        if result == 1:
            pbar.update()
        elif isinstance(result, dict):
            count += 1

            for k, v in result.items():
                all_res[k].append(v)

            master_pbar.update()
    
        if count == len(gameloaddir):
            break

    wall_time = time.time() - start_time
    return all_res, flags, step, count, wall_time


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument(
        "--gameloaddir",
        type=str,
        default=None,
        help="can be a single path or a list of paths",
    )
    parser.add_argument("--results_path", type=Path, default="data.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_ray", type=bool, default=False)
    parser.add_argument("--score_target", type=float, default=5000)
    parser.add_argument("--env", type=str, default="challenge")
    # render
    parser.add_argument(
        "--render", type=bool, default=False, help="Disables env.render()."
    )
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
    parser.add_argument("--save_ttyrec_every", type=int, default=0)
    parser.add_argument("--savedir", type=Path, default=None)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--exp_kind", type=str, default="eval")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]
    gameloaddir = variant["gameloaddir"]

    kwargs = dict(
        path=checkpoint_dir,
        device=variant["device"],
        score_target=variant["score_target"],
        env=variant["env"],
        gameloaddir=gameloaddir,
        render=variant["render"],
        render_mode=variant["render_mode"],
        print_frames_separately=variant["print_frames_separately"],
        savedir=variant["savedir"],
        save_ttyrec_every=variant["save_ttyrec_every"],
    )

    print(f"Gameloaddir :{gameloaddir}")
    print(f"Evaluating checkpoint {checkpoint_dir}")

    if isinstance(gameloaddir, list):
        if variant["use_ray"]:
            data, flags, step, count, wall_time = ray_evaluations(**kwargs)
        else:
            data, flags, step, count, wall_time = multiple_evaluations(**kwargs)
    else:
        data, flags, step, count, wall_time = single_evaluation(**kwargs)

    data = pd.DataFrame(data)
    data.to_csv("eval.csv")
    results = results_to_dict(data)
    results["eval/count"] = count
    results["eval/wall_time"] = wall_time
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
