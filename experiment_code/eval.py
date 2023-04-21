import os
import argparse

from pathlib import Path

import moolib
import numpy as np
import omegaconf
import torch
import tqdm

import hackrl.core
import hackrl.environment
import hackrl.models
from hackrl.core import nest

ENVS = None


def load_model_and_flags(path, device):
    load_data = torch.load(path, map_location=torch.device(device))
    flags = omegaconf.OmegaConf.create(load_data["flags"])
    flags.device = device
    model = hackrl.models.create_model(flags, device)
    if flags.use_kickstarting:
        print("Kickstarting")
        t_data = torch.load(flags.kickstarting_path)
        t_flags = omegaconf.OmegaConf.create(t_data["flags"])
        teacher = hackrl.models.create_model(t_flags, flags.device)
        # teacher.load_state_dict(load_data["learner_state"]["model"])
        model = hackrl.models.KickStarter(
            model, teacher, run_teacher_hs=flags.run_teacher_hs
        )
    model.load_state_dict(load_data["learner_state"]["model"])
    return model, flags


def generate_envpool_rollouts(
    model, 
    flags, 
    rollouts = 1024, 
    batch_size=512,
    num_actor_cpus = 20,
    num_actor_batches = 2,
    pbar_idx=0, 
    score_target=10000
):
    global ENVS
    # NB: We do NOT want to generate the first N rollouts from B batch
    # of envs since this will bias short episodes.
    # Instead lets just allocate some episodes to each env
    split = rollouts // (batch_size * num_actor_batches)
    flags.batch_size = batch_size
    device = flags.device

    ENVS = moolib.EnvPool(
        lambda: hackrl.environment.create_env(flags),
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
    results = [None, None]
    grand_pbar = tqdm.tqdm(position=0, leave=True)
    pbar = tqdm.tqdm(
        total=batch_size * num_actor_batches * split, position=pbar_idx + 1, leave=True
    )

    action = torch.zeros((num_actor_batches, batch_size)).long().to(device)
    hs = [model.initial_state(batch_size) for _ in range(num_actor_batches)]
    hs = nest.map(lambda x: x.to(device), hs)

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
    return len(returns), np.mean(returns), np.std(returns), np.median(returns)


def find_checkpoint(path, min_steps, device):
    # versions = []
    # flags = omegaconf.OmegaConf.create(
    #     torch.load(path / "checkpoint.tar", map_location=torch.device(device))["flags"]
    # )
    # for f in os.listdir(path):
    #     ff = str(f)
    #     if ff.startswith("checkpoint_v"):
    #         v = int(ff.replace("checkpoint_v", "").replace(".tar", ""))
    #         versions.append(v)
    # desired_v = min_steps / (flags.batch_size * flags.unroll_length)
    # for v in sorted(versions):
    #     allowed_paths = [
    #         "/checkpoint/ehambro/20220531/meek-binturong",
    #         "/checkpoint/ehambro/20220531/adamant-viper",
    #         "/checkpoint/ehambro/20220531/hallowed-bat",
    #         "/checkpoint/ehambro/20220531/celadon-llama",
    #     ]
    #     if v > desired_v or v > 122070.325 or (path in allowed_paths and v > 18000):
    #         return f"{path}/checkpoint_v{v}.tar"
    # return f"{path}/checkpoint_v{v}.tar"

    print("Returning checkpoint.tar")
    return f"{path}/checkpoint.tar"


def evaluate_folder(name, path, min_steps, device, pbar_idx, output_dir, **kwargs):
    p_ckpt = find_checkpoint(path, min_steps, device)
    if not p_ckpt:
        print(f"Not yet: {name} - {path}")
        return (
            name,
            path,
            -1,
            -1,
            -1,
            -1,
        )
    print(f"{pbar_idx} {name} Using: {p_ckpt}")
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    model, flags = load_model_and_flags(p_ckpt, device)
    returns = generate_envpool_rollouts(
        model=model, 
        flags=flags, 
        pbar_idx=pbar_idx, 
        **kwargs,
    )
    return (name, p_ckpt) + returns


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--min_steps", type=int, default=1_000_000_000)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score_target", type=float)
    return parser.parse_known_args(args=args)[0]


def main(
    name: str, 
    checkpoint_dir: Path, 
    output_dir: Path, 
    rollouts: int,
    min_steps: int, 
    device: str, 
    **kwargs
):
    results = (name, checkpoint_dir, -1, -1, -1)

    results = evaluate_folder( 
        name=name, 
        path=checkpoint_dir, 
        min_steps=min_steps, 
        device=device, 
        pbar_idx=0, 
        output_dir=output_dir,
        rollouts=rollouts,
        **kwargs
    )

    print(
        f"{results[0]} Done {results[1]}  Mean {results[3]} ± {results[4]}  | Median {results[5]}"
    )
    if results[2] > -2:
        data = (
            min_steps,
            rollouts,
        ) + results
        os.makedirs(f"{output_dir}/{name}/", exist_ok=True)
        with open(f"{output_dir}/{name}/{checkpoint_dir.split('/')[-1]}.txt", "w") as f:
            f.write(",".join(str(d) for d in data) + "\n")
    print("done")


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)
