from random_word import RandomWords

from mrunner.helpers.specification_helper import (
    create_experiments_helper,
    get_combinations,
)

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AMZN-KLAA",
    "num_actor_cpus": 20,
    "total_steps": 100_000_000,
    "group": "monk-APPO-AMZN-KLAA",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": True,
    "kickstarting_loss_bc": 0.01,
    "use_kickstarting_bc": True,
    "kickstarting_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/mrunner_scratch/nle/16_10-15_35-peaceful_khorana/2023-16-10-monk-appo-t-baseline_3g50_0/checkpoint/hackrl/nle/2023_16_10_monk-APPO-T_baseline_0_cavitate/checkpoint_v100000000",
    "model_checkpoint_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/mrunner_scratch/nle/16_10-15_35-peaceful_khorana/2023-16-10-monk-appo-t-baseline_3g50_0/checkpoint/hackrl/nle/2023_16_10_monk-APPO-T_baseline_0_cavitate/checkpoint_v100000000",
    "dataset": "autoascend",
    "use_resnet": True,
    "model": "NetHackNetTtyrec",
    "unfreeze_actor_steps": 50_000_000,
    "sampling_type": "softmax",
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "adam_learning_rate": [0.00006, 0.00002, 0.000006, 0.000002, 0.0000006, 0.0000002],
        "kickstarting_loss_bc": [0.5, 0.2, 0.1, 0.05],
        "kickstarting_decay_bc": [1.0, 0.9999, 0.99996, 0.99998],
        "appo_clip_policy": [0.2],
        "baseline_cost": [1.0],
        "reward_clip": [True],
        "actor_batch_size": [128],
        "batch_size": [64],
        "virtual_batch_size": [64],
        "ttyrec_batch_size": [256],
        "unroll_length": [32],
        "ttyrec_unroll_length": [32],
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    r = RandomWords().get_random_word()
    cfg["group"] = [f"{name}_{e}_{r}"]
    final_grid.append(dict(cfg))


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=final_grid,
)
