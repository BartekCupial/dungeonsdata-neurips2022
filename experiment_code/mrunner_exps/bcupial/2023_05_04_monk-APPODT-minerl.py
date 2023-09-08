from pathlib import Path

from mrunner.helpers.specification_helper import (
    create_experiments_helper,
    get_combinations,
)
from random_word import RandomWords


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPODT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPODT",
    "character": "mon-hum-neu-mal",
    "model": "DecisionTransformer",
    "use_timesteps": True,
    "return_to_go": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "grad_norm_clipping": 4,
    "n_layer": 6,
    "n_head": 8,
    "hidden_dim": 512,
    "warmup_steps": 10000,
    "weight_decay": 0.01,
    "use_kickstarting": True,
    "kickstarting_path": "/tscratch/nle/30_04-06_07-relaxed_spence/2023-04-30-pretrain-no-returns_usal_1/checkpoint/hackrl/nle/2023_04_30_pretrain_no_returns_1/checkpoint.tar",
    "use_checkpoint_actor": True,
    "model_checkpoint_path": "/tscratch/nle/30_04-06_07-relaxed_spence/2023-04-30-pretrain-no-returns_usal_1/checkpoint/hackrl/nle/2023_04_30_pretrain_no_returns_1/checkpoint.tar",
    "use_returns": False,
}


# params different between exps
params_grid = [
    {
        "num_actor_cpus": [20],
        "warmup_steps": [1],
        "adam_learning_rate": [0.00002],
        # "baseline_learning_rate": [0.001],
        "weight_decay": [0.04],
        "batch_size": [64],
        "actor_batch_size": [128],
        "virtual_batch_size": [128, 256, 512, 1024],
        "unroll_length": [128],
        "discounting": [0.999],
        "lambda_gae": [0.95],
        "appo_clip_policy": [0.02],
        "grad_norm_clipping": [5],
        "kickstarting_loss": [0.2],
        "kickstarting_decay": [0.9995],
        "entropy_cost": [0.0],
        "ppg_sleep": [True],
        "ppg_sleep_sample_reuse": [6],
        "ppg_sleep_cycles": [2],
        "ppg_baseline_cost": [1.0],
        "ppg_kl_loss": [1.0],
        "n_layer": [3],
        "n_head": [4],
        "seed": [0],  # reduced number of seeds
    },
    {
        "num_actor_cpus": [20],
        "warmup_steps": [1],
        "adam_learning_rate": [0.00002],
        # "baseline_learning_rate": [0.001],
        "weight_decay": [0.04],
        "batch_size": [128],
        "actor_batch_size": [256],
        "virtual_batch_size": [128, 256, 512, 1024],
        "unroll_length": [32],
        "discounting": [0.999],
        "lambda_gae": [0.95],
        "appo_clip_policy": [0.02],
        "grad_norm_clipping": [5],
        "kickstarting_loss": [0.2],
        "kickstarting_decay": [0.9995],
        "entropy_cost": [0.0],
        "ppg_sleep": [True],
        "ppg_sleep_sample_reuse": [6],
        "ppg_sleep_cycles": [2],
        "ppg_baseline_cost": [1.0],
        "ppg_kl_loss": [1.0],
        "n_layer": [3],
        "n_head": [4],
        "seed": [0],  # reduced number of seeds
    },
    # {
    #     "warmup_steps": [10000],
    #     "adam_learning_rate": [0.0001],
    #     # "baseline_learning_rate": [0.001],
    #     "weight_decay": [0.01],
    #     "batch_size": [128],
    #     "actor_batch_size": [256],
    #     "virtual_batch_size": [128],
    #     "unroll_length": [32],
    #     "ttyrec_unroll_length": [32],
    #     "discounting": [0.999],
    #     "lambda_gae": [0.95],
    #     "appo_clip_policy": [0.01],
    #     "grad_norm_clipping": [4],
    #     "entropy_cost": [0.0],
    #     "ppg_sleep": [True],
    #     "ppg_sleep_sample_reuse": [6],
    #     "ppg_sleep_cycles": [2],
    #     "ppg_baseline_cost": [1.0],
    #     "ppg_kl_loss": [1.0],
    #     "n_layer": [3],
    #     "n_head": [4],
    #     "seed": [0], # reduced number of seeds
    # },
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
