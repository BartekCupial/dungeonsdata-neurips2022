from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations
from random_word import RandomWords


# name = globals()["script"][:-3]
name = "local"

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-AA-TrXL",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 256,
    "batch_size": 128,
    "ttyrec_batch_size": 512,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    'group': name,
    "character": "mon-hum-neu-mal",
    "model": "TransfoXL",
    "return_to_go": True,
    "use_returns": False,
    "use_timesteps": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "n_layer": 4,
    "n_head": 16,
    "grad_norm_clipping": 4,
    "hidden_dim": 512,
    "warmup_steps": 10000,
    "weight_decay": 0.04,
    "linear_time_embeddings": True,
}


# params different between exps
params_grid = [
    {
        "seed": [0],
        "ttyrec_unroll_length": [context_size],
        "unroll_length": [context_size],
        "mem_len": [memory_size],
        "hidden_dim": [128, 256, 512, 1024, 2048],
        "actor_batch_size": [batch_size * 2],
        "batch_size": [batch_size],
        "virtual_batch_size": [batch_size * virtual_batch_size],
        "ttyrec_batch_size": [batch_size * 2],
    }
    for context_size, memory_size in zip([32, 64, 128], [64, 128, 256])
    for batch_size in [40, 64, 128]
    for virtual_batch_size in [1, 2, 4, 8]
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