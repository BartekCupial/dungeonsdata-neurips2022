from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-AA-BC",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 256,
    "batch_size": 128,
    "ttyrec_batch_size": 512,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    'group': "monk-AA-BC",
    "character": "mon-hum-neu-mal",

    # watcher arguments
    "eval_watcher": True,
    "eval_rollouts": 1024,
    "eval_batch_size": 256,
    "eval_checkpoint_step": 100_000_000,
    "eval_max_step": 2_000_000_000,
    "eval_device": "cuda:0",
}


# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    cfg["group"] = [f"{name}_{e}"]
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
