from random_word import RandomWords

from mrunner.helpers.specification_helper import (
    create_experiments_helper,
    get_combinations,
)


name = globals()["script"][:-3]

# params for all exps
config = {
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AA-KL",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-AA-KL",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
    "ttyrec_batch_size": 256,
    "kickstarting_loss_bc": 0.5,
    "use_kickstarting_bc": True,
}


# params different between exps
params_grid = [
    {
        "exp_tags": [f"{name}-4"],
        "seed": list(range(2, 3)),
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "kickstarting_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip4/checkpoint.tar"
        ],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip4/checkpoint.tar"
        ],
        "omitted_dlvls": [4],
    },
    {
        "exp_tags": [f"{name}-3"],
        "seed": list(range(3)),
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "kickstarting_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip3/checkpoint.tar"
        ],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip3/checkpoint.tar"
        ],
        "omitted_dlvls": [3],
    },
    {
        "exp_tags": [f"{name}-2"],
        "seed": list(range(3)),
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "kickstarting_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip2/checkpoint.tar"
        ],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip2/checkpoint.tar"
        ],
        "omitted_dlvls": [2],
    },
    {
        "exp_tags": [f"{name}-1"],
        "seed": list(range(3)),
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "kickstarting_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip1/checkpoint.tar"
        ],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip1/checkpoint.tar"
        ],
        "omitted_dlvls": [1],
    },
    {
        "exp_tags": [f"{name}-0"],
        "seed": list(range(3)),
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "kickstarting_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip0/checkpoint.tar"
        ],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/mrunner_scratch/checkpoints_nle/skip0/checkpoint.tar"
        ],
        "omitted_dlvls": [0],
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
    exclude_git_files=False,
)
