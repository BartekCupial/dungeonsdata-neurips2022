from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.rollout import parse_args as rollout_parse_args


PARSE_ARGS_DICT = {
    "eval": eval_parse_args,
    "rollout": rollout_parse_args,
}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))
    res.update(config)
    return res


name = globals()["script"][:-3]


# params for all exps
config = {
    "exp_tags": [name],
    "run_kind": "rollout",
    "name": "rollout",
    "num_actor_cpus": 1,
    "num_actor_batches": 1,
    "rollouts": 1000,
    "batch_size": 1,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/file",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "seed": list(range(10)),
        "checkpoint_dir": [
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/AMZN/checkpoint_v0"
        ],
        "savedir": ["/nle/nld-amzn/nld_data"],
        "device": ["cpu"],
    },
]


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_eval.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
)
