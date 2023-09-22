from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.eval_save import parse_args as eval_save_parse_args
from hackrl.rollout import parse_args as rollout_parse_args


PARSE_ARGS_DICT = {
    "eval": eval_parse_args,
    "eval_array": eval_array_parse_args,
    "eval_save": eval_save_parse_args,
    "rollout": rollout_parse_args,
}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))
    res.update(config)
    return res


name = globals()["script"][:-3]
# name = "local"

# params for all exps
config = {
    "exp_tags": [name],
    "run_kind": "eval_save",
    "name": name,
    "wandb": True,
    "render": False,
    "device": "cpu",
    "checkpoint_dir": "/path/to/checkpoint/dir",
    "save_ttyrec_every": 1,
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "seed": list(range(5)), # how many processes to spawn
        # "checkpoint_dir": ["/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-KL-T/checkpoint.tar"],
        "checkpoint_dir": ["/home/bartek/Workspace/data/nethack_checkpoints/AMZN/checkpoint_v0"],
        "savedir": ["nle_data"],
        "gameloaddir": [[None] * 10], # num_rollouts
        "use_ray": [False],
        "device": ["cuda"],
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
