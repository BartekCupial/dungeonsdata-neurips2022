from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.rollout import parse_args as rollout_parse_args
from hackrl.eval_forgetting import parse_args as eval_forgetting_parse_args
from hackrl.utils.pamiko import get_checkpoint_paths

PARSE_ARGS_DICT = {
    "eval": eval_parse_args,
    "eval_array": eval_array_parse_args,
    "eval_forgetting": eval_forgetting_parse_args,
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
    "run_kind": "eval_forgetting",
    "name": "eval_forgetting",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "checkpoint_step": 100_000_000,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/dir",
    "dbfilename": "/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db",
    "forgetting_dataset": "autoascend",
    "batch_size": 128,
    "n_batches": 2**10,
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "batch_size": [4],
        "checkpoint_dir": ["/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-KS"],
        "teacher_path": ["/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-BC/checkpoint.tar"],
        "checkpoint_step": [100_000_000],
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