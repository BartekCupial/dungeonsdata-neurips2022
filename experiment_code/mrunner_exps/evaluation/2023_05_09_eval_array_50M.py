from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.rollout import parse_args as rollout_parse_args
from hackrl.utils.pamiko import get_checkpoint_paths

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
    "exp_kind": [name],
    "run_kind": "eval",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)

root_dirs = [
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/03_05-13_31-relaxed_cori"
    ),
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-11_07-elastic_mccarthy"
    ),
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_51-gallant_sammet"
    ),
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_51-youthful_kare"
    ),
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_50-elastic_bhabha"
    ),
    Path(
        "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_50-modest_euclid"
    ),
]
checkpoint_step = 50000000

params_grid = []
for root_dir in root_dirs:
    group_paths = get_checkpoint_paths(root_dir)
    for group_path in group_paths:
        group_path = Path(group_path)
        print(group_path)
        params_grid.append(
            {
                "checkpoint_dir": [str(group_path / f"checkpoint_v{checkpoint_step}")],
            }
        )


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
