from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.eval_save import parse_args as eval_save_parse_args
from hackrl.rollout import parse_args as rollout_parse_args
from hackrl.utils.pamiko import get_checkpoint_paths, get_save_paths

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
    "run_kind": "eval_save",
    "name": "eval_save",
    "checkpoint_step": 100_000_000,
    "wandb": True,
    "render": False,
    "device": "cpu",
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)


aa_bc = Path(
    "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_03-condescending_turing"
)
appo_t = Path(
    "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_02-priceless_meninsky"
)
appo_aa_kl_t = Path(
    "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_03-pedantic_hawking"
)

save_root_path = Path("/net/ascratch/people/plgbartekcupial")

cpaths = []
for ckpt_paths in [
    get_checkpoint_paths(aa_bc),
    get_checkpoint_paths(appo_t),
    get_checkpoint_paths(appo_aa_kl_t),
]:
    paths = []
    for path in ckpt_paths:
        for i in range(0, 2_000_000_000, 100_000_000):
            paths.append(Path(path) / f"checkpoint_v{i}")
        paths.append(Path(path) / "checkpoint.tar")
    cpaths.append(paths)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": paths,
        "gameloaddir": [[None] * 1000],
        "name": [f"save_1"],
    }
    for paths in cpaths
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
