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
    "exp_tags": [name],
    "run_kind": "eval_save",
    "name": name,
    "wandb": True,
    "render": False,
    "device": "cpu",
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)

appo = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_02-cranky_fermi")
appo_t = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_02-priceless_meninsky")
appo_aa_ks_t = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_03-modest_goldstine")
appo_aa_klbc_t = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_03-pedantic_hawking")
appo_aa_ceaa_t = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/12_05-09_03-nervous_wiles")
appo_aa_klaa_t = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/old_scratch/nle/16_09-08_00-hardcore_jackson")

save_root_path = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/gamesavedir")

cpaths = []
for ckpt_paths in [
    get_checkpoint_paths(appo),
    get_checkpoint_paths(appo_t),
    get_checkpoint_paths(appo_aa_ks_t),
    get_checkpoint_paths(appo_aa_klbc_t),
    # get_checkpoint_paths(appo_aa_ceaa_t), # uncomment if needed
    [path for path in get_checkpoint_paths(appo_aa_klaa_t) if any(f'T_{i}' in path for i in range(3, 25, 5))], 
]:
    paths = []
    for path in ckpt_paths:
        for i in range(0, 100_000_000, 10_000_000):
            paths.append(Path(path) / f"checkpoint_v{i}")
        for i in range(100_000_000, 2_000_000_000, 100_000_000):
            paths.append(Path(path) / f"checkpoint_v{i}")
    cpaths.append(paths)

folders = []
folders.append(None)
folders.append("sokoban")
for i in range(2, 10):
    folder = f"saves{i}"
    folders.append(folder)

expected_saves = 200
save_paths = []
for folder in folders:
    if folder is None:
        saves = [None] * expected_saves
        save_paths.append(saves)
    else:
        saves = get_save_paths(save_root_path / folder, host="ares.cyfronet.pl")
        if len(saves) < expected_saves:
            saves = saves * ((expected_saves // len(saves)) + 1)
        saves = [save_root_path / folder / s for e, s in enumerate(saves) if e < expected_saves]
        save_paths.append(saves)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": paths,
        "gameloaddir": [spaths],
        "name": [f"{name}_{folder}"],
    }
    for paths in cpaths
    for spaths, folder in zip(save_paths, folders)
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
