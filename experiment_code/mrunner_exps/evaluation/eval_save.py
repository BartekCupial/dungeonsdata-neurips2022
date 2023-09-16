from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.eval_save import parse_args as eval_save_parse_args
from hackrl.rollout import parse_args as rollout_parse_args
from hackrl.utils.pamiko import get_checkpoint_paths

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

# params for all exps
config = {
    "run_kind": "eval_save",
    "name": "eval_save",
    "wandb": False,
    "group": "monk-APPODT",
    "checkpoint_dir": "/path/to/checkpoint/file",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": [
            # "/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-KL-T/checkpoint.tar"
            "/home/bartek/12_05-09_03-nervous_wiles/2023-11-05-monk-appo-aa-bc-t_jisi_2/checkpoint/hackrl/nle/2023_11_05_monk-APPO-AA-BC-T_2_ostentate/checkpoint_v10000000"
        ],
        "env": ["challenge"],
        "wandb": [False],
        "gameloaddir": [
            [
                # "/home/bartek/Workspace/autoascend/saves2/ahudajsetduj",
                # "/home/bartek/Workspace/autoascend/saves/sokoban_hgebprhoavfy",
                # "/home/bartek/Workspace/autoascend/saves2/gvmxflekyghg",
                # "/home/bartek/Workspace/autoascend/saves2/amfzqwtcuwct",
                # "/home/bartek/Workspace/autoascend/saves2/awoekdexqiid",
                # "/home/bartek/Workspace/autoascend/saves2/bckndsihafsr",
                # "/home/bartek/Workspace/autoascend/saves2/bgnsgmmpfiev",
                # None,
                # None,
                # None,
                # None,
                # None,
                # None,
                *list(Path("/home/bartek/Workspace/data/sokoban").iterdir()),
            ]
        ],
        # "gameloaddir": ["/home/bartek/Workspace/autoascend/saves2/ahudajsetduj"],
        "render": [False],
        "device": ["cpu"],
        "use_ray": [False],
        "print_frames_separately": [False],
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
