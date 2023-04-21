from pathlib import Path
import numpy as np
from mrunner.helpers.specification_helper import create_experiments_helper
from hackrl.eval import parse_args

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_prefix": [name],
    "name": "eval_monk-AA-DT-40k-newembeds",
    # "checkpoint_dir": "/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-DT-40k-newembeds",
    "checkpoint_dir": "/checkpoint/hackrl/nle/monk-AA-DT-40k-newembeds",
    "min_steps": 100000000,
    "output_dir": "DT_results",
    "rollouts": 2048,
    "batch_size": 512,
}

args = vars(parse_args())
args.update(config)

# params different between exps
params_grid = {
    "score_target": list(np.logspace(2.0, 5.0, num=10)),
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_eval.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=args,
    params_grid=params_grid,
)
