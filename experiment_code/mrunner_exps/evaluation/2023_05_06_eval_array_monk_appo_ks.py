from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.rollout import parse_args as rollout_parse_args


PARSE_ARGS_DICT = {"eval": eval_parse_args, "eval_array": eval_array_parse_args, "rollout": rollout_parse_args}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))
    res.update(config)
    return res

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "run_kind": "eval",
    "name": "eval",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)

root_dir = Path("/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_51-gallant_sammet")
group_pattern = "monk-APPO-AA-KS_"
checkpoint_step = 100_000_000

# params different between exps
params_grid = []
for i in range(5):
    seed_dir = root_dir / f"monk-appo-aa-ks_49dv_{i}/checkpoint/hackrl/nle/monk-APPO-AA-KS_{i}_leapers"
    params_grid.append(
        {
            "checkpoint_dir": [str(seed_dir / "checkpoint.tar")],
            "group": [f"monk-APPO-AA-KS_{i}_leapers"],
            "step": [checkpoint_step * 20],
        } 
    )

    for ckpt in range(1, 20):
        step = ckpt * checkpoint_step
        chpt_i = seed_dir / f"checkpoint_v{step}"
        params_grid.append(
            {
                "checkpoint_dir": [str(chpt_i)],
                "group": [f"monk-APPO-AA-KS_{i}_leapers"],
                "step": [step],
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