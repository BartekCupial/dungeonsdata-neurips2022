from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.rollout import parse_args as rollout_parse_args
from hackrl.utils.pamiko import get_checkpoint_paths

PARSE_ARGS_DICT = {
    "eval": eval_parse_args,
    "eval_array": eval_array_parse_args,
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
    "run_kind": "eval",
    "name": "eval",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "checkpoint_step": 100_000_000,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)


ckpt_paths = [
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_02-priceless_meninsky/2023-11-05-monk-appo-t_qphj_0/checkpoint/hackrl/nle/2023_11_05_monk-APPO-T_0_veratrine"),
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_02-priceless_meninsky/2023-11-05-monk-appo-t_qphj_1/checkpoint/hackrl/nle/2023_11_05_monk-APPO-T_1_polyzoarial"),
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_02-priceless_meninsky/2023-11-05-monk-appo-t_qphj_3/checkpoint/hackrl/nle/2023_11_05_monk-APPO-T_3_maizes"),
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_02-priceless_meninsky/2023-11-05-monk-appo-t_qphj_4/checkpoint/hackrl/nle/2023_11_05_monk-APPO-T_4_sandpiper"),
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_03-pedantic_hawking/2023-11-05-monk-appo-aa-kl-t_nmjx_0/checkpoint/hackrl/nle/2023_11_05_monk-APPO-AA-KL-T_0_stillroom"),
    Path("/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_03-modest_goldstine/2023-11-05-monk-appo-aa-ks-t_k8zh_2/checkpoint/hackrl/nle/2023_11_05_monk-APPO-AA-KS-T_2_postglenoid"),
]

paths = []
for path in ckpt_paths:
    for i in range(0, 2_000_000_000, 100_000_000):
        paths.append(Path(path) / f"checkpoint_v{i}")
    paths.append(Path(path) / "checkpoint.tar")


# params different between exps
params_grid = [
    {"checkpoint_dir": paths},
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