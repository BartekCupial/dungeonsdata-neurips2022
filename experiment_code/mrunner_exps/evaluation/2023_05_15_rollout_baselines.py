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
    "rollouts": 1024,
    "batch_size": 1,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/file",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": [
            "/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_03-pedantic_hawking/2023-11-05-monk-appo-aa-kl-t_nmjx_0/checkpoint/hackrl/nle/2023_11_05_monk-APPO-AA-KL-T_0_stillroom/checkpoint.tar"
        ],
        "savedir": ["/nle/nld-appo-aa-kl-t/nld_data"],
    },
    {
        "checkpoint_dir": [
            "/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_03-modest_goldstine/2023-11-05-monk-appo-aa-ks-t_k8zh_0/checkpoint/hackrl/nle/2023_11_05_monk-APPO-AA-KS-T_0_hexapodal/checkpoint.tar"
        ],
        "savedir": ["/nle/nld-appo-aa-ks-t/nld_data"],
    },
    {
        "checkpoint_dir": [
            "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/03_05-13_31-relaxed_cori/monk-appo_4z1a_4/checkpoint/hackrl/nle/monk-APPO_4/checkpoint.tar"
        ],
        "savedir": ["/nle/nld-appo/nld_data"],
    },
    {
        "checkpoint_dir": [
            "/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/12_05-09_02-priceless_meninsky/2023-11-05-monk-appo-t_qphj_2/checkpoint/hackrl/nle/2023_11_05_monk-APPO-T_2_arsenetted/checkpoint.tar"
        ],
        "savedir": ["/nle/nld-appo-t/nld_data"],
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
