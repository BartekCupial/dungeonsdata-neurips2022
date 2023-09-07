import argparse
import ast
import numbers
import pandas as pd
import wandb
import json

from pathlib import Path

import numpy as np

api = wandb.Api()


def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d


def unfreeze(d):
    if isinstance(d, frozenset):
        return {key: unfreeze(value) for key, value in d}
    elif isinstance(d, tuple):
        return list(value for value in d)
    return d


def create_dataframe(filters):
    runs = api.runs("gmum/nle", filters=filters)
    data = []
    configs = {}
    for run in runs:
        df = {}
        for key, value in ast.literal_eval(run.summary.__repr__()).items():
            if isinstance(value, numbers.Number):
                df[key] = value
            elif isinstance(value, list):
                if key == "returns":
                    key = "return"
                
                df[f"eval/mean_episode_{key}"] = np.mean(value)
                df[f"eval/std_episode_{key}"] = np.std(value)
                df[f"eval/median_episode_{key}"] = np.median(value)

        df["group"] = run.config["group"]
        data.append(df)
        configs[run.config["group"]] = run.config

    return pd.DataFrame(data), configs


def log_group(group, df, config):
    wandb.init(
        project="nle",
        group=group,
        config=config,
        entity="gmum",
        name=f"eval_stitch_{group}",
    )
    df = df[df["group"] == group]
    df = df.sort_values(["_step"])

    for index, row in df.iterrows():
        logs = row.to_dict()
        logs["global/env_train_steps"] = logs["_step"]
        del logs["group"]
        wandb.log(logs, step=int(logs["_step"]))

    wandb.finish()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    return parser.parse_known_args(args=args)[0]


def main(variant):
    filters = variant["filters"]
    df, configs = create_dataframe(filters)
    df = df.sort_values(["_step"])

    groups = df["group"].unique()
    for group in groups:
        config = configs[group]
        config.update(variant)
        log_group(group, df, config)


if __name__ == "__main__":
    args = vars(parse_args())
    with open(args["json"], "r+") as file:
        variant = json.load(file)

    main(variant=variant)
