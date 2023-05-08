import argparse
import ast
import numbers
import pandas as pd
import wandb
import json

from pathlib import Path

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
    for run in runs:
        df = {}
        for key, value in ast.literal_eval(run.summary.__repr__()).items():
            if key.startswith("_"):
                continue

            if isinstance(value, numbers.Number):
                df[key] = value

        df["step"] = run.config["step"]
        df["group"] = run.config["group"]
        data.append(df)

    return pd.DataFrame(data)


def log_group(group, df, variant):
    variant["group"] = group

    wandb.init(
        project="nle",
        group="monk-APPO",
        config=variant,
        entity="gmum",
        name=f"eval_stitch_{group}",
    )
    df = df[df["group"] == group]
    df = df.sort_values(["step"])

    for index, row in df.iterrows():
        logs = row.to_dict()
        del logs["group"]
        wandb.log(logs, step=logs["step"])

    wandb.finish()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    return parser.parse_known_args(args=args)[0]


def main(variant):
    filters = {"$and": [variant["filters"]]}
    df = create_dataframe(filters)
    df = df.sort_values(["step"])

    groups = df["group"].unique()
    for group in groups:
        log_group(group, df, variant)


if __name__ == "__main__":
    args = vars(parse_args())
    with open(args["json"], "r+") as file:
        variant = json.load(file)

    main(variant=variant)
