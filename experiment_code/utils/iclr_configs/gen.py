import json

from pathlib import Path


exp_tag = "2023_09_22_eval_full2"
methods = [
    "monk-APPO",
    # "monk-APPO",
    # "monk-APPO-AA-KLAA",
    # "monk-APPO-AA-KL",
    # "monk-APPO-AA-KS",
    # "monk-APPO-AA-BC",
]

folder_names = [
    "appo",
    # "appo_t",
    # "appo_klaa_t",
    # "appo_klbc_t",
    # "appo_ks_t",
    # "appo_ceaa_t",
]

folders = []
folders.append(None)
folders.append("sokoban")
for i in range(2, 10):
    folder = f"saves{i}"
    folders.append(folder)


for method, folder_name in zip(methods, folder_names):
    for folder in folders:
        file_name = "saves1" if folder is None else folder
        path = Path("/home/bartek/Workspace/CW/dungeonsdata-neurips2022/experiment_code/utils/iclr_configs") / folder_name / f"{file_name}.json"
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as file:

            d = {
                "exp_kind": "eval_stitch",
                "filters": {
                    "$or": [
                        {
                            "config.exp_kind": "eval",
                            "config.name": f'{exp_tag}_{folder}',
                            "config.exp_tags": [exp_tag],
                            "config.exp_point": method,
                        }
                    ]
                }
            }

            json.dump(d, file, indent=4)
