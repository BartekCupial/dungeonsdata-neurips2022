#!/bin/bash


python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo_ks_T.json & \
python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo_bc_T.json & \
python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo_T.json
# python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo_ks.json
# python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo_bc.json
# python -m experiment_code.utils.stitch_evals --json experiment_code/utils/stitch_configs/stitch_monk_appo.json
