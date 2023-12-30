#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-AA-BC-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-AA-BCEL-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-AA-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO.py