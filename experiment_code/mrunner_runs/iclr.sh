#!/bin/bash

ssh-add


mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/ICLR_baselines/2023_20_09_monk-AA-BC.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/ICLR_baselines/2023_20_09_monk-APPO-AA-CEAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/ICLR_baselines/2023_20_09_monk-APPO-AA-KLAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/ICLR_baselines/2023_20_09_monk-APPO-AA-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/ICLR_baselines/2023_20_09_monk-APPO-T.py