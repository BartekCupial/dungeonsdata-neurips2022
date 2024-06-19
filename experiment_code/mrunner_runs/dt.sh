#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_dungeons_1gpu run mrunner_exps/DT/monk-AA-DT.py 