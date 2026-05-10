#!/usr/bin/env bash
set -euo pipefail

python main_train.py gnn_config_mos2.ini > out_train_mos2.txt
mv -f convergence.png convergence_mos2.png

python main_eval.py gnn_config_mos2.ini > out_eval_mos2.txt
