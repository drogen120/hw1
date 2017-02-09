#!/bin/bash
mkdir data
mkdir log
mkdir model
mkdir figures
bash get_data.sh
bash run_BC.sh
bash run_BC3.sh
bash run_dagger.sh
python gen_figure1.py HalfCheetah
python gen_figure1.py Hopper
python gen_figure2.py HalfCheetah
python gen_figure2.py Hopper
