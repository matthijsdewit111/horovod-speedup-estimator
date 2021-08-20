#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
python ../../hse/tool.py slater_jastrow_wrapper SlaterJastrowWrapperH2 -is 6 -sa h2_prediction.png
python ../../hse/tool.py slater_jastrow_wrapper SlaterJastrowWrapperCH4 -is 30 -sa ch4_prediction.png
