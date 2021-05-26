#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
python ../hse/tool.py some_module MySmallNet -is 100
