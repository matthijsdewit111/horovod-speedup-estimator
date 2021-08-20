#!/bin/bash

for n in {1..32}; do
  sbatch -p normal -n "$n" run_real_speedup.sh
done
