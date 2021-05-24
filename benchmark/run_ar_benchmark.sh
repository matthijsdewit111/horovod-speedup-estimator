#!/bin/bash

for n in {1..32}; do
  sbatch -p normal -n "$n" run_ar_"$1".sh
done
