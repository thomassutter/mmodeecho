#!/bin/bash

# command line parameters
# =======================
seed=(13 25 32 47 54)

num_epochs=90

# resources
mem_tot=20480
t=04:00
n=1
ngpu=1
mem=$((mem_tot / n))

for S in "${seed[@]}"; do
  # submit job
  bsub -n $n -W $t -R "rusage[mem=${mem}, ngpus_excl_p=${ngpu}]" -R "select[gpu_model0==GeForceGTX1080]" \
  "python echonet video --seed $S --num_epochs ${num_epochs}"
done
