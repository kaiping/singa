#!/bin/bash
set -e
set -x
for nw in  6
do
  echo $nw
  for threshold in 5000000
    do
      echo $threshold
      mpirun -np 9 -hostfile examples/imagenet12/hostfile -nooversubscribe \
      ./lapis_test.bin -system_conf=examples/imagenet12/system.conf \
      -model_conf=examples/imagenet12/model.conf --v=3 --data_dir=tmp \
      --table_buffer=20 --block_size=10 --workers=$nw --threshold=$threshold --iterations=5
      echo $nw >>log.comm
      echo $threshold>>log.comm
      cat log_variance_*>>log.comm
      cat throughput_* >>log.comm
    done
done
