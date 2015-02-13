#!/bin/bash

hostfile="examples/mnist/hostfile"
hosts=(`cat $hostfile |cut -d ' ' -f 1`)
count=0
for i in ${hosts[@]}
do
  if [ $1 == "start" ]
  then
    cmd="ssh $i \"cd ~/program/singa;\
      ./build/singa -procsID=$count \
      -hostfile=examples/mnist/hostfile \
      -cluster_conf=examples/mnist/cluster.conf \
      -model_conf=examples/mnist/mlp.conf\""
    echo $cmd
    #cmd
  fi
  if [ $1 == "stop" ]
  then
    echo "ssh $i \"kill ./singa\""
    ssh $i "killall -q singa"
  fi
  count=$(($count+1))
  if [ $count -eq $2 ]
  then
    exit
  fi
done


