#!/bin/bash
folder="/data1/wangwei/lapis/validation/*"
#folder="/tmp/lapis.bin.INFO"
hostfile="examples/imagenet12/hostfile"
if [ $# -eq 0 ]
then
  echo "must provide argument, [chmod,ps, ssh,ls,cat, cp, delete, create or reset] + hostfile"
  exit
fi
hosts=(`cat $hostfile |cut -d ' ' -f 1`)

if [ $1 == "cp" ]
then
  for i in {0..18}
  do
      echo ${hosts[i]} ${hosts[i+19]}
      ssh ${hosts[i]} "scp -r wangwei@${hosts[i+19]}:/data1/wangwei/lapis/train-lmdb /data1/wangwei/lapis/train-lmdb" &
  done
fi

for i in ${hosts[@]}
do
  if [ $1 == "ps" ]
  then
    echo "ssh $i"
    ssh $i "ps ax|pgrep $2"
  fi
  if [ $1 == "chmod" ]
  then
    echo "ssh $i"
    ssh $i "chmod 644 $folder"
  fi
  if [ $1 == "cat" ]
  then
    echo "ssh $i"
    ssh $i "cat $folder"
  fi
  if [ $1 == "ls" ]
  then
    echo "ssh $i"
    ssh $i "ls -l $folder "
  fi
  if [ $1 == "ssh" ]
  then
    echo "ssh $i"
    ssh $i "exit"
  fi
  if [ $1 == "delete" -o $1 == "reset" ]
  then
    echo "delete $folder on $i"
    ssh $i "rm -rf $folder"
  fi
  if [ $1 == "create" -o $1 == "reset" ]
  then
    echo "create $folder on $i"
    ssh $i "mkdir -p $folder"
  fi
  if [ $1 == "kill" ]
  then
    echo "kill $2 on $i"
    ssh $i "ps ax|pgrep $2|xargs kill"
  fi
done
