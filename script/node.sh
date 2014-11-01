#!/bin/bash
folder="/data1/wangwei/lapis/local_cp"
#folder="/tmp/lapis.bin.INFO"
hostfile="examples/imagenet12/rack2"
if [ $# -eq 0 ]
then
  echo "must provide argument, [chmod, ssh,ls,cat, delete, create or reset] + hostfile"
  exit
fi
for i in `cat $hostfile |cut -d ' ' -f 1`
do
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
    ssh $i "ls $folder |wc -l"
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
    echo "kill $3 on $i"
    ssh $i "ps aux|grep $3 |cut -d ' ' -f 3 |xargs kill"
  fi
done
