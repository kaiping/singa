#!/bin/bash
if [ $# -ne 3 ]
then
  echo "must provide 3 arguments, rid file, hdfs folder, local folder"
  exit
fi
mkdir -p $3
imgs=(`cat $1 |cut -d ' ' -f 1`)
num=0
files=''
for img in ${imgs[@]}
do
  imgpath=$2/$img
  localpath=$3/$img
  if [ ! -e $localpath ]
  then
    echo "hadoop fs -get $imgpath $3"
    #hadoop fs -get $imgpath $3
  fi
done
