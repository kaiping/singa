!/bin/bash
folder="/data1/wangwei/lapis"
if [ $# -eq 0 ]
  then
    echo "must provide argument, delete, create or reset"
    exit
fi
for i in `cat hostfile |cut -d ' ' -f 1`
do
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
done
