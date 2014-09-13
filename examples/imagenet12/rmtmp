for i in `cat hostfile |cut -d ' ' -f 1`
do
  echo $i
  ssh $i "rm -rf /data/tmp/*"
done
