for i in {0..23}
do
  echo "group{"
  x=$[$i*2]
  echo "start: $x"
  x=$[$x+2]
  echo "end:$x"
  echo "}"
done
