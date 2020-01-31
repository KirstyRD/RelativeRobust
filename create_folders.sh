from=$1
to=$2
sequ=$(expr $2 - $1 + 1)
folder=$3

for y in $(seq $sequ)
do
	x=$(expr $y + $from - 1)
	mkdir $folder/$x
	mkdir $folder/${x}a
done
