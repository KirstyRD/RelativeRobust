#!/bin/bash
folder=data/$6

#output results fie
outfile_rob=$1
outfile_adv=$2
from=$3
to=$4
sequ=$(expr $4 - $3 + 1)
radius=$5

# create folders
for x in $(seq $sequ)
do
    y=$(expr  $x + $from - 1)
    rm -rf $folder/$y
    rm -rf $folder/${y}a
done

./create_folders.sh $from $to $folder

for x in $(seq $sequ)
do
    y=$(expr $x + $from - 1)
    #run DLV
    echo $y
    python2 DLV.py $y 20 10 20 $radius $folder
    
    #check folder
    adv="F"
    class="N"
    #get right string length
    len_number=${#y}
    len_folder=${#folder}
    len=$(expr  3 + $len_number + $len_folder)
    len1=$len
    len2=$len+18
    #loop through adversarial folder
    for file in $folder/${y}a/*
    do
#	echo $file
#	echo "${file:$len1:3}"
#	echo "${file:$len2:3}"
	if [ "${file:$len1:1}" != "*" ] && [ "${file:$len1:3}" != "${file:$len2:3}" ]
	then
	    adv="T"
	    advclass="${file:$len2:3}"
            class="${file:$len1:3}"
	else
	    adv="F"
	    class="${file:$len1:3}"
	fi
    done
    
    if [ $adv == "F" ]
    then
	#get number of images                                                                                                                  
        num="$(ls $folder/$y | wc -l)"
	if [ "$num" != 0 ]
	   then
	   echo "$y $class $num" >> $outfile_rob
	else
	    echo "misclassified"
	fi
	rm -r $folder/${y}a
	
    else
	echo "$y $class $advclass" >> $outfile_adv
        rm -r $folder/$y
    fi
done