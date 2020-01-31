#!/bin/bash


#number of images
infile=$1
#output results fie
outfile=$2
folder=$3

echo "index,class,num_images,num_correct,robust" > $outfile
#folder=data/cifar10_pic_relrob_r1


while IFS= read -r line
do
    IFS=' ' # comma is set as delimiter
    read -ra ADDR <<< "$line" # line is read into an array
    index=${ADDR[0]}
    class=${ADDR[1]}
    num="$(ls $folder/$index | wc -l)"
    let num-=1
    echo "$index, $class, $num, $folder, $outfile"
    if [ $class != "*" ]
    then
       python3 run_tf.py $index $class $num $folder $outfile
    fi
done <"$infile"


#get number of images
#		num="$(ls $folder/$x | wc -l)"
		#run TFLite
#		echo "ROBUST: $x $class $num" >> $outprint
#		python3 run_tf.py $x $class $num $outfile >> $outprint
#		rm -r $folder/$x

#    else
#		echo "non robust image" >> $outprint
#		echo $advfile >> $outprint
#		rm -r $folder/$x
#		rm -r $folder/${x}a
