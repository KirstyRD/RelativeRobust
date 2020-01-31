#!/bin/bash


#number of images
infile=$1
#output results fie
outfile=$2
echo "index,class,adv_class,qnn_class,robust" > $outfile
folder=$3

while IFS= read -r line
do
    IFS=' ' # comma is set as delimiter
    read -ra ADDR <<< "$line" # line is read into an array
    index=${ADDR[0]}
    class=${ADDR[1]}
    advclass=${ADDR[2]}
    let num-=1
    echo "$index, $class, $advclass"
    python3 run_tf_single.py $folder $index $class $advclass $outfile
done <"$infile"
