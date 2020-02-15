#!/bin/bash                                                                                          
from=$1
to=$2
manip=$3 #256 128 64 32 16 8 4
r=$4 #10 20                                                                                          

folder="r${r}_m${manip}"
file_robust1="r${r}_m${manip}_robust.csv"
file_robust2="r${r}_m${manip}_robust_tflite.csv"
file_adv1="r${r}_m${manip}_adv.csv"
file_adv2="r${r}_m${manip}_adv_tflite.csv"

mkdir data/$folder

./gen_both.sh $file_robust1 $file_adv1 $from $to $manip $r $folder
./get_morerob.sh $file_adv1 $file_adv2 data/$folder                                                 
./get_rel_rob.sh $file_robust1 $file_robust2 data/$folder
