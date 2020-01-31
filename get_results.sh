#!/bin/bash
from=$1
to=$2

mkdir data/cifar_r1

./gen_both.sh r1_robust.csv r1_adversarial.csv $from $to 128 cifar_r1
./get_morerob.sh r1_adversarial.csv r1_adversarial_tflite.csv data/cifar_r1
./get_rel_rob.sh r1_robust.csv r1_robust_tflite.csv data/cifar_r1
