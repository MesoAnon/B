#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

dataset=eurlexsum_validation
python eval.py $dataset 0 > log_$dataset.txt

# dataset=eurlexsum_test
# python eval.py $dataset 0 > log_$dataset.txt

dataset=multilex_long
python eval.py $dataset 0 > log_$dataset.txt