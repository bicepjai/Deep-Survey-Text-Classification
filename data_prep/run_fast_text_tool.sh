#!/bin/bash

echo "running fasttext .... "
input_file="/home/bicepjai/Projects/dsotc/data_prep/processed/stage1/all_text.txt"
output_dir="/home/bicepjai/Projects/dsotc/data_prep/processed/stage1/pretrained_word_vectors"

FASTTEXT=/home/bicepjai/Programs/fasttext/fasttext

run_fast_text_sg () {
   echo "$FASTTEXT skipgram -minCount 1 -dim $1 -epoch $2 -input $3 -output $4/ft_sg_${dim}d_${epoch}e"
   $FASTTEXT skipgram -minCount 1 -dim $1 -epoch $2 -input $3 -output $4/ft_sg_{$dim}d_{$epoch}e
}

run_fast_text_cbow () {
   echo "$FASTTEXT cbow -minCount 1 -dim $1 -epoch $2 -input $3 -output $4/ft_cbow_${dim}d_${epoch}e"
   $FASTTEXT cbow -minCount 1 -dim $1 -epoch $2 -input $3 -output $4/ft_sg_{$dim}d_{$epoch}e
}

epoch=20

# skipgram runs
dim=100
run_fast_text_sg $dim $epoch $input_file $output_dir

dim=200
run_fast_text_sg $dim $epoch $input_file $output_dir

dim=300
run_fast_text_sg $dim $epoch $input_file $output_dir

# cbow runs
dim=100
run_fast_text_cbow $dim $epoch $input_file $output_dir

dim=200
run_fast_text_cbow $dim $epoch $input_file $output_dir

dim=300
run_fast_text_cbow $epoch $dim $input_file $output_dir
