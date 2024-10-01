#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh

if [ ! -d ./answers ]; then
    mkdir ./answers
fi

test_size=1510
datasets=(multilex_tiny multilex_short multilex_long eurlexsum)
declare -A summary_length=( ["multilex_tiny"]=25 ["multilex_short"]=130 ["multilex_long"]=650 ["eurlexsum"]=600 )
for dataset in ${datasets[@]}; do
    if [ -f results_$dataset.pickle ]; then
        continue
    fi

    conda activate facilex_caselaw
    if [ ! -d ./answers/$dataset ]; then
        mkdir ./answers/$dataset
    fi

    prompt_types=(basic detailed cod)
    for prompt_type in ${prompt_types[@]}; do
        for is_equal in 0; do
            python groq_functions.py $dataset deepinfra gemma $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
            python groq_functions.py $dataset groq gemma2 $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
            python groq_functions.py $dataset deepinfra mistral $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
            python groq_functions.py $dataset deepinfra llama3 $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
            # python groq_functions.py $dataset deepinfra mixtral $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
            # python groq_functions.py $dataset openai gpt-3.5 $prompt_type $test_size $is_equal ${summary_length[$dataset]} &
        done
    done
    wait
    # ################################
    # conda deactivate
    # conda activate facilex_caselaw
    # python eval.py $dataset
    # python eval_mover.py $dataset
    # wait
    # conda deactivate
done

# shutdown now