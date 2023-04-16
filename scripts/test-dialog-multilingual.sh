#!/bin/bash
for tag in en de zh fr; do

    data_path=/home/wyang/work/multilingual_mt_dia/data/data-bin/dialog-test-$tag
    model_path=/home/wyang/work/multilingual_mt_dia/models/union_many_to_many
    # data args

    data_args="$data_path --skip-invalid-size-inputs-valid-test"


    # task args
    task="--user-dir ../utils --task my_dialogue"
    task_args="$task --source-lang src --target-lang tgt"


    # gpu agrs
    gpu=2,3

    if [ ! -d $model_path ]; then
        mkdir -p $model_path
    fi


    checkpoint="checkpoint100" # checkpoint... or checkpoint_last or checkpoint_best


    beam="--beam 5"
    # lenpen="--lenpen 0.6"
    remove_bpe="--remove-bpe sentencepiece"
    result_path="$model_path/result_$tag"

    decoding_args="$beam $lenpen $remove_bpe --path $model_path/$checkpoint.pt"
    if [ ! -d $result_path ]; then
        mkdir -p $result_path
    fi


    CUDA_VISIBLE_DEVICES=$gpu fairseq-generate $data_args $task_args $decoding_args --source-lang src --target-lang tgt > $result_path/generate-test-${checkpoint}.txt
    o=$(mktemp) && cat $result_path/generate-test-${checkpoint}.txt | grep ^T | cut -f2 > $o
    cat $result_path/generate-test-${checkpoint}.txt | grep ^H | cut -f3 | sacrebleu $o -m bleu --force > $result_path/bleu-${checkpoint}.txt

done