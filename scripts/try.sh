#!/bin/bash

LANG_PAIR=(
    "en-zh"
    "en-de"
    "en-fr"
    "zh-en"
    "de-en"
    "fr-en"
)
text=/home/wyang/work/multilingual_mt_dia/data/2017-01-trnted/texts/
ori=/home/wyang/work/multilingual_mt_dia/data/ori/ 
mkdir $ori

for split in train test; do
    if [ $split = "train" ]; then
        echo 12345
    fi
done
# for lang in ${LANG_PAIR[@]}; do
#     src=${lang:0:2} #str:start:length
#     tgt=${lang:3:2}
#     echo $src
#     echo $tgt
#     cp $text/$src/$tgt/$lang.tgz $ori
    
# done





