#!/bin/bash
SCRIPTS=/home/wyang/tools/mosesdecoder/scripts
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
SPM_TRAIN=/home/wyang/tools/fairseq/scripts/spm_train.py
SPM_ENCODE=/home/wyang/tools/fairseq/scripts/spm_encode.py

LANG_PAIR=(
    "en-zh"
    "en-de"
    "en-fr"
    "zh-en"
    "de-en"
    "fr-en"
)
LANGS=(en zh de fr)

TMP=/home/wyang/work/multilingual_mt_dia/data/tmp
ORI=/home/wyang/work/multilingual_mt_dia/data/ori
BPE=/home/wyang/work/multilingual_mt_dia/data/bpe
DATA_BIN=/home/wyang/work/multilingual_mt_dia/data/data-bin


rm -r $BPE
mkdir -p $BPE

rm -r $DATA_BIN
mkdir -p $DATA_BIN


VOCAB_SIZE=32000


python "$SPM_TRAIN" \
    --input=$TMP/train.all \
    --model_prefix=$BPE/spm.bpe \
    --vocab_size=$VOCAB_SIZE \
    --character_coverage=1.0 \
    --model_type=bpe \
    --num_threads=45 \
    --shuffle_input_sentence

for split in train valid test; do
    for lang in ${LANG_PAIR[@]}; do
        src=${lang:0:2}
        tgt=${lang:3:2}
        echo ${split} ${lang}
        python "$SPM_ENCODE" \
            --model $BPE/spm.bpe.model \
            --output_format=piece \
            --inputs $TMP/${split}.${lang}.${src} $TMP/${split}.${lang}.${tgt} \
            --outputs ${BPE}/${split}.${lang}.bpe.unclean.${src} ${BPE}/${split}.${lang}.bpe.unclean.${tgt}
        perl $CLEAN -ratio 1.5 ${BPE}/${split}.${lang}.bpe.unclean ${src} ${tgt} ${BPE}/${split}.${lang}.bpe 1 256
        rm ${BPE}/${split}.${lang}.bpe.unclean.*
    done
done

for split in train valid; do  
    echo ${split} dialogue
    python "$SPM_ENCODE" \
        --model $BPE/spm.bpe.model \
        --output_format=piece \
        --inputs $TMP/${split}.dialog-en.src $TMP/${split}.dialog-en.tgt \
        --outputs ${BPE}/${split}.dialog-en.bpe.src ${BPE}/${split}.dialog-en.bpe.tgt
    
    # perl $CLEAN -ratio 1.5 ${BPE}/${split}.dialog.bpe.unclean src tgt ${BPE}/${split}.dialog.bpe 1 256
    # rm ${BPE}/${split}.dialog.bpe.unclean.*
done

for split in test;do
    for lang in ${LANGS[@]};do
        echo ${split} dialogue
        python "$SPM_ENCODE" \
            --model $BPE/spm.bpe.model \
            --output_format=piece \
            --inputs $TMP/${split}.dialog-$lang.src $TMP/${split}.dialog-$lang.tgt \
            --outputs ${BPE}/${split}.dialog-$lang.bpe.src ${BPE}/${split}.dialog-$lang.bpe.tgt
        # perl $CLEAN -ratio 1.5 ${BPE}/${split}.dialog.bpe.unclean src tgt ${BPE}/${split}.dialog.bpe 1 256
        # rm ${BPE}/${split}.dialog.bpe.unclean.*
    done
done




cut -f1 $BPE/spm.bpe.vocab | tail -n +4 | sed "s/$/ 100/g" > $DATA_BIN/dict.txt
for lang in ${LANG_PAIR[@]}; do
    src=${lang:0:2}
    tgt=${lang:3:2}
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $BPE/train.${lang}.bpe \
        --validpref $BPE/valid.${lang}.bpe \
        --testpref $BPE/test.${lang}.bpe \
        --destdir $DATA_BIN \
        --srcdict $DATA_BIN/dict.txt \
        --tgtdict $DATA_BIN/dict.txt
done


fairseq-preprocess \
        --source-lang src \
        --target-lang tgt \
        --trainpref $BPE/train.dialog-en.bpe \
        --validpref $BPE/valid.dialog-en.bpe \
        --testpref $BPE/test.dialog-en.bpe \
        --destdir $DATA_BIN \
        --srcdict $DATA_BIN/dict.txt \
        --tgtdict $DATA_BIN/dict.txt

for lang in ${LANGS[@]}; do
    fairseq-preprocess \
            --source-lang src \
            --target-lang tgt \
            --testpref $BPE/test.dialog-$lang.bpe \
            --destdir $DATA_BIN/dialog-test-$lang/ \
            --srcdict $DATA_BIN/dict.txt \
            --tgtdict $DATA_BIN/dict.txt
done