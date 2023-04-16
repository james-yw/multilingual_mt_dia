#!/bin/bash
UTILS=/home/wyang/work/multilingual_mt_dia/utils

SCRIPTS=/home/wyang/tools/mosesdecoder/scripts
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl


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
DIALOG=/home/wyang/work/multilingual_mt_dia/data/dialog_data/DailyDialog

rm -r $TMP
mkdir -p $TMP
# if [ ! -d "$TMP"] || [ ! -d "$ORI"]; then
#     mkdir $TMP
#     mkdir $ORI
# fi


#  Prepare translation data
for lang in ${LANG_PAIR[@]}; do
    src=${lang:0:2} #str:start:length
    tgt=${lang:3:2}
    
    # Input & clean
    for l in $src $tgt; do
        cd $ORI
        tar zxvf $lang.tgz
        cd ..
        f=train.tags.$lang.$l

        cat $ORI/$lang/$f \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > $TMP/$f
    done

    # lowercase
    for l in $src $tgt; do
        perl $LC < $TMP/train.tags.$lang.$l > $TMP/train.$lang.$l
        rm $TMP/train.tags.$lang.$l
    done

    # pre-processing valid & test data
    echo "pre-processing valid & test data..."
    for l in $src $tgt; do
        for o in `ls $ORI/$lang/IWSLT17.TED*.$l.xml`; do
            fname=${o##*/}
            f=$TMP/${fname%.*}
            echo $o $f
            grep '<seg id' $o | \
                sed -e 's/<seg id="[0-9]*">\s*//g' | \
                sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" | \
            perl $LC > $f
            echo ""
        done
    done

    echo "creating train, valid, test..."
    for l in $src $tgt; do
        mv $TMP/train.$lang.$l $TMP/train-valid.$lang.$l
        awk '{if (NR%23 == 0)  print $0; }' $TMP/train-valid.$lang.$l > $TMP/valid.$lang.$l
        awk '{if (NR%23 != 0)  print $0; }' $TMP/train-valid.$lang.$l > $TMP/train.$lang.$l
        rm $TMP/train-valid.$lang.$l 
        cat $TMP/IWSLT17.TED.dev2010.$lang.$l \
            $TMP/IWSLT17.TED.tst2010.$lang.$l \
            $TMP/IWSLT17.TED.tst2011.$lang.$l \
            $TMP/IWSLT17.TED.tst2012.$lang.$l \
            $TMP/IWSLT17.TED.tst2013.$lang.$l \
            $TMP/IWSLT17.TED.tst2014.$lang.$l \
            $TMP/IWSLT17.TED.tst2015.$lang.$l \
            > $TMP/test.$lang.$l
        rm $TMP/IWSLT17.TED*.$lang.$l 
    done
done

# Prepare dialogue data
python $UTILS/dialog_data_process.py $DIALOG

for split in train valid; do
    for l in src tgt; do
        cat $DIALOG/$l-$split.txt > $TMP/$split.dialog-en.$l
    done
done
for split in test; do
    for l in src tgt; do
        for lang in ${LANGS[@]};do
            cat $DIALOG/$l-$split-$lang.txt > $TMP/$split.dialog-$lang.$l
        done
    done
done


echo "counting..."

for lang in ${LANG_PAIR[@]}; do
    src=${lang:0:2}
    tgt=${lang:3:2}
    for split in train valid test; do
        for l in $src $tgt; do
            wc -l $TMP/$split.$lang.$l >> $TMP/data.log
        done
    done
done


for split in train valid; do
    for l in src tgt; do  
        wc -l $TMP/$split.dialog-en.$l >> $TMP/data.log 
    done
done
for split in test; do
    for l in src tgt; do
        for lang in ${LANGS[@]};do
            wc -l $TMP/$split.dialog-$lang.$l >> $TMP/data.log
        done
    done
done


TRAIN=$TMP/train.all
for lang in ${LANG_PAIR[@]}; do
    src=${lang:0:2} #str:start:length
    tgt=${lang:3:2}
    for l in $src $tgt; do
        cat $TMP/train.$lang.$l >> $TRAIN
    done
done

for l in src tgt; do
    cat  $TMP/train.dialog-en.$l >> $TRAIN
done


wc -l $TMP/train.all >> $TMP/data.log


