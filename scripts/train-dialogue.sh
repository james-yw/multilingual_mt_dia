#!/bin/bash

data_path=/home/wyang/work/multilingual_mt_dia/data/data-bin
model_path=/home/wyang/work/multilingual_mt_dia/models/dialog



# langs=(es fr de pt)

# langs_comma="en,es,fr,de,pt"
# lang_pairs="en-es,en-fr,en-de,en-pt"


# data args
data_args="$data_path --skip-invalid-size-inputs-valid-test"




# model args
arch="--arch transformer"
extra_arch="--share-all-embeddings --attention-dropout 0.1 --dropout 0.1"
model_args="$arch $extra_arch"

# task args
task="--user-dir ../utils --task my_dialogue"
task_args="$task --source-lang src --target-lang tgt"

# training args
patience="--patience -1"
update_freq="--update-freq 1"
max_tokens="--max-tokens 16384"
max_epoch="--max-epoch 50"
max_update="--max-update 50000"
save_dir="--save-dir $model_path"
valid_after="--validate-after-updates 0"
training_args="$update_freq $max_tokens $max_epoch $patience $max_update $save_dir $valid_after"

# criterion args
criterion="--criterion label_smoothed_cross_entropy"
label_smoothing="--label-smoothing 0.1"
criterion_args="$criterion $label_smoothing"

# optimizer args
lr="--lr 0.0007"
optimizer="--optimizer adam"
weight_decay="--weight-decay 0.0"
adam_betas="--adam-betas (0.9,0.98)"
warmup_updates="--warmup-updates 4000"
warmup_init_lr="--warmup-init-lr 1e-07"
lr_scheduler="--lr-scheduler inverse_sqrt"
optimizer_args="$optimizer $adam_betas $lr_scheduler $warmup_init_lr $warmup_updates $lr $weight_decay"

# fp16 and distribute args
fp16="--fp16"
ddp_backend="--ddp-backend no_c10d"
fp16_init_scale="--fp16-init-scale 8"
fp16_scale_tolerance="--fp16-scale-tolerance 0"
fp16_args="$fp16 $ddp_backend $fp16_init_scale $fp16_scale_tolerance"

# logging args
log_interval="--log-interval 1"
keep_interval_updates="--keep-interval-updates 5"
save_interval_updates="--save-interval-updates 99999"
extra_logging="--no-progress-bar --validate-interval 1 --save-interval 5"
logging_args="$log_interval $save_interval_updates $keep_interval_updates $extra_logging"

# gpu agrs
gpu=2,3

if [ ! -d $model_path ]; then
    mkdir -p $model_path
fi

# CUDA_VISIBLE_DEVICES=$gpu fairseq-train $data_args  $model_args $task_args $training_args $criterion_args $optimizer_args $fp16_args $logging_args > $model_path/log.txt

checkpoint="checkpoint_best" # checkpoint... or checkpoint_last or checkpoint_best

beam="--beam 5"
# lenpen="--lenpen 0.6"
remove_bpe="--remove-bpe sentencepiece"
result_path="$model_path/result_average"

decoding_args="$beam $lenpen $remove_bpe --path $model_path/$checkpoint.pt"
if [ ! -d $result_path ]; then
    mkdir -p $result_path
fi

CUDA_VISIBLE_DEVICES=$gpu fairseq-generate $data_args $task_args $decoding_args --source-lang src --target-lang tgt > $result_path/generate-test-${checkpoint}.txt
o=$(mktemp) && cat $result_path/generate-test-${checkpoint}.txt | grep ^T | cut -f2 > $o
cat $result_path/generate-test-${checkpoint}.txt | grep ^H | cut -f3 | sacrebleu $o -m bleu --force > $result_path/bleu-${checkpoint}.txt
