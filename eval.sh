#!/usr/bin/env bash
source model/misc/utils.sh
# start of configurations ->
topk=1
ds_name="ut-zap50k"
model_path=/path/to/model  # e.g. ./mitstates2_best/best.state
data_path=/path/to/data
# <- end of configurations

if [[ $ds_name = "ut-zap50k" ]]; then
    req_mem=2500
else
    req_mem=18000
fi

command_str="REQ_MEM=${req_mem} python eval.py \
    --cuda \
    --dataset ${ds_name} \
    --data_path ${data_path} \
    --split natural-split \
    --pre_feat \
    --no-pbar \
    --test_only \
    --model_path ${model_path} \
    --topk ${topk}"

bash -c "$command_str"
