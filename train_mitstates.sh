#!/usr/bin/env bash
source model/misc/utils.sh
data_path="/path/to/data"
lca=10.0
ica=10.0
ta=20.0
ra=5.0
ita=5.0
step_a=0.01
lr=0.000005
batch_size=128
rank_margin=0.5
class_margin=1.0
seed=0

ds_name=mitstates
if [[ $ds_name = "ut-zap50k" ]]; then
    save_model_to="snapshots_utzap50k"
else
    save_model_to="snapshots_mit"
fi

lws="'{\"lca\": ${lca}, \"lco\": ${lca}, \"ica\": ${ica}, \"ico\": ${ica}, "
lws="${lws}\"ra\": ${ra}, \"ro\": ${ra}, \"step_a\": ${step_a}, \"step_o\": ${step_a}, "
lws="${lws}""\"ta\": ${ta}, \"to\": ${ta}, \"ita\": ${ita}, \"ito\": ${ita}, \"cmargin\": ${class_margin}}'"

echo ${lws}
str="python train.py \
    --cuda \
    --lr ${lr} \
    --data_path ${data_path} \
    --meta_samples 1.0 \
    --dataset ${ds_name} \
    --split natural-split \
    --batch_size ${batch_size} \
    --patience 20 \
    --max_epoch 200 \
    --pre_feat \
    --seed ${seed} \
    --kneg 1 \
    --save_model_to ${save_model_to} \
    --no-pbar \
    --rank_margin ${rank_margin} \
    --loss_weights ${lws}"
echo ${str}
bash -c ${str}