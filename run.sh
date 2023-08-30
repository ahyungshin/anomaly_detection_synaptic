#!/bin/bash

fe=20
pc=1
a=0.0001
epoch=1

# dataset
dataset_list=(MSL PSM SMAP SMD SWAT WADI)
channel_list=(55 25 25 38 51 123)
anormly_ratio_list=(1 1 1 0.5 0.5 1)
seed_list=(127 128 105 115 114 128)

# hyperparameters
bs_list=(128 128 128 128 128 64)
cluster_t_list=(0.1 0.1 0.1 0.1 0.1 0.1)
cluster_w_list=(1 1 1 1 1 1)
pool_list=(20 10 15 10 10 10)
prompt_list=(10 10 20 10 10 10)

for i in "${!dataset_list[@]}"; do
        python -u main.py \
                --seed ${seed_list[i]} \
                --anormly_ratio ${anormly_ratio_list[i]} \
                --num_epochs $epoch \
                --batch_size ${bs_list[i]}  \
                --mode train \
                --dataset ${dataset_list[i]}  \
                --data_path /path/to/dataset/${dataset_list[i]} \
                --input_c ${channel_list[i]}    \
                --output_c ${channel_list[i]} \
                --cluster_t ${cluster_t_list[i]} \
                --consist_coeff ${cluster_w_list[i]} \
                --fe_epochs $fe \
                --ploss_coeff $pc \
                --ae_lr $a \
                --pool_size ${pool_list[i]} \
                --prompt_num ${prompt_list[i]} \
                --use_p_noise

        python -u main.py \
                --seed ${seed_list[i]} \
                --anormly_ratio ${anormly_ratio_list[i]}  \
                --num_epochs $epoch      \
                --batch_size ${bs_list[i]}     \
                --mode test    \
                --dataset ${dataset_list[i]}   \
                --data_path /path/to/dataset/${dataset_list[i]}  \
                --input_c ${channel_list[i]}    \
                --output_c ${channel_list[i]}  \
                --pretrained_model 20 \
                --fe_epochs $fe \
                --ploss_coeff $pc \
                --ae_lr $a \
                --pool_size ${pool_list[i]} \
                --prompt_num ${prompt_list[i]} \
                --use_p_noise
done
