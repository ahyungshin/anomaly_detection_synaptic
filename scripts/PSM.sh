seed="1"
dataset="PSM"
channel="25"

cluster_t="1"
cluster_w="5"
pool_size_list=(1 3 5 7 10 12 15 20 25 30)
ploss_coeff_list=(0.1 0.2 0.5 1 2 5 10 20 50 100 125 150)

for pc in ${ploss_coeff_list[*]}; do
    for p in ${pool_size_list[*]}; do
                    python -u main.py \
                            --seed $seed \
                            --anormly_ratio 1 \
                            --num_epochs 1 \
                            --batch_size 128  \
                            --mode train \
                            --dataset $dataset  \
                            --data_path /data/pmy0792/dataset/AD_datasets/$dataset \
                            --input_c $channel    \
                            --output_c $channel \
                            --cluster_t $cluster_t \
                            --cluster_w $cluster_w \
                            --ploss_coeff $pc \
                            --pool_size $p \


                    python -u main.py \
                        --seed $seed \
                        --anormly_ratio 1  \
                        --num_epochs 1      \
                        --batch_size 128     \
                        --mode test    \
                        --dataset $dataset   \
                        --data_path /data/pmy0792/dataset/AD_datasets/$dataset  \
                        --input_c $channel    \
                        --output_c $channel  \
                        --pretrained_model 20 \
                        --ploss_coeff $pc \
                        --pool_size $p \

    done
done