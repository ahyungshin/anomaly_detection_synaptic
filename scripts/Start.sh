# # MSL
python -u main.py --seed 0 --anormly_ratio 1 --num_epochs 1   --batch_size 128  --mode train --dataset MSL  --data_path /data/pmy0792/dataset/AD_datasets/MSL --input_c 55    --output_c 55 --cluster_t 1 --cluster_w 10
python -u main.py --seed 0 --anormly_ratio 1  --num_epochs 1      --batch_size 128     --mode test    --dataset MSL   --data_path /data/pmy0792/dataset/AD_datasets/MSL  --input_c 55    --output_c 55  --pretrained_model 20

# PSM
python -u main.py --seed 1 --anormly_ratio 1 --num_epochs 1    --batch_size 128  --mode train --dataset PSM  --data_path /data/pmy0792/dataset/AD_datasets/PSM --input_c 25    --output_c 25 --cluster_t 1 --cluster_w 0.1
python -u main.py --seed 1 --anormly_ratio 1  --num_epochs 1       --batch_size 128     --mode test    --dataset PSM   --data_path /data/pmy0792/dataset/AD_datasets/PSM  --input_c 25    --output_c 25  --pretrained_model 20

# SMAP
python -u main.py --seed 35 --anormly_ratio 1 --num_epochs 1   --batch_size 128  --mode train --dataset SMAP  --data_path /data/pmy0792/dataset/AD_datasets/SMAP --input_c 25    --output_c 25 --cluster_t 0.1 --cluster_w 100
python -u main.py --seed 35 --anormly_ratio 1  --num_epochs 1        --batch_size 128     --mode test    --dataset SMAP   --data_path /data/pmy0792/dataset/AD_datasets/SMAP  --input_c 25    --output_c 25  --pretrained_model 20

# SMD
python -u main.py --seed 0 --anormly_ratio 0.5 --num_epochs 1   --batch_size 128  --mode train --dataset SMD  --data_path /data/pmy0792/dataset/AD_datasets/SMD   --input_c 38 --cluster_t 0.1 --cluster_w 10
python -u main.py --seed 0 --anormly_ratio 0.5 --num_epochs 1   --batch_size 128     --mode test    --dataset SMD   --data_path /data/pmy0792/dataset/AD_datasets/SMD     --input_c 38     --pretrained_model 20

# SWAT
python -u main.py --seed 9 --anormly_ratio 1 --num_epochs 1   --batch_size 128  --mode train --dataset SWAT  --data_path /data/pmy0792/dataset/AD_datasets/SWAT --input_c 51    --output_c 51 --cluster_t 1 --cluster_w 5
python -u main.py --seed 9 --anormly_ratio 1  --num_epochs 1      --batch_size 128     --mode test    --dataset SWAT   --data_path /data/pmy0792/dataset/AD_datasets/SWAT  --input_c 51    --output_c 51  --pretrained_model 20

# WADI
python -u main.py --seed 1 --anormly_ratio 1 --num_epochs 1   --batch_size 128  --mode train --dataset WADI  --data_path /data/pmy0792/dataset/AD_datasets/WADI --input_c 123    --output_c 123 --cluster_t 1 --cluster_w 5
python -u main.py --seed 1 --anormly_ratio 1  --num_epochs 1      --batch_size 128     --mode test    --dataset WADI   --data_path /data/pmy0792/dataset/AD_datasets/WADI  --input_c 123    --output_c 123  --pretrained_model 20
