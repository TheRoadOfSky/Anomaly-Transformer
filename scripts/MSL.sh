export CUDA_VISIBLE_DEVICES=0

#python main.py --anormly_ratio 4 --num_epochs 10   --batch_size 16  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55  --model_d 128 --ff_d 128
python main.py --anormly_ratio 4  --num_epochs 10      --batch_size 16     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20 --model_d 128 --ff_d 128




