seq_len=168
model_name=DLinear

root_path=./dataset/mydata_v1/
data_path=h57.csv
model_id_name=h57
data_name=custom

random_seed=2021
for pred_len in 24 168
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 57\
      --dropout 0.2\
      --train_epochs 20\
      --patience 10\
      --itr 1 --learning_rate 0.001
done