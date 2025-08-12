random_seed=2025
seq_len=168
model_name=Informer

root_path=./dataset/mydata_v1/
data_path=h57.csv
model_id_name=h57
data_name=custom
f=57
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
      --enc_in $f \
      --dec_in $f \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --c_out $f \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 20\
      --patience 10\
      --itr 1 --learning_rate 0.001
done