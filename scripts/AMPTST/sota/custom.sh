# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather_with_nonTST" ]; then
    mkdir ./logs/weather_with_nonTST
fi
pred_len=168
model_name=TSTDLSTM

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2025
for seq_len in 168
do
    python -u run_longExp.py \
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
      --enc_in 21\
      --dropout 0.2\
      --des exp \
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/weather_with_nonTST/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done