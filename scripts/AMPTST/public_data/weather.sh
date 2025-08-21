export CUDA_VISIBLE_DEVICES=0
model_name=AMPTST

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
train_epochs=20
patience=10
f=21
data_path=weather.csv
des=CM
for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path $data_path \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --checkpoints ./checkpoints/publicdata \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --n_heads 8 \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --top_k 5 \
    --enc_in $f \
    --dec_in $f \
    --c_out $f \
    --des $des \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --moving_avg 25 \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --channel_independence 0 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --pf 0
done