export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/othermodel" ]; then
    mkdir ./logs/othermodel
fi
model_name=TimeXer
f=57
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_168_168 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 168 \
  --e_layers 3 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 1024 \
  --batch_size 4 \
  --itr 1
