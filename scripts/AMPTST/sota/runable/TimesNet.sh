export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
f=57
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 168 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


