export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
f=57
seq_len=96
for pred_len in 24 48 96 168
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_$seq_len_$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --top_k 5 \
  --batch_size 16 \
  --des 'Exp_PS10_withsamehyperparameter' \
  --itr 1 \
  --ps_lambda 10.0 \
  --use_ps_loss 0 \
  --head_or_projection 1
done

