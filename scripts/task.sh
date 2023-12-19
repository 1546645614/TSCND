### ETTh1
python -u run.py --model TSCNDOrigin --data ETTh1 --features S --seq_len 336 --label_len 24 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --hid_dim 64

python -u run.py --model TSCNDOrigin --data ETTh1 --features S --seq_len 336 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --hid_dim 64

python -u run.py --model TSCNDOrigin --data ETTh1 --features S --seq_len 512 --label_len 24 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --hid_dim 64 --dropout 0.4

python -u run.py --model TSCNDOrigin --data ETTh1 --features S --seq_len 512 --label_len 24 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --hid_dim 64 --dropout 0.3

python -u run.py --model TSCNDOrigin --data ETTh1 --features S --seq_len 720 --label_len 24 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --hid_dim 64 --dropout 0.3

### ETTh2
python -u run.py --model TSCNDOrigin --data ETTh2 --features S --seq_len 336 --label_len 24 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ETTh2 --features S --seq_len 512 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ETTh2 --features S --seq_len 512 --label_len 24 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --dropout 0.2

python -u run.py --model TSCNDOrigin --data ETTh2 --features S --seq_len 512 --label_len 24 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --dropout 0.2

python -u run.py --model TSCNDOrigin --data ETTh2 --features S --seq_len 512 --label_len 24 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --dropout 0.5

### ECL
python -u run.py --model TSCNDOrigin --data ECL --features S --seq_len 512 --label_len 24 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ECL --features S --seq_len 512 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ECL --features S --seq_len 512 --label_len 24 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ECL --features S --seq_len 336 --label_len 24 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data ECL --features S --seq_len 336 --label_len 24 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

### WTH
python -u run.py --model TSCNDOrigin --data WTH --features S --seq_len 336 --label_len 24 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data WTH --features S --seq_len 168 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001

python -u run.py --model TSCNDOrigin --data WTH --features S --seq_len 168 --label_len 24 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --dropout 0.3

python -u run.py --model TSCNDOrigin --data WTH --features S --seq_len 168 --label_len 24 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001 --dropout 0.3

python -u run.py --model TSCNDOrigin --data WTH --features S --seq_len 336 --label_len 24 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.001
