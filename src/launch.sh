set -e
python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --num_deep_layer 1
python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --num_deep_layer 0

python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --custom_loss --alpha 0.01 --num_deep_layer 1
python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --custom_loss --alpha 0.1 --num_deep_layer 1

python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --custom_loss --alpha 0.01 --num_deep_layer 2
python3 main.py --epoch 500 --batch 100 --valpct 0.2 --num_thread 8 --l2_reg 0.001 --num_var 2 --num_const 2 --num_prob 100000 --log --custom_loss --alpha 0.1 --num_deep_layer 2