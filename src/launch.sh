set -e
python3 main.py --epoch 1 --batch 100 --valpct 0.2 --num_thread 1 --l2_reg 0.001 --num_var 5 --num_const 8 --num_prob 10000 --log --custom_loss --alpha 0.01 --num_deep_layer 2