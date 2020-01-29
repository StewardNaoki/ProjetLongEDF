set -e
python3 main.py --epoch 20 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8 --log --custom_loss --num_deep_layer 2