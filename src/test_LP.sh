set -e

python3 main_LP.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8 --num_prob 10 --loss MSE --alpha 0.1 --log  --num_deep_layer 1
python3 main_LP.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8 --num_prob 10 --loss GCL --alpha 0.1  --log   --num_deep_layer 2
python3 main_LP.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8 --num_prob 10 --loss PCL --alpha 0.1  --log   --num_deep_layer 2
# python3 main.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8                                           --log  --num_deep_layer 3
# python3 main.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --num_var 5 --num_const 8 --num_prob 10 --loss GCL --alpha 0.1  --log  --num_deep_layer 3

# cd ../ && ./clear_log.sh
