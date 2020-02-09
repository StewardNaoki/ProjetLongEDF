set -e
num_epoch=400
num_batch=100
num_thread=8

num_var=2
num_const=2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE              --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 10000 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 0.1   --num_deep_layer 2



num_var=5
num_const=8

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE              --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 10000 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 0.1   --num_deep_layer 2



num_var=20
num_const=40

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE              --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 10000 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 0.1   --num_deep_layer 2