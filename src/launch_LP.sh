set -e
num_epoch=350
num_batch=128
num_thread=8

num_var=2
num_const=2

# python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE  --alpha 10       --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 100 --num_deep_layer 2

# python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 10   --num_deep_layer 2



num_var=5
num_const=8

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss MSE --alpha 10 --num_neur 256 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 100 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 10 --num_neur 256 --num_deep_layer 2



num_var=20
num_const=10

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE --alpha 10 --num_neur 256 --dropout --num_deep_layer 4

# python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 10000 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 10 --num_neur 256  --dropout --num_deep_layer 4


num_var=20
num_const=40

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log  --loss MSE --alpha 10 --num_neur 256 --dropout --num_deep_layer 4

# python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss PCL --alpha 10000 --num_deep_layer 2

python3 main.py --epoch $num_epoch --batch $num_batch --valpct 0.2 --num_thread $num_thread --l2_reg 0.001 --num_var $num_var --num_const $num_const --num_prob 100000 --log --loss GCL --alpha 10 --num_neur 256  --dropout --num_deep_layer 4