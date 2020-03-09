set -e
num_epoch=350
num_batch=128
num_thread=8
alpha=10
beta=10

num_neur=16
num_layer=1
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max 40000 --loss MSE --num_deep_layer $num_layer--num_neur $num_neur --alpha $alpha --beta $beta

num_neur=16
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max 40000 --loss MSE --num_deep_layer $num_layer--num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max 40000 --loss MSE --num_deep_layer $num_layer--num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=3
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max 40000 --loss MSE --num_deep_layer $num_layer--num_neur $num_neur --alpha $alpha --beta $beta