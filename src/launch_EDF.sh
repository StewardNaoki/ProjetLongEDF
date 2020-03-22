set -e
num_epoch=350
num_batch=32
num_thread=8
alpha=0.0
beta=0.0
num_data=40000

num_neur=32
num_layer=1
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

alpha=0.001
beta=0.001

num_neur=32
num_layer=1
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta