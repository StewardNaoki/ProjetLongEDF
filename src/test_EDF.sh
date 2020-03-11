set -e
num_epoch=350
num_batch=128
num_thread=1
alpha=0.1
beta=0.1

num_neur=16
num_layer=1
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max 1000 --loss MSE --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta


