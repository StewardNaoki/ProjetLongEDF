set -e
num_epoch=350
num_batch=32
num_thread=8
alpha=0.0
beta=0.0
num_data=40000
# (16,1) (16,2) (32,2) (32,2)
num_neur=128
num_layer=4
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

num_neur=16
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

num_neur=32
num_layer=3
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta