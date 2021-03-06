set -e
num_epoch=1
num_batch=32
num_thread=1
# alpha=0.001
# beta=0.001
alpha=0.0
beta=0.0
num_data=5000
loss=PCL

num_neur=16
num_layer=2
python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss $loss --network CNN --dropout --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

# num_neur=16
# num_layer=2
# python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

# num_neur=32
# num_layer=1
# python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

# num_neur=32
# num_layer=2
# python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

# num_neur=64
# num_layer=1
# python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta

# num_neur=64
# num_layer=2
# python3 main_EDF.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log --num_json_max $num_data --loss MSE --network CNN --num_deep_layer $num_layer --num_neur $num_neur --alpha $alpha --beta $beta


