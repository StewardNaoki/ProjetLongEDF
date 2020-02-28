set -e
# cd ../ && ./clear_log.sh
# cd ../src
python3 main_EDF.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --log --num_json_max 10 --num_deep_layer 1 --log --num_neur 128 --loss MSE
# python3 main_EDF.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --log --num_json_max 10 --num_deep_layer 1 --num_neur 128 --loss GCL
# python3 main_EDF.py --epoch 1 --batch 100 --valpct 0.2 --l2_reg 0.001 --log --num_json_max 10 --num_deep_layer 1 --num_neur 128 --loss PCL


