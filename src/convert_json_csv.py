
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os

N = 94
P_MIN = 1200
P_MAX = 7000
HOUSE_MAX = 9000
V_MAX = 40000
DT= 0.5

PATH_DATA = "./../DATA/"
OUTPUT_VECTOR_SIZE = 94

def generate_csv(path_json_dir, input_file_path , num_json_max ):
    print("Opening json folder: ", path_json_dir)
    dict_input = {"house_pmax": [], "vehicle_pmax": [],"vehicle_energy_need": [],"house_cons": [], "opt_charging_profile_step1": []}

    num_json = 0
    with tqdm(total=num_json_max) as pbar:
        for filename in os.listdir(path_json_dir):
            # print("Filename: ",filename)
            if num_json >= num_json_max:
                break

            pbar.update(1)
            pbar.set_description("Filename {}".format(filename))

            house_pmax=[]
            vehicle_pmax=[]
            house_cons=[]
            vehicle_energy_need=[]
            opt_charging_profile_step1=[]

            df=pd.read_json(path_json_dir+filename)

            house_pmax.append(df['house_pmax'][0])
            vehicle_pmax.append(df['vehicle_pmax'][0])
            vehicle_energy_need.append(df['vehicle_energy_need'][0])

            for i in range(OUTPUT_VECTOR_SIZE):
                house_cons.append(df['house_cons'][i])
                opt_charging_profile_step1.append(df['opt_charging_profile_step1'][i])

            dict_input["house_pmax"].append(house_pmax)
            dict_input["vehicle_pmax"].append(vehicle_pmax)
            dict_input["vehicle_energy_need"].append(vehicle_energy_need)

            dict_input["house_cons"].append(house_cons)
            ### Solution
            dict_input["opt_charging_profile_step1"].append(opt_charging_profile_step1)

            num_json+= 1
    # print(dict_input.head())
    df_input = pd.DataFrame(dict_input)
    df_input.to_csv(input_file_path, index=False)







def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="input",
                        help="file name (default: input)")
    parser.add_argument("--path_json_dir", type=str, default="./../DATA/json_data/",
                        help="Path to json files (default: ./../DATA/json_data/")
    parser.add_argument("--num_json_max", type=int, default=10,
                        help="Maximum number of json file to load in csv (default: 10)")
    # parser.add_argument("--num_prob", type=int, default=10,
    #                     help="number of problems to generate (default: 10)")

    args = parser.parse_args()

    file_path = PATH_DATA + args.file_name+"NJM{}".format(args.num_json_max) + ".csv"
    generate_csv(args.path_json_dir, file_path, args.num_json_max )
    # generate_csv(file_path, args.num_var,
    #              args.num_const, args.num_prob)


if __name__ == "__main__":
    main()